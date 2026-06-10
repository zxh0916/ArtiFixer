#!/usr/bin/env python3
"""Run ArtiFixer on arbitrary-length videos by chunking and stitching.

This is a pragmatic video adapter for ArtiFixer reconstructed_colmap inference:
  1. Decode input video into fixed-size chunks (default 81 frames).
  2. Build minimal reconstructed_colmap-style scene inputs for each chunk.
  3. Run model_eval.run_inference over all chunks.
  4. Trim padded tail frames, concatenate enhanced chunks, and optionally create a
     side-by-side original/enhanced comparison.

It does not replace true scene-side ArtiFixer inputs; it uses the source frames as
renders/context plus white opacity, which is useful for arbitrary video smoke tests
and visual enhancement passes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ChunkSpec:
    scene_id: str
    start: int
    actual_frames: int
    padded_frames: int


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def capture(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def ffprobe_json(video: Path) -> dict:
    raw = capture([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration",
        "-of", "json", str(video),
    ])
    data = json.loads(raw)
    if not data.get("streams"):
        raise RuntimeError(f"No video stream found in {video}")
    return data["streams"][0]


def parse_fps(rate: str) -> float:
    if "/" in rate:
        num, den = rate.split("/", 1)
        return float(num) / float(den)
    return float(rate)


def frame_count(video: Path) -> int:
    stream = ffprobe_json(video)
    nb = stream.get("nb_frames")
    if nb and nb != "N/A":
        return int(nb)
    # fallback: count decoded frames
    out = capture([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(video),
    ])
    return int(out)


def extract_chunk_frames(video: Path, images_dir: Path, start: int, count: int) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    for p in images_dir.glob("*.png"):
        p.unlink()
    # select absolute frame indices in [start, start+count). setpts keeps encoding simple.
    expr = f"select='between(n,{start},{start + count - 1})',setpts=N/FRAME_RATE/TB"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf", expr, "-vsync", "0", str(images_dir / "%05d.png")])
    files = sorted(images_dir.glob("*.png"))
    # ffmpeg's image2 pattern starts at 00001 unless start_number is set; normalize to 00000.
    tmp = images_dir.parent / f".{images_dir.name}_renaming"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    for i, p in enumerate(files):
        shutil.move(str(p), tmp / f"{i:05d}.png")
    for p in sorted(tmp.glob("*.png")):
        shutil.move(str(p), images_dir / p.name)
    tmp.rmdir()
    return len(list(images_dir.glob("*.png")))


def make_prompt_h5(path: Path, caption: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        # ArtiFixer loader expects a bfloat16 embedding stored as uint16.
        ds = f.create_dataset("00000", data=np.zeros((1, 4096), dtype=np.uint16))
        ds.attrs["caption"] = caption


def build_chunk_scene(
    *,
    video: Path,
    prep_root: Path,
    spec: ChunkSpec,
    width: int,
    height: int,
    num_views: int,
    fov_degrees: float,
    caption: str,
) -> dict:
    scene_root = prep_root / spec.scene_id
    images = scene_root / "images"
    renders = scene_root / "renders"
    opacity = scene_root / "opacity"
    captions = scene_root / "captions"
    for d in (images, renders, opacity, captions):
        d.mkdir(parents=True, exist_ok=True)

    decoded = extract_chunk_frames(video, images, spec.start, spec.actual_frames)
    if decoded != spec.actual_frames:
        raise RuntimeError(f"Chunk {spec.scene_id}: expected {spec.actual_frames} decoded frames, got {decoded}")

    # Pad the final chunk by duplicating its last real frame so ArtiFixer sees a fixed chunk length.
    last_frame = images / f"{spec.actual_frames - 1:05d}.png"
    for i in range(spec.actual_frames, spec.padded_frames):
        shutil.copy2(last_frame, images / f"{i:05d}.png")

    for i in range(spec.padded_frames):
        src = images / f"{i:05d}.png"
        shutil.copy2(src, renders / f"{i:05d}.png")
        Image.new("L", (width, height), 255).save(opacity / f"{i:05d}.png")

    make_prompt_h5(captions / "prompt.h5", caption)

    # Synthetic but stable camera path. Target frames are 0..chunk-1; context frames are appended.
    fov = math.radians(fov_degrees)
    fl = (width / 2.0) / math.tan(fov / 2.0)
    frames: list[dict] = []
    for i in range(spec.padded_frames):
        y = (spec.start + i) * 0.05
        frames.append({
            "file_path": f"images/{i:05d}.png",
            "transform_matrix": [[1, 0, 0, 0.0], [0, 1, 0, y], [0, 0, 1, 1.5], [0, 0, 0, 1]],
        })

    context_src = np.linspace(0, spec.padded_frames - 1, num_views, dtype=int).tolist()
    selected_indices: list[int] = []
    for j, src_i in enumerate(context_src):
        idx = spec.padded_frames + j
        selected_indices.append(idx)
        y = (spec.start + src_i) * 0.05
        frames.append({
            "file_path": f"images/{src_i:05d}.png",
            "transform_matrix": [[1, 0, 0, 0.0], [0, 1, 0, y], [0, 0, 1, 1.5], [0, 0, 0, 1]],
        })

    transforms = {
        "camera_model": "OPENCV",
        "w": width,
        "h": height,
        "fl_x": fl,
        "fl_y": fl,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "frames": frames,
    }
    (scene_root / "transforms.json").write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    (scene_root / "selected_indices.json").write_text(json.dumps(selected_indices), encoding="utf-8")
    (scene_root / "target_indices.json").write_text(json.dumps(list(range(spec.padded_frames))), encoding="utf-8")
    (scene_root / "chunk_meta.json").write_text(json.dumps(spec.__dict__, indent=2), encoding="utf-8")

    return {
        "transforms_path": str(scene_root / "transforms.json"),
        "image_root": str(scene_root),
        "render_dir": str(renders),
        "opacity_dir": str(opacity),
        "selected_indices_path": str(scene_root / "selected_indices.json"),
        "target_indices_path": str(scene_root / "target_indices.json"),
        "prompt_path": str(captions / "prompt.h5"),
        "camera_scale": 1.0,
        "has_gt": False,
    }


def checkpoint_name(checkpoint_pt: Path) -> str:
    return checkpoint_pt.stem


def output_mode_dir(save_dir: Path, checkpoint_pt: Path, num_views: int, sink_size: int) -> Path:
    return save_dir / checkpoint_name(checkpoint_pt) / f"distilled_views_reconstructed_colmap_{num_views}_evenly_spaced_sink{sink_size}_trajectory"


def concat_trimmed_chunks(*, out_root: Path, mode_dir: Path, specs: list[ChunkSpec], fps: float, make_comparison: bool, source_video: Path) -> tuple[Path, Path | None]:
    stitch_dir = out_root / "stitched"
    pred_frames = stitch_dir / "pred_frames"
    pred_frames.mkdir(parents=True, exist_ok=True)
    for p in pred_frames.glob("*.png"):
        p.unlink()

    global_idx = 0
    for spec in specs:
        pred_dir = mode_dir / spec.scene_id / "frames" / "batch_0000" / "pred"
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"Missing ArtiFixer pred frames: {pred_dir}")
        for local_idx in range(spec.actual_frames):
            src = pred_dir / f"{local_idx:05d}.png"
            if not src.is_file():
                raise FileNotFoundError(src)
            shutil.copy2(src, pred_frames / f"{global_idx:06d}.png")
            global_idx += 1

    enhanced = stitch_dir / "enhanced_full.mp4"
    run([
        "ffmpeg", "-y", "-v", "error", "-framerate", f"{fps:.8f}", "-i", str(pred_frames / "%06d.png"),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(enhanced),
    ])

    comparison = None
    if make_comparison:
        comparison = stitch_dir / "before_after_hstack_full.mp4"
        run([
            "ffmpeg", "-y", "-v", "error", "-i", str(source_video), "-i", str(enhanced),
            "-filter_complex", "[0:v]setpts=PTS-STARTPTS,scale=320:180[left];[1:v]setpts=PTS-STARTPTS,scale=320:180[right];[left][right]hstack=inputs=2,format=yuv420p[v]",
            "-map", "[v]", "-r", f"{fps:.8f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-movflags", "+faststart", str(comparison),
        ])
    return enhanced, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument("--work_dir", type=Path, required=True, help="Working/output directory for prepared chunks and stitched video")
    parser.add_argument("--artifixer_repo", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model_id", type=Path, required=True)
    parser.add_argument("--checkpoint_pt", type=Path, required=True)
    parser.add_argument("--chunk_frames", type=int, default=81)
    parser.add_argument("--num_views", type=int, default=12)
    parser.add_argument("--sink_size", type=int, default=7)
    parser.add_argument("--output_fps", type=float, default=None, help="Defaults to input FPS")
    parser.add_argument("--scene_prefix", default="chunk")
    parser.add_argument("--caption", default="A drone camera video through a 3D scene. Enhance reconstruction artifacts and improve temporal consistency.")
    parser.add_argument("--fov_degrees", type=float, default=86.0)
    parser.add_argument("--replace_if_exists", action="store_true")
    parser.add_argument("--skip_inference", action="store_true", help="Only prepare split/chunks and stitch existing outputs")
    parser.add_argument("--no_comparison", action="store_true")
    args = parser.parse_args()

    args.video = args.video.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    prep_root = args.work_dir / "prep"
    infer_save_dir = args.work_dir / "artifixer_out"

    stream = ffprobe_json(args.video)
    width = int(stream["width"])
    height = int(stream["height"])
    fps = args.output_fps or parse_fps(stream["r_frame_rate"])
    total = frame_count(args.video)
    print(f"Input: {args.video} width={width} height={height} fps={fps:.6f} frames={total}", flush=True)

    specs: list[ChunkSpec] = []
    for chunk_idx, start in enumerate(range(0, total, args.chunk_frames)):
        actual = min(args.chunk_frames, total - start)
        specs.append(ChunkSpec(scene_id=f"{args.scene_prefix}_{chunk_idx:04d}", start=start, actual_frames=actual, padded_frames=args.chunk_frames))

    split = {"test": {}}
    for spec in specs:
        split["test"][spec.scene_id] = build_chunk_scene(
            video=args.video,
            prep_root=prep_root,
            spec=spec,
            width=width,
            height=height,
            num_views=args.num_views,
            fov_degrees=args.fov_degrees,
            caption=args.caption,
        )
    split_path = prep_root / "split.json"
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Prepared {len(specs)} chunks: {split_path}", flush=True)

    if not args.skip_inference:
        env = os.environ.copy()
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("DIFFUSERS_OFFLINE", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        cmd = [
            str(args.python), "-u", "-m", "model_eval.run_inference",
            "--model_id", str(args.model_id),
            "--evalset", "reconstructed_colmap",
            "--checkpoint_pt", str(args.checkpoint_pt),
            "--save_dir", str(infer_save_dir),
            "--split_path", str(split_path),
            "--render_trajectory", "trajectory",
            "--neighbor_selection_mode", "evenly_spaced",
            "--num_views", str(args.num_views),
            "--sink_size", str(args.sink_size),
            "--output_fps", str(int(round(fps))),
        ]
        if args.replace_if_exists:
            cmd.append("--replace_if_exists")
        run(cmd, cwd=args.artifixer_repo, env=env)

    mode_dir = output_mode_dir(infer_save_dir, args.checkpoint_pt, args.num_views, args.sink_size)
    enhanced, comparison = concat_trimmed_chunks(
        out_root=args.work_dir,
        mode_dir=mode_dir,
        specs=specs,
        fps=fps,
        make_comparison=not args.no_comparison,
        source_video=args.video,
    )
    print("Enhanced:", enhanced, flush=True)
    if comparison:
        print("Comparison:", comparison, flush=True)


if __name__ == "__main__":
    main()
