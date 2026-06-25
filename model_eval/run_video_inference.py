# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ArtiFixer inference for arbitrary-length videos.

This wrapper mirrors model_eval.run_inference's common CLI shape, but accepts a
single input video instead of a prepared evalset. Internally it chunks the video
(default 81 frames), runs normal ArtiFixer reconstructed_colmap inference on the
chunks, trims padding from the last chunk, then stitches the enhanced frames back
to a full-length MP4.

It can also process a directory of numbered trajectory folders, e.g. a DiffAero
``trajs`` directory containing ``0/rgb.mp4``, ``1/rgb.mp4``, ... . In batch mode
intermediate files are kept in each trajectory's hidden work directory while the
final enhanced videos are copied next to the source ``rgb.mp4``.

Examples:
    python -m model_eval.run_video_inference \
        --video /path/to/input.mp4 \
        --checkpoint_pt /path/to/artifixer-14b.pt \
        --model_id /path/to/Wan2.1-T2V-14B-Diffusers \
        --save_dir /path/to/output \
        --num_views 12 \
        --output_fps 30 \
        --replace_if_exists

    python -m model_eval.run_video_inference \
        --video_dir /path/to/trajs \
        --checkpoint_pt /path/to/artifixer-14b.pt \
        --model_id /path/to/Wan2.1-T2V-14B-Diffusers \
        --num_views 12 \
        --output_fps 30 \
        --replace_if_exists
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

from model_eval.checkpoint_loading import add_checkpoint_args, validate_checkpoint_args
from model_training.utils.train_utils import get_eval_common_opts


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_chunk_runner():
    runner_path = _repo_root() / "scripts" / "run_artifixer_video_chunks.py"
    spec = importlib.util.spec_from_file_location("_artifixer_video_chunk_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load chunk runner from {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = get_eval_common_opts()
    add_checkpoint_args(parser)

    # New video inputs. Everything else intentionally resembles run_inference.py.
    inputs = parser.add_argument_group("arbitrary video inputs")
    inputs.add_argument("--video", type=Path, default=None, help="Input MP4/video to enhance with ArtiFixer")
    inputs.add_argument(
        "--video_dir",
        type=Path,
        default=None,
        help="Directory containing numeric subdirectories with a video file, e.g. trajs/0/rgb.mp4, trajs/1/rgb.mp4",
    )
    inputs.add_argument("--video_name", default="rgb.mp4", help="Video filename to read inside each numeric --video_dir child")
    inputs.add_argument("--max_videos", type=int, default=None, help="Optional batch-mode limit for smoke tests")

    outputs = parser.add_argument_group("arbitrary video outputs")
    outputs.add_argument("--save_dir", type=Path, default=None, help="Single-video working/output directory for prepared chunks, ArtiFixer outputs, and stitched videos")
    outputs.add_argument("--work_dir_name", default=".artifixer_work", help="Batch-mode hidden work directory created inside each numeric child")
    outputs.add_argument("--enhanced_name", default="rgb_artifixer.mp4", help="Batch-mode enhanced video filename copied next to the source video")
    outputs.add_argument("--comparison_name", default="rgb_before_after_artifixer.mp4", help="Batch-mode side-by-side video filename copied next to the source video")
    outputs.add_argument("--continue_on_error", action="store_true", help="Batch mode: keep processing later videos if one fails")

    parser.add_argument("--num_views", default=12, type=int, help="Number of context views per chunk")
    parser.add_argument("--output_fps", default=None, type=int, help="Output FPS. Defaults to input video's FPS")
    parser.add_argument("--replace_if_exists", action="store_true")
    parser.add_argument("--sink_size", default=7, type=int)
    parser.add_argument("--neighbor_selection_mode", default="evenly_spaced", choices=["evenly_spaced"], help="Only evenly_spaced is supported for arbitrary-video chunks")

    # Keep the familiar name from run_inference. For this wrapper it is the video chunk size.
    parser.add_argument("--bidirectional_chunk_size", default=81, type=int, help="Frames per ArtiFixer chunk; padding is trimmed after stitching")
    parser.add_argument("--chunk_frames", default=None, type=int, help="Alias for --bidirectional_chunk_size")

    # Accepted for run_inference familiarity. Some are passed through only when the helper supports them;
    # unsupported values are rejected clearly rather than silently ignored.
    parser.add_argument("--render_trajectory", default="trajectory", choices=["trajectory"], help="Arbitrary video mode always renders the prepared chunk trajectory")
    parser.add_argument("--scene_id", default=None, help="Optional prefix for generated chunk scene ids")
    parser.add_argument("--caption", default="A drone camera video through a 3D scene. Enhance reconstruction artifacts and improve temporal consistency.")
    parser.add_argument("--fov_degrees", type=float, default=86.0)
    parser.add_argument("--no_comparison", action="store_true", help="Skip side-by-side original/enhanced video")
    parser.add_argument("--skip_inference", action="store_true", help="Prepare/stitch only; useful when chunk outputs already exist")

    # These exist on run_inference.py; keep them so command templates port cleanly, but fail on unsupported modes.
    parser.add_argument("--inference_pipeline", default="kv_cache", choices=["kv_cache"], help="Only kv_cache is supported by this wrapper")
    parser.add_argument("--context_parallel_size", default=1, type=int)
    parser.add_argument("--save_frame_outputs_only", action="store_true")
    parser.add_argument("--distributed_timeout_minutes", default=60, type=int)
    parser.add_argument("--num_inference_steps", default=4, type=int)
    parser.add_argument("--frames_per_block", default=7, type=int)
    parser.add_argument("--local_attn_size", default=21, type=int)
    parser.add_argument("--render_diagnostics", action="store_true")
    parser.add_argument("--max_neighbors_per_encode", default=None, type=int)
    parser.add_argument("--output_suffix", default="", type=str)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_checkpoint_args(parser, args)
    if (args.video is None) == (args.video_dir is None):
        parser.error("Specify exactly one of --video or --video_dir")
    if args.video is not None and args.save_dir is None:
        parser.error("--save_dir is required with --video")
    if args.video_dir is not None and args.save_dir is not None:
        parser.error("--save_dir is single-video only; batch mode writes to <numeric_child>/<work_dir_name>")
    if args.checkpoint_dir is not None:
        parser.error("run_video_inference currently supports --checkpoint_pt only; export checkpoint_dir to .pt first")
    if args.context_parallel_size != 1:
        parser.error("run_video_inference currently supports --context_parallel_size=1 only")
    if args.save_frame_outputs_only:
        parser.error("--save_frame_outputs_only is not supported because stitching needs pred frames and videos")
    if args.output_suffix:
        parser.error("--output_suffix is not supported in arbitrary-video mode; choose a different --save_dir or batch output names")
    if args.max_neighbors_per_encode is not None and args.max_neighbors_per_encode <= 0:
        parser.error("--max_neighbors_per_encode must be positive when set")
    if args.chunk_frames is not None:
        args.bidirectional_chunk_size = args.chunk_frames
    if args.bidirectional_chunk_size <= 0:
        parser.error("--bidirectional_chunk_size / --chunk_frames must be positive")
    if args.num_views <= 0:
        parser.error("--num_views must be positive")
    if args.max_videos is not None and args.max_videos <= 0:
        parser.error("--max_videos must be positive when set")
    return args


def _run_chunk_runner(runner, args: argparse.Namespace, *, video: Path, save_dir: Path, scene_prefix: str) -> tuple[Path, Path | None]:
    runner_args = [
        "--video", str(video),
        "--work_dir", str(save_dir),
        "--artifixer_repo", str(_repo_root()),
        "--python", sys.executable,
        "--model_id", str(args.model_id),
        "--checkpoint_pt", str(args.checkpoint_pt),
        "--chunk_frames", str(args.bidirectional_chunk_size),
        "--num_views", str(args.num_views),
        "--sink_size", str(args.sink_size),
        "--scene_prefix", scene_prefix,
        "--caption", args.caption,
        "--fov_degrees", str(args.fov_degrees),
    ]
    if args.output_fps is not None:
        runner_args.extend(["--output_fps", str(args.output_fps)])
    if args.replace_if_exists:
        runner_args.append("--replace_if_exists")
    if args.skip_inference:
        runner_args.append("--skip_inference")
    if args.no_comparison:
        runner_args.append("--no_comparison")
    if args.max_neighbors_per_encode is not None:
        runner_args.extend(["--max_neighbors_per_encode", str(args.max_neighbors_per_encode)])

    # Reuse the chunk runner's tested implementation while exposing a run_inference-like CLI.
    old_argv = sys.argv
    try:
        sys.argv = ["run_artifixer_video_chunks.py", *runner_args]
        runner.main()
    finally:
        sys.argv = old_argv

    enhanced = save_dir / "stitched" / "enhanced_full.mp4"
    comparison = None if args.no_comparison else save_dir / "stitched" / "before_after_hstack_full.mp4"
    if not enhanced.is_file():
        raise FileNotFoundError(f"Expected enhanced video was not created: {enhanced}")
    if comparison is not None and not comparison.is_file():
        raise FileNotFoundError(f"Expected comparison video was not created: {comparison}")
    return enhanced, comparison


def _numeric_video_jobs(video_dir: Path, video_name: str, max_videos: int | None) -> list[tuple[Path, Path]]:
    if not video_dir.is_dir():
        raise NotADirectoryError(video_dir)
    jobs: list[tuple[Path, Path]] = []
    for child in sorted((p for p in video_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
        video = child / video_name
        if video.is_file():
            jobs.append((child, video))
            if max_videos is not None and len(jobs) >= max_videos:
                break
        else:
            print(f"Skipping {child}: missing {video_name}", flush=True)
    if not jobs:
        raise FileNotFoundError(f"No numeric child directories containing {video_name} found under {video_dir}")
    return jobs


def _copy_batch_outputs(args: argparse.Namespace, *, enhanced: Path, comparison: Path | None, traj_dir: Path) -> tuple[Path, Path | None]:
    enhanced_out = traj_dir / args.enhanced_name
    comparison_out = None if comparison is None else traj_dir / args.comparison_name
    targets = [enhanced_out]
    if comparison_out is not None:
        targets.append(comparison_out)
    existing = [p for p in targets if p.exists()]
    if existing and not args.replace_if_exists:
        raise FileExistsError(
            "Refusing to overwrite existing batch output(s) without --replace_if_exists: "
            + ", ".join(str(p) for p in existing)
        )
    shutil.copy2(enhanced, enhanced_out)
    if comparison is not None and comparison_out is not None:
        shutil.copy2(comparison, comparison_out)
    return enhanced_out, comparison_out


def _run_single(args: argparse.Namespace, runner) -> None:
    scene_prefix = args.scene_id or "chunk"
    enhanced, comparison = _run_chunk_runner(runner, args, video=args.video, save_dir=args.save_dir, scene_prefix=scene_prefix)
    print("Enhanced:", enhanced, flush=True)
    if comparison:
        print("Comparison:", comparison, flush=True)


def _run_batch(args: argparse.Namespace, runner) -> None:
    jobs = _numeric_video_jobs(args.video_dir, args.video_name, args.max_videos)
    print(f"Found {len(jobs)} video(s) in numeric child directories under {args.video_dir}", flush=True)
    failures: list[tuple[Path, BaseException]] = []
    for idx, (traj_dir, video) in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] Enhancing {video}", flush=True)
        try:
            work_dir = traj_dir / args.work_dir_name
            scene_prefix = args.scene_id or f"{traj_dir.name}_chunk"
            enhanced, comparison = _run_chunk_runner(runner, args, video=video, save_dir=work_dir, scene_prefix=scene_prefix)
            enhanced_out, comparison_out = _copy_batch_outputs(args, enhanced=enhanced, comparison=comparison, traj_dir=traj_dir)
            print("Enhanced:", enhanced_out, flush=True)
            if comparison_out:
                print("Comparison:", comparison_out, flush=True)
        except BaseException as exc:  # noqa: BLE001 - report and optionally continue for long batch jobs.
            if not args.continue_on_error:
                raise
            failures.append((video, exc))
            print(f"FAILED {video}: {exc}", flush=True)
    if failures:
        details = "\n".join(f"  {video}: {exc}" for video, exc in failures)
        raise RuntimeError(f"{len(failures)} batch video(s) failed:\n{details}")


def main(args: argparse.Namespace) -> None:
    runner = _load_chunk_runner()
    if args.video_dir is not None:
        _run_batch(args, runner)
    else:
        _run_single(args, runner)


if __name__ == "__main__":
    main(parse_args())
