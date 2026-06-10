# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ArtiFixer inference for arbitrary-length videos.

This wrapper mirrors model_eval.run_inference's common CLI shape, but accepts a
single input video instead of a prepared evalset. Internally it chunks the video
(default 81 frames), runs normal ArtiFixer reconstructed_colmap inference on the
chunks, trims padding from the last chunk, then stitches the enhanced frames back
to a full-length MP4.

Example:
    python -m model_eval.run_video_inference \
        --video /path/to/input.mp4 \
        --checkpoint_pt /path/to/artifixer-14b.pt \
        --model_id /path/to/Wan2.1-T2V-14B-Diffusers \
        --save_dir /path/to/output \
        --num_views 12 \
        --output_fps 30 \
        --replace_if_exists
"""

from __future__ import annotations

import argparse
import importlib.util
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

    # New video input. Everything else intentionally resembles run_inference.py.
    parser.add_argument("--video", type=Path, required=True, help="Input MP4/video to enhance with ArtiFixer")
    parser.add_argument("--save_dir", type=Path, required=True, help="Directory for prepared chunks, ArtiFixer outputs, and stitched videos")

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
    if args.checkpoint_dir is not None:
        parser.error("run_video_inference currently supports --checkpoint_pt only; export checkpoint_dir to .pt first")
    if args.context_parallel_size != 1:
        parser.error("run_video_inference currently supports --context_parallel_size=1 only")
    if args.save_frame_outputs_only:
        parser.error("--save_frame_outputs_only is not supported because stitching needs pred frames and videos")
    if args.output_suffix:
        parser.error("--output_suffix is not supported in arbitrary-video mode; choose a different --save_dir")
    if args.max_neighbors_per_encode is not None and args.max_neighbors_per_encode <= 0:
        parser.error("--max_neighbors_per_encode must be positive when set")
    if args.chunk_frames is not None:
        args.bidirectional_chunk_size = args.chunk_frames
    if args.bidirectional_chunk_size <= 0:
        parser.error("--bidirectional_chunk_size / --chunk_frames must be positive")
    if args.num_views <= 0:
        parser.error("--num_views must be positive")
    return args


def main(args: argparse.Namespace) -> None:
    runner = _load_chunk_runner()
    scene_prefix = args.scene_id or "chunk"
    runner_args = [
        "--video", str(args.video),
        "--work_dir", str(args.save_dir),
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

    # Reuse the chunk runner's tested implementation while exposing a run_inference-like CLI.
    old_argv = sys.argv
    try:
        sys.argv = ["run_artifixer_video_chunks.py", *runner_args]
        runner.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main(parse_args())
