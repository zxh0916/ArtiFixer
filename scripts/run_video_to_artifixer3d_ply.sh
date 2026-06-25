#!/usr/bin/env bash
set -euo pipefail

# End-to-end 104.mp4 -> ArtiFixer3D+ PLY pipeline with optimized 3DGS parameters.
# The script is idempotent by default: completed stages are reused and the final
# PLY is not overwritten unless FORCE=1 is set.

WORKSPACE=${WORKSPACE:-/home/dataset-assist-0/chl_ws}
REPO_DIR=${REPO_DIR:-$WORKSPACE/ArtiFixer}
DATA_ROOT=${DATA_ROOT:-$WORKSPACE/artifixer-data}
SCENE_ID=${SCENE_ID:-104}
VIDEO=${VIDEO:-$WORKSPACE/video/${SCENE_ID}.mp4}

PYTHON=${PYTHON:-$WORKSPACE/miniconda3/envs/artifixer/bin/python}
export CUDA_HOME=${CUDA_HOME:-$WORKSPACE/miniconda3/envs/artifixer}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export DIFFUSERS_OFFLINE=${DIFFUSERS_OFFLINE:-1}
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/runtime-batchcom}

ROOT="$DATA_ROOT/scene_${SCENE_ID}"
RAW="$ROOT/colmap_raw"
UNDIST="$ROOT/colmap_undist"
COLMAP_DIR="$ROOT/colmap_dir"
PREP="$ROOT/prepared"
PREP960="$ROOT/prepared_960_optimized"
LOG_DIR="$DATA_ROOT/logs"
LOG="$LOG_DIR/scene_${SCENE_ID}_video_to_ply.log"

CKPT=${CKPT:-$DATA_ROOT/checkpoints/artifixer-14b.pt}
MODEL=${MODEL:-$DATA_ROOT/Wan2.1-T2V-14B-Diffusers}
CONFIG_OPT=apps/colmap_3dgut_sparse_mcmc_lpips_optimized
CONFIG_OPT_FILE="$REPO_DIR/thirdparty/3DGRUT-ArtiFixer/configs/apps/colmap_3dgut_sparse_mcmc_lpips_optimized.yaml"
RECON_STEPS=20000
ARTIFIXER3D_STEPS=30000
LONG_SIDE=960
PLY_OUT="$ROOT/${SCENE_ID}_artifixer3d_final_960.ply"
OPTIMIZED_MARKER="$ROOT/.${SCENE_ID}_artifixer3d_final_960.optimized"
FORCE=${FORCE:-0}

mkdir -p "$ROOT" "$LOG_DIR" "$XDG_RUNTIME_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "VIDEO_TO_PLY_START $(date -Iseconds)"
echo "scene_id=$SCENE_ID video=$VIDEO output=$PLY_OUT"

if [[ -f "$PLY_OUT" && -f "$OPTIMIZED_MARKER" && "$FORCE" != "1" ]]; then
  echo "FINAL_PLY_EXISTS $PLY_OUT"
  echo "Set FORCE=1 to rerun and overwrite downstream outputs."
  echo "VIDEO_TO_PLY_DONE $(date -Iseconds)"
  exit 0
fi

need_file() {
  test -f "$1" || { echo "Missing required file: $1" >&2; exit 1; }
}

need_dir() {
  test -d "$1" || { echo "Missing required directory: $1" >&2; exit 1; }
}

need_file "$VIDEO"
need_file "$CKPT"
need_dir "$MODEL"
command -v ffmpeg >/dev/null
command -v colmap >/dev/null

if [[ ! -f "$CONFIG_OPT_FILE" ]]; then
  mkdir -p "$(dirname "$CONFIG_OPT_FILE")"
  cat > "$CONFIG_OPT_FILE" <<'YAML'
# @package _global_

# Optimized tuning: fewer floaters, tighter MCMC, stronger opacity/scale regularization.
defaults:
  - colmap_3dgut_sparse_mcmc_lpips
  - _self_

strategy:
  opacity_threshold: 0.01
  relocate:
    end_iteration: 20000
  perturb:
    noise_lr: 300000.0
  add:
    end_iteration: 20000
    max_n_gaussians: 800000

loss:
  lambda_opacity: 0.02
  lambda_scale: 0.03
YAML
fi

cd "$REPO_DIR"

echo "STAGE_1_EXTRACT_FRAMES $(date -Iseconds)"
mkdir -p "$RAW/images"
if ! compgen -G "$RAW/images/*.jpg" >/dev/null; then
  ffmpeg -hide_banner -loglevel error -i "$VIDEO" -vf fps=2 -q:v 2 "$RAW/images/%06d.jpg"
fi
echo "frames=$(compgen -G "$RAW/images/*.jpg" | wc -l)"

echo "STAGE_2_COLMAP_SFM $(date -Iseconds)"
if [[ ! -f "$RAW/sparse/0/images.bin" ]]; then
  rm -f "$RAW/database.db"
  rm -rf "$RAW/sparse"
  mkdir -p "$RAW/sparse"
  colmap feature_extractor \
    --database_path "$RAW/database.db" \
    --image_path "$RAW/images" \
    --ImageReader.single_camera 1 \
    --SiftExtraction.max_image_size 1600 \
    --SiftExtraction.use_gpu 0
  colmap exhaustive_matcher \
    --database_path "$RAW/database.db" \
    --SiftMatching.use_gpu 0
  colmap mapper \
    --database_path "$RAW/database.db" \
    --image_path "$RAW/images" \
    --output_path "$RAW/sparse"
fi
need_file "$RAW/sparse/0/images.bin"

echo "STAGE_3_COLMAP_UNDISTORT $(date -Iseconds)"
if [[ ! -f "$COLMAP_DIR/sparse/0/images.bin" ]]; then
  rm -rf "$UNDIST"
  colmap image_undistorter \
    --image_path "$RAW/images" \
    --input_path "$RAW/sparse/0" \
    --output_path "$UNDIST" \
    --output_type COLMAP \
    --max_image_size 1600

  sparse_src="$UNDIST/sparse/0"
  if [[ ! -f "$sparse_src/images.bin" ]]; then
    sparse_src="$UNDIST/sparse"
  fi
  mkdir -p "$COLMAP_DIR/images" "$COLMAP_DIR/sparse/0"
  cp -a "$UNDIST/images/"* "$COLMAP_DIR/images/"
  cp "$sparse_src/cameras.bin" "$sparse_src/images.bin" "$sparse_src/points3D.bin" "$COLMAP_DIR/sparse/0/"
fi
need_file "$COLMAP_DIR/sparse/0/images.bin"

echo "STAGE_4_PREPARE_AND_SAMPLE $(date -Iseconds)"
if [[ ! -f "$PREP/selected_indices.json" ]]; then
  "$PYTHON" -m data_processing.prepare_colmap_artifixer_inputs \
    --colmap_dir "$COLMAP_DIR" \
    --output_root "$PREP" \
    --phases prepare --replace
  "$PYTHON" -m data_processing.sparse_recon.half_covisibility_sampling \
    --dataset_path "$PREP/3dgrut_input/prepared" \
    --output_path "$PREP"
  "$PYTHON" - <<PY
import json
from pathlib import Path

prep = Path("$PREP")
indices = json.loads((prep / "half_covisibility_sampled_indices_0.json").read_text())
transforms = json.loads((prep / "3dgrut_input/prepared/nerfstudio/transforms.json").read_text())
names = [Path(transforms["frames"][i]["file_path"]).name for i in indices]
(prep / "selected_train_images.txt").write_text("\n".join(names) + "\n")
print(f"selected_train_images={len(names)} total_frames={len(transforms['frames'])}")
PY
fi
need_file "$PREP/selected_train_images.txt"

echo "STAGE_5_BASE_3DGRUT_20K $(date -Iseconds)"
BASE_CKPT="$PREP/3dgrut_runs/prepared/prepared/ours_${RECON_STEPS}/ckpt_${RECON_STEPS}.pt"
if [[ ! -f "$BASE_CKPT" ]]; then
  "$PYTHON" -m data_processing.prepare_colmap_artifixer_inputs \
    --colmap_dir "$COLMAP_DIR" \
    --output_root "$PREP" \
    --selected_image_names_file "$PREP/selected_train_images.txt" \
    --phases prepare,reconstruct,render,scale \
    --reconstruction_steps "$RECON_STEPS"
fi
need_file "$BASE_CKPT"

echo "STAGE_6_LIGHT_CAPTION $(date -Iseconds)"
if [[ ! -f "$PREP/captions/prepared/caption.h5" ]]; then
  "$PYTHON" - <<PY
import h5py
import numpy as np
import torch
from pathlib import Path
from data_processing.captioning.generate_captions import generate_text_embedding, get_text_encoder_and_tokenizer

caption = (
    "A drone camera video flying through an outdoor 3D scene with buildings, roads, "
    "vegetation and urban structures. The view moves smoothly forward with natural lighting "
    "and realistic textures across the environment."
)
model_id = "$MODEL"
out = Path("$PREP") / "captions/prepared/caption.h5"
out.parent.mkdir(parents=True, exist_ok=True)
te, tok = get_text_encoder_and_tokenizer(model_id)
emb = generate_text_embedding(caption, te, tok, 512)
tmp = out.with_suffix(".tmp")
with h5py.File(tmp, "w") as hf:
    ds = hf.create_dataset("00000", data=emb)
    ds.attrs["caption"] = caption
    ds.attrs["image_indices"] = np.array([0], dtype=np.int64)
tmp.rename(out)
del te
torch.cuda.empty_cache()
print(f"caption_h5={out}")
PY
fi

"$PYTHON" -m data_processing.prepare_colmap_artifixer_inputs \
  --colmap_dir "$COLMAP_DIR" \
  --output_root "$PREP" \
  --selected_image_names_file "$PREP/selected_train_images.txt" \
  --phases render,scale \
  --reconstruction_steps "$RECON_STEPS"
need_file "$PREP/split.json"

echo "STAGE_7_PREPARE_960_OPTIMIZED $(date -Iseconds)"
if [[ ! -f "$PREP960/3dgrut_input/prepared/nerfstudio/transforms.json" || "$FORCE" == "1" ]]; then
  rm -rf "$PREP960"
  "$PYTHON" - <<PY
from pathlib import Path
from PIL import Image
import json
import os
import shutil

src = Path("$PREP")
dst = Path("$PREP960")
long_side = int("$LONG_SIDE")
recon_steps = int("$RECON_STEPS")
render_subdir = f"ours_{recon_steps}"

def ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def link_or_copy(src_path: Path, dst_path: Path) -> None:
    ensure(dst_path.parent)
    try:
        os.link(src_path, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)

for rel in ["split.json", "selected_indices.json", "selected_images.txt", "selected_train_images.txt"]:
    if (src / rel).exists():
        link_or_copy(src / rel, dst / rel)

for rel in ["metric_alignment", "captions"]:
    if (src / rel).exists():
        shutil.copytree(src / rel, dst / rel)

ensure(dst / "3dgrut_runs/prepared")
os.symlink(src / "3dgrut_runs/prepared/prepared", dst / "3dgrut_runs/prepared/prepared")

render_src_root = src / f"recon_results/prepared/reconstruction/prepared/{render_subdir}"
render_dst_root = dst / f"recon_results/prepared/reconstruction/prepared/{render_subdir}"
ensure(render_dst_root)
if (render_src_root / "selected_indices.json").exists():
    link_or_copy(render_src_root / "selected_indices.json", render_dst_root / "selected_indices.json")

with Image.open(src / "3dgrut_input/prepared/images/000001.jpg") as im:
    ow, oh = im.size
scale = long_side / max(ow, oh)
nw = int(round(ow * scale / 16) * 16)
nh = int(round(oh * scale / 16) * 16)
sx, sy = nw / ow, nh / oh
print(f"LOWRES from {ow}x{oh} to {nw}x{nh}, scale=({sx:.6f},{sy:.6f})", flush=True)

resample_rgb = Image.Resampling.LANCZOS
resample_mask = Image.Resampling.BILINEAR

def resize_dir(src_dir: Path, dst_dir: Path) -> None:
    ensure(dst_dir)
    files = [p for p in sorted(src_dir.glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    for i, path in enumerate(files, 1):
        with Image.open(path) as im:
            resample = resample_mask if im.mode in {"L", "I;16", "1"} else resample_rgb
            out = im.resize((nw, nh), resample)
            save_kwargs = {"quality": 95} if path.suffix.lower() in {".jpg", ".jpeg"} else {}
            out.save(dst_dir / path.name, **save_kwargs)
        if i % 50 == 0 or i == len(files):
            print(f"resized {i}/{len(files)}: {src_dir.name}", flush=True)

resize_dir(src / "3dgrut_input/prepared/images", dst / "3dgrut_input/prepared/images")
resize_dir(render_src_root / "renders", render_dst_root / "renders")
resize_dir(render_src_root / "opacity", render_dst_root / "opacity")

transforms = json.loads((src / "3dgrut_input/prepared/nerfstudio/transforms.json").read_text())
for obj in [transforms, *transforms.get("frames", [])]:
    if "w" in obj:
        obj["w"] = nw
    if "h" in obj:
        obj["h"] = nh
    if "fl_x" in obj:
        obj["fl_x"] *= sx
    if "fl_y" in obj:
        obj["fl_y"] *= sy
    if "cx" in obj:
        obj["cx"] *= sx
    if "cy" in obj:
        obj["cy"] *= sy
trans_dst = dst / "3dgrut_input/prepared/nerfstudio/transforms.json"
ensure(trans_dst.parent)
trans_dst.write_text(json.dumps(transforms, indent=2) + "\\n")

sparse_src = src / "3dgrut_input/prepared/sparse/0"
sparse_dst = dst / "3dgrut_input/prepared/sparse/0"
ensure(sparse_dst)
for name in ["cameras.bin", "images.bin", "points3D.bin"]:
    shutil.copy2(sparse_src / name, sparse_dst / name)

print(f"LOWRES_PREPARED={dst}", flush=True)
PY
fi
need_file "$PREP960/split.json"
need_file "$PREP960/3dgrut_input/prepared/sparse/0/images.bin"

echo "STAGE_8_ARTIFIXER_INFERENCE $(date -Iseconds)"
AF_FRAMES=""
if compgen -G "$ROOT/artifixer_out_960_optimized/*/*/prepared/frames/batch_0000/pred" >/dev/null; then
  for candidate in "$ROOT"/artifixer_out_960_optimized/*/*/prepared/frames/batch_0000/pred; do
    AF_FRAMES="$candidate"
    break
  done
fi
if [[ -z "$AF_FRAMES" || "$FORCE" == "1" ]]; then
  "$PYTHON" -u -m model_eval.run_inference \
    --evalset reconstructed_colmap \
    --checkpoint_pt "$CKPT" \
    --model_id "$MODEL" \
    --save_dir "$ROOT/artifixer_out_960_optimized" \
    --split_path "$PREP960/split.json" \
    --render_trajectory val_frames \
    --num_views 4 --sink_size 7 \
    --max_neighbors_per_encode 1 --local_attn_size 7 \
    --replace_if_exists
  for candidate in "$ROOT"/artifixer_out_960_optimized/*/*/prepared/frames/batch_0000/pred; do
    AF_FRAMES="$candidate"
    break
  done
fi
need_dir "$AF_FRAMES"
echo "ARTIFIXER_FRAMES=$AF_FRAMES"

echo "STAGE_9_ARTIFIXER3D_OPTIMIZED $(date -Iseconds)"
AF3D_CKPT="$PREP960/artifixer3d/runs/prepared/prepared/ours_${ARTIFIXER3D_STEPS}/ckpt_${ARTIFIXER3D_STEPS}.pt"
if [[ ! -f "$AF3D_CKPT" || "$FORCE" == "1" ]]; then
  "$PYTHON" -m data_processing.run_artifixer3d \
    --scene_root "$PREP960" \
    --artifixer_frames_dir "$AF_FRAMES" \
    --phases distill,render,prepare_artifixer3d_plus \
    --artifixer3d_steps "$ARTIFIXER3D_STEPS" \
    --config_name "$CONFIG_OPT" \
    --replace
fi
need_file "$AF3D_CKPT"

echo "STAGE_10_ARTIFIXER3D_PLUS $(date -Iseconds)"
if [[ ! -d "$ROOT/artifixer3d_plus_out_960_optimized" || "$FORCE" == "1" ]]; then
  "$PYTHON" -u -m model_eval.run_inference \
    --evalset reconstructed_colmap \
    --checkpoint_pt "$CKPT" \
    --model_id "$MODEL" \
    --save_dir "$ROOT/artifixer3d_plus_out_960_optimized" \
    --split_path "$PREP960/split_artifixer3d_plus.json" \
    --render_trajectory all_frames \
    --num_views 4 --max_neighbors_per_encode 1 --local_attn_size 7 \
    --replace_if_exists
fi

echo "STAGE_11_EXPORT_PLY $(date -Iseconds)"
DIST_DIR="$PREP960/artifixer3d/distillation_input/prepared"
need_dir "$DIST_DIR"
cd "$REPO_DIR/thirdparty/3DGRUT-ArtiFixer"
"$PYTHON" train.py --config-name "$CONFIG_OPT" \
  path="$DIST_DIR" \
  resume="$AF3D_CKPT" \
  export_ply.enabled=True \
  export_ply.path="$PLY_OUT" \
  test_last=True \
  n_iterations="$ARTIFIXER3D_STEPS"

need_file "$PLY_OUT"
touch "$OPTIMIZED_MARKER"
echo "PLY_DONE $PLY_OUT $(date -Iseconds)"
echo "VIDEO_TO_PLY_DONE $(date -Iseconds)"
