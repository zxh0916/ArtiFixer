#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build an x86_64 Conda environment for ArtiFixer.
#
# Usage:
#   bash scripts/setup_conda_cuda13_x86.sh [--cu13] [ENV_NAME]
#
# Default mode reuses the system CUDA/PyTorch defaults and does not force the
# CUDA13/cu130 stack. Pass --cu13 for the Dockerfile.cuda13-equivalent setup.
#
# Examples:
#   bash scripts/setup_conda_cuda13_x86.sh arti
#   bash scripts/setup_conda_cuda13_x86.sh --cu13 artifixer-cu13
#   conda activate artifixer-cu13
#   python tests/container_sanity_check.py
#   python tests/test_flash_attn.py
#
# Useful overrides:
#   PYTHON_VERSION=3.12
#   CUDA_HOME=/usr/local/cuda-13.0       # used in --cu13 mode, or current CUDA_HOME otherwise
#   CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh
#   TORCH_VERSION=2.11.0                # --cu13 default; otherwise latest compatible torch
#   PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
#   PIP_EXTRA_INDEX_URL=https://pypi.ngc.nvidia.com
#   TORCH_INDEX_URL=$PIP_INDEX_URL      # default uses Tsinghua PyPI mirror
#   TORCHVISION_VERSION=0.26.0          # default for TORCH_VERSION=2.11.0
#   SLANGC_GITHUB_PROXY=https://ghfast.top/
#   FLASH_ATTN3_MODE=auto|source|skip   # auto tries binary first, then source
#   FLASH_ATTN4_SPEC='flash-attn-4[cu13]' # installed only with --cu13 unless overridden
#   FLASH_ATTN_MAX_JOBS=16
#   FLASH_ATTN_NVCC_THREADS=2
#   RUN_SANITY=1                        # run tests/container_sanity_check.py at the end

set -euo pipefail

USE_CU13=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      sed -n "1,31p" "$0"
      exit 0
      ;;
    --cu13)
      USE_CU13=1
      shift
      ;;
    --)
      shift
      POSITIONAL+=("$@")
      break
      ;;
    -*)
      printf "\n\033[1;31m[ERROR]\033[0m Unknown option: %s\n" "$1" >&2
      exit 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

ENV_NAME="${1:-artifixer}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://pypi.ngc.nvidia.com}"
SLANGC_GITHUB_PROXY="${SLANGC_GITHUB_PROXY:-https://ghfast.top/}"
export SLANGC_GITHUB_PROXY
if [[ "$USE_CU13" == "1" ]]; then
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
  TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-$PIP_INDEX_URL}"
  TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
  FLASH_ATTN4_SPEC="${FLASH_ATTN4_SPEC:-flash-attn-4[cu13]}"
else
  CUDA_HOME="${CUDA_HOME:-}"
  TORCH_VERSION="${TORCH_VERSION:-}"
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-$PIP_INDEX_URL}"
  TORCHVISION_VERSION="${TORCHVISION_VERSION:-}"
  FLASH_ATTN4_SPEC="${FLASH_ATTN4_SPEC:-}"
fi
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
FLASH_ATTN3_MODE="${FLASH_ATTN3_MODE:-auto}"
FLASH_ATTN_MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-16}"
FLASH_ATTN_NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS:-2}"
RUN_SANITY="${RUN_SANITY:-0}"
TORCH_CUDA_ARCH_LIST_VALUE="${TORCH_CUDA_ARCH_LIST:-8.0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THREEDGRUT_ROOT="$REPO_ROOT/thirdparty/3DGRUT-ArtiFixer"

log() { printf '\n\033[1;36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\n\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
fatal() { printf '\n\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fatal "Missing required command: $1"
}

if [[ "$(uname -m)" != "x86_64" ]]; then
  fatal "This script is for x86_64. Use Dockerfile.cuda13-aarch64 logic on ARM64."
fi

[[ -f "$REPO_ROOT/Dockerfile.cuda13" ]] || fatal "Run from the ArtiFixer repo, or keep this script under scripts/. Missing $REPO_ROOT/Dockerfile.cuda13"
[[ -d "$THREEDGRUT_ROOT" ]] || fatal "Missing submodule: $THREEDGRUT_ROOT. Run: git submodule update --init --recursive"
[[ -f "$CONDA_SH" ]] || fatal "Cannot find conda shell hook: $CONDA_SH"
REQUESTED_CUDA_HOME="$CUDA_HOME"
if [[ "$USE_CU13" == "1" ]]; then
  [[ -d "$CUDA_HOME" ]] || fatal "CUDA_HOME does not exist: $CUDA_HOME. Install CUDA13 first or set CUDA_HOME."
  [[ -x "$CUDA_HOME/bin/nvcc" ]] || fatal "nvcc not found at $CUDA_HOME/bin/nvcc"
else
  if [[ -z "$CUDA_HOME" && -x /usr/local/cuda/bin/nvcc ]]; then
    CUDA_HOME=/usr/local/cuda
  fi
  if [[ -n "$CUDA_HOME" && ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    fatal "CUDA_HOME is set but nvcc is not executable at $CUDA_HOME/bin/nvcc"
  fi
fi

for c in git curl wget gcc-11 g++-11; do require_cmd "$c"; done

# System libraries used by Dockerfile.cuda13. We do not apt-install here because
# this script is intended to work without sudo; fail early with actionable info.
if command -v dpkg >/dev/null 2>&1; then
  for pkg in build-essential libgl1-mesa-dev libglib2.0-0; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
      fatal "Missing system package '$pkg'. Install once with sudo apt-get install $pkg"
    fi
  done
fi

# shellcheck source=/dev/null
source "$CONDA_SH"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "Conda env '$ENV_NAME' already exists; reusing it"
else
  log "Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

set +u
conda activate "$ENV_NAME"
set -u
if [[ -n "$REQUESTED_CUDA_HOME" ]]; then
  CUDA_HOME="$REQUESTED_CUDA_HOME"
fi

# Runtime/compiler environment. Keep this in activate.d so later shells inherit it.
log "Writing conda activation hooks"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/artifixer_cuda13.sh" <<EOF
export _ARTIFIXER_OLD_CUDA_HOME="\${CUDA_HOME:-}"
export _ARTIFIXER_OLD_PATH="\${PATH:-}"
export _ARTIFIXER_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export _ARTIFIXER_OLD_PYTHONPATH="\${PYTHONPATH:-}"
if [[ -n "$CUDA_HOME" ]]; then
  export CUDA_HOME="$CUDA_HOME"
  export PATH="$CUDA_HOME/bin:\$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"
fi
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST_VALUE"
export FORCE_CUDA=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$REPO_ROOT:$THREEDGRUT_ROOT:\${PYTHONPATH:-}"
EOF
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/artifixer_cuda13.sh" <<'EOF'
if [ -n "${_ARTIFIXER_OLD_CUDA_HOME+x}" ]; then export CUDA_HOME="$_ARTIFIXER_OLD_CUDA_HOME"; else unset CUDA_HOME; fi
if [ -n "${_ARTIFIXER_OLD_PATH+x}" ]; then export PATH="$_ARTIFIXER_OLD_PATH"; fi
if [ -n "${_ARTIFIXER_OLD_LD_LIBRARY_PATH+x}" ]; then export LD_LIBRARY_PATH="$_ARTIFIXER_OLD_LD_LIBRARY_PATH"; else unset LD_LIBRARY_PATH; fi
if [ -n "${_ARTIFIXER_OLD_PYTHONPATH+x}" ]; then export PYTHONPATH="$_ARTIFIXER_OLD_PYTHONPATH"; else unset PYTHONPATH; fi
unset _ARTIFIXER_OLD_CUDA_HOME _ARTIFIXER_OLD_PATH _ARTIFIXER_OLD_LD_LIBRARY_PATH _ARTIFIXER_OLD_PYTHONPATH
unset CC CXX TORCH_CUDA_ARCH_LIST FORCE_CUDA PYTORCH_CUDA_ALLOC_CONF
EOF
# Apply current activation hook immediately.
# shellcheck source=/dev/null
set +u
source "$CONDA_PREFIX/etc/conda/activate.d/artifixer_cuda13.sh"
set -u
export PIP_INDEX_URL PIP_EXTRA_INDEX_URL

log "Environment summary"
python - <<'PY'
import os, platform, sys
print('python', sys.version)
print('platform', platform.platform(), platform.machine())
print('CONDA_PREFIX', os.environ.get('CONDA_PREFIX'))
print('CUDA_HOME', os.environ.get('CUDA_HOME'))
print('CC', os.environ.get('CC'))
print('CXX', os.environ.get('CXX'))
print('TORCH_CUDA_ARCH_LIST', os.environ.get('TORCH_CUDA_ARCH_LIST'))
PY
if [[ -n "$CUDA_HOME" ]]; then
  nvcc --version | tail -4
else
  warn "CUDA_HOME not set; CUDA extensions may fail to build until CUDA is available"
fi

log "Installing conda build helpers (cmake, ninja)"
conda install -y -c conda-forge cmake ninja

python -m pip install --upgrade pip "setuptools<82" wheel packaging
TORCHVISION_SPEC="torchvision"
if [[ -n "${TORCHVISION_VERSION:-}" ]]; then
  TORCHVISION_SPEC="torchvision==$TORCHVISION_VERSION"
fi
if python - "$TORCH_VERSION" "$TORCHVISION_VERSION" <<'PYTORCHCHECK'
import sys
want_torch, want_tv = sys.argv[1], sys.argv[2]
try:
    import torch, torchvision
except Exception:
    raise SystemExit(1)
ok = True
if want_torch:
    ok = ok and torch.__version__.split('+', 1)[0] == want_torch
if want_tv:
    ok = ok and torchvision.__version__.split('+', 1)[0] == want_tv
print('existing torch', torch.__version__, 'torchvision', torchvision.__version__)
raise SystemExit(0 if ok else 1)
PYTORCHCHECK
then
  log "Reusing existing matching PyTorch/Torchvision install"
else
  if [[ -n "$TORCH_VERSION" && -n "$TORCH_INDEX_URL" ]]; then
    log "Installing PyTorch $TORCH_VERSION from $TORCH_INDEX_URL (pip mirror: $PIP_INDEX_URL)"
    python -m pip install --no-cache-dir "torch==$TORCH_VERSION" "$TORCHVISION_SPEC" --index-url "$TORCH_INDEX_URL"
  elif [[ -n "$TORCH_VERSION" ]]; then
    log "Installing PyTorch $TORCH_VERSION from pip mirror: $PIP_INDEX_URL"
    python -m pip install --no-cache-dir "torch==$TORCH_VERSION" "$TORCHVISION_SPEC"
  else
    log "Installing PyTorch from pip mirror: $PIP_INDEX_URL"
    python -m pip install --no-cache-dir torch "$TORCHVISION_SPEC"
  fi
fi
python -m pip uninstall -y flash-attn flash-attn-3 flash_attn_3 opencv-python || true

log "Installing 3DGRUT requirements and editable package"
cd "$THREEDGRUT_ROOT"
# fused-ssim's setup imports torch while computing build requirements.  pip's
# default PEP517 build isolation hides the already-installed torch, so install
# the rest of requirements first, then build fused-ssim with --no-build-isolation.
REQ_NO_FUSED="/tmp/artifixer_3dgrut_requirements_no_fused_ssim.txt"
grep -v 'git+https://github.com/rahul-goel/fused-ssim' requirements.txt > "$REQ_NO_FUSED"
python -m pip install -r "$REQ_NO_FUSED"
python -m pip install --no-build-isolation "fused_ssim @ git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157"
# 3DGRUT's install_slangc.sh downloads directly from GitHub. On machines where
# GitHub release assets are slow, pre-fill the exact tarball path it expects
# using an optional proxy or explicit mirror URL, then let the upstream script
# do its normal version check and extraction.
SLANGC_VERSION="${SLANGC_VERSION:-2026.5.2}"
case "$(uname -m)" in
  x86_64|amd64) SLANGC_ARCH="x86_64" ;;
  aarch64|arm64) SLANGC_ARCH="aarch64" ;;
  *) fatal "Unsupported platform for slangc: $(uname -m)" ;;
esac
SLANGC_TARBALL="/tmp/slang-${SLANGC_VERSION}-linux-${SLANGC_ARCH}.tar.gz"
SLANGC_UPSTREAM_URL="https://github.com/shader-slang/slang/releases/download/v${SLANGC_VERSION}/slang-${SLANGC_VERSION}-linux-${SLANGC_ARCH}.tar.gz"
if [[ ! -f "$SLANGC_TARBALL" ]]; then
  if [[ -n "${SLANGC_DOWNLOAD_URL:-}" ]]; then
    log "Downloading slangc via SLANGC_DOWNLOAD_URL"
    wget -O "$SLANGC_TARBALL" "$SLANGC_DOWNLOAD_URL"
  elif [[ -n "${SLANGC_GITHUB_PROXY:-}" ]]; then
    log "Downloading slangc via proxy: $SLANGC_GITHUB_PROXY"
    wget -O "$SLANGC_TARBALL" "${SLANGC_GITHUB_PROXY}${SLANGC_UPSTREAM_URL}"
  fi
fi
bash scripts/install_slangc.sh "$CONDA_PREFIX"
python -m pip install -e .

verify_fa3() {
  python - <<'PY'
import importlib
importlib.import_module('flash_attn_3')
importlib.import_module('flash_attn_interface')
print('FA3 import verification ok')
PY
}

install_fa3_binary_if_available() {
  log "Trying FA3 prebuilt wheel first (if one exists for this Python/PyTorch/CUDA combo)"
  # PyPI currently has only a placeholder flash-attn-3 wheel on many combos; verify strictly.
  if python -m pip install --only-binary=:all: flash-attn-3 flash_attn_3; then
    if verify_fa3; then
      log "FA3 prebuilt wheel works"
      return 0
    fi
  fi
  warn "No working FA3 prebuilt wheel found; falling back to source build from Dao-AILab/flash-attention/hopper"
  python -m pip uninstall -y flash-attn-3 flash_attn_3 || true
  return 1
}

install_fa3_from_source() {
  log "Building FA3 from source (this can take a while)"
  rm -rf /tmp/flash-attention
  git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention
  cd /tmp/flash-attention/hopper
  MAX_JOBS="$FLASH_ATTN_MAX_JOBS" \
  NVCC_THREADS="$FLASH_ATTN_NVCC_THREADS" \
  python -m pip install --no-build-isolation . 2>&1 | awk '!/^ptxas info/'
  cd /
  rm -rf /tmp/flash-attention
  verify_fa3
}

case "$FLASH_ATTN3_MODE" in
  skip)
    warn "Skipping FA3 install because FLASH_ATTN3_MODE=skip"
    ;;
  source)
    install_fa3_from_source
    ;;
  auto)
    install_fa3_binary_if_available || install_fa3_from_source
    ;;
  *)
    fatal "Invalid FLASH_ATTN3_MODE=$FLASH_ATTN3_MODE (expected auto|source|skip)"
    ;;
esac

if [[ "$USE_CU13" == "1" ]]; then
  log "Installing FA4 using prebuilt PyPI wheel: $FLASH_ATTN4_SPEC"
  python -m pip install --pre --only-binary=:all: "$FLASH_ATTN4_SPEC"
  # Dockerfile.cuda13 workaround: flash-attn-4 can pull a cuda-python 13.x namespace
  # stub missing cuda.bindings.driver on x86; pin the monolithic package used there.
  python -m pip install --force-reinstall --no-deps "cuda-python==12.6.2.post1"
else
  warn "Skipping FA4 CUDA13 wheel; pass --cu13 to install flash-attn-4[cu13]"
fi

log "Installing HuggingFace and ArtiFixer runtime/training dependencies"
python -m pip install accelerate==1.13.0 diffusers==0.37.1 transformers==5.5.0 ftfy
python -m pip install \
  einops scipy wandb tqdm Pillow matplotlib opencv-python-headless \
  pyyaml torchmetrics imageio-ffmpeg h5py av torch-fidelity \
  git+https://github.com/microsoft/MoGe.git

log "Smoke imports matching Dockerfile.cuda13"
cd "$REPO_ROOT"
python - <<'PY'
import importlib, os, torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0), 'cap', torch.cuda.get_device_capability(0))
mods = ['cv2', 'diffusers', 'transformers', 'accelerate', 'threedgrut']
if os.environ.get('FLASH_ATTN3_MODE') != 'skip':
    mods[:0] = ['flash_attn_interface', 'flash_attn_3']
for mod in mods:
    importlib.import_module(mod)
    print(mod, 'ok')
from moge.model.v2 import MoGeModel
print('MoGe ok')
PY
if [[ "$USE_CU13" == "1" ]]; then
  python -m pip show flash-attn-4 | grep Version && echo 'FA4 ok'
fi

if [[ "$RUN_SANITY" == "1" ]]; then
  log "Running tests/container_sanity_check.py"
  python tests/container_sanity_check.py
fi

log "Done. Activate with: conda activate $ENV_NAME"
