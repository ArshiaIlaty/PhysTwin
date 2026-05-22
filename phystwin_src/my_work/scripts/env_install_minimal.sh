#!/bin/bash
# Minimal PhysTwin install for inference + force visualization.
# Pip-only (avoids HPC conda-repo flakiness). Skips TRELLIS, Grounded-SAM-2,
# RealSense, SDXL. Run from the repo root after `conda activate <env>` so the
# right pip/python are first on PATH.
set -euo pipefail

PIP_FLAGS="--retries 10 --timeout 60 --no-cache-dir"

echo "[$(date)] === Stage 1: numpy ==="
pip install $PIP_FLAGS "numpy==1.26.4"

echo "[$(date)] === Stage 2: warp + basic deps ==="
pip install $PIP_FLAGS warp-lang
pip install $PIP_FLAGS usd-core matplotlib
pip install $PIP_FLAGS "pyglet<2"
pip install $PIP_FLAGS open3d
pip install $PIP_FLAGS trimesh rtree pyrender

echo "[$(date)] === Stage 3: pytorch 2.4.0 + cu121 (official wheels) ==="
pip install $PIP_FLAGS \
  --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.4.0" "torchvision==0.19.0" "torchaudio==2.4.0"

echo "[$(date)] === Stage 4: smaller utility libs ==="
pip install $PIP_FLAGS stannum termcolor fvcore wandb moviepy imageio cma pynput atomics plyfile einops
pip install $PIP_FLAGS opencv-python==4.10.0.84
pip install $PIP_FLAGS scikit-learn pandas tqdm pyyaml

echo "[$(date)] === Stage 5: pytorch3d wheel (py310 cu121 pyt240) ==="
pip install $PIP_FLAGS pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html \
  --no-index || {
    echo "Falling back: build pytorch3d from source"
    pip install $PIP_FLAGS "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"
}

echo "[$(date)] === Stage 6: gaussian-splatting deps + compiled submodules ==="
pip install $PIP_FLAGS gsplat==1.4.0 kornia
# These compile against CUDA; need `module load cuda/12.2.0` (or matching) before running this script.
# --no-build-isolation: setup.py imports torch, which is only available in this env.
# TORCH_CUDA_ARCH_LIST: required when compiling on a CPU node (torch can't auto-detect arch
# without a visible GPU). Covers HPC3's V100/A30/A100/A40/A6000/L40S.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;8.9}"
pip install $PIP_FLAGS --no-build-isolation ./gaussian_splatting/submodules/diff-gaussian-rasterization/
pip install $PIP_FLAGS --no-build-isolation ./gaussian_splatting/submodules/simple-knn/

echo "[$(date)] === DONE ==="
