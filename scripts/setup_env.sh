#!/bin/bash
# Build verl/.venv from verl v0.9.1's own lock (vLLM + FSDP extras). Written for a RunPod node, where
# everything lives on the network volume (/workspace); change UV_CACHE_DIR / TMPDIR below elsewhere.
set -eo pipefail
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_LINK_MODE=copy
export TMPDIR=/workspace/tmp
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
cd "$PROJECT_DIR/verl"
date
# flash-attn 2.8.3 for torch 2.11 is gone from the verl wheelhouse; verl falls back to pure-torch padding utils
uv sync --frozen --all-packages --extra vllm --extra fsdp --no-install-package flash-attn
date
.venv/bin/python -c "import torch, vllm, transformers, verl, fla; print('torch', torch.__version__, '| vllm', vllm.__version__, '| transformers', transformers.__version__, '| fla', fla.__version__)"
# Extra deps used by hack_nucleation scripts
uv pip install --no-config --python .venv/bin/python fire python-dotenv
# Disable only CCCL's nvcc-vs-runtime-headers version check for TileLang JIT kernels
PROFILE=.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc.profile
grep -q CCCL_DISABLE_CTK_COMPATIBILITY_CHECK "$PROFILE" || \
    sed -i 's|^INCLUDES\(.*\)$|INCLUDES\1 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK|' "$PROFILE"
# On Blackwell (sm_100a) vLLM's flashinfer JIT-builds the trtllm-gen FMHA module with CUDA_HOME=nvidia/cu13
# (set in scripts/train_runpod_2xb200.sh). Its link step wants $CUDA_HOME/lib64/libcudart.so, which the pip
# wheel ships only as lib/libcudart.so.13.
CU13=.venv/lib/python3.12/site-packages/nvidia/cu13
[ -e "$CU13/lib64" ] || ln -s lib "$CU13/lib64"
[ -e "$CU13/lib/libcudart.so" ] || ln -s libcudart.so.13 "$CU13/lib/libcudart.so"
