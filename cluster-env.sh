#!/usr/bin/env bash

# This file is sourced by all AUS cluster jobs. Keep every writable cache,
# temporary directory, run artifact, and tool directory under the only path the
# user authorized for writes: /shared/b00090279/.

MEMORY_ALIGN_SHARED_ROOT=/shared/b00090279
MEMORY_ALIGN_PROJECT="$MEMORY_ALIGN_SHARED_ROOT/memory_align"

case "$MEMORY_ALIGN_PROJECT" in
    "$MEMORY_ALIGN_SHARED_ROOT"/*) ;;
    *)
        echo "Invalid cluster project path: $MEMORY_ALIGN_PROJECT" >&2
        return 2 2>/dev/null || exit 2
        ;;
esac

# W&B sweep commands use ``/usr/bin/env python`` for their child training
# process. Put the project environment first so the agent and every child run
# are guaranteed to use the same interpreter and dependency set.
export PATH="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin:/opt/slurm/bin:$MEMORY_ALIGN_SHARED_ROOT/.local/bin:$PATH"

export TMPDIR="$MEMORY_ALIGN_SHARED_ROOT/.tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$MEMORY_ALIGN_SHARED_ROOT/.cache"
export XDG_CONFIG_HOME="$MEMORY_ALIGN_SHARED_ROOT/.config"
export XDG_DATA_HOME="$MEMORY_ALIGN_SHARED_ROOT/.local/share"
export XDG_STATE_HOME="$MEMORY_ALIGN_SHARED_ROOT/.local/state"
export PIP_CACHE_DIR="$MEMORY_ALIGN_SHARED_ROOT/.cache/pip"
export PYTHONPYCACHEPREFIX="$MEMORY_ALIGN_SHARED_ROOT/.cache/pycache"
export PYTHONUSERBASE="$MEMORY_ALIGN_SHARED_ROOT/.local"
export UV_CACHE_DIR="$MEMORY_ALIGN_SHARED_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$MEMORY_ALIGN_SHARED_ROOT/.local/uv/python"
export UV_TOOL_DIR="$MEMORY_ALIGN_SHARED_ROOT/.local/uv/tools"
export UV_TOOL_BIN_DIR="$MEMORY_ALIGN_SHARED_ROOT/.local/bin"

export WANDB_CACHE_DIR="$MEMORY_ALIGN_SHARED_ROOT/.cache/wandb"
export WANDB_ARTIFACT_DIR="$MEMORY_ALIGN_SHARED_ROOT/.cache/wandb-artifacts"
export WANDB_CONFIG_DIR="$MEMORY_ALIGN_SHARED_ROOT/.config/wandb"
export WANDB_DATA_DIR="$MEMORY_ALIGN_SHARED_ROOT/.cache/wandb-data"
export WANDB_DIR="$MEMORY_ALIGN_PROJECT/wandb"
export WANDB_ENTITY=osuwaidi-khalifa-university
export WANDB_PROJECT=MAL_benchmark

export HF_HOME="$MEMORY_ALIGN_SHARED_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$MEMORY_ALIGN_SHARED_ROOT/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$MEMORY_ALIGN_SHARED_ROOT/.cache/huggingface/transformers"
export TORCH_HOME="$MEMORY_ALIGN_SHARED_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$MEMORY_ALIGN_SHARED_ROOT/.cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$MEMORY_ALIGN_SHARED_ROOT/.cache/torchinductor"
export TRITON_CACHE_DIR="$MEMORY_ALIGN_SHARED_ROOT/.cache/triton"
export CUDA_CACHE_PATH="$MEMORY_ALIGN_SHARED_ROOT/.cache/cuda"
export MPLCONFIGDIR="$MEMORY_ALIGN_SHARED_ROOT/.cache/matplotlib"
export JOBLIB_TEMP_FOLDER="$MEMORY_ALIGN_SHARED_ROOT/.tmp/joblib"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

umask 077
mkdir -p \
    "$TMPDIR" \
    "$JOBLIB_TEMP_FOLDER" \
    "$XDG_CACHE_HOME" \
    "$XDG_CONFIG_HOME" \
    "$XDG_DATA_HOME" \
    "$XDG_STATE_HOME" \
    "$UV_PYTHON_INSTALL_DIR" \
    "$UV_TOOL_DIR" \
    "$UV_TOOL_BIN_DIR" \
    "$WANDB_CACHE_DIR" \
    "$WANDB_ARTIFACT_DIR" \
    "$WANDB_CONFIG_DIR" \
    "$WANDB_DATA_DIR" \
    "$WANDB_DIR" \
    "$MEMORY_ALIGN_PROJECT/logs" \
    "$MEMORY_ALIGN_PROJECT/outputs"
