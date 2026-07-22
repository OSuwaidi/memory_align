#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=500:00:00
#SBATCH --job-name=wb-agents

set -euo pipefail

PROJECT=/shared/b00090279/memory_align
SWEEP=osuwaidi-khalifa-university/FINAL_MAL_CIFAR10/mpj6tnpm

cd "$PROJECT"

# sbatch does not source ~/.bashrc; pull in the /shared path redirects
# (uv python dir, wandb caches, etc.) explicitly.
. /shared/b00090279/shared-paths.sh

# ---------------------------------------------------------------------------
# Node-local venv.
#
# The shared venv on NFS breaks whenever `uv sync` runs on the login node
# while a job reads it: uv unlinks in-use .so files, NFS silly-renames them
# to .nfsXXXX, and torch dies with "libcudnn.so.9: cannot open shared object
# file". Fix: each job copies the venv to the node's local NVMe (/scratch)
# and runs from there, so login-node syncs can never touch a running job.
#
# The shared venv is treated as read-only reference. To update packages:
#   1. ensure `squeue --me` is empty and no memory_align/.venv procs run
#   2. `uv sync --all-groups` on the login node
#   3. resubmit — new jobs copy the fresh venv.
# ---------------------------------------------------------------------------
SRC_VENV="$PROJECT/.venv"
LOCAL_ROOT="/scratch/$USER/wb-${SLURM_JOB_ID:-manual}"
LOCAL_VENV="$LOCAL_ROOT/.venv"

cleanup() { rm -rf "$LOCAL_ROOT" 2>/dev/null || true; }
trap cleanup EXIT

echo "[$(hostname)] copying venv -> $LOCAL_VENV"
mkdir -p "$LOCAL_ROOT"
# -a preserves symlinks/perms; the venv's interpreter symlink points at the
# /shared uv python, which every node can still reach over NFS.
rsync -a --delete "$SRC_VENV/" "$LOCAL_VENV/"

# Repoint the venv to its new absolute location so console-scripts and
# `python` resolve locally instead of back to /shared/.../.venv.
sed -i "s|^home = .*|home = $(readlink -f "$LOCAL_VENV")/bin|" "$LOCAL_VENV/pyvenv.cfg" 2>/dev/null || true
export VIRTUAL_ENV="$LOCAL_VENV"
export PATH="$LOCAL_VENV/bin:$PATH"
unset PYTHONHOME

# Preflight: fail fast with a clear message rather than cryptic import errors.
"$LOCAL_VENV/bin/python" -c "import torch; assert torch.cuda.is_available()" || {
    echo "[FATAL] venv broken or no GPU on $(hostname). On the login node with no jobs running: uv sync --all-groups" >&2
    exit 1
}

# Run the agent from the node-local venv. No uv here: uv would try to
# re-resolve against $PROJECT and defeat the point. Call the entry point
# directly out of the local venv.
# Do NOT set CUDA_VISIBLE_DEVICES; SLURM sets it from --gres=gpu:1.
echo "[$(hostname)] starting agent for $SWEEP"
exec "$LOCAL_VENV/bin/wandb" agent --forward-signals "$SWEEP"
