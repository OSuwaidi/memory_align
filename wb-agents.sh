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

set -e

cd /shared/b00090279/memory_align

# sbatch does not source ~/.bashrc, so pull in the /shared path redirects
# (uv python dir, wandb caches, etc.) explicitly.
. /shared/b00090279/shared-paths.sh

# The venv lives on shared NFS and every agent reads from it concurrently.
# A re-sync would unlink .so files out from under running jobs; NFS then
# silly-renames them to .nfsXXXX and torch dies with "libcudnn.so.9: cannot
# open shared object file". --no-sync/UV_FROZEN keep agents read-only.
# Run `uv sync` manually on the login node ONLY when no jobs are queued.
export UV_FROZEN=1

# Fail fast with a clear message rather than 3 cryptic import errors.
.venv/bin/python -c "import torch; assert torch.cuda.is_available()" || {
    echo "[FATAL] venv broken or no GPU on $(hostname). Run 'uv sync --all-groups' on the login node with no jobs running." >&2
    exit 1
}

# Do NOT set CUDA_VISIBLE_DEVICES here; SLURM sets it from --gres=gpu:1.
uv run --no-sync wandb agent --forward-signals \
    osuwaidi-khalifa-university/FINAL_MAL_CIFAR10/mpj6tnpm