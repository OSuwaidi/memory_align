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
SWEEP=osuwaidi-khalifa-university/FINAL_MAL_CIFAR100/0vo7fyc8

cd "$PROJECT"

# sbatch does not source ~/.bashrc; pull in the /shared path redirects
# (uv python dir, wandb caches, etc.) explicitly.
. /shared/b00090279/shared-paths.sh

# ---------------------------------------------------------------------------
# Immutable versioned venv on shared NFS -> no per-job copy.
#
# .venv is a symlink to .venv-<uvlock-hash>/, and venvs are never mutated in
# place: `./sync-venv.sh` builds a NEW versioned dir and flips the symlink,
# so a running job's venv is never unlinked out from under it. That means all
# nodes can run straight off the shared venv with zero copying and zero risk
# of the "libcudnn.so.9: cannot open shared object file" NFS corruption.
#
# To update packages:  ./sync-venv.sh   (safe to run even while jobs run),
# then resubmit; new jobs pick up the new .venv target, old jobs keep theirs.
# ---------------------------------------------------------------------------
VENV="$PROJECT/.venv"

# Resolve the symlink ONCE at job start and pin to that concrete version for
# the whole run, so a mid-run sync (which flips the .venv symlink) can't
# switch this job to a different venv underneath it.
VENV_REAL=$(readlink -f "$VENV")
export VIRTUAL_ENV="$VENV_REAL"
export PATH="$VENV_REAL/bin:$PATH"
unset PYTHONHOME

# Preflight: run a REAL cuDNN convolution, not just an import check — a
# poisoned venv can import torch and report cuda.is_available()=True while
# every conv fails with CUDNN_STATUS_NOT_INITIALIZED, burning sweep runs.
"$VENV_REAL/bin/python" -c "
import torch, torch.nn as nn
assert torch.cuda.is_available(), 'CUDA not available'
nn.Conv2d(3, 16, 3, padding=1).cuda()(torch.randn(2, 3, 32, 32, device='cuda'))
torch.cuda.synchronize()
" || {
    echo "[FATAL] venv broken or no GPU on $(hostname). On the login node: ./sync-venv.sh --rebuild" >&2
    exit 1
}

# Run the agent from the pinned venv. Invoke via `python -m wandb` (by the
# resolved versioned path, not the .venv symlink) so that even if a sync
# flips the .venv symlink mid-run, THIS job stays on its own venv. No uv
# here: uv would try to re-resolve against $PROJECT and could flip the venv.
# Do NOT set CUDA_VISIBLE_DEVICES; SLURM sets it from --gres=gpu:1.
echo "[$(hostname)] using venv $VENV_REAL"
echo "[$(hostname)] starting agent for $SWEEP"
exec "$VENV_REAL/bin/python" -m wandb agent --forward-signals "$SWEEP"
