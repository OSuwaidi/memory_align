#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=500:00:00
#SBATCH --job-name=mal-wb-agent
#SBATCH --output=/shared/b00090279/memory_align/logs/agent-%A_%a.out
#SBATCH --error=/shared/b00090279/memory_align/logs/agent-%A_%a.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
EXPECTED_SWEEP_PREFIX=osuwaidi-khalifa-university/MAL_benchmark/
SWEEP_PATH=${1:?"usage: sbatch --array=1-15 wb-agents.sh <entity/project/sweep-id>"}

case "$SWEEP_PATH" in
    "$EXPECTED_SWEEP_PREFIX"*) ;;
    *)
        echo "Refusing unexpected sweep path: $SWEEP_PATH" >&2
        echo "Expected prefix: $EXPECTED_SWEEP_PREFIX" >&2
        exit 2
        ;;
esac

# Every path capable of receiving job-created files is redirected beneath the
# user's explicitly authorized /shared directory.
. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"

CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
if [[ ! -x "$CLUSTER_PYTHON" ]]; then
    echo "Cluster environment is missing: $CLUSTER_PYTHON" >&2
    exit 1
fi

RESOLVED_PYTHON=$(command -v python)
if [[ "$(readlink -f "$RESOLVED_PYTHON")" != "$(readlink -f "$CLUSTER_PYTHON")" ]]; then
    echo "Sweep child interpreter mismatch: python resolves to $RESOLVED_PYTHON, expected $CLUSTER_PYTHON" >&2
    exit 1
fi

# Fail before claiming a sweep run if the allocated GPU or CUDA runtime is bad.
"$CLUSTER_PYTHON" - <<'PY'
import torch
from torch import nn

assert torch.cuda.is_available(), "CUDA is unavailable in the SLURM allocation"
device = torch.device("cuda")
nn.Conv2d(3, 16, 3, padding=1).to(device)(torch.randn(2, 3, 32, 32, device=device))
torch.cuda.synchronize()
print(
    f"CUDA preflight passed: torch={torch.__version__}, "
    f"device={torch.cuda.get_device_name(0)}, cudnn={torch.backends.cudnn.version()}"
)
PY

echo "[$(hostname)] array task ${SLURM_ARRAY_TASK_ID:-single} starting W&B agent for $SWEEP_PATH"
exec "$CLUSTER_PYTHON" -m wandb agent --forward-signals "$SWEEP_PATH"
