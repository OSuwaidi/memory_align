#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=168:00:00
#SBATCH --job-name=agm-llm-agent
#SBATCH --output=/shared/b00090279/memory_align/logs/llm-pretrain-agent-%A_%a.out
#SBATCH --error=/shared/b00090279/memory_align/logs/llm-pretrain-agent-%A_%a.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
EXPECTED_SWEEP_PREFIX=osuwaidi-khalifa-university/MAL_benchmark/
SWEEP_PATH=${1:?"usage: sbatch --array=1-15 wb-llm-pretrain-agent.sh <entity/project/sweep-id>"}

case "$SWEEP_PATH" in
    "$EXPECTED_SWEEP_PREFIX"*) ;;
    *)
        echo "Refusing unexpected sweep path: $SWEEP_PATH" >&2
        exit 2
        ;;
esac

. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"

CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
[[ -x "$CLUSTER_PYTHON" ]] || {
    echo "Cluster environment is missing: $CLUSTER_PYTHON" >&2
    exit 1
}

"$CLUSTER_PYTHON" - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable in the allocation"
assert torch.cuda.is_bf16_supported(), "The GPU does not support bfloat16"
print(f"CUDA preflight: torch={torch.__version__}, device={torch.cuda.get_device_name(0)}")
PY

exec "$CLUSTER_PYTHON" -m wandb agent --forward-signals "$SWEEP_PATH"
