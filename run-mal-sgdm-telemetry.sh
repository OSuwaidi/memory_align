#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=mal-sgdm-telemetry
#SBATCH --output=/shared/b00090279/memory_align/logs/mal-sgdm-telemetry-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/mal-sgdm-telemetry-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
OUTPUT_DIRECTORY="$MEMORY_ALIGN_PROJECT/outputs/mal-sgdm-telemetry-resnet18-cifar10-seed42-${SLURM_JOB_ID}"
RECEIPT="$MEMORY_ALIGN_PROJECT/logs/mal-sgdm-telemetry-${SLURM_JOB_ID}.env"

. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"

if [[ ! -x "$CLUSTER_PYTHON" ]]; then
    echo "Cluster environment is missing: $CLUSTER_PYTHON" >&2
    exit 1
fi
if [[ -e "$OUTPUT_DIRECTORY" ]]; then
    echo "Refusing to overwrite telemetry output: $OUTPUT_DIRECTORY" >&2
    exit 1
fi

"$CLUSTER_PYTHON" -m unittest checks.mal_sgdm_telemetry_checks
"$CLUSTER_PYTHON" download_datasets.py \
    --task cifar10 \
    --cifar10_dir "$MEMORY_ALIGN_PROJECT/data"

printf 'TELEMETRY_JOB_ID=%q\nOUTPUT_DIRECTORY=%q\nENTITY=%q\nPROJECT=%q\n' \
    "$SLURM_JOB_ID" "$OUTPUT_DIRECTORY" "$ENTITY_NAME" "$PROJECT_NAME" >"$RECEIPT"

"$CLUSTER_PYTHON" tasks/mal_sgdm_telemetry.py train \
    --output "$OUTPUT_DIRECTORY" \
    --data-dir "$MEMORY_ALIGN_PROJECT/data" \
    --device cuda \
    --epochs 200 \
    --batch-size 256 \
    --lr 0.1 \
    --weight-decay 0.0005 \
    --schedule cosine \
    --warmup-epochs 5 \
    --min-lr 0.00001 \
    --norm group \
    --augmentation repo \
    --amp-dtype bfloat16 \
    --float32-precision tf32 \
    --seed 42 \
    --split-seed 20260901 \
    --workers 4 \
    --flush-steps 64 \
    --log-every 100 \
    --late-fraction 0.2 \
    --wandb-mode online \
    --wandb-entity "$ENTITY_NAME" \
    --wandb-project "$PROJECT_NAME" \
    --wandb-name "MAL-SGDM gate telemetry · ResNet18/CIFAR-10 · seed 42"

"$CLUSTER_PYTHON" - "$OUTPUT_DIRECTORY" "$RECEIPT" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
receipt = Path(sys.argv[2])
run = json.loads((output / "run.json").read_text())
analysis = json.loads((output / "analysis" / "analysis.json").read_text())
assert run["status"] == "completed", run
assert run["completed_steps"] == run["planned_steps"] == analysis["persisted_steps"]
assert analysis["complete_tensor_step_coverage"]
assert analysis["observed_tensor_step_observations"] == analysis["expected_tensor_step_observations"]
wandb_info = run.get("wandb", {})
with receipt.open("a") as handle:
    handle.write(f"WANDB_RUN_ID={wandb_info.get('run_id', '')}\n")
    handle.write(f"WANDB_RUN_URL={wandb_info.get('run_url', '')}\n")
    handle.write(f"PERSISTED_STEPS={analysis['persisted_steps']}\n")
    handle.write(f"TENSOR_STEP_OBSERVATIONS={analysis['observed_tensor_step_observations']}\n")
print(f"Telemetry run validated: {output}")
PY

echo "MAL-SGDM telemetry complete: $RECEIPT"
