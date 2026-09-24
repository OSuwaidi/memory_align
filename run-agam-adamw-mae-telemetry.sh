#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --job-name=agam-adamw-telemetry
#SBATCH --output=/shared/b00090279/memory_align/logs/agam-adamw-telemetry-master-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/agam-adamw-telemetry-master-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY=osuwaidi-khalifa-university
PROJECT=MAL_benchmark
CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
ACTIVE_JOB=

. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"
mkdir -p logs outputs/agam-adamw-mae-telemetry outputs/mae

cleanup() {
    if [[ -n "$ACTIVE_JOB" ]] && squeue -h -j "$ACTIVE_JOB" 2>/dev/null | grep -q .; then
        scancel "$ACTIVE_JOB"
    fi
}
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 130' INT

if [[ ! -x "$CLUSTER_PYTHON" ]]; then
    echo "Cluster Python is missing: $CLUSTER_PYTHON" >&2
    exit 1
fi
if [[ ! -d "$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200/train" ]]; then
    "$CLUSTER_PYTHON" download_datasets.py --task tiny-imagenet --tiny_imagenet_dir "$MEMORY_ALIGN_PROJECT/data"
fi

CREATION_OUTPUT=$("$CLUSTER_PYTHON" sweeps/agam_adamw_mae_telemetry_sweep.py \
    tasks/mae_pretrain.py \
    --sweep_name "agam-adamw-mae-gate-telemetry-${SLURM_JOB_ID}" \
    --project_name "$PROJECT" \
    --data_dir "$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200" \
    --output_dir "$MEMORY_ALIGN_PROJECT/outputs/mae" \
    --telemetry_output_dir "$MEMORY_ALIGN_PROJECT/outputs/agam-adamw-mae-telemetry")
printf '%s\n' "$CREATION_OUTPUT"
SWEEP_PATH=$(printf '%s\n' "$CREATION_OUTPUT" | sed -nE 's/^SWEEP_PATH=([^[:space:]]+)$/\1/p' | tail -n 1)
case "$SWEEP_PATH" in
    "$ENTITY/$PROJECT/"*) ;;
    *) echo "Could not resolve the telemetry sweep path." >&2; exit 1 ;;
esac

RECEIPT="$MEMORY_ALIGN_PROJECT/logs/agam-adamw-mae-telemetry-${SLURM_JOB_ID}.env"
printf 'SWEEP_PATH=%q\nEXPECTED_RUNS=1\n' "$SWEEP_PATH" >"$RECEIPT"
SUBMISSION=$(sbatch --parsable --array=1-1 "$MEMORY_ALIGN_PROJECT/wb-agents.sh" "$SWEEP_PATH" 1)
ACTIVE_JOB=${SUBMISSION%%;*}
printf 'GPU_JOB=%q\n' "$ACTIVE_JOB" >>"$RECEIPT"

while squeue -h -j "$ACTIVE_JOB" 2>/dev/null | grep -q .; do
    states=$(squeue -h -j "$ACTIVE_JOB" -o '%T' 2>/dev/null | sort | uniq -c | xargs || true)
    echo "$(date -Is) telemetry GPU job $ACTIVE_JOB: $states"
    sleep 60
done
ACTIVE_JOB=

for attempt in {1..12}; do
    if "$CLUSTER_PYTHON" sweeps/validate_sweep.py "$SWEEP_PATH" --expected_runs 1; then
        break
    fi
    if (( attempt == 12 )); then
        echo "Telemetry sweep did not finish successfully." >&2
        exit 1
    fi
    sleep 10
done

RUN_ID=$("$CLUSTER_PYTHON" -c \
    'import sys, wandb; sweep=wandb.Api(timeout=120).sweep(sys.argv[1]); runs=list(sweep.runs); assert len(runs)==1 and runs[0].state=="finished"; print(runs[0].id)' \
    "$SWEEP_PATH")
RUN_DIRECTORY="$MEMORY_ALIGN_PROJECT/outputs/agam-adamw-mae-telemetry/$RUN_ID"
"$CLUSTER_PYTHON" - "$RUN_DIRECTORY" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run = json.loads((root / "run.json").read_text())
summary = json.loads((root / "analysis" / "summary.json").read_text())
index = json.loads((root / "telemetry" / "index.json").read_text())
assert run["status"] == summary["status"] == "complete"
assert run["completed_steps"] == run["planned_steps"] == index["completed_steps"] == summary["completed_steps"]
assert summary["valid_tensor_step_observations"] > 0
for stem in (
    "gate_evolution_overview",
    "gate_evolution_by_parameter_kind",
    "gate_depth_heatmaps",
    "late_gate_profile_by_parameter_kind",
    "tensor_vs_global_gate_counterfactual",
):
    assert (root / "analysis" / "figures" / f"{stem}.pdf").is_file()
    assert (root / "analysis" / "figures" / f"{stem}.png").is_file()
print(f"Validated complete AGAM-AdamW MAE telemetry: {root}")
PY
printf 'RUN_ID=%q\nRUN_DIRECTORY=%q\n' "$RUN_ID" "$RUN_DIRECTORY" >>"$RECEIPT"
echo "AGAM-AdamW MAE telemetry complete: $RECEIPT"
