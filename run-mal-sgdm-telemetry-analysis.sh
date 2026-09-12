#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=04:00:00
#SBATCH --job-name=mal-telemetry-analysis
#SBATCH --output=/shared/b00090279/memory_align/logs/mal-sgdm-telemetry-analysis-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/mal-sgdm-telemetry-analysis-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
SUITE_DIRECTORY=${1:?Usage: sbatch run-mal-sgdm-telemetry-analysis.sh SUITE_DIRECTORY}
OUTPUT_DIRECTORY="$SUITE_DIRECTORY/aggregate"
RECEIPT="$MEMORY_ALIGN_PROJECT/logs/mal-sgdm-telemetry-analysis-${SLURM_JOB_ID}.env"

. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"

case "$SUITE_DIRECTORY" in
    "$MEMORY_ALIGN_PROJECT"/outputs/mal-sgdm-telemetry-suite-*) ;;
    *) echo "Unexpected suite directory: $SUITE_DIRECTORY" >&2; exit 2 ;;
esac
if [[ ! -x "$CLUSTER_PYTHON" ]]; then
    echo "Cluster environment is missing: $CLUSTER_PYTHON" >&2
    exit 1
fi

"$CLUSTER_PYTHON" analysis/analyze_mal_sgdm_telemetry_suite.py \
    --suite "$SUITE_DIRECTORY" \
    --output "$OUTPUT_DIRECTORY" \
    --wandb-mode online \
    --wandb-entity "$ENTITY_NAME" \
    --wandb-project "$PROJECT_NAME"

"$CLUSTER_PYTHON" - "$OUTPUT_DIRECTORY" <<'PY'
import csv
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
manifest = json.loads((output / "analysis_manifest.json").read_text())
assert manifest["n_independent_runs"] == 6, manifest
assert manifest["n_independent_runs_per_condition"] == 3, manifest
assert not manifest["synthetic_pipeline_check"], manifest
with (output / "run_manifest.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 6, len(rows)
assert all(int(row["completed_steps"]) == 33_200 for row in rows)
for stem in (
    "01_beta_eff_model_evolution",
    "01_gate_q_model_evolution",
    "02_beta_eff_depth_evolution",
    "03_beta_eff_kind_distributions_equal_tensor",
    "03_beta_eff_kind_distributions_numel_weighted",
    "04_beta_eff_stage_threshold_occupancy",
    "05_beta_eff_tensor_map",
    "06_training_performance_context",
):
    for extension in ("png", "pdf", "svg"):
        assert (output / "figures" / f"{stem}.{extension}").is_file(), (stem, extension)
print(f"Validated six-run telemetry analysis at {output}")
PY

printf 'ANALYSIS_JOB_ID=%q\nSUITE_DIRECTORY=%q\nOUTPUT_DIRECTORY=%q\nENTITY=%q\nPROJECT=%q\n' \
    "$SLURM_JOB_ID" "$SUITE_DIRECTORY" "$OUTPUT_DIRECTORY" "$ENTITY_NAME" "$PROJECT_NAME" >"$RECEIPT"

echo "MAL-SGDM telemetry analysis complete: $RECEIPT"
