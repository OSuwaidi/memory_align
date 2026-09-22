#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=168:00:00
#SBATCH --job-name=agam-adamw-bias-mae
#SBATCH --output=/shared/b00090279/memory_align/logs/agam-adamw-bias-mae-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/agam-adamw-bias-mae-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
DATA_DIR="$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200"
OUTPUT_DIR="$MEMORY_ALIGN_PROJECT/outputs/agam-adamw-standard-correction-mae"
PRETRAIN_EXPECTED_RUNS=3
FINETUNE_EXPECTED_RUNS=3
ACTIVE_AGENT_JOB_ID=""

. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"

cancel_active_agents() {
    if [[ -n "$ACTIVE_AGENT_JOB_ID" ]] && squeue -h -j "$ACTIVE_AGENT_JOB_ID" 2>/dev/null | grep -q .; then
        scancel "$ACTIVE_AGENT_JOB_ID"
    fi
}
trap 'cancel_active_agents; exit 143' TERM
trap 'cancel_active_agents; exit 130' INT

extract_value() {
    local key=$1
    local output=$2
    local value
    value=$(printf '%s\n' "$output" | sed -nE "s/^${key}=(.*)$/\\1/p" | tail -n 1)
    [[ -n "$value" ]] || { echo "Could not extract $key." >&2; return 1; }
    printf '%s\n' "$value"
}

validate_sweep() {
    local sweep_path=$1
    local expected_runs=$2
    for _attempt in {1..18}; do
        if "$CLUSTER_PYTHON" sweeps/validate_sweep.py "$sweep_path" --expected_runs "$expected_runs"; then
            return 0
        fi
        sleep 10
    done
    return 1
}

wait_for_job() {
    local job_id=$1
    while squeue -h -j "$job_id" 2>/dev/null | grep -q .; do
        echo "$(date -Is) array $job_id remains active."
        sleep 60
    done
}

PRETRAIN_OUTPUT=$(
    "$CLUSTER_PYTHON" sweeps/agam_adamw_mae_bias_correction_sweep.py \
        tasks/mae_pretrain.py \
        --sweep_name "agam-adamw-standard-correction-mae-${SLURM_JOB_ID}" \
        --project_name "$PROJECT_NAME" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR"
)
printf '%s\n' "$PRETRAIN_OUTPUT"
PRETRAIN_PATH=$(extract_value SWEEP_PATH "$PRETRAIN_OUTPUT")
[[ "$(extract_value EXPECTED_RUNS "$PRETRAIN_OUTPUT")" == "$PRETRAIN_EXPECTED_RUNS" ]]

PRETRAIN_SUBMISSION=$(sbatch --parsable --array="1-${PRETRAIN_EXPECTED_RUNS}" --time=72:00:00 \
    --job-name=agam-adamw-bias-pretrain "$MEMORY_ALIGN_PROJECT/wb-agents.sh" "$PRETRAIN_PATH" 1)
ACTIVE_AGENT_JOB_ID=${PRETRAIN_SUBMISSION%%;*}
echo "Submitted pretraining array $ACTIVE_AGENT_JOB_ID for $PRETRAIN_PATH."
wait_for_job "$ACTIVE_AGENT_JOB_ID"
ACTIVE_AGENT_JOB_ID=""
validate_sweep "$PRETRAIN_PATH" "$PRETRAIN_EXPECTED_RUNS"

mapfile -t SOURCE_RUN_IDS < <(
    "$CLUSTER_PYTHON" - "$PRETRAIN_PATH" <<'PY'
import sys
import wandb

sweep = wandb.Api(timeout=180).sweep(sys.argv[1])
runs = sorted(sweep.runs, key=lambda run: int(run.config["seed"]))
if len(runs) != 3 or any(run.state != "finished" for run in runs):
    raise SystemExit("Expected exactly three finished source runs.")
for run in runs:
    print(run.id)
PY
)
[[ "${#SOURCE_RUN_IDS[@]}" == "$PRETRAIN_EXPECTED_RUNS" ]]

SOURCE_ARGS=()
for source_run_id in "${SOURCE_RUN_IDS[@]}"; do
    SOURCE_ARGS+=(--source_run_id "$source_run_id")
done
FINETUNE_OUTPUT=$(
    "$CLUSTER_PYTHON" sweeps/mae_selected_sources_finetune_sweep.py \
        tasks/mae_finetune_eval.py \
        "${SOURCE_ARGS[@]}" \
        --expected_optimizer MAL_AdamW \
        --expected_seeds 42 1337 2026 \
        --expected_batch_size 1024 \
        --sweep_name "agam-adamw-standard-correction-selected-finetune-${SLURM_JOB_ID}" \
        --project_name "$PROJECT_NAME" \
        --data_dir "$DATA_DIR" \
        --comparison_group agam_adamw_mae_first_moment_correction_finetune_v1 \
        --study_stage standard_first_moment_correction_end_to_end_finetune
)
printf '%s\n' "$FINETUNE_OUTPUT"
FINETUNE_PATH=$(extract_value SWEEP_PATH "$FINETUNE_OUTPUT")
[[ "$(extract_value EXPECTED_RUNS "$FINETUNE_OUTPUT")" == "$FINETUNE_EXPECTED_RUNS" ]]

RECEIPT="$MEMORY_ALIGN_PROJECT/logs/agam-adamw-bias-mae-${SLURM_JOB_ID}.env"
printf 'PRETRAIN_PATH=%q\nFINETUNE_PATH=%q\n' "$PRETRAIN_PATH" "$FINETUNE_PATH" >"$RECEIPT"

FINETUNE_SUBMISSION=$(sbatch --parsable --array="1-${FINETUNE_EXPECTED_RUNS}" --time=12:00:00 \
    --job-name=agam-adamw-bias-ft "$MEMORY_ALIGN_PROJECT/wb-agents.sh" "$FINETUNE_PATH" 1)
ACTIVE_AGENT_JOB_ID=${FINETUNE_SUBMISSION%%;*}
echo "Submitted fine-tune array $ACTIVE_AGENT_JOB_ID for $FINETUNE_PATH."
wait_for_job "$ACTIVE_AGENT_JOB_ID"
ACTIVE_AGENT_JOB_ID=""
validate_sweep "$FINETUNE_PATH" "$FINETUNE_EXPECTED_RUNS"
echo "AGAM-AdamW correction ablation completed. Receipt: $RECEIPT"
