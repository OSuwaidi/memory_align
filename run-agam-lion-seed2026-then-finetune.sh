#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=72:00:00
#SBATCH --job-name=agam-lion-3seed-ft
#SBATCH --output=/shared/b00090279/memory_align/logs/agam-lion-3seed-ft-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/agam-lion-3seed-ft-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
DATA_DIR="$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200"
SEED2026_SCREEN_PATH=${1:?"missing AGAM-Lion seed-2026 screen path"}
FINETUNE_EXPECTED_RUNS=6
ACTIVE_AGENT_JOB_ID=""

. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"

case "$SEED2026_SCREEN_PATH" in
    "$ENTITY_NAME/$PROJECT_NAME/"*) ;;
    *) echo "Refusing unexpected sweep path: $SEED2026_SCREEN_PATH" >&2; exit 2 ;;
esac

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
    [[ -n "$value" ]] || {
        echo "Could not extract $key from sweep creation output." >&2
        return 1
    }
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

validate_sweep "$SEED2026_SCREEN_PATH" 2

FINETUNE_OUTPUT=$(
    "$CLUSTER_PYTHON" sweeps/mae_lion_finetune_confirmation_sweep.py \
        tasks/mae_finetune_eval.py \
        --screen_path "$ENTITY_NAME/$PROJECT_NAME/70dkclas" \
        --screen_path "$ENTITY_NAME/$PROJECT_NAME/4w6e5si4" \
        --screen_path "$ENTITY_NAME/$PROJECT_NAME/9ynl0vx9" \
        --screen_path "$SEED2026_SCREEN_PATH" \
        --expected_screen_runs 23 \
        --expected_seeds 42 1337 2026 \
        --lion_source_run_id pcqfobwq \
        --lion_source_run_id ajqbhr9a \
        --lion_source_run_id ztvfc3up \
        --sweep_name "lion-agam-lion-three-seed-selected-mae-finetune-${SLURM_JOB_ID}" \
        --project_name "$PROJECT_NAME" \
        --data_dir "$DATA_DIR"
)
printf '%s\n' "$FINETUNE_OUTPUT"
FINETUNE_PATH=$(extract_value SWEEP_PATH "$FINETUNE_OUTPUT")
[[ "$(extract_value EXPECTED_RUNS "$FINETUNE_OUTPUT")" == "$FINETUNE_EXPECTED_RUNS" ]]

RECEIPT="$MEMORY_ALIGN_PROJECT/logs/agam-lion-3seed-ft-${SLURM_JOB_ID}.env"
printf 'SEED2026_SCREEN_PATH=%q\nFINETUNE_PATH=%q\n' \
    "$SEED2026_SCREEN_PATH" "$FINETUNE_PATH" >"$RECEIPT"
printf '%s\n' "$FINETUNE_OUTPUT" | sed -n '/^SELECTED_AGAM_CONFIG=/p;/^SOURCE_RUN_IDS=/p;/^SCREEN_PATHS=/p' >>"$RECEIPT"

FINETUNE_SUBMISSION=$(sbatch \
    --parsable \
    --array="1-${FINETUNE_EXPECTED_RUNS}" \
    --time=12:00:00 \
    --job-name=lion-agam-lion-3seed-ft \
    "$MEMORY_ALIGN_PROJECT/wb-agents.sh" \
    "$FINETUNE_PATH" \
    1)
ACTIVE_AGENT_JOB_ID=${FINETUNE_SUBMISSION%%;*}
echo "Submitted matched six-run Lion/AGAM-Lion fine-tune array $ACTIVE_AGENT_JOB_ID for $FINETUNE_PATH."

while squeue -h -j "$ACTIVE_AGENT_JOB_ID" 2>/dev/null | grep -q .; do
    echo "$(date -Is) fine-tune array $ACTIVE_AGENT_JOB_ID remains active."
    sleep 60
done
ACTIVE_AGENT_JOB_ID=""
validate_sweep "$FINETUNE_PATH" "$FINETUNE_EXPECTED_RUNS"
echo "Three-seed Lion versus AGAM-Lion fine-tuning completed. Receipt: $RECEIPT"
