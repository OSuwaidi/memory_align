#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=240:00:00
#SBATCH --job-name=agam-lion-select-ft
#SBATCH --output=/shared/b00090279/memory_align/logs/agam-lion-select-ft-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/agam-lion-select-ft-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
DATA_DIR="$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200"

BASE_SCREEN_PATH=${1:?"missing original 18-cell screen path"}
FINAL_SEED42_SCREEN_PATH=${2:?"missing final-screen seed-42 path"}
BASE_RECOVERY_015_PATH=${3:?"missing WD=0.15 recovery path"}
BASE_RECOVERY_025_PATH=${4:?"missing WD=0.25 recovery path"}
FINAL_SEED1337_SCREEN_PATH=${5:?"missing final-screen seed-1337 path"}
SUPERSEDED_MASTER_ONE=${6:-}
SUPERSEDED_MASTER_TWO=${7:-}

COMBINED_SCREEN_EXPECTED_RUNS=24
FINETUNE_EXPECTED_RUNS=4
ACTIVE_AGENT_JOB_ID=""

. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"

[[ -x "$CLUSTER_PYTHON" ]] || {
    echo "Cluster environment is missing: $CLUSTER_PYTHON" >&2
    exit 1
}
[[ -d "$DATA_DIR/train" && -d "$DATA_DIR/val" ]] || {
    echo "Tiny-ImageNet is missing or incomplete beneath $DATA_DIR." >&2
    exit 1
}
for sweep_path in \
    "$BASE_SCREEN_PATH" \
    "$FINAL_SEED42_SCREEN_PATH" \
    "$BASE_RECOVERY_015_PATH" \
    "$BASE_RECOVERY_025_PATH" \
    "$FINAL_SEED1337_SCREEN_PATH"; do
    case "$sweep_path" in
        "$ENTITY_NAME/$PROJECT_NAME/"*) ;;
        *) echo "Refusing unexpected sweep path: $sweep_path" >&2; exit 2 ;;
    esac
done

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
    for _attempt in {1..12}; do
        if "$CLUSTER_PYTHON" sweeps/validate_sweep.py "$sweep_path" --expected_runs "$expected_runs"; then
            return 0
        fi
        sleep 10
    done
    return 1
}

# The dependency ensures the source and recovery GPU arrays have all exited.
# Each recovery sweep must be complete before model selection is allowed.
validate_sweep "$FINAL_SEED42_SCREEN_PATH" 3
validate_sweep "$BASE_RECOVERY_015_PATH" 2
validate_sweep "$BASE_RECOVERY_025_PATH" 1
validate_sweep "$FINAL_SEED1337_SCREEN_PATH" 3

FINETUNE_OUTPUT=$(
    "$CLUSTER_PYTHON" sweeps/mae_lion_finetune_confirmation_sweep.py \
        tasks/mae_finetune_eval.py \
        --screen_path "$BASE_SCREEN_PATH" \
        --screen_path "$FINAL_SEED42_SCREEN_PATH" \
        --screen_path "$BASE_RECOVERY_015_PATH" \
        --screen_path "$BASE_RECOVERY_025_PATH" \
        --screen_path "$FINAL_SEED1337_SCREEN_PATH" \
        --expected_screen_runs "$COMBINED_SCREEN_EXPECTED_RUNS" \
        --sweep_name "lion-agam-lion-final-selected-mae-finetune-${SLURM_JOB_ID}" \
        --project_name "$PROJECT_NAME" \
        --data_dir "$DATA_DIR"
)
printf '%s\n' "$FINETUNE_OUTPUT"
FINETUNE_PATH=$(extract_value SWEEP_PATH "$FINETUNE_OUTPUT")
[[ "$(extract_value EXPECTED_RUNS "$FINETUNE_OUTPUT")" == "$FINETUNE_EXPECTED_RUNS" ]]

RECEIPT="$MEMORY_ALIGN_PROJECT/logs/agam-lion-select-ft-${SLURM_JOB_ID}.env"
printf 'BASE_SCREEN_PATH=%q\nFINAL_SEED42_SCREEN_PATH=%q\nBASE_RECOVERY_015_PATH=%q\nBASE_RECOVERY_025_PATH=%q\nFINAL_SEED1337_SCREEN_PATH=%q\nFINETUNE_PATH=%q\n' \
    "$BASE_SCREEN_PATH" \
    "$FINAL_SEED42_SCREEN_PATH" \
    "$BASE_RECOVERY_015_PATH" \
    "$BASE_RECOVERY_025_PATH" \
    "$FINAL_SEED1337_SCREEN_PATH" \
    "$FINETUNE_PATH" >"$RECEIPT"
printf '%s\n' "$FINETUNE_OUTPUT" | sed -n '/^SELECTED_AGAM_CONFIG=/p;/^SOURCE_RUN_IDS=/p;/^SCREEN_PATHS=/p' >>"$RECEIPT"

FINETUNE_SUBMISSION=$(sbatch \
    --parsable \
    --array="1-${FINETUNE_EXPECTED_RUNS}" \
    --job-name=lion-agam-lion-mae-ft \
    "$MEMORY_ALIGN_PROJECT/wb-agents.sh" \
    "$FINETUNE_PATH" \
    1)
ACTIVE_AGENT_JOB_ID=${FINETUNE_SUBMISSION%%;*}
echo "Submitted four one-run selected-checkpoint fine-tune agents as array $ACTIVE_AGENT_JOB_ID for $FINETUNE_PATH."

while squeue -h -j "$ACTIVE_AGENT_JOB_ID" 2>/dev/null | grep -q .; do
    echo "$(date -Is) fine-tune array $ACTIVE_AGENT_JOB_ID remains active."
    sleep 60
done
ACTIVE_AGENT_JOB_ID=""
validate_sweep "$FINETUNE_PATH" "$FINETUNE_EXPECTED_RUNS"

# These coordinators were deliberately frozen before either could perform
# stale selection. They are safe to remove once the replacement has finished.
for old_master in "$SUPERSEDED_MASTER_ONE" "$SUPERSEDED_MASTER_TWO"; do
    if [[ -n "$old_master" ]] && squeue -h -j "$old_master" 2>/dev/null | grep -q .; then
        scancel --signal=KILL "$old_master"
        echo "Terminated superseded stopped coordinator $old_master."
    fi
done

echo "Final Lion versus AGAM-Lion pipeline completed. Receipt: $RECEIPT"
