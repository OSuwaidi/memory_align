#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=500:00:00
#SBATCH --job-name=agam-lion-final-then-ft
#SBATCH --output=/shared/b00090279/memory_align/logs/agam-lion-final-then-ft-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/agam-lion-final-then-ft-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
DATA_DIR="$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200"
BASE_SCREEN_PATH=${1:?"usage: sbatch run-agam-lion-final-screen-and-finetune.sh <base-screen-path> [old-master-job] [base-agent-job]"}
SUPERSEDED_MASTER_JOB_ID=${2:-}
BASE_AGENT_JOB_ID=${3:-}
FINAL_SCREEN_AGENT_COUNT=6
FINAL_SCREEN_EXPECTED_RUNS=6
COMBINED_SCREEN_EXPECTED_RUNS=24
FINETUNE_AGENT_COUNT=4
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
case "$BASE_SCREEN_PATH" in
    "$ENTITY_NAME/$PROJECT_NAME/"*) ;;
    *) echo "Refusing unexpected base screen path: $BASE_SCREEN_PATH" >&2; exit 2 ;;
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

wait_for_job() {
    local job_id=$1
    local label=$2
    local queue_snapshot
    while :; do
        queue_snapshot=$(squeue -h -j "$job_id" -o '%T' 2>/dev/null || true)
        [[ -z "$queue_snapshot" ]] && break
        echo "$(date -Is) $label array $job_id: $(printf '%s\n' "$queue_snapshot" | sort | uniq -c | xargs)"
        sleep 60
    done
    sacct -n -X -j "$job_id" --format=JobID,State,ExitCode,Elapsed -P 2>/dev/null || true
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

FINAL_SCREEN_OUTPUT=$(
    "$CLUSTER_PYTHON" sweeps/agam_lion_mae_extended_sweep.py \
        tasks/mae_pretrain.py \
        --sweep_name "agam-lion-mae-final-probe-screen-${SLURM_JOB_ID}" \
        --project_name "$PROJECT_NAME" \
        --data_dir "$DATA_DIR" \
        --output_dir "$MEMORY_ALIGN_PROJECT/outputs/agam-lion-mae-final-probe-screen" \
        --base_lrs 0.0001 \
        --weight_decays 0.3 0.5 0.75 \
        --seeds 42 1337 \
        --comparison_group "agam_lion_mae_final_probe_screen_v1" \
        --study_stage "agam_lion_mae_final_probe_screen"
)
printf '%s\n' "$FINAL_SCREEN_OUTPUT"
FINAL_SCREEN_PATH=$(extract_value SWEEP_PATH "$FINAL_SCREEN_OUTPUT")
[[ "$(extract_value EXPECTED_RUNS "$FINAL_SCREEN_OUTPUT")" == "$FINAL_SCREEN_EXPECTED_RUNS" ]]

case "$FINAL_SCREEN_PATH" in
    "$ENTITY_NAME/$PROJECT_NAME/"*) ;;
    *) echo "Refusing unexpected final screen path: $FINAL_SCREEN_PATH" >&2; exit 2 ;;
esac

FINAL_SCREEN_SUBMISSION=$(sbatch \
    --parsable \
    --array="1-${FINAL_SCREEN_AGENT_COUNT}" \
    --job-name=agam-lion-final-screen \
    "$MEMORY_ALIGN_PROJECT/wb-agents.sh" \
    "$FINAL_SCREEN_PATH")
ACTIVE_AGENT_JOB_ID=${FINAL_SCREEN_SUBMISSION%%;*}
echo "Submitted six final-screen agents as array $ACTIVE_AGENT_JOB_ID for $FINAL_SCREEN_PATH."
wait_for_job "$ACTIVE_AGENT_JOB_ID" "final probe screen"
ACTIVE_AGENT_JOB_ID=""
validate_sweep "$FINAL_SCREEN_PATH" "$FINAL_SCREEN_EXPECTED_RUNS" || {
    echo "Final screen $FINAL_SCREEN_PATH did not finish with exactly six successful runs." >&2
    exit 1
}

if [[ -n "$BASE_AGENT_JOB_ID" ]] && squeue -h -j "$BASE_AGENT_JOB_ID" 2>/dev/null | grep -q .; then
    wait_for_job "$BASE_AGENT_JOB_ID" "base probe screen"
fi
validate_sweep "$BASE_SCREEN_PATH" 18 || {
    echo "Base screen $BASE_SCREEN_PATH did not finish with exactly 18 successful runs." >&2
    exit 1
}

# The stopped predecessor is safe to terminate only after its screen agents
# have exited; SIGKILL avoids running its trap or launching its stale selector.
if [[ -n "$SUPERSEDED_MASTER_JOB_ID" ]] && squeue -h -j "$SUPERSEDED_MASTER_JOB_ID" 2>/dev/null | grep -q .; then
    scancel --signal=KILL "$SUPERSEDED_MASTER_JOB_ID"
    echo "Terminated superseded stopped coordinator $SUPERSEDED_MASTER_JOB_ID."
fi

FINETUNE_OUTPUT=$(
    "$CLUSTER_PYTHON" sweeps/mae_lion_finetune_confirmation_sweep.py \
        tasks/mae_finetune_eval.py \
        --screen_path "$BASE_SCREEN_PATH" \
        --screen_path "$FINAL_SCREEN_PATH" \
        --expected_screen_runs "$COMBINED_SCREEN_EXPECTED_RUNS" \
        --sweep_name "lion-agam-lion-final-selected-mae-finetune-${SLURM_JOB_ID}" \
        --project_name "$PROJECT_NAME" \
        --data_dir "$DATA_DIR"
)
printf '%s\n' "$FINETUNE_OUTPUT"
FINETUNE_PATH=$(extract_value SWEEP_PATH "$FINETUNE_OUTPUT")
[[ "$(extract_value EXPECTED_RUNS "$FINETUNE_OUTPUT")" == "$FINETUNE_EXPECTED_RUNS" ]]

case "$FINETUNE_PATH" in
    "$ENTITY_NAME/$PROJECT_NAME/"*) ;;
    *) echo "Refusing unexpected fine-tune path: $FINETUNE_PATH" >&2; exit 2 ;;
esac

RECEIPT="$MEMORY_ALIGN_PROJECT/logs/agam-lion-final-then-ft-${SLURM_JOB_ID}.env"
printf 'BASE_SCREEN_PATH=%q\nFINAL_SCREEN_PATH=%q\nFINETUNE_PATH=%q\n' \
    "$BASE_SCREEN_PATH" "$FINAL_SCREEN_PATH" "$FINETUNE_PATH" >"$RECEIPT"
printf '%s\n' "$FINETUNE_OUTPUT" | sed -n '/^SELECTED_AGAM_CONFIG=/p;/^SOURCE_RUN_IDS=/p;/^SCREEN_PATHS=/p' >>"$RECEIPT"

FINETUNE_SUBMISSION=$(sbatch \
    --parsable \
    --array="1-${FINETUNE_AGENT_COUNT}" \
    --job-name=lion-agam-lion-mae-ft \
    "$MEMORY_ALIGN_PROJECT/wb-agents.sh" \
    "$FINETUNE_PATH")
ACTIVE_AGENT_JOB_ID=${FINETUNE_SUBMISSION%%;*}
echo "Submitted four selected-checkpoint fine-tune agents as array $ACTIVE_AGENT_JOB_ID for $FINETUNE_PATH."
wait_for_job "$ACTIVE_AGENT_JOB_ID" "selected-checkpoint fine-tune"
ACTIVE_AGENT_JOB_ID=""
validate_sweep "$FINETUNE_PATH" "$FINETUNE_EXPECTED_RUNS" || {
    echo "Fine-tune sweep $FINETUNE_PATH did not finish with exactly four successful runs." >&2
    exit 1
}

echo "Final two-stage Lion comparison completed. Receipt: $RECEIPT"
