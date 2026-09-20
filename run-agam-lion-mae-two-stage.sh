#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=500:00:00
#SBATCH --job-name=agam-lion-mae-two-stage
#SBATCH --output=/shared/b00090279/memory_align/logs/agam-lion-mae-two-stage-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/agam-lion-mae-two-stage-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
DATA_DIR="$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200"
SCREEN_AGENT_COUNT=15
SCREEN_EXPECTED_RUNS=18
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

SCREEN_OUTPUT=$(
    "$CLUSTER_PYTHON" sweeps/agam_lion_mae_extended_sweep.py \
        tasks/mae_pretrain.py \
        --sweep_name "agam-lion-mae-probe-screen-${SLURM_JOB_ID}" \
        --project_name "$PROJECT_NAME" \
        --data_dir "$DATA_DIR" \
        --output_dir "$MEMORY_ALIGN_PROJECT/outputs/agam-lion-mae-probe-screen"
)
printf '%s\n' "$SCREEN_OUTPUT"
SCREEN_PATH=$(extract_value SWEEP_PATH "$SCREEN_OUTPUT")
[[ "$(extract_value EXPECTED_RUNS "$SCREEN_OUTPUT")" == "$SCREEN_EXPECTED_RUNS" ]]

case "$SCREEN_PATH" in
    "$ENTITY_NAME/$PROJECT_NAME/"*) ;;
    *) echo "Refusing unexpected screen path: $SCREEN_PATH" >&2; exit 2 ;;
esac

SCREEN_SUBMISSION=$(sbatch \
    --parsable \
    --array="1-${SCREEN_AGENT_COUNT}" \
    --job-name=agam-lion-probe-screen \
    "$MEMORY_ALIGN_PROJECT/wb-agents.sh" \
    "$SCREEN_PATH")
ACTIVE_AGENT_JOB_ID=${SCREEN_SUBMISSION%%;*}
echo "Submitted 15 probe-screen agents as array $ACTIVE_AGENT_JOB_ID for $SCREEN_PATH."
wait_for_job "$ACTIVE_AGENT_JOB_ID" "probe screen"
ACTIVE_AGENT_JOB_ID=""
validate_sweep "$SCREEN_PATH" "$SCREEN_EXPECTED_RUNS" || {
    echo "Probe screen $SCREEN_PATH did not finish with exactly 18 successful runs." >&2
    exit 1
}

FINETUNE_OUTPUT=$(
    "$CLUSTER_PYTHON" sweeps/mae_lion_finetune_confirmation_sweep.py \
        tasks/mae_finetune_eval.py \
        --screen_path "$SCREEN_PATH" \
        --sweep_name "lion-agam-lion-selected-mae-finetune-${SLURM_JOB_ID}" \
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

RECEIPT="$MEMORY_ALIGN_PROJECT/logs/agam-lion-mae-two-stage-${SLURM_JOB_ID}.env"
printf 'SCREEN_PATH=%q\nFINETUNE_PATH=%q\n' "$SCREEN_PATH" "$FINETUNE_PATH" >"$RECEIPT"
printf '%s\n' "$FINETUNE_OUTPUT" | sed -n '/^SELECTED_AGAM_CONFIG=/p;/^SOURCE_RUN_IDS=/p' >>"$RECEIPT"

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

echo "Two-stage Lion comparison completed. Receipt: $RECEIPT"
