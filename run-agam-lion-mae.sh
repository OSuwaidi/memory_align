#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=500:00:00
#SBATCH --job-name=agam-lion-mae-master
#SBATCH --output=/shared/b00090279/memory_align/logs/agam-lion-mae-master-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/agam-lion-mae-master-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
DATA_DIR="$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200"
SCREEN_AGENT_COUNT=7
CONFIRMATION_AGENT_COUNT=2

SCREEN_JOB_ID=""
LION_JOB_ID=""
AGAM_LION_JOB_ID=""

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
    local job_id
    for job_id in "$SCREEN_JOB_ID" "$LION_JOB_ID" "$AGAM_LION_JOB_ID"; do
        if [[ -n "$job_id" ]] && squeue -h -j "$job_id" 2>/dev/null | grep -q .; then
            scancel "$job_id"
        fi
    done
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

validate_sweep_path() {
    local sweep_path=$1
    case "$sweep_path" in
        "$ENTITY_NAME/$PROJECT_NAME/"*) ;;
        *)
            echo "Refusing unexpected sweep path: $sweep_path" >&2
            exit 2
            ;;
    esac
}

submit_agents() {
    local sweep_path=$1
    local count=$2
    local job_name=$3
    local submission
    submission=$(sbatch \
        --parsable \
        --array="1-${count}" \
        --job-name="$job_name" \
        "$MEMORY_ALIGN_PROJECT/wb-agents.sh" \
        "$sweep_path")
    printf '%s\n' "${submission%%;*}"
}

wait_for_job() {
    local job_id=$1
    local description=$2
    local queue_snapshot
    while :; do
        queue_snapshot=$(squeue -h -j "$job_id" -o '%T' 2>/dev/null || true)
        [[ -z "$queue_snapshot" ]] && break
        echo "$(date -Is) $description $job_id: $(printf '%s\n' "$queue_snapshot" | sort | uniq -c | xargs)"
        sleep 60
    done
    sacct -n -X -j "$job_id" --format=JobID,State,ExitCode,Elapsed -P 2>/dev/null || true
}

validate_finished_sweep() {
    local sweep_path=$1
    local expected_runs=$2
    for _attempt in {1..12}; do
        if "$CLUSTER_PYTHON" sweeps/validate_sweep.py "$sweep_path" --expected_runs "$expected_runs"; then
            return 0
        fi
        sleep 10
    done
    echo "Sweep $sweep_path did not finish with exactly $expected_runs successful runs." >&2
    return 1
}

SWEEP_RECORD="$MEMORY_ALIGN_PROJECT/logs/agam-lion-mae-sweeps-${SLURM_JOB_ID}.env"
SELECTION_RECEIPT="$MEMORY_ALIGN_PROJECT/logs/agam-lion-mae-selection-${SLURM_JOB_ID}.json"
: >"$SWEEP_RECORD"

SCREEN_OUTPUT=$("$CLUSTER_PYTHON" sweeps/agam_lion_mae_sweep.py \
    tasks/mae_pretrain.py \
    --stage screen \
    --sweep_name "agam-lion-mae-screen-${SLURM_JOB_ID}" \
    --project_name "$PROJECT_NAME" \
    --data_dir "$DATA_DIR" \
    --output_dir "$MEMORY_ALIGN_PROJECT/outputs/agam-lion-mae/screen")
printf '%s\n' "$SCREEN_OUTPUT"
SCREEN_SWEEP_PATH=$(extract_value SWEEP_PATH "$SCREEN_OUTPUT")
SCREEN_EXPECTED_RUNS=$(extract_value EXPECTED_RUNS "$SCREEN_OUTPUT")
validate_sweep_path "$SCREEN_SWEEP_PATH"
[[ "$SCREEN_EXPECTED_RUNS" == "8" ]] || {
    echo "The Lion screen must contain exactly 8 runs; found $SCREEN_EXPECTED_RUNS." >&2
    exit 1
}
printf 'SCREEN_SWEEP_PATH=%q\nSCREEN_EXPECTED_RUNS=%q\n' \
    "$SCREEN_SWEEP_PATH" "$SCREEN_EXPECTED_RUNS" >>"$SWEEP_RECORD"

# Seven free GPUs execute the eight-cell equal-budget screen. The first agent
# that finishes a cell claims the eighth run.
SCREEN_JOB_ID=$(submit_agents "$SCREEN_SWEEP_PATH" "$SCREEN_AGENT_COUNT" agam-lion-screen)
printf 'SCREEN_AGENT_JOB_ID=%q\n' "$SCREEN_JOB_ID" >>"$SWEEP_RECORD"
echo "Submitted seven screen agents as array $SCREEN_JOB_ID."
wait_for_job "$SCREEN_JOB_ID" "Lion/AGAM-Lion screen array"
SCREEN_JOB_ID=""
validate_finished_sweep "$SCREEN_SWEEP_PATH" "$SCREEN_EXPECTED_RUNS"

CONFIRMATION_OUTPUT=$("$CLUSTER_PYTHON" sweeps/agam_lion_mae_sweep.py \
    tasks/mae_pretrain.py \
    --stage confirmation \
    --sweep_name "agam-lion-mae-confirm-${SLURM_JOB_ID}" \
    --project_name "$PROJECT_NAME" \
    --data_dir "$DATA_DIR" \
    --output_dir "$MEMORY_ALIGN_PROJECT/outputs/agam-lion-mae/confirmation" \
    --source_sweep "$SCREEN_SWEEP_PATH" \
    --selection_receipt "$SELECTION_RECEIPT")
printf '%s\n' "$CONFIRMATION_OUTPUT"

LION_SWEEP_PATH=$(extract_value SWEEP_PATH_LION "$CONFIRMATION_OUTPUT")
AGAM_LION_SWEEP_PATH=$(extract_value SWEEP_PATH_AGAM_LION "$CONFIRMATION_OUTPUT")
LION_EXPECTED_RUNS=$(extract_value EXPECTED_RUNS_LION "$CONFIRMATION_OUTPUT")
AGAM_LION_EXPECTED_RUNS=$(extract_value EXPECTED_RUNS_AGAM_LION "$CONFIRMATION_OUTPUT")
validate_sweep_path "$LION_SWEEP_PATH"
validate_sweep_path "$AGAM_LION_SWEEP_PATH"
[[ "$LION_EXPECTED_RUNS" == "2" && "$AGAM_LION_EXPECTED_RUNS" == "2" ]] || {
    echo "Each confirmation sweep must contain exactly two new seed runs." >&2
    exit 1
}
printf 'SELECTION_RECEIPT=%q\nLION_SWEEP_PATH=%q\nAGAM_LION_SWEEP_PATH=%q\n' \
    "$SELECTION_RECEIPT" "$LION_SWEEP_PATH" "$AGAM_LION_SWEEP_PATH" >>"$SWEEP_RECORD"

LION_JOB_ID=$(submit_agents "$LION_SWEEP_PATH" "$CONFIRMATION_AGENT_COUNT" lion-mae-confirm)
AGAM_LION_JOB_ID=$(submit_agents "$AGAM_LION_SWEEP_PATH" "$CONFIRMATION_AGENT_COUNT" agam-lion-confirm)
printf 'LION_AGENT_JOB_ID=%q\nAGAM_LION_AGENT_JOB_ID=%q\n' \
    "$LION_JOB_ID" "$AGAM_LION_JOB_ID" >>"$SWEEP_RECORD"
echo "Submitted Lion confirmation array $LION_JOB_ID and AGAM-Lion confirmation array $AGAM_LION_JOB_ID."

wait_for_job "$LION_JOB_ID" "Lion confirmation array"
wait_for_job "$AGAM_LION_JOB_ID" "AGAM-Lion confirmation array"
LION_JOB_ID=""
AGAM_LION_JOB_ID=""
validate_finished_sweep "$LION_SWEEP_PATH" "$LION_EXPECTED_RUNS"
validate_finished_sweep "$AGAM_LION_SWEEP_PATH" "$AGAM_LION_EXPECTED_RUNS"

echo "Lion/AGAM-Lion MAE experiment completed. Receipt: $SWEEP_RECORD"
