#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=250:00:00
#SBATCH --job-name=mal-adamw-recursive
#SBATCH --output=/shared/b00090279/memory_align/logs/adamw-recursive-master-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/adamw-recursive-master-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
SOURCE_SWEEP_PATH=${1:?"usage: run-mal-adamw-in-place-followup.sh <entity/project/sweep-id>"}
AGENT_COUNT=15
EXPECTED_SOURCE_RUNS=120
EXPECTED_FOLLOWUP_RUNS=80
MAX_AGENT_ROUNDS=3
CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
ACTIVE_AGENT_JOB_ID=""

case "$SOURCE_SWEEP_PATH" in
    "$ENTITY_NAME/$PROJECT_NAME/"*) ;;
    *)
        echo "Refusing unexpected source sweep path: $SOURCE_SWEEP_PATH" >&2
        exit 2
        ;;
esac

. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"

cancel_active_agents() {
    local signal_name=${1:-TERM}
    if [[ -n "$ACTIVE_AGENT_JOB_ID" ]] && squeue -h -j "$ACTIVE_AGENT_JOB_ID" 2>/dev/null | grep -q .; then
        echo "Master received $signal_name; cancelling agent array $ACTIVE_AGENT_JOB_ID" >&2
        scancel "$ACTIVE_AGENT_JOB_ID"
    fi
}

trap 'cancel_active_agents TERM; exit 143' TERM
trap 'cancel_active_agents INT; exit 130' INT

extract_sweep_path() {
    local creation_output=$1
    local sweep_path
    sweep_path=$(printf '%s\n' "$creation_output" | sed -nE 's|.*wandb agent --forward-signals ([^[:space:]]+).*|\1|p' | tail -n 1)
    case "$sweep_path" in
        "$ENTITY_NAME/$PROJECT_NAME/"*) printf '%s\n' "$sweep_path" ;;
        *)
            echo "Could not extract the expected sweep path from:" >&2
            printf '%s\n' "$creation_output" >&2
            return 1
            ;;
    esac
}

submit_agents() {
    local sweep_path=$1
    local submission
    submission=$(sbatch \
        --parsable \
        --array="1-${AGENT_COUNT}" \
        --job-name=mal-adamw-recursive \
        "$MEMORY_ALIGN_PROJECT/wb-agents.sh" \
        "$sweep_path")
    printf '%s\n' "${submission%%;*}"
}

wait_for_agents() {
    local job_id=$1
    local queue_snapshot
    echo "Waiting for agent array $job_id"
    while :; do
        queue_snapshot=$(squeue -h -j "$job_id" -o '%T' 2>/dev/null || true)
        [[ -z "$queue_snapshot" ]] && break
        echo "$(date -Is) array $job_id: $(printf '%s\n' "$queue_snapshot" | sort | uniq -c | xargs)"
        sleep 60
    done
    sacct -n -X -j "$job_id" --format=JobID,State,ExitCode -P 2>/dev/null || true
}

sweep_state() {
    "$CLUSTER_PYTHON" - "$1" <<'PY'
import sys
import wandb

print(wandb.Api(timeout=180).sweep(sys.argv[1]).state)
PY
}

run_agents_until_complete() {
    local sweep_path=$1
    local expected_runs=$2
    local round
    local state

    for ((round = 1; round <= MAX_AGENT_ROUNDS; round++)); do
        ACTIVE_AGENT_JOB_ID=$(submit_agents "$sweep_path")
        echo "Submitted $sweep_path as a ${AGENT_COUNT}-GPU array $ACTIVE_AGENT_JOB_ID (round $round)"
        wait_for_agents "$ACTIVE_AGENT_JOB_ID"
        ACTIVE_AGENT_JOB_ID=""

        for _attempt in {1..12}; do
            if "$CLUSTER_PYTHON" sweeps/validate_sweep.py \
                "$sweep_path" \
                --expected_runs "$expected_runs"; then
                return 0
            fi
            sleep 10
        done

        state=$(sweep_state "$sweep_path")
        if [[ "$state" == "FINISHED" || "$state" == "CANCELED" ]]; then
            echo "Sweep reached $state without $expected_runs finished runs." >&2
            return 1
        fi
        echo "Sweep remains $state; submitting a recovery agent array."
    done
    echo "Sweep did not complete after $MAX_AGENT_ROUNDS agent rounds: $sweep_path" >&2
    return 1
}

if [[ ! -x "$CLUSTER_PYTHON" ]]; then
    echo "Cluster environment is missing: $CLUSTER_PYTHON" >&2
    exit 1
fi

"$CLUSTER_PYTHON" sweeps/validate_sweep.py \
    "$SOURCE_SWEEP_PATH" \
    --expected_runs "$EXPECTED_SOURCE_RUNS"

SOURCE_SWEEP_ID=${SOURCE_SWEEP_PATH##*/}
SELECTION_DIR="$MEMORY_ALIGN_PROJECT/logs/mal-adamw-selection-${SOURCE_SWEEP_ID}-${SLURM_JOB_ID}"
"$CLUSTER_PYTHON" analysis/select_mal_adamw_structure.py \
    "$SOURCE_SWEEP_PATH" \
    --output_dir "$SELECTION_DIR" \
    --backfill_metadata

creation_output=$("$CLUSTER_PYTHON" sweeps/mal_confirmatory_sweep.py \
    tasks/tiny_imagenet_classification.py \
    --experiment adamw-in-place \
    --selection_file "$SELECTION_DIR/selected_configs.txt" \
    --source_sweep "$SOURCE_SWEEP_PATH" \
    --sweep_name "mal-adamw-in-place-${SLURM_JOB_ID}" \
    --project_name "$PROJECT_NAME" \
    --tiny_imagenet_dir "$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200")
printf '%s\n' "$creation_output"
FOLLOWUP_SWEEP_PATH=$(extract_sweep_path "$creation_output")

SWEEP_RECORD="$MEMORY_ALIGN_PROJECT/logs/adamw-recursive-sweep-${SLURM_JOB_ID}.env"
printf 'SOURCE_SWEEP_PATH=%q\n' "$SOURCE_SWEEP_PATH" >"$SWEEP_RECORD"
printf 'SELECTION_DIR=%q\n' "$SELECTION_DIR" >>"$SWEEP_RECORD"
printf 'FOLLOWUP_SWEEP_PATH=%q\n' "$FOLLOWUP_SWEEP_PATH" >>"$SWEEP_RECORD"
printf 'FOLLOWUP_EXPECTED_RUNS=%q\n' "$EXPECTED_FOLLOWUP_RUNS" >>"$SWEEP_RECORD"

run_agents_until_complete "$FOLLOWUP_SWEEP_PATH" "$EXPECTED_FOLLOWUP_RUNS"
echo "MAL-AdamW recursive follow-up completed successfully: $FOLLOWUP_SWEEP_PATH"
