#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --job-name=mal-adamw-mae-ext
#SBATCH --output=/shared/b00090279/memory_align/logs/mal-adamw-mae-ext-master-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/mal-adamw-mae-ext-master-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
AGENT_COUNT=15
EXPECTED_RUNS=15
MAX_AGENT_ROUNDS=3
CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
ACTIVE_AGENT_JOB_ID=""

. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"

[[ -x "$CLUSTER_PYTHON" ]] || {
    echo "Cluster environment is missing: $CLUSTER_PYTHON" >&2
    exit 1
}
[[ -d "$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200" ]] || {
    echo "Tiny-ImageNet is missing beneath the authorized shared directory." >&2
    exit 1
}

cancel_active_agents() {
    local signal_name=${1:-TERM}
    if [[ -n "$ACTIVE_AGENT_JOB_ID" ]] && squeue -h -j "$ACTIVE_AGENT_JOB_ID" 2>/dev/null | grep -q .; then
        echo "Master received $signal_name; cancelling agent array $ACTIVE_AGENT_JOB_ID" >&2
        scancel "$ACTIVE_AGENT_JOB_ID"
    fi
}

trap 'cancel_active_agents TERM; exit 143' TERM
trap 'cancel_active_agents INT; exit 130' INT

extract_value() {
    local key=$1
    local creation_output=$2
    printf '%s\n' "$creation_output" | sed -nE "s/^${key}=(.+)$/\\1/p" | tail -n 1
}

submit_agents() {
    local sweep_path=$1
    local submission
    submission=$(sbatch \
        --parsable \
        --array="1-${AGENT_COUNT}" \
        --job-name=mal-adamw-mae-ext \
        "$MEMORY_ALIGN_PROJECT/wb-agents.sh" \
        "$sweep_path")
    printf '%s\n' "${submission%%;*}"
}

wait_for_agents() {
    local job_id=$1
    local queue_snapshot
    while :; do
        queue_snapshot=$(squeue -h -j "$job_id" -o '%T' 2>/dev/null || true)
        [[ -z "$queue_snapshot" ]] && break
        echo "$(date -Is) array $job_id: $(printf '%s\n' "$queue_snapshot" | sort | uniq -c | xargs)"
        sleep 60
    done
    sacct -n -X -j "$job_id" --format=JobID,State,ExitCode,Elapsed -P 2>/dev/null || true
}

sweep_state() {
    "$CLUSTER_PYTHON" - "$1" <<'PY'
import sys
import wandb

print(wandb.Api(timeout=180).sweep(sys.argv[1]).state)
PY
}

creation_output=$("$CLUSTER_PYTHON" sweeps/mal_adamw_mae_structural_extensions_sweep.py \
    tasks/mae_pretrain.py \
    --sweep_name "mal-adamw-mae-structural-extensions-${SLURM_JOB_ID}" \
    --project_name "$PROJECT_NAME" \
    --data_dir "$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200" \
    --output_dir "$MEMORY_ALIGN_PROJECT/outputs/mae-structural-extensions")
printf '%s\n' "$creation_output"

SWEEP_PATH=$(extract_value SWEEP_PATH "$creation_output")
CREATED_EXPECTED_RUNS=$(extract_value EXPECTED_RUNS "$creation_output")
case "$SWEEP_PATH" in
    "$ENTITY_NAME/$PROJECT_NAME/"*) ;;
    *)
        echo "Could not extract a valid sweep path from sweep creation output." >&2
        exit 1
        ;;
esac
[[ "$CREATED_EXPECTED_RUNS" == "$EXPECTED_RUNS" ]] || {
    echo "Expected $EXPECTED_RUNS runs, sweep reported $CREATED_EXPECTED_RUNS." >&2
    exit 1
}

SWEEP_RECORD="$MEMORY_ALIGN_PROJECT/logs/mal-adamw-mae-ext-sweep-${SLURM_JOB_ID}.env"
printf 'SWEEP_PATH=%q\nEXPECTED_RUNS=%q\n' "$SWEEP_PATH" "$EXPECTED_RUNS" >"$SWEEP_RECORD"

for ((round = 1; round <= MAX_AGENT_ROUNDS; round++)); do
    ACTIVE_AGENT_JOB_ID=$(submit_agents "$SWEEP_PATH")
    printf 'AGENT_JOB_%s=%q\n' "$round" "$ACTIVE_AGENT_JOB_ID" >>"$SWEEP_RECORD"
    echo "Submitted $AGENT_COUNT GPU agents as array $ACTIVE_AGENT_JOB_ID (round $round)."
    wait_for_agents "$ACTIVE_AGENT_JOB_ID"
    ACTIVE_AGENT_JOB_ID=""

    for _attempt in {1..12}; do
        if "$CLUSTER_PYTHON" sweeps/validate_sweep.py "$SWEEP_PATH" --expected_runs "$EXPECTED_RUNS"; then
            echo "MAL-AdamW MAE structural extension sweep completed. Receipt: $SWEEP_RECORD"
            exit 0
        fi
        sleep 10
    done

    state=$(sweep_state "$SWEEP_PATH")
    if [[ "$state" == "FINISHED" || "$state" == "CANCELED" ]]; then
        echo "Sweep reached terminal state $state without $EXPECTED_RUNS finished runs." >&2
        exit 1
    fi
done

echo "Sweep did not complete after $MAX_AGENT_ROUNDS agent rounds: $SWEEP_PATH" >&2
exit 1
