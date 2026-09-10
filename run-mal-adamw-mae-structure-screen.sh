#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=500:00:00
#SBATCH --job-name=mal-adamw-screen
#SBATCH --output=/shared/b00090279/memory_align/logs/adamw-screen-master-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/adamw-screen-master-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
SOURCE_SWEEP_PATH=${1:?'usage: run-mal-adamw-mae-structure-screen.sh <source-sweep-path> <source-agent-job-id>'}
SOURCE_AGENT_JOB_ID=${2:?'usage: run-mal-adamw-mae-structure-screen.sh <source-sweep-path> <source-agent-job-id>'}
EXPECTED_SOURCE_MAL_RUNS=24
EXPECTED_SOURCE_MAL_FINISHED=9
AGENT_COUNT=15
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
[[ "$SOURCE_AGENT_JOB_ID" =~ ^[0-9]+$ ]] || {
    echo "Source agent job id must be numeric: $SOURCE_AGENT_JOB_ID" >&2
    exit 2
}

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
        echo "Master received $signal_name; cancelling structure-screen array $ACTIVE_AGENT_JOB_ID" >&2
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

extract_expected_runs() {
    local creation_output=$1
    local expected_runs
    expected_runs=$(printf '%s\n' "$creation_output" | sed -nE 's/^EXPECTED_RUNS=([0-9]+)$/\1/p' | tail -n 1)
    [[ "$expected_runs" =~ ^[0-9]+$ ]] || {
        echo "Could not extract EXPECTED_RUNS from sweep creation output." >&2
        return 1
    }
    printf '%s\n' "$expected_runs"
}

submit_agents() {
    local sweep_path=$1
    local submission
    submission=$(sbatch \
        --parsable \
        --array="1-${AGENT_COUNT}" \
        --job-name=mal-adamw-screen \
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

validate_source_subset() {
    local cancellation_record=$1
    "$CLUSTER_PYTHON" - "$SOURCE_SWEEP_PATH" "$cancellation_record" <<'PY'
import json
import sys
from collections import Counter

import wandb

sweep_path, record_path = sys.argv[1:]
runs = list(wandb.Api(timeout=180).sweep(sweep_path).runs)
with open(record_path, encoding="utf-8") as handle:
    record = json.load(handle)

expected_baselines = {"AdamW", "AM_AdamW", "AdaTAMW"}
optimizer_counts = Counter(dict(run.config).get("optimizer") for run in runs)
for optimizer in expected_baselines:
    selected = [run for run in runs if dict(run.config).get("optimizer") == optimizer]
    states = Counter(run.state for run in selected)
    if len(selected) != 24 or states != Counter({"finished": 24}):
        raise SystemExit(f"Incomplete baseline {optimizer}: count={len(selected)}, states={dict(states)}")

mal_runs = [run for run in runs if dict(run.config).get("optimizer") == "MAL_AdamW"]
mal_finished = {run.id for run in mal_runs if run.state == "finished"}
initial_finished = set(record["initial_finished_ids"])
excluded_runs = set(record["excluded_nonfinished_ids"])
if len(mal_runs) != 24:
    raise SystemExit(f"Expected 24 allocated MAL_AdamW runs, found {len(mal_runs)}")
if mal_finished != initial_finished or len(initial_finished) != 9:
    raise SystemExit(
        f"The preserved MAL result set changed: initial={len(initial_finished)}, current_finished={len(mal_finished)}"
    )
if len(excluded_runs) != 15 or excluded_runs & initial_finished:
    raise SystemExit(f"Invalid MAL exclusion receipt: excluded={len(excluded_runs)}")
if {run.id for run in mal_runs} != initial_finished | excluded_runs:
    raise SystemExit("Not every non-finished MAL run is covered by the stop receipt.")
print(
    json.dumps(
        {
            "sweep_path": sweep_path,
            "baseline_finished": {optimizer: 24 for optimizer in sorted(expected_baselines)},
            "mal_finished_preserved": len(initial_finished),
            "mal_runs_stopped_or_terminally_excluded": len(excluded_runs),
            "all_optimizer_counts": dict(optimizer_counts),
        },
        sort_keys=True,
    )
)
PY
}

run_screen_agents_until_complete() {
    local sweep_path=$1
    local expected_runs=$2
    local round
    local state
    for ((round = 1; round <= MAX_AGENT_ROUNDS; round++)); do
        ACTIVE_AGENT_JOB_ID=$(submit_agents "$sweep_path")
        printf 'SCREEN_AGENT_JOB_%s=%q\n' "$round" "$ACTIVE_AGENT_JOB_ID" >>"$SWEEP_RECORD"
        echo "Submitted focused MAL-AdamW sweep as GPU array $ACTIVE_AGENT_JOB_ID (round $round)."
        wait_for_job "$ACTIVE_AGENT_JOB_ID" "structure-screen array"
        ACTIVE_AGENT_JOB_ID=""

        for _attempt in {1..12}; do
            if "$CLUSTER_PYTHON" sweeps/validate_sweep.py "$sweep_path" --expected_runs "$expected_runs"; then
                return 0
            fi
            sleep 10
        done
        state=$("$CLUSTER_PYTHON" - "$sweep_path" <<'PY'
import sys
import wandb
print(wandb.Api(timeout=180).sweep(sys.argv[1]).state)
PY
)
        if [[ "$state" == "FINISHED" || "$state" == "CANCELED" ]]; then
            echo "Focused sweep reached $state without $expected_runs finished runs." >&2
            return 1
        fi
        echo "Focused sweep remains $state; submitting a recovery array."
    done
    echo "Focused sweep did not complete after $MAX_AGENT_ROUNDS agent rounds." >&2
    return 1
}

SWEEP_RECORD="$MEMORY_ALIGN_PROJECT/logs/mal-adamw-screen-sweeps-${SLURM_JOB_ID}.env"
SOURCE_SWEEP_ID=${SOURCE_SWEEP_PATH##*/}
CANCELLATION_RECORD="$MEMORY_ALIGN_PROJECT/logs/${SOURCE_SWEEP_ID}-mal-stop-${SLURM_JOB_ID}.json"
: >"$SWEEP_RECORD"
printf 'SOURCE_SWEEP_PATH=%q\nSOURCE_AGENT_JOB_ID=%q\n' "$SOURCE_SWEEP_PATH" "$SOURCE_AGENT_JOB_ID" >>"$SWEEP_RECORD"

"$CLUSTER_PYTHON" sweeps/stop_sweep_optimizer_runs.py \
    "$SOURCE_SWEEP_PATH" \
    --optimizer MAL_AdamW \
    --expected-optimizer-runs "$EXPECTED_SOURCE_MAL_RUNS" \
    --expected-finished-at-start "$EXPECTED_SOURCE_MAL_FINISHED" \
    --poll-seconds 20 \
    --record-path "$CANCELLATION_RECORD" &
CULLER_PID=$!

wait_for_job "$SOURCE_AGENT_JOB_ID" "source baseline array"
wait "$CULLER_PID"
validate_source_subset "$CANCELLATION_RECORD"

creation_output=$("$CLUSTER_PYTHON" sweeps/mal_adamw_mae_structure_sweep.py \
    tasks/mae_pretrain.py \
    --sweep_name "mal-adamw-mae-structure-screen-${SLURM_JOB_ID}" \
    --project_name "$PROJECT_NAME" \
    --data_dir "$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200" \
    --output_dir "$MEMORY_ALIGN_PROJECT/outputs/mae-structure-screen")
printf '%s\n' "$creation_output" >&2
SCREEN_SWEEP_PATH=$(extract_sweep_path "$creation_output")
SCREEN_EXPECTED_RUNS=$(extract_expected_runs "$creation_output")
[[ "$SCREEN_EXPECTED_RUNS" == "24" ]] || {
    echo "Focused structure screen must contain exactly 24 runs, found $SCREEN_EXPECTED_RUNS." >&2
    exit 1
}
printf 'SCREEN_SWEEP_PATH=%q\nSCREEN_EXPECTED_RUNS=%q\n' "$SCREEN_SWEEP_PATH" "$SCREEN_EXPECTED_RUNS" >>"$SWEEP_RECORD"

run_screen_agents_until_complete "$SCREEN_SWEEP_PATH" "$SCREEN_EXPECTED_RUNS"
echo "Focused MAL-AdamW structure screen completed. Receipt: $SWEEP_RECORD"
