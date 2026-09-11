#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=500:00:00
#SBATCH --job-name=adamal-mae-screen
#SBATCH --output=/shared/b00090279/memory_align/logs/adamal-screen-master-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/adamal-screen-master-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
SOURCE_SWEEP_PATH=${1:?'usage: run-mal-adamw-mae-structure-screen.sh <recovery-sweep-path> <recovery-agent-job-id> <prior-sweep-path>'}
SOURCE_AGENT_JOB_ID=${2:?'missing source-agent-job-id'}
PRIOR_SWEEP_PATH=${3:?'missing prior-sweep-path'}
ADAMAL_AGENT_COUNT=12
FIXED_AGENT_COUNT=3
MAX_AGENT_ROUNDS=3
CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
ADAMAL_AGENT_JOB_ID=""
FIXED_AGENT_JOB_ID=""

case "$SOURCE_SWEEP_PATH" in
    "$ENTITY_NAME/$PROJECT_NAME/"*) ;;
    *)
        echo "Refusing unexpected source sweep path: $SOURCE_SWEEP_PATH" >&2
        exit 2
        ;;
esac
case "$PRIOR_SWEEP_PATH" in
    "$ENTITY_NAME/$PROJECT_NAME/"*) ;;
    *)
        echo "Refusing unexpected prior sweep path: $PRIOR_SWEEP_PATH" >&2
        exit 2
        ;;
esac
[[ "$SOURCE_AGENT_JOB_ID" =~ ^[0-9]+$ ]] || {
    echo "SLURM job id must be numeric: $SOURCE_AGENT_JOB_ID" >&2
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
    local job_id
    for job_id in "$ADAMAL_AGENT_JOB_ID" "$FIXED_AGENT_JOB_ID"; do
        if [[ -n "$job_id" ]] && squeue -h -j "$job_id" 2>/dev/null | grep -q .; then
            echo "Master received $signal_name; cancelling screen array $job_id" >&2
            scancel "$job_id"
        fi
    done
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
    local agent_count=$2
    local job_name=$3
    local submission
    submission=$(sbatch \
        --parsable \
        --array="1-${agent_count}" \
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

validate_source_subset() {
    "$CLUSTER_PYTHON" - "$SOURCE_SWEEP_PATH" "$PRIOR_SWEEP_PATH" <<'PY'
import sys
from collections import Counter

import wandb

recovery_sweep_path, prior_sweep_path = sys.argv[1:]
api = wandb.Api(timeout=180)
recovery_runs = list(api.sweep(recovery_sweep_path).runs)
prior_runs = list(api.sweep(prior_sweep_path).runs)

expected_baselines = {"AdamW", "AM_AdamW", "AdaTAMW"}
recovery_optimizers = Counter(dict(run.config).get("optimizer") for run in recovery_runs)
if not set(recovery_optimizers).issubset(expected_baselines):
    raise SystemExit(f"Unexpected optimizers in baseline recovery sweep: {dict(recovery_optimizers)}")

def signature(run):
    config = dict(run.config)
    return (
        config.get("optimizer"),
        config.get("batch_size"),
        config.get("base_lr"),
        config.get("weight_decay"),
        config.get("seed"),
        config.get("use_scheduler"),
    )

finished_runs = {
    run.id: run
    for run in (*prior_runs, *recovery_runs)
    if run.state == "finished" and dict(run.config).get("optimizer") in expected_baselines
}
for optimizer in expected_baselines:
    selected = [run for run in finished_runs.values() if dict(run.config).get("optimizer") == optimizer]
    signatures = [signature(run) for run in selected]
    if len(selected) != 24 or len(set(signatures)) != 24:
        raise SystemExit(
            f"Incomplete or duplicate baseline {optimizer}: finished={len(selected)}, "
            f"unique_signatures={len(set(signatures))}"
        )
print(f"Validated 72 unique finished baseline cells across {prior_sweep_path} and {recovery_sweep_path}.")
PY
}

create_screen() {
    local screen=$1
    local sweep_name=$2
    local output_dir=$3
    local creation_output
    creation_output=$("$CLUSTER_PYTHON" sweeps/mal_adamw_mae_structure_sweep.py \
        tasks/mae_pretrain.py \
        --screen "$screen" \
        --sweep_name "$sweep_name" \
        --project_name "$PROJECT_NAME" \
        --data_dir "$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200" \
        --output_dir "$output_dir")
    printf '%s\n' "$creation_output" >&2
    CREATED_SWEEP_PATH=$(extract_sweep_path "$creation_output")
    CREATED_EXPECTED_RUNS=$(extract_expected_runs "$creation_output")
}

validate_with_recovery() {
    local sweep_path=$1
    local expected_runs=$2
    local agent_count=$3
    local job_name=$4
    local receipt_prefix=$5
    local round state recovery_job_id
    for ((round = 1; round < MAX_AGENT_ROUNDS; round++)); do
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
            echo "Sweep $sweep_path reached $state without $expected_runs finished runs." >&2
            return 1
        fi
        recovery_job_id=$(submit_agents "$sweep_path" "$agent_count" "$job_name")
        printf '%s_RECOVERY_JOB_%s=%q\n' "$receipt_prefix" "$round" "$recovery_job_id" >>"$SWEEP_RECORD"
        wait_for_job "$recovery_job_id" "$receipt_prefix recovery array"
    done
    echo "Sweep $sweep_path did not validate after recovery." >&2
    return 1
}

SWEEP_RECORD="$MEMORY_ALIGN_PROJECT/logs/adamal-screen-sweeps-${SLURM_JOB_ID}.env"
: >"$SWEEP_RECORD"
printf 'SOURCE_SWEEP_PATH=%q\nSOURCE_AGENT_JOB_ID=%q\nPRIOR_SWEEP_PATH=%q\n' \
    "$SOURCE_SWEEP_PATH" "$SOURCE_AGENT_JOB_ID" "$PRIOR_SWEEP_PATH" >>"$SWEEP_RECORD"

wait_for_job "$SOURCE_AGENT_JOB_ID" "source baseline array"
validate_source_subset

create_screen \
    adamal \
    "adamal-mae-structure-screen-${SLURM_JOB_ID}" \
    "$MEMORY_ALIGN_PROJECT/outputs/adamal-mae-structure-screen"
ADAMAL_SWEEP_PATH=$CREATED_SWEEP_PATH
ADAMAL_EXPECTED_RUNS=$CREATED_EXPECTED_RUNS
[[ "$ADAMAL_EXPECTED_RUNS" == "24" ]] || {
    echo "AdaMAL screen must contain exactly 24 runs, found $ADAMAL_EXPECTED_RUNS." >&2
    exit 1
}

create_screen \
    fixed-control \
    "mal-adamw-fixed-control-${SLURM_JOB_ID}" \
    "$MEMORY_ALIGN_PROJECT/outputs/mal-adamw-fixed-control"
FIXED_SWEEP_PATH=$CREATED_SWEEP_PATH
FIXED_EXPECTED_RUNS=$CREATED_EXPECTED_RUNS
[[ "$FIXED_EXPECTED_RUNS" == "3" ]] || {
    echo "Fixed MAL-AdamW control must contain exactly 3 runs, found $FIXED_EXPECTED_RUNS." >&2
    exit 1
}

printf 'ADAMAL_SWEEP_PATH=%q\nADAMAL_EXPECTED_RUNS=%q\nFIXED_SWEEP_PATH=%q\nFIXED_EXPECTED_RUNS=%q\n' \
    "$ADAMAL_SWEEP_PATH" "$ADAMAL_EXPECTED_RUNS" "$FIXED_SWEEP_PATH" "$FIXED_EXPECTED_RUNS" >>"$SWEEP_RECORD"

# The two arrays run concurrently and occupy exactly 12 + 3 = 15 GPUs.
ADAMAL_AGENT_JOB_ID=$(submit_agents "$ADAMAL_SWEEP_PATH" "$ADAMAL_AGENT_COUNT" adamal-mae)
FIXED_AGENT_JOB_ID=$(submit_agents "$FIXED_SWEEP_PATH" "$FIXED_AGENT_COUNT" mal-adamw-fixed)
printf 'ADAMAL_AGENT_JOB_ID=%q\nFIXED_AGENT_JOB_ID=%q\n' \
    "$ADAMAL_AGENT_JOB_ID" "$FIXED_AGENT_JOB_ID" >>"$SWEEP_RECORD"
echo "Submitted AdaMAL array $ADAMAL_AGENT_JOB_ID (12 GPUs) and fixed-control array $FIXED_AGENT_JOB_ID (3 GPUs)."

wait_for_job "$FIXED_AGENT_JOB_ID" "fixed-control array"
wait_for_job "$ADAMAL_AGENT_JOB_ID" "AdaMAL array"
ADAMAL_AGENT_JOB_ID=""
FIXED_AGENT_JOB_ID=""

validate_with_recovery "$ADAMAL_SWEEP_PATH" "$ADAMAL_EXPECTED_RUNS" "$ADAMAL_AGENT_COUNT" adamal-mae ADAMAL
validate_with_recovery "$FIXED_SWEEP_PATH" "$FIXED_EXPECTED_RUNS" "$FIXED_AGENT_COUNT" mal-adamw-fixed FIXED

echo "AdaMAL structure screen and matched fixed-gradient control completed. Receipt: $SWEEP_RECORD"
