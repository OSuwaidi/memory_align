#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=240:00:00
#SBATCH --job-name=agm-llm-pretrain
#SBATCH --output=/shared/b00090279/memory_align/logs/llm-pretrain-master-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/llm-pretrain-master-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
AGENT_COUNT=15
MAX_AGENT_ROUNDS=2
CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
TOKEN_DATA_DIR="$MEMORY_ALIGN_PROJECT/data/fineweb-edu-smollm2"
RUN_OUTPUT_DIR="$MEMORY_ALIGN_PROJECT/outputs/llm-pretrain"
ACTIVE_AGENT_JOB_ID=""

. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"

[[ -x "$CLUSTER_PYTHON" ]] || {
    echo "Cluster environment is missing: $CLUSTER_PYTHON" >&2
    exit 1
}

cancel_active_agents() {
    local signal_name=${1:-TERM}
    if [[ -n "$ACTIVE_AGENT_JOB_ID" ]] && squeue -h -j "$ACTIVE_AGENT_JOB_ID" 2>/dev/null | grep -q .; then
        echo "Master received $signal_name; cancelling GPU array $ACTIVE_AGENT_JOB_ID" >&2
        scancel "$ACTIVE_AGENT_JOB_ID"
    fi
}
trap 'cancel_active_agents TERM; exit 143' TERM
trap 'cancel_active_agents INT; exit 130' INT

extract_value() {
    local key=$1
    local output=$2
    printf '%s\n' "$output" | sed -nE "s/^${key}=(.+)$/\\1/p" | tail -n 1
}

submit_agents() {
    local sweep_path=$1
    local job_name=$2
    local submission
    submission=$(sbatch \
        --parsable \
        --array="1-${AGENT_COUNT}" \
        --job-name="$job_name" \
        "$MEMORY_ALIGN_PROJECT/wb-llm-pretrain-agent.sh" \
        "$sweep_path")
    printf '%s\n' "${submission%%;*}"
}

wait_for_job() {
    local job_id=$1
    while :; do
        local states
        states=$(squeue -h -j "$job_id" -o '%T' 2>/dev/null || true)
        [[ -z "$states" ]] && break
        echo "$(date -Is) job $job_id: $(printf '%s\n' "$states" | sort | uniq -c | xargs)"
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

run_agents_until_complete() {
    local sweep_path=$1
    local expected_runs=$2
    local job_name=$3
    local receipt_key=$4
    local round state
    for ((round = 1; round <= MAX_AGENT_ROUNDS; round++)); do
        ACTIVE_AGENT_JOB_ID=$(submit_agents "$sweep_path" "$job_name")
        printf '%s_AGENT_JOB_%s=%q\n' "$receipt_key" "$round" "$ACTIVE_AGENT_JOB_ID" >>"$SWEEP_RECEIPT"
        wait_for_job "$ACTIVE_AGENT_JOB_ID"
        ACTIVE_AGENT_JOB_ID=""
        for _attempt in {1..12}; do
            if "$CLUSTER_PYTHON" sweeps/validate_sweep.py "$sweep_path" --expected_runs "$expected_runs"; then
                return 0
            fi
            sleep 10
        done
        state=$(sweep_state "$sweep_path")
        if [[ "$state" == "FINISHED" || "$state" == "CANCELED" ]]; then
            echo "Sweep ended as $state without exactly $expected_runs successful runs." >&2
            return 1
        fi
    done
    return 1
}

create_sweep() {
    local stage=$1
    local sweep_name=$2
    local selected_configs=${3:-}
    local command=(
        "$CLUSTER_PYTHON" sweeps/llm_pretrain_sweep.py
        tasks/llm_pretrain.py
        --stage "$stage"
        --sweep_name "$sweep_name"
        --project_name "$PROJECT_NAME"
        --data_dir "$TOKEN_DATA_DIR"
        --output_dir "$RUN_OUTPUT_DIR"
        --micro_batch_size "$MICRO_BATCH_SIZE"
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    )
    if [[ -n "$selected_configs" ]]; then
        command+=(--selected_configs "$selected_configs")
    fi
    local output
    output=$("${command[@]}")
    printf '%s\n' "$output" >&2
    CREATED_SWEEP_PATH=$(extract_value SWEEP_PATH "$output")
    CREATED_EXPECTED_RUNS=$(extract_value EXPECTED_RUNS "$output")
    case "$CREATED_SWEEP_PATH" in
        "$ENTITY_NAME/$PROJECT_NAME/"*) ;;
        *) echo "Invalid sweep path: $CREATED_SWEEP_PATH" >&2; return 1 ;;
    esac
    [[ "$CREATED_EXPECTED_RUNS" =~ ^[0-9]+$ ]]
}

mkdir -p "$TOKEN_DATA_DIR" "$RUN_OUTPUT_DIR" "$MEMORY_ALIGN_PROJECT/logs"
export TOKENIZERS_PARALLELISM=true

# One pinned, document-disjoint corpus is shared by every optimizer and seed.
"$CLUSTER_PYTHON" prepare_fineweb_edu.py \
    --output_dir "$TOKEN_DATA_DIR" \
    --train_tokens 1499463680 \
    --eval_tokens 8388608

# Calibrate only memory capacity. The effective batch remains 262,144 tokens
# under either branch, so this fallback cannot change the scientific recipe.
if smoke_submission=$(sbatch --parsable --wait "$MEMORY_ALIGN_PROJECT/run-llm-pretrain-smoke.sh" 4); then
    MICRO_BATCH_SIZE=4
    GRADIENT_ACCUMULATION_STEPS=32
else
    echo "Micro-batch 4 failed preflight; retrying the same effective batch with micro-batch 2." >&2
    smoke_submission=$(sbatch --parsable --wait "$MEMORY_ALIGN_PROJECT/run-llm-pretrain-smoke.sh" 2)
    MICRO_BATCH_SIZE=2
    GRADIENT_ACCUMULATION_STEPS=64
fi
echo "GPU memory preflight completed: $smoke_submission"

SWEEP_RECEIPT="$MEMORY_ALIGN_PROJECT/logs/llm-pretraining-sweeps-${SLURM_JOB_ID}.env"
: >"$SWEEP_RECEIPT"
printf 'MICRO_BATCH_SIZE=%q\nGRADIENT_ACCUMULATION_STEPS=%q\n' \
    "$MICRO_BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" >>"$SWEEP_RECEIPT"

# Phase 1: 4 optimizers x 3 LRs x 2 WDs x 2 paired seeds = 48
# 299,892,736-token screens (20% of the confirmation horizon, modulo a batch).
create_sweep screen "agm-smollm2-360m-fineweb-screen-${SLURM_JOB_ID}"
SCREEN_SWEEP_PATH=$CREATED_SWEEP_PATH
SCREEN_EXPECTED_RUNS=$CREATED_EXPECTED_RUNS
printf 'SCREEN_SWEEP_PATH=%q\nSCREEN_EXPECTED_RUNS=%q\n' \
    "$SCREEN_SWEEP_PATH" "$SCREEN_EXPECTED_RUNS" >>"$SWEEP_RECEIPT"
run_agents_until_complete "$SCREEN_SWEEP_PATH" "$SCREEN_EXPECTED_RUNS" agm-llm-screen SCREEN

# Select only from final dev loss averaged over the two screening seeds.
SELECTION_RECEIPT="$RUN_OUTPUT_DIR/selected-hparams-${SLURM_JOB_ID}.json"
"$CLUSTER_PYTHON" analysis/select_llm_pretrain_hparams.py \
    "$SCREEN_SWEEP_PATH" \
    --output "$SELECTION_RECEIPT"
printf 'SELECTION_RECEIPT=%q\n' "$SELECTION_RECEIPT" >>"$SWEEP_RECEIPT"

# Phase 2: one selected cell x 4 optimizers x 3 seeds = 12 full 1.499B-token runs.
create_sweep confirmation "agm-smollm2-360m-fineweb-confirmation-${SLURM_JOB_ID}" "$SELECTION_RECEIPT"
CONFIRMATION_SWEEP_PATH=$CREATED_SWEEP_PATH
CONFIRMATION_EXPECTED_RUNS=$CREATED_EXPECTED_RUNS
printf 'CONFIRMATION_SWEEP_PATH=%q\nCONFIRMATION_EXPECTED_RUNS=%q\n' \
    "$CONFIRMATION_SWEEP_PATH" "$CONFIRMATION_EXPECTED_RUNS" >>"$SWEEP_RECEIPT"
run_agents_until_complete "$CONFIRMATION_SWEEP_PATH" "$CONFIRMATION_EXPECTED_RUNS" agm-llm-full CONFIRMATION

echo "FineWeb-Edu pre-training benchmark completed. Receipt: $SWEEP_RECEIPT"
