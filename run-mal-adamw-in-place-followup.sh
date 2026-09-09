#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=500:00:00
#SBATCH --job-name=mal-adamw-pipeline
#SBATCH --output=/shared/b00090279/memory_align/logs/adamw-pipeline-master-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/adamw-pipeline-master-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
SOURCE_SWEEP_PATH=${1:?"usage: run-mal-adamw-in-place-followup.sh <entity/project/source-sweep-id>"}
AGENT_COUNT=15
EXPECTED_SOURCE_RUNS=120
MAX_AGENT_ROUNDS=3
CLUSTER_VENV="$MEMORY_ALIGN_PROJECT/.cluster-venv"
CLUSTER_PYTHON="$CLUSTER_VENV/bin/python"
UV_BIN=/shared/b00090279/.local/bin/uv
LOCK_HASH=$(sha256sum "$MEMORY_ALIGN_PROJECT/uv.lock" | cut -c1-16)
ENVIRONMENT_MARKER="$CLUSTER_VENV/.mal-uv-lock-$LOCK_HASH"
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

prepare_python_environment() {
    if [[ -x "$CLUSTER_PYTHON" && -f "$ENVIRONMENT_MARKER" ]]; then
        echo "Reusing cluster environment for uv.lock $LOCK_HASH at $CLUSTER_VENV"
        return
    fi
    if [[ ! -x "$UV_BIN" ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | env \
            UV_INSTALL_DIR=/shared/b00090279/.local/bin \
            UV_NO_MODIFY_PATH=1 \
            sh
    fi
    "$UV_BIN" python install 3.14
    UV_PROJECT_ENVIRONMENT="$CLUSTER_VENV" "$UV_BIN" sync \
        --frozen \
        --python 3.14 \
        --compile-bytecode
    touch "$ENVIRONMENT_MARKER"
}

prepare_inputs() {
    "$CLUSTER_PYTHON" download_datasets.py \
        --task tiny-imagenet \
        --tiny_imagenet_dir "$MEMORY_ALIGN_PROJECT/data"
    "$CLUSTER_PYTHON" download_datasets.py \
        --task llm \
        --llm_cache_dir "$MEMORY_ALIGN_PROJECT/data/llm_cache"

    # The short supervised selection task uses ImageNet-pretrained timm weights.
    "$CLUSTER_PYTHON" - <<'PY'
import timm

model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=200)
print(f"Cached pretrained ViT-Tiny weights ({sum(p.numel() for p in model.parameters()):,} parameters).")
PY
}

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

record_sweep() {
    local key=$1
    local sweep_path=$2
    local expected_runs=$3
    printf '%s=%q\n' "$key" "$sweep_path" >>"$SWEEP_RECORD"
    printf '%s_EXPECTED_RUNS=%q\n' "$key" "$expected_runs" >>"$SWEEP_RECORD"
}

submit_agents() {
    local sweep_path=$1
    local job_name=$2
    local submission
    submission=$(sbatch \
        --parsable \
        --array="1-${AGENT_COUNT}" \
        --job-name="$job_name" \
        "$MEMORY_ALIGN_PROJECT/wb-agents.sh" \
        "$sweep_path")
    printf '%s\n' "${submission%%;*}"
}

wait_for_agents() {
    local job_id=$1
    local sweep_path=$2
    local queue_snapshot
    echo "Waiting for ${AGENT_COUNT}-GPU array $job_id ($sweep_path)"
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

run_agents_until_complete() {
    local sweep_path=$1
    local expected_runs=$2
    local job_name=$3
    local record_key=$4
    local round
    local state

    for ((round = 1; round <= MAX_AGENT_ROUNDS; round++)); do
        ACTIVE_AGENT_JOB_ID=$(submit_agents "$sweep_path" "$job_name")
        printf '%s_AGENT_JOB_%s=%q\n' "$record_key" "$round" "$ACTIVE_AGENT_JOB_ID" >>"$SWEEP_RECORD"
        echo "Submitted $sweep_path as a ${AGENT_COUNT}-GPU array $ACTIVE_AGENT_JOB_ID (round $round)"
        wait_for_agents "$ACTIVE_AGENT_JOB_ID" "$sweep_path"
        ACTIVE_AGENT_JOB_ID=""

        for _attempt in {1..12}; do
            if "$CLUSTER_PYTHON" sweeps/validate_sweep.py "$sweep_path" --expected_runs "$expected_runs"; then
                return 0
            fi
            sleep 10
        done

        state=$(sweep_state "$sweep_path")
        if [[ "$state" == "FINISHED" || "$state" == "CANCELED" ]]; then
            echo "Sweep reached terminal state $state without $expected_runs finished runs." >&2
            return 1
        fi
        echo "Sweep remains $state; submitting a recovery array."
    done
    echo "Sweep did not complete after $MAX_AGENT_ROUNDS agent rounds: $sweep_path" >&2
    return 1
}

create_confirmatory_sweep() {
    local experiment=$1
    local sweep_name=$2
    local selection_file=$3
    local creation_output
    creation_output=$("$CLUSTER_PYTHON" sweeps/mal_confirmatory_sweep.py \
        tasks/tiny_imagenet_classification.py \
        --experiment "$experiment" \
        --selection_file "$selection_file" \
        --source_sweep "$SOURCE_SWEEP_PATH" \
        --sweep_name "$sweep_name" \
        --project_name "$PROJECT_NAME" \
        --tiny_imagenet_dir "$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200")
    printf '%s\n' "$creation_output" >&2
    CREATED_SWEEP_PATH=$(extract_sweep_path "$creation_output")
    CREATED_EXPECTED_RUNS=$(extract_expected_runs "$creation_output")
}

create_mae_sweep() {
    local mal_config=$1
    local creation_output
    creation_output=$("$CLUSTER_PYTHON" sweeps/adamw_vit_sweep.py \
        tasks/mae_pretrain.py \
        --sweep_name "mal-adamw-mae-tiny-imagenet-${SLURM_JOB_ID}" \
        --project_name "$PROJECT_NAME" \
        --data_dir "$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200" \
        --output_dir "$MEMORY_ALIGN_PROJECT/outputs/mae" \
        --mal_config "$mal_config")
    printf '%s\n' "$creation_output" >&2
    CREATED_SWEEP_PATH=$(extract_sweep_path "$creation_output")
    CREATED_EXPECTED_RUNS=$(extract_expected_runs "$creation_output")
}

create_llm_sweep() {
    local mal_config=$1
    local creation_output
    creation_output=$("$CLUSTER_PYTHON" sweeps/llm_finetune_sweep.py \
        tasks/llm_finetune.py \
        --family adamw \
        --sweep_name "mal-adamw-smollm2-wikitext-${SLURM_JOB_ID}" \
        --project_name "$PROJECT_NAME" \
        --cache_dir "$MEMORY_ALIGN_PROJECT/data/llm_cache" \
        --mal_config "$mal_config")
    printf '%s\n' "$creation_output" >&2
    CREATED_SWEEP_PATH=$(extract_sweep_path "$creation_output")
    CREATED_EXPECTED_RUNS=$(extract_expected_runs "$creation_output")
}

prepare_python_environment
prepare_inputs

SWEEP_RECORD="$MEMORY_ALIGN_PROJECT/logs/adamw-pipeline-sweeps-${SLURM_JOB_ID}.env"
: >"$SWEEP_RECORD"
printf 'SOURCE_SWEEP_PATH=%q\n' "$SOURCE_SWEEP_PATH" >>"$SWEEP_RECORD"

"$CLUSTER_PYTHON" sweeps/validate_sweep.py \
    "$SOURCE_SWEEP_PATH" \
    --expected_runs "$EXPECTED_SOURCE_RUNS"

SOURCE_SWEEP_ID=${SOURCE_SWEEP_PATH##*/}
SOURCE_SELECTION_DIR="$MEMORY_ALIGN_PROJECT/outputs/mal-adamw-source-selection-${SOURCE_SWEEP_ID}-${SLURM_JOB_ID}"
"$CLUSTER_PYTHON" analysis/select_mal_adamw_structure.py \
    "$SOURCE_SWEEP_PATH" \
    --output_dir "$SOURCE_SELECTION_DIR" \
    --backfill_metadata
printf 'SOURCE_SELECTION_DIR=%q\n' "$SOURCE_SELECTION_DIR" >>"$SWEEP_RECORD"

# Phase 1: 40 runs. Two source finalists x pwr {0.5, 1.0} x two
# learning rates x five seeds, with the warmup+cosine schedule only.
create_confirmatory_sweep \
    adamw-in-place \
    "mal-adamw-in-place-scheduled-${SLURM_JOB_ID}" \
    "$SOURCE_SELECTION_DIR/selected_configs.txt"
SCHEDULED_SWEEP_PATH=$CREATED_SWEEP_PATH
SCHEDULED_EXPECTED_RUNS=$CREATED_EXPECTED_RUNS
record_sweep SCHEDULED_IN_PLACE_SWEEP_PATH "$SCHEDULED_SWEEP_PATH" "$SCHEDULED_EXPECTED_RUNS"
run_agents_until_complete \
    "$SCHEDULED_SWEEP_PATH" \
    "$SCHEDULED_EXPECTED_RUNS" \
    mal-adamw-ip-s \
    SCHEDULED_IN_PLACE

FINAL_SELECTION_DIR="$MEMORY_ALIGN_PROJECT/outputs/mal-adamw-final-selection-${SLURM_JOB_ID}"
"$CLUSTER_PYTHON" analysis/select_final_mal_adamw.py \
    "$SOURCE_SWEEP_PATH" \
    "$SCHEDULED_SWEEP_PATH" \
    --source_selection_file "$SOURCE_SELECTION_DIR/selected_configs.txt" \
    --output_dir "$FINAL_SELECTION_DIR"
. "$FINAL_SELECTION_DIR/decision.env"

# Phase 2 is conditional: one 10-run scheduler-free stress test is permitted
# only when recursive state first beats the transient source finalist under the
# scheduled protocol.
if [[ "$RUN_UNSCHEDULED" == "1" ]]; then
    create_confirmatory_sweep \
        adamw-in-place-unscheduled \
        "mal-adamw-in-place-unscheduled-${SLURM_JOB_ID}" \
        "$FINAL_SELECTION_DIR/unscheduled_candidate.txt"
    UNSCHEDULED_SWEEP_PATH=$CREATED_SWEEP_PATH
    UNSCHEDULED_EXPECTED_RUNS=$CREATED_EXPECTED_RUNS
    record_sweep UNSCHEDULED_IN_PLACE_SWEEP_PATH "$UNSCHEDULED_SWEEP_PATH" "$UNSCHEDULED_EXPECTED_RUNS"
    run_agents_until_complete \
        "$UNSCHEDULED_SWEEP_PATH" \
        "$UNSCHEDULED_EXPECTED_RUNS" \
        mal-adamw-ip-u \
        UNSCHEDULED_IN_PLACE
    "$CLUSTER_PYTHON" analysis/select_final_mal_adamw.py \
        "$SOURCE_SWEEP_PATH" \
        "$SCHEDULED_SWEEP_PATH" \
        --source_selection_file "$SOURCE_SELECTION_DIR/selected_configs.txt" \
        --unscheduled_sweep_path "$UNSCHEDULED_SWEEP_PATH" \
        --output_dir "$FINAL_SELECTION_DIR"
fi

FINAL_MAL_CONFIG=$(<"$FINAL_SELECTION_DIR/final_config.txt")
printf 'FINAL_SELECTION_DIR=%q\nFINAL_MAL_CONFIG=%q\n' \
    "$FINAL_SELECTION_DIR" "$FINAL_MAL_CONFIG" >>"$SWEEP_RECORD"
echo "Selected final MAL-AdamW configuration: $FINAL_MAL_CONFIG"

# Phase 3: 96-run MAE pretraining benchmark = four AdamW-family optimizers x
# two batches x two base LRs x two weight decays x three seeds.
create_mae_sweep "$FINAL_MAL_CONFIG"
record_sweep ADAMW_MAE_SWEEP_PATH "$CREATED_SWEEP_PATH" "$CREATED_EXPECTED_RUNS"
run_agents_until_complete \
    "$CREATED_SWEEP_PATH" \
    "$CREATED_EXPECTED_RUNS" \
    mal-adamw-mae \
    ADAMW_MAE

# Phase 4 starts only after MAE is complete: 36 full-parameter SmolLM2 runs =
# four AdamW-family optimizers x three LR multipliers x three seeds.
create_llm_sweep "$FINAL_MAL_CONFIG"
record_sweep ADAMW_LLM_SWEEP_PATH "$CREATED_SWEEP_PATH" "$CREATED_EXPECTED_RUNS"
run_agents_until_complete \
    "$CREATED_SWEEP_PATH" \
    "$CREATED_EXPECTED_RUNS" \
    mal-adamw-llm \
    ADAMW_LLM

echo "MAL-AdamW selection, MAE, and LLM phases completed. Sweep receipt: $SWEEP_RECORD"
