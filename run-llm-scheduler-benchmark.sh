#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=500:00:00
#SBATCH --job-name=postship-benchmark
#SBATCH --output=/shared/b00090279/memory_align/logs/postship-master-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/postship-master-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
SOURCE_SWEEP_PATH=${1:-$ENTITY_NAME/$PROJECT_NAME/9565gqxx}
VISION_SOURCE_SWEEP_PATH=${2:-$ENTITY_NAME/$PROJECT_NAME/c72berzj}
MAL_CONFIG=False,1.0,none,attenuate,update,complement
MAL_SGDM_CONFIG=False,1.0,False,attenuate
TOTAL_GPU_AGENTS=15
MAX_RECOVERY_ROUNDS=2
CLUSTER_VENV="$MEMORY_ALIGN_PROJECT/.cluster-venv"
CLUSTER_PYTHON="$CLUSTER_VENV/bin/python"
UV_BIN=/shared/b00090279/.local/bin/uv
LOCK_HASH=$(sha256sum "$MEMORY_ALIGN_PROJECT/uv.lock" | cut -c1-16)
ENVIRONMENT_MARKER="$CLUSTER_VENV/.mal-uv-lock-$LOCK_HASH"

declare -A SWEEP_PATHS=()
declare -A EXPECTED_RUNS=()
declare -A AGENT_COUNTS=()
declare -A AGENT_JOB_IDS=()
ACTIVE_AGENT_JOB_IDS=()

for source_sweep in "$SOURCE_SWEEP_PATH" "$VISION_SOURCE_SWEEP_PATH"; do
    case "$source_sweep" in
        "$ENTITY_NAME/$PROJECT_NAME/"*) ;;
        *)
            echo "Refusing unexpected source sweep path: $source_sweep" >&2
            exit 2
            ;;
    esac
done

. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"

cancel_active_agents() {
    local signal_name=${1:-TERM}
    local job_id
    for job_id in "${ACTIVE_AGENT_JOB_IDS[@]:-}"; do
        if [[ -n "$job_id" ]] && squeue -h -j "$job_id" 2>/dev/null | grep -q .; then
            echo "Master received $signal_name; cancelling GPU array $job_id" >&2
            scancel "$job_id"
        fi
    done
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

extract_sweep_path() {
    local creation_output=$1
    local path
    path=$(printf '%s\n' "$creation_output" | sed -nE 's|.*wandb agent --forward-signals ([^[:space:]]+).*|\1|p' | tail -n 1)
    case "$path" in
        "$ENTITY_NAME/$PROJECT_NAME/"*) printf '%s\n' "$path" ;;
        *)
            echo "Could not extract a sweep path from:" >&2
            printf '%s\n' "$creation_output" >&2
            return 1
            ;;
    esac
}

extract_expected_runs() {
    local creation_output=$1
    local expected
    expected=$(printf '%s\n' "$creation_output" | sed -nE 's/^EXPECTED_RUNS=([0-9]+)$/\1/p' | tail -n 1)
    [[ "$expected" =~ ^[0-9]+$ ]] || {
        echo "Could not extract EXPECTED_RUNS from sweep creation output." >&2
        return 1
    }
    printf '%s\n' "$expected"
}

create_llm_sweep() {
    local key=$1
    local expected=$2
    local agent_count=$3
    local name=$4
    shift 4
    local creation_output
    creation_output=$("$CLUSTER_PYTHON" sweeps/llm_finetune_sweep.py \
        tasks/llm_finetune.py \
        --sweep_name "$name" \
        --project_name "$PROJECT_NAME" \
        --cache_dir "$MEMORY_ALIGN_PROJECT/data/llm_cache" \
        --epochs 5 \
        --batch_sizes 32 \
        --weight_decay 0.0 \
        --mal_config "$MAL_CONFIG" \
        "$@")
    printf '%s\n' "$creation_output"
    SWEEP_PATHS[$key]=$(extract_sweep_path "$creation_output")
    EXPECTED_RUNS[$key]=$(extract_expected_runs "$creation_output")
    AGENT_COUNTS[$key]=$agent_count
    if [[ "${EXPECTED_RUNS[$key]}" != "$expected" ]]; then
        echo "$key expected $expected runs, but its creator reported ${EXPECTED_RUNS[$key]}." >&2
        exit 1
    fi
    printf '%s=%q\n%s_EXPECTED_RUNS=%q\n' \
        "$key" "${SWEEP_PATHS[$key]}" "$key" "${EXPECTED_RUNS[$key]}" >>"$SWEEP_RECORD"
}

create_cifar_sweep() {
    local key=$1
    local expected=$2
    local agent_count=$3
    local name=$4
    local creation_output
    creation_output=$("$CLUSTER_PYTHON" sweeps/cifar_heatmap_sweep.py \
        tasks/cifar_train.py \
        --experiment cifar100-scheduler-ablation \
        --sweep_name "$name" \
        --project_name "$PROJECT_NAME" \
        --data_dir "$MEMORY_ALIGN_PROJECT/data" \
        --epochs 200 \
        --split_seed 20260901 \
        --amp_dtype bfloat16 \
        --float32_precision tf32 \
        --mal_sgdm_config "$MAL_SGDM_CONFIG")
    printf '%s\n' "$creation_output"
    SWEEP_PATHS[$key]=$(extract_sweep_path "$creation_output")
    EXPECTED_RUNS[$key]=$(extract_expected_runs "$creation_output")
    AGENT_COUNTS[$key]=$agent_count
    if [[ "${EXPECTED_RUNS[$key]}" != "$expected" ]]; then
        echo "$key expected $expected runs, but its creator reported ${EXPECTED_RUNS[$key]}." >&2
        exit 1
    fi
    printf '%s=%q\n%s_EXPECTED_RUNS=%q\n' \
        "$key" "${SWEEP_PATHS[$key]}" "$key" "${EXPECTED_RUNS[$key]}" >>"$SWEEP_RECORD"
}

submit_agents() {
    local key=$1
    local job_name=$2
    local submission
    submission=$(sbatch \
        --parsable \
        --array="1-${AGENT_COUNTS[$key]}" \
        --job-name="$job_name" \
        "$MEMORY_ALIGN_PROJECT/wb-agents.sh" \
        "${SWEEP_PATHS[$key]}")
    printf '%s\n' "${submission%%;*}"
}

wait_for_active_agents() {
    local any_active
    local job_id
    local states
    while :; do
        any_active=0
        for job_id in "${ACTIVE_AGENT_JOB_IDS[@]}"; do
            states=$(squeue -h -j "$job_id" -o '%T' 2>/dev/null || true)
            if [[ -n "$states" ]]; then
                any_active=1
                echo "$(date -Is) GPU array $job_id: $(printf '%s\n' "$states" | sort | uniq -c | xargs)"
            fi
        done
        ((any_active == 0)) && break
        sleep 60
    done
    for job_id in "${ACTIVE_AGENT_JOB_IDS[@]}"; do
        sacct -n -X -j "$job_id" --format=JobID,State,ExitCode,Elapsed -P 2>/dev/null || true
    done
    ACTIVE_AGENT_JOB_IDS=()
}

sweep_state() {
    "$CLUSTER_PYTHON" - "${SWEEP_PATHS[$1]}" <<'PY'
import sys
import wandb

print(wandb.Api(timeout=180).sweep(sys.argv[1]).state)
PY
}

validate_or_recover() {
    local key=$1
    local job_name=$2
    local round
    local recovery_job_id
    local state
    for ((round = 0; round <= MAX_RECOVERY_ROUNDS; round++)); do
        for _attempt in {1..12}; do
            if "$CLUSTER_PYTHON" sweeps/validate_sweep.py \
                "${SWEEP_PATHS[$key]}" \
                --expected_runs "${EXPECTED_RUNS[$key]}"; then
                return 0
            fi
            sleep 10
        done
        state=$(sweep_state "$key")
        if [[ "$state" == "FINISHED" || "$state" == "CANCELED" ]]; then
            echo "$key reached terminal state $state without ${EXPECTED_RUNS[$key]} finished runs." >&2
            return 1
        fi
        ((round < MAX_RECOVERY_ROUNDS)) || break
        recovery_job_id=$(submit_agents "$key" "$job_name")
        printf '%s_RECOVERY_JOB_%s=%q\n' "$key" "$((round + 1))" "$recovery_job_id" >>"$SWEEP_RECORD"
        ACTIVE_AGENT_JOB_IDS=("$recovery_job_id")
        wait_for_active_agents
    done
    echo "$key did not validate after recovery." >&2
    return 1
}

launch_phase() {
    local phase=$1
    shift
    local keys=("$@")
    local allocated=0
    local key
    local job_id
    ACTIVE_AGENT_JOB_IDS=()
    for key in "${keys[@]}"; do
        allocated=$((allocated + AGENT_COUNTS[$key]))
    done
    if ((allocated <= 0 || allocated > TOTAL_GPU_AGENTS)); then
        echo "$phase must allocate between 1 and $TOTAL_GPU_AGENTS GPU agents; found $allocated." >&2
        exit 1
    fi
    echo "$phase allocates $allocated/$TOTAL_GPU_AGENTS GPUs; no idle duplicate agents are submitted."
    for key in "${keys[@]}"; do
        job_id=$(submit_agents "$key" "${phase}-${key}")
        AGENT_JOB_IDS[$key]=$job_id
        ACTIVE_AGENT_JOB_IDS+=("$job_id")
        printf '%s_AGENT_JOB=%q\n' "$key" "$job_id" >>"$SWEEP_RECORD"
        echo "Submitted $key (${SWEEP_PATHS[$key]}) on ${AGENT_COUNTS[$key]} GPU agents as $job_id."
    done
    wait_for_active_agents
    for key in "${keys[@]}"; do
        validate_or_recover "$key" "${phase}-${key}-recovery"
    done
}

prepare_python_environment
"$CLUSTER_PYTHON" download_datasets.py \
    --task llm \
    --llm_cache_dir "$MEMORY_ALIGN_PROJECT/data/llm_cache"
"$CLUSTER_PYTHON" sweeps/validate_sweep.py "$SOURCE_SWEEP_PATH" --expected_runs 54
"$CLUSTER_PYTHON" download_datasets.py \
    --task cifar100 \
    --cifar100_dir "$MEMORY_ALIGN_PROJECT/data"
"$CLUSTER_PYTHON" sweeps/validate_sweep.py "$VISION_SOURCE_SWEEP_PATH" --expected_runs 540

SWEEP_RECORD="$MEMORY_ALIGN_PROJECT/logs/postship-sweeps-${SLURM_JOB_ID}.env"
: >"$SWEEP_RECORD"
printf 'SOURCE_SWEEP_PATH=%q\nVISION_SOURCE_SWEEP_PATH=%q\nMAL_CONFIG=%q\nMAL_SGDM_CONFIG=%q\n' \
    "$SOURCE_SWEEP_PATH" "$VISION_SOURCE_SWEEP_PATH" "$MAL_CONFIG" "$MAL_SGDM_CONFIG" >>"$SWEEP_RECORD"

# Phase 1 adds only the shipped MAL-AdamW to the completed scheduled pilot.
# The requested 0.3/1/3/10 multiplier grid brackets its plausible optimum
# without rerunning 27 already-complete AdamW/AM-AdamW/AdaTAMW runs.
create_llm_sweep \
    MAL_SCHEDULED_SWEEP_PATH 12 12 \
    "mal-adamw-smollm2-scheduled-completion-${SLURM_JOB_ID}" \
    --optimizers MAL_AdamW \
    --lr_multipliers 0.3 1.0 3.0 10.0 \
    --use_scheduler True
launch_phase llm-mal-on MAL_SCHEDULED_SWEEP_PATH

# Phase 2 is the LLM scheduler ablation. The common 0.3/1/3 grid gives exact
# scheduled counterparts for every optimizer; MAL additionally retains 10x.
create_llm_sweep \
    BASE_UNSCHEDULED_SWEEP_PATH 27 9 \
    "adamw-am-adatamw-smollm2-scheduler-free-${SLURM_JOB_ID}" \
    --optimizers AdamW AM_AdamW AdaTAMW \
    --lr_multipliers 0.3 1.0 3.0 \
    --use_scheduler False
create_llm_sweep \
    MAL_UNSCHEDULED_SWEEP_PATH 12 6 \
    "mal-adamw-smollm2-scheduler-free-${SLURM_JOB_ID}" \
    --optimizers MAL_AdamW \
    --lr_multipliers 0.3 1.0 3.0 10.0 \
    --use_scheduler False
launch_phase llm-off BASE_UNSCHEDULED_SWEEP_PATH MAL_UNSCHEDULED_SWEEP_PATH

ANALYSIS_DIR="$MEMORY_ALIGN_PROJECT/outputs/llm-scheduler-ablation-${SLURM_JOB_ID}"
"$CLUSTER_PYTHON" analysis/analyze_llm_scheduler_ablation.py \
    --scheduled_sweeps \
        "$SOURCE_SWEEP_PATH" \
        "${SWEEP_PATHS[MAL_SCHEDULED_SWEEP_PATH]}" \
    --unscheduled_sweeps \
        "${SWEEP_PATHS[BASE_UNSCHEDULED_SWEEP_PATH]}" \
        "${SWEEP_PATHS[MAL_UNSCHEDULED_SWEEP_PATH]}" \
    --mal_config "$MAL_CONFIG" \
    --backfill_metadata \
    --output_dir "$ANALYSIS_DIR"
printf 'ANALYSIS_DIR=%q\n' "$ANALYSIS_DIR" >>"$SWEEP_RECORD"

# Phase 3 adds the complementary vision stress test after the LLM ablation.
# It exactly matches the canonical BS=256, LR=0.1, WD=5e-4 cells in c72berzj,
# changing only warmup+cosine to constant LR for four SGDM-family optimizers.
create_cifar_sweep \
    CIFAR_UNSCHEDULED_SWEEP_PATH 12 12 \
    "sgdm-family-resnet50-cifar100-scheduler-free-${SLURM_JOB_ID}"
launch_phase cifar100-off CIFAR_UNSCHEDULED_SWEEP_PATH

CIFAR_ANALYSIS_DIR="$MEMORY_ALIGN_PROJECT/outputs/cifar100-scheduler-ablation-${SLURM_JOB_ID}"
"$CLUSTER_PYTHON" analysis/analyze_cifar_scheduler_ablation.py \
    --scheduled_sweep "$VISION_SOURCE_SWEEP_PATH" \
    --unscheduled_sweep "${SWEEP_PATHS[CIFAR_UNSCHEDULED_SWEEP_PATH]}" \
    --output_dir "$CIFAR_ANALYSIS_DIR"
printf 'CIFAR_ANALYSIS_DIR=%q\n' "$CIFAR_ANALYSIS_DIR" >>"$SWEEP_RECORD"

echo "Post-shipping benchmark sequence finished. Receipt: $SWEEP_RECORD"
echo "LLM analysis: $ANALYSIS_DIR/report.md"
echo "CIFAR-100 analysis: $CIFAR_ANALYSIS_DIR/report.md"
