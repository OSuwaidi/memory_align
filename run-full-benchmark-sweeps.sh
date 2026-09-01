#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=500:00:00
#SBATCH --job-name=mal-full-benchmark
#SBATCH --output=/shared/b00090279/memory_align/logs/full-benchmark-master-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/full-benchmark-master-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
AGENT_COUNT=15
MAX_AGENT_ROUNDS=3
CLUSTER_VENV="$MEMORY_ALIGN_PROJECT/.cluster-venv"
CLUSTER_PYTHON="$CLUSTER_VENV/bin/python"
UV_BIN=/shared/b00090279/.local/bin/uv
LOCK_HASH=$(sha256sum "$MEMORY_ALIGN_PROJECT/uv.lock" | cut -c1-16)
ENVIRONMENT_MARKER="$CLUSTER_VENV/.mal-uv-lock-$LOCK_HASH"
ACTIVE_AGENT_JOB_ID=""
DEFAULT_MAL_ADAMW_CONFIG="False,1.0,moment,replace,False,moment"

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
    "$CLUSTER_PYTHON" download_datasets.py --task cifar10 --cifar10_dir "$MEMORY_ALIGN_PROJECT/data"
    "$CLUSTER_PYTHON" download_datasets.py --task cifar100 --cifar100_dir "$MEMORY_ALIGN_PROJECT/data"
    "$CLUSTER_PYTHON" download_datasets.py --task tiny-imagenet --tiny_imagenet_dir "$MEMORY_ALIGN_PROJECT/data"
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
    echo "Waiting for 15-GPU array $job_id ($sweep_path)"
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
        echo "Submitted $sweep_path as 15-GPU array $ACTIVE_AGENT_JOB_ID (round $round)"
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
            echo "Sweep reached terminal state $state without the required $expected_runs finished runs." >&2
            return 1
        fi
        echo "Sweep remains $state after agent round $round; submitting another 15-agent recovery round."
    done
    echo "Sweep did not complete after $MAX_AGENT_ROUNDS agent rounds: $sweep_path" >&2
    return 1
}

create_cifar_sweep() {
    local experiment=$1
    local sweep_name=$2
    local mal_config=${3:-}
    local creation_output
    local command=(
        "$CLUSTER_PYTHON" sweeps/cifar_heatmap_sweep.py
        tasks/cifar_train.py
        --experiment "$experiment"
        --sweep_name "$sweep_name"
        --project_name "$PROJECT_NAME"
        --data_dir "$MEMORY_ALIGN_PROJECT/data"
    )
    if [[ -n "$mal_config" ]]; then
        command+=(--mal_sgdm_config "$mal_config")
    fi
    creation_output=$("${command[@]}")
    printf '%s\n' "$creation_output" >&2
    CREATED_SWEEP_PATH=$(extract_sweep_path "$creation_output")
    CREATED_EXPECTED_RUNS=$(extract_expected_runs "$creation_output")
}

create_vit_sweep() {
    local family=$1
    local sweep_name=$2
    local mal_config=$3
    local creation_output
    creation_output=$("$CLUSTER_PYTHON" "sweeps/${family}_vit_sweep.py" \
        tasks/mae_pretrain.py \
        --sweep_name "$sweep_name" \
        --project_name "$PROJECT_NAME" \
        --data_dir "$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200" \
        --mal_config "$mal_config")
    printf '%s\n' "$creation_output" >&2
    CREATED_SWEEP_PATH=$(extract_sweep_path "$creation_output")
    CREATED_EXPECTED_RUNS=$(extract_expected_runs "$creation_output")
}

prepare_python_environment
prepare_inputs

SWEEP_RECORD="$MEMORY_ALIGN_PROJECT/logs/full-benchmark-sweeps-${SLURM_JOB_ID}.env"
: >"$SWEEP_RECORD"

# Phase 1: 735 runs = 5 optimizer cases x 7 BS x 7 LR x 3 seeds.
create_cifar_sweep cifar10-screen "mal-cifar10-resnet18-heatmap-${SLURM_JOB_ID}"
record_sweep CIFAR10_HEATMAP_SWEEP_PATH "$CREATED_SWEEP_PATH" "$CREATED_EXPECTED_RUNS"
run_agents_until_complete "$CREATED_SWEEP_PATH" "$CREATED_EXPECTED_RUNS" mal-c10-hm CIFAR10_HEATMAP

SELECTION_DIR="$MEMORY_ALIGN_PROJECT/outputs/mal-sgdm-selection-${SLURM_JOB_ID}"
"$CLUSTER_PYTHON" analysis/select_mal_sgdm_heatmap.py \
    "$CREATED_SWEEP_PATH" \
    --output_dir "$SELECTION_DIR"
. "$SELECTION_DIR/selection.env"
printf 'MAL_SGDM_VARIANT=%q\nMAL_SGDM_CONFIG=%q\n' \
    "$MAL_SGDM_VARIANT" "$MAL_SGDM_CONFIG" >>"$SWEEP_RECORD"
echo "Selected MAL-SGDM $MAL_SGDM_VARIANT: $MAL_SGDM_CONFIG"

# Phase 2: 540 runs = 5 SGDM-family optimizers x 6 BS x 6 LR x 3 seeds.
create_cifar_sweep \
    cifar100-benchmark \
    "mal-cifar100-resnet50-heatmap-${SLURM_JOB_ID}" \
    "$MAL_SGDM_CONFIG"
record_sweep CIFAR100_HEATMAP_SWEEP_PATH "$CREATED_SWEEP_PATH" "$CREATED_EXPECTED_RUNS"
run_agents_until_complete "$CREATED_SWEEP_PATH" "$CREATED_EXPECTED_RUNS" mal-c100-hm CIFAR100_HEATMAP

# Phase 3: 120 runs = 5 optimizers x 2 BS x 2 LR x 2 WD x 3 seeds.
create_vit_sweep sgdm "mal-mae-sgdm-benchmark-${SLURM_JOB_ID}" "$MAL_SGDM_CONFIG"
record_sweep SGDM_MAE_SWEEP_PATH "$CREATED_SWEEP_PATH" "$CREATED_EXPECTED_RUNS"
run_agents_until_complete "$CREATED_SWEEP_PATH" "$CREATED_EXPECTED_RUNS" mal-mae-sgdm SGDM_MAE

# Phase 4: 120 runs with the shipped AdamW replacement/raw-moment geometry.
create_vit_sweep adamw "mal-mae-adamw-benchmark-${SLURM_JOB_ID}" "$DEFAULT_MAL_ADAMW_CONFIG"
record_sweep ADAMW_MAE_SWEEP_PATH "$CREATED_SWEEP_PATH" "$CREATED_EXPECTED_RUNS"
run_agents_until_complete "$CREATED_SWEEP_PATH" "$CREATED_EXPECTED_RUNS" mal-mae-adamw ADAMW_MAE

echo "All four benchmark phases completed. Sweep receipt: $SWEEP_RECORD"
