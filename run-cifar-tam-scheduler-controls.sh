#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=500:00:00
#SBATCH --job-name=cifar-paired-controls
#SBATCH --output=/shared/b00090279/memory_align/logs/cifar-controls-master-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/cifar-controls-master-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
CIFAR10_SOURCE_SWEEP=${1:-$ENTITY_NAME/$PROJECT_NAME/bps6rkim}
CIFAR100_SOURCE_SWEEP=${2:-$ENTITY_NAME/$PROJECT_NAME/c72berzj}
UPSTREAM_MASTER_JOB=${3:-31443}
UPSTREAM_RECEIPT=${4:-$MEMORY_ALIGN_PROJECT/logs/postship-sweeps-${UPSTREAM_MASTER_JOB}.env}
MAL_SGDM_CONFIG=False,1.0,False,attenuate
TOTAL_GPU_AGENTS=15
MAX_RECOVERY_ROUNDS=2
CLUSTER_VENV="$MEMORY_ALIGN_PROJECT/.cluster-venv"
CLUSTER_PYTHON="$CLUSTER_VENV/bin/python"

declare -A SWEEP_PATHS=()
declare -A EXPECTED_RUNS=()
declare -A AGENT_COUNTS=()
ACTIVE_AGENT_JOB_IDS=()

. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"

cancel_active_agents() {
    local job_id
    for job_id in "${ACTIVE_AGENT_JOB_IDS[@]:-}"; do
        if [[ -n "$job_id" ]] && squeue -h -j "$job_id" 2>/dev/null | grep -q .; then
            scancel "$job_id"
        fi
    done
}
trap 'cancel_active_agents; exit 143' TERM
trap 'cancel_active_agents; exit 130' INT

extract_sweep_path() {
    local output=$1
    local path
    path=$(printf '%s\n' "$output" | sed -nE 's|.*wandb agent --forward-signals ([^[:space:]]+).*|\1|p' | tail -n 1)
    case "$path" in
        "$ENTITY_NAME/$PROJECT_NAME/"*) printf '%s\n' "$path" ;;
        *) echo "Could not extract sweep path" >&2; return 1 ;;
    esac
}

extract_expected_runs() {
    local output=$1
    printf '%s\n' "$output" | sed -nE 's/^EXPECTED_RUNS=([0-9]+)$/\1/p' | tail -n 1
}

create_cifar_sweep() {
    local key=$1
    local experiment=$2
    local expected=$3
    local agents=$4
    local name=$5
    local output
    output=$("$CLUSTER_PYTHON" sweeps/cifar_heatmap_sweep.py \
        tasks/cifar_train.py \
        --experiment "$experiment" \
        --sweep_name "$name" \
        --project_name "$PROJECT_NAME" \
        --data_dir "$MEMORY_ALIGN_PROJECT/data" \
        --epochs 200 \
        --split_seed 20260901 \
        --amp_dtype bfloat16 \
        --float32_precision tf32 \
        --mal_sgdm_config "$MAL_SGDM_CONFIG")
    printf '%s\n' "$output"
    SWEEP_PATHS[$key]=$(extract_sweep_path "$output")
    EXPECTED_RUNS[$key]=$(extract_expected_runs "$output")
    AGENT_COUNTS[$key]=$agents
    if [[ "${EXPECTED_RUNS[$key]}" != "$expected" ]]; then
        echo "$key expected $expected runs; creator returned ${EXPECTED_RUNS[$key]}." >&2
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

wait_for_agents() {
    local active
    local job_id
    local states
    while :; do
        active=0
        for job_id in "${ACTIVE_AGENT_JOB_IDS[@]}"; do
            states=$(squeue -h -j "$job_id" -o '%T' 2>/dev/null || true)
            if [[ -n "$states" ]]; then
                active=1
                echo "$(date -Is) GPU array $job_id: $(printf '%s\n' "$states" | sort | uniq -c | xargs)"
            fi
        done
        ((active == 0)) && break
        sleep 60
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
    local round
    local state
    local job_id
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
            echo "$key ended as $state without a complete grid." >&2
            return 1
        fi
        ((round < MAX_RECOVERY_ROUNDS)) || break
        job_id=$(submit_agents "$key" "$key-recovery")
        printf '%s_RECOVERY_%s=%q\n' "$key" "$round" "$job_id" >>"$SWEEP_RECORD"
        ACTIVE_AGENT_JOB_IDS=("$job_id")
        wait_for_agents
    done
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
        echo "$phase requested $allocated GPU agents; allowed range is 1-$TOTAL_GPU_AGENTS." >&2
        exit 1
    fi
    for key in "${keys[@]}"; do
        job_id=$(submit_agents "$key" "$phase-$key")
        ACTIVE_AGENT_JOB_IDS+=("$job_id")
        printf '%s_AGENT_JOB=%q\n' "$key" "$job_id" >>"$SWEEP_RECORD"
        echo "Submitted $key (${SWEEP_PATHS[$key]}) with ${AGENT_COUNTS[$key]} agents as $job_id."
    done
    wait_for_agents
    for key in "${keys[@]}"; do
        validate_or_recover "$key"
    done
}

wait_for_upstream_master() {
    while squeue -h -j "$UPSTREAM_MASTER_JOB" 2>/dev/null | grep -q .; do
        echo "$(date -Is) waiting for upstream master $UPSTREAM_MASTER_JOB to release its 12 GPUs"
        sleep 60
    done
    for _attempt in {1..10}; do
        [[ -f "$UPSTREAM_RECEIPT" ]] && return 0
        sleep 10
    done
    echo "Upstream receipt not found: $UPSTREAM_RECEIPT" >&2
    return 1
}

"$CLUSTER_PYTHON" sweeps/validate_sweep.py "$CIFAR10_SOURCE_SWEEP" --expected_runs 735
"$CLUSTER_PYTHON" sweeps/validate_sweep.py "$CIFAR100_SOURCE_SWEEP" --expected_runs 540
"$CLUSTER_PYTHON" download_datasets.py --task cifar10 --cifar10_dir "$MEMORY_ALIGN_PROJECT/data"

SWEEP_RECORD="$MEMORY_ALIGN_PROJECT/logs/cifar-paired-sweeps-${SLURM_JOB_ID}.env"
: >"$SWEEP_RECORD"
printf 'CIFAR10_SOURCE_SWEEP=%q\nCIFAR100_SOURCE_SWEEP=%q\nMAL_SGDM_CONFIG=%q\nUPSTREAM_MASTER_JOB=%q\n' \
    "$CIFAR10_SOURCE_SWEEP" "$CIFAR100_SOURCE_SWEEP" "$MAL_SGDM_CONFIG" "$UPSTREAM_MASTER_JOB" >>"$SWEEP_RECORD"

# Only three GPUs are presently free.  Use them immediately to add the exact
# scheduled SGDM and fixed-gate TAM controls missing from bps6rkim.
create_cifar_sweep \
    CIFAR10_SCHEDULED_CONTROLS \
    cifar10-scheduled-controls \
    6 3 \
    "resnet18-cifar10-scheduled-controls-${SLURM_JOB_ID}"
launch_phase cifar10-on-controls CIFAR10_SCHEDULED_CONTROLS

# The running 12-GPU CIFAR-100 job is left undisturbed.  Once its master exits,
# all 15 GPUs are allocated: 12 to CIFAR-10 and 3 to the longer ResNet-50
# fixed-gate controls.
wait_for_upstream_master
set +u
. "$UPSTREAM_RECEIPT"
set -u
if [[ -z "${CIFAR_UNSCHEDULED_SWEEP_PATH:-}" ]]; then
    echo "CIFAR_UNSCHEDULED_SWEEP_PATH missing from $UPSTREAM_RECEIPT" >&2
    exit 1
fi
"$CLUSTER_PYTHON" sweeps/validate_sweep.py "$CIFAR_UNSCHEDULED_SWEEP_PATH" --expected_runs 12

create_cifar_sweep \
    CIFAR10_UNSCHEDULED \
    cifar10-scheduler-ablation \
    15 12 \
    "sgdm-family-resnet18-cifar10-scheduler-free-${SLURM_JOB_ID}"
create_cifar_sweep \
    CIFAR100_TAM_CONTROLS \
    cifar100-tam-controls \
    6 3 \
    "tam-fixed-control-resnet50-cifar100-${SLURM_JOB_ID}"
launch_phase paired-off-and-tam CIFAR10_UNSCHEDULED CIFAR100_TAM_CONTROLS

CIFAR10_ANALYSIS="$MEMORY_ALIGN_PROJECT/outputs/cifar10-paired-ablation-${SLURM_JOB_ID}"
"$CLUSTER_PYTHON" analysis/analyze_cifar_paired_ablation.py \
    --task_label ResNet18/CIFAR-10 \
    --scheduled_sweeps "$CIFAR10_SOURCE_SWEEP" "${SWEEP_PATHS[CIFAR10_SCHEDULED_CONTROLS]}" \
    --unscheduled_sweeps "${SWEEP_PATHS[CIFAR10_UNSCHEDULED]}" \
    --optimizers SGDM AM_MSGD TAM_SGDM TAM_baseline MAL_SGDM \
    --batch_size 256 \
    --learning_rate 0.1 \
    --weight_decay 0.0005 \
    --mal_config "$MAL_SGDM_CONFIG" \
    --output_dir "$CIFAR10_ANALYSIS"

CIFAR100_ANALYSIS="$MEMORY_ALIGN_PROJECT/outputs/cifar100-paired-ablation-${SLURM_JOB_ID}"
"$CLUSTER_PYTHON" analysis/analyze_cifar_paired_ablation.py \
    --task_label ResNet50/CIFAR-100 \
    --scheduled_sweeps "$CIFAR100_SOURCE_SWEEP" "${SWEEP_PATHS[CIFAR100_TAM_CONTROLS]}" \
    --unscheduled_sweeps "$CIFAR_UNSCHEDULED_SWEEP_PATH" "${SWEEP_PATHS[CIFAR100_TAM_CONTROLS]}" \
    --optimizers SGDM AM_MSGD TAM_SGDM TAM_baseline MAL_SGDM \
    --batch_size 256 \
    --learning_rate 0.1 \
    --weight_decay 0.0005 \
    --mal_config "$MAL_SGDM_CONFIG" \
    --output_dir "$CIFAR100_ANALYSIS"

printf 'CIFAR10_ANALYSIS=%q\nCIFAR100_ANALYSIS=%q\n' \
    "$CIFAR10_ANALYSIS" "$CIFAR100_ANALYSIS" >>"$SWEEP_RECORD"
echo "Paired CIFAR scheduler/TAM controls complete: $SWEEP_RECORD"
