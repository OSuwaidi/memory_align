#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=500:00:00
#SBATCH --job-name=mal-sweep-master
#SBATCH --output=/shared/b00090279/memory_align/logs/master-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/master-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
AGENT_COUNT=15
CLUSTER_VENV="$MEMORY_ALIGN_PROJECT/.cluster-venv"
CLUSTER_PYTHON="$CLUSTER_VENV/bin/python"
UV_BIN=/shared/b00090279/.local/bin/uv
LOCK_HASH=$(sha256sum "$MEMORY_ALIGN_PROJECT/uv.lock" | cut -c1-16)
ENVIRONMENT_MARKER="$CLUSTER_VENV/.mal-uv-lock-$LOCK_HASH"
ACTIVE_AGENT_JOB_ID=""

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
        echo "Installing uv beneath /shared/b00090279"
        curl -LsSf https://astral.sh/uv/install.sh | env \
            UV_INSTALL_DIR=/shared/b00090279/.local/bin \
            UV_NO_MODIFY_PATH=1 \
            sh
    fi

    "$UV_BIN" python install 3.14
    echo "Synchronizing the committed uv.lock in frozen mode"
    UV_PROJECT_ENVIRONMENT="$CLUSTER_VENV" "$UV_BIN" sync \
        --frozen \
        --python 3.14 \
        --compile-bytecode

    "$CLUSTER_PYTHON" - <<'PY'
from importlib.metadata import version
import sys

import numpy
import sklearn
import timm
import torch
import torchvision
import wandb

assert sys.version_info >= (3, 14)
assert torch.__version__.startswith("2.11.0+cu128"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
print(
    "Cluster imports passed: "
    f"python={sys.version.split()[0]}, torch={torch.__version__}, "
    f"torchvision={torchvision.__version__}, cuda={torch.version.cuda}, "
    f"timm={timm.__version__}, wandb={version('wandb')}, "
    f"numpy={numpy.__version__}, sklearn={sklearn.__version__}"
)
PY
    touch "$ENVIRONMENT_MARKER"
    echo "Built cluster environment for uv.lock $LOCK_HASH at $CLUSTER_VENV"
}

prepare_cifar100() {
    echo "Preparing CIFAR-100 once from the verified Hugging Face mirror"
    "$CLUSTER_PYTHON" download_datasets.py \
        --task cifar100 \
        --cifar100_dir "$MEMORY_ALIGN_PROJECT/data"
}

prepare_tiny_imagenet() {
    echo "Preparing Tiny-ImageNet once before concurrent agents start"
    "$CLUSTER_PYTHON" download_datasets.py \
        --task tiny-imagenet \
        --tiny_imagenet_dir "$MEMORY_ALIGN_PROJECT/data"
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

create_sweep() {
    local sweep_script=$1
    local training_script=$2
    local sweep_name=$3
    local creation_output

    creation_output=$("$CLUSTER_PYTHON" "$sweep_script" "$training_script" \
        --sweep_name "$sweep_name" \
        --project_name "$PROJECT_NAME")
    printf '%s\n' "$creation_output" >&2
    extract_sweep_path "$creation_output"
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
    local accounting_records=""
    local task_records=""
    local task_count=0
    local attempt

    echo "Waiting for agent array $job_id ($sweep_path)"
    while :; do
        queue_snapshot=$(squeue -h -j "$job_id" -o '%T' 2>/dev/null || true)
        [[ -z "$queue_snapshot" ]] && break
        echo "$(date -Is) array $job_id: $(printf '%s\n' "$queue_snapshot" | sort | uniq -c | xargs)"
        sleep 60
    done

    for attempt in {1..12}; do
        accounting_records=$(sacct -n -X -j "$job_id" --format=JobID,State,ExitCode -P 2>/dev/null || true)
        task_records=$(printf '%s\n' "$accounting_records" | awk -F'|' -v prefix="${job_id}_" 'index($1, prefix) == 1 && $1 ~ /_[0-9]+$/')
        task_count=$(printf '%s\n' "$task_records" | sed '/^$/d' | wc -l)
        [[ "$task_count" -eq "$AGENT_COUNT" ]] && break
        sleep 5
    done

    printf '%s\n' "$accounting_records"
    if [[ "$task_count" -ne "$AGENT_COUNT" ]]; then
        echo "Expected $AGENT_COUNT accounted array tasks for $job_id, found $task_count" >&2
        return 1
    fi
    if printf '%s\n' "$task_records" | awk -F'|' '$2 !~ /^COMPLETED/ { failed = 1 } END { exit failed ? 0 : 1 }'; then
        echo "At least one agent task in array $job_id did not complete successfully" >&2
        return 1
    fi
    echo "All $AGENT_COUNT agents completed for $sweep_path"
}

prepare_python_environment
prepare_cifar100

SWEEP_RECORD="$MEMORY_ALIGN_PROJECT/logs/sweep-paths-${SLURM_JOB_ID}.env"

SGDM_SWEEP_PATH=$(create_sweep \
    sweeps/cifar_resnet_sweep.py \
    tasks/cifar_train.py \
    "mal-sgdm-structure-${SLURM_JOB_ID}")
printf 'SGDM_SWEEP_PATH=%q\n' "$SGDM_SWEEP_PATH" >"$SWEEP_RECORD"

ACTIVE_AGENT_JOB_ID=$(submit_agents "$SGDM_SWEEP_PATH" mal-sgdm-agent)
echo "Submitted first sweep $SGDM_SWEEP_PATH as 15-GPU array $ACTIVE_AGENT_JOB_ID"
wait_for_agents "$ACTIVE_AGENT_JOB_ID" "$SGDM_SWEEP_PATH"
ACTIVE_AGENT_JOB_ID=""

prepare_tiny_imagenet

ADAMW_SWEEP_PATH=$(create_sweep \
    sweeps/adamw_vit_sweep.py \
    tasks/mae_pretrain.py \
    "mal-adamw-structure-${SLURM_JOB_ID}")
printf 'ADAMW_SWEEP_PATH=%q\n' "$ADAMW_SWEEP_PATH" >>"$SWEEP_RECORD"

ACTIVE_AGENT_JOB_ID=$(submit_agents "$ADAMW_SWEEP_PATH" mal-adamw-agent)
echo "Submitted second sweep $ADAMW_SWEEP_PATH as 15-GPU array $ACTIVE_AGENT_JOB_ID"
wait_for_agents "$ACTIVE_AGENT_JOB_ID" "$ADAMW_SWEEP_PATH"
ACTIVE_AGENT_JOB_ID=""

echo "Both sequential MAL structural sweeps completed successfully. Sweep paths: $SWEEP_RECORD"
