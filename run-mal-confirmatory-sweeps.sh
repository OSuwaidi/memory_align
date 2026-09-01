#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=250:00:00
#SBATCH --job-name=mal-confirm-master
#SBATCH --output=/shared/b00090279/memory_align/logs/confirm-master-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/confirm-master-%j.err

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

    # Populate the shared Hugging Face/timm cache once before 15 agents start.
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

create_sweep() {
    local experiment=$1
    local sweep_name=$2
    local creation_output
    creation_output=$("$CLUSTER_PYTHON" sweeps/mal_confirmatory_sweep.py \
        tasks/tiny_imagenet_classification.py \
        --experiment "$experiment" \
        --sweep_name "$sweep_name" \
        --project_name "$PROJECT_NAME" \
        --tiny_imagenet_dir "$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200")
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
}

run_phase() {
    local experiment=$1
    local sweep_name=$2
    local agent_job_name=$3
    local record_key=$4
    local sweep_path

    sweep_path=$(create_sweep "$experiment" "$sweep_name")
    printf '%s=%q\n' "$record_key" "$sweep_path" >>"$SWEEP_RECORD"
    ACTIVE_AGENT_JOB_ID=$(submit_agents "$sweep_path" "$agent_job_name")
    echo "Submitted $sweep_path as a ${AGENT_COUNT}-GPU array $ACTIVE_AGENT_JOB_ID"
    wait_for_agents "$ACTIVE_AGENT_JOB_ID" "$sweep_path"
    ACTIVE_AGENT_JOB_ID=""
}

prepare_python_environment
prepare_inputs

SWEEP_RECORD="$MEMORY_ALIGN_PROJECT/logs/confirmatory-sweep-paths-${SLURM_JOB_ID}.env"
: >"$SWEEP_RECORD"

# 20 runs: unscaled versus norm-preserving replacement for MAL-SGDM.
run_phase \
    sgdm-resnet50 \
    "mal-confirmatory-sgdm-resnet50-${SLURM_JOB_ID}" \
    mal-conf-sgdm \
    SGDM_CONFIRMATORY_SWEEP_PATH

# 30 runs: unscaled, step-norm, and raw-moment replacement for MAL-AdamW.
run_phase \
    adamw-vit \
    "mal-confirmatory-adamw-vit-${SLURM_JOB_ID}" \
    mal-conf-adamw \
    ADAMW_CONFIRMATORY_SWEEP_PATH

echo "Both final confirmatory sweeps completed successfully. Sweep paths: $SWEEP_RECORD"
