#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=500:00:00
#SBATCH --job-name=mal-sgdm-structure
#SBATCH --output=/shared/b00090279/memory_align/logs/mal-sgdm-structure-master-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/mal-sgdm-structure-master-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
SOURCE_SWEEP=${1:-$ENTITY_NAME/$PROJECT_NAME/52y7g41m}
EXPECTED_RUNS=9
GPU_AGENTS=9
CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
ACTIVE_JOB=

. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"

cleanup() {
    if [[ -n "$ACTIVE_JOB" ]] && squeue -h -j "$ACTIVE_JOB" 2>/dev/null | grep -q .; then
        scancel "$ACTIVE_JOB"
    fi
}
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 130' INT

"$CLUSTER_PYTHON" sweeps/validate_sweep.py "$SOURCE_SWEEP" --expected_runs 12
"$CLUSTER_PYTHON" download_datasets.py --task cifar100 --cifar100_dir "$MEMORY_ALIGN_PROJECT/data"

CREATION_OUTPUT=$("$CLUSTER_PYTHON" sweeps/cifar_heatmap_sweep.py \
    tasks/cifar_train.py \
    --experiment cifar100-mal-structure-scheduler-free \
    --sweep_name "mal-sgdm-resnet50-cifar100-scheduler-free-structure-${SLURM_JOB_ID}" \
    --project_name "$PROJECT_NAME" \
    --data_dir "$MEMORY_ALIGN_PROJECT/data" \
    --epochs 200 \
    --split_seed 20260901 \
    --amp_dtype bfloat16 \
    --float32_precision tf32)
printf '%s\n' "$CREATION_OUTPUT"

SWEEP_PATH=$(printf '%s\n' "$CREATION_OUTPUT" | sed -nE 's|.*wandb agent --forward-signals ([^[:space:]]+).*|\1|p' | tail -n 1)
CREATED_RUNS=$(printf '%s\n' "$CREATION_OUTPUT" | sed -nE 's/^EXPECTED_RUNS=([0-9]+)$/\1/p' | tail -n 1)
case "$SWEEP_PATH" in
    "$ENTITY_NAME/$PROJECT_NAME/"*) ;;
    *) echo "Could not resolve the created sweep path." >&2; exit 1 ;;
esac
if [[ "$CREATED_RUNS" != "$EXPECTED_RUNS" ]]; then
    echo "Sweep creator reported $CREATED_RUNS runs; expected $EXPECTED_RUNS." >&2
    exit 1
fi

RECEIPT="$MEMORY_ALIGN_PROJECT/logs/mal-sgdm-structure-sweep-${SLURM_JOB_ID}.env"
printf 'SOURCE_SWEEP=%q\nMAL_STRUCTURE_SWEEP=%q\nEXPECTED_RUNS=%q\n' \
    "$SOURCE_SWEEP" "$SWEEP_PATH" "$EXPECTED_RUNS" >"$RECEIPT"

submit_agents() {
    local submission
    submission=$(sbatch \
        --parsable \
        --array="1-$GPU_AGENTS" \
        --job-name=mal-sgdm-structure-agents \
        "$MEMORY_ALIGN_PROJECT/wb-agents.sh" \
        "$SWEEP_PATH")
    printf '%s\n' "${submission%%;*}"
}

wait_for_agents() {
    local states
    while :; do
        states=$(squeue -h -j "$ACTIVE_JOB" -o '%T' 2>/dev/null || true)
        [[ -z "$states" ]] && break
        echo "$(date -Is) GPU array $ACTIVE_JOB: $(printf '%s\n' "$states" | sort | uniq -c | xargs)"
        sleep 60
    done
}

validate_sweep() {
    for _attempt in {1..12}; do
        if "$CLUSTER_PYTHON" sweeps/validate_sweep.py "$SWEEP_PATH" --expected_runs "$EXPECTED_RUNS"; then
            return 0
        fi
        sleep 10
    done
    return 1
}

ACTIVE_JOB=$(submit_agents)
printf 'AGENT_JOB=%q\n' "$ACTIVE_JOB" >>"$RECEIPT"
wait_for_agents
if ! validate_sweep; then
    echo "Initial agents did not complete the grid; submitting one recovery array." >&2
    ACTIVE_JOB=$(submit_agents)
    printf 'RECOVERY_AGENT_JOB=%q\n' "$ACTIVE_JOB" >>"$RECEIPT"
    wait_for_agents
    validate_sweep
fi
ACTIVE_JOB=

ANALYSIS_DIR="$MEMORY_ALIGN_PROJECT/outputs/mal-sgdm-cifar100-structure-${SLURM_JOB_ID}"
"$CLUSTER_PYTHON" analysis/analyze_mal_sgdm_structure_ablation.py \
    --source_sweep "$SOURCE_SWEEP" \
    --ablation_sweep "$SWEEP_PATH" \
    --output_dir "$ANALYSIS_DIR"
printf 'ANALYSIS_DIR=%q\n' "$ANALYSIS_DIR" >>"$RECEIPT"
echo "MAL-SGDM structure ablation complete: $RECEIPT"
