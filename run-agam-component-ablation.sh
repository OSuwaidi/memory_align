#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=48:00:00
#SBATCH --job-name=agam-components
#SBATCH --output=/shared/b00090279/memory_align/logs/agam-components-master-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/agam-components-master-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY=osuwaidi-khalifa-university
PROJECT=MAL_benchmark
EXPECTED_RUNS=15
GPU_AGENTS=15
MAX_RECOVERY_ROUNDS=2
CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
ACTIVE_JOB=

. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"
mkdir -p logs outputs

cleanup() {
    if [[ -n "$ACTIVE_JOB" ]] && squeue -h -j "$ACTIVE_JOB" 2>/dev/null | grep -q .; then
        scancel "$ACTIVE_JOB"
    fi
}
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 130' INT

# These completed sweeps establish the exact scheduled and constant-LR cells
# that motivated this component study. Refuse to launch against partial sources.
"$CLUSTER_PYTHON" sweeps/validate_sweep.py "$ENTITY/$PROJECT/c72berzj" --expected_runs 540
"$CLUSTER_PYTHON" sweeps/validate_sweep.py "$ENTITY/$PROJECT/52y7g41m" --expected_runs 12
"$CLUSTER_PYTHON" download_datasets.py \
    --task cifar100 \
    --cifar100_dir "$MEMORY_ALIGN_PROJECT/data"

CREATION_OUTPUT=$("$CLUSTER_PYTHON" sweeps/agam_component_ablation_sweep.py \
    tasks/cifar_train.py \
    --sweep_name "agam-components-resnet50-cifar100-${SLURM_JOB_ID}" \
    --project_name "$PROJECT" \
    --data_dir "$MEMORY_ALIGN_PROJECT/data")
printf '%s\n' "$CREATION_OUTPUT"

SWEEP_PATH=$(printf '%s\n' "$CREATION_OUTPUT" | sed -nE 's/^SWEEP_PATH=([^[:space:]]+)$/\1/p' | tail -n 1)
CREATED_RUNS=$(printf '%s\n' "$CREATION_OUTPUT" | sed -nE 's/^EXPECTED_RUNS=([0-9]+)$/\1/p' | tail -n 1)
case "$SWEEP_PATH" in
    "$ENTITY/$PROJECT/"*) ;;
    *) echo "Could not resolve the created W&B sweep path." >&2; exit 1 ;;
esac
if [[ "$CREATED_RUNS" != "$EXPECTED_RUNS" ]]; then
    echo "Sweep creator returned $CREATED_RUNS runs; expected $EXPECTED_RUNS." >&2
    exit 1
fi

RECEIPT="$MEMORY_ALIGN_PROJECT/logs/agam-components-${SLURM_JOB_ID}.env"
printf 'SWEEP_PATH=%q\nEXPECTED_RUNS=%q\nGPU_AGENTS=%q\n' \
    "$SWEEP_PATH" "$EXPECTED_RUNS" "$GPU_AGENTS" >"$RECEIPT"

submit_agents() {
    local submission
    submission=$(sbatch \
        --parsable \
        --array="1-$GPU_AGENTS" \
        --job-name=agam-component-agents \
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
    local attempt
    for attempt in {1..12}; do
        if "$CLUSTER_PYTHON" sweeps/validate_sweep.py \
            "$SWEEP_PATH" \
            --expected_runs "$EXPECTED_RUNS"; then
            return 0
        fi
        sleep 10
    done
    return 1
}

for ((round = 0; round <= MAX_RECOVERY_ROUNDS; round++)); do
    ACTIVE_JOB=$(submit_agents)
    printf 'AGENT_JOB_%s=%q\n' "$round" "$ACTIVE_JOB" >>"$RECEIPT"
    wait_for_agents
    ACTIVE_JOB=
    if validate_sweep; then
        break
    fi
    if ((round == MAX_RECOVERY_ROUNDS)); then
        echo "AGAM component sweep remains incomplete after recovery." >&2
        exit 1
    fi
    echo "Submitting recovery agents for unfinished W&B cells." >&2
done

ANALYSIS_DIR="$MEMORY_ALIGN_PROJECT/outputs/agam-component-ablation-${SLURM_JOB_ID}"
"$CLUSTER_PYTHON" analysis/analyze_agam_component_ablation.py \
    --sweep "$SWEEP_PATH" \
    --output_dir "$ANALYSIS_DIR"
printf 'ANALYSIS_DIR=%q\n' "$ANALYSIS_DIR" >>"$RECEIPT"
echo "AGAM component ablation complete: $RECEIPT"
