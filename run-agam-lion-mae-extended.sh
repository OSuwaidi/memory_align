#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=500:00:00
#SBATCH --job-name=agam-lion-mae-ext-master
#SBATCH --output=/shared/b00090279/memory_align/logs/agam-lion-mae-ext-master-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/agam-lion-mae-ext-master-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
DATA_DIR="$MEMORY_ALIGN_PROJECT/data/tiny-imagenet-200"
AGENT_COUNT=15
EXPECTED_RUNS=18
AGENT_JOB_ID=""

. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"

[[ -x "$CLUSTER_PYTHON" ]] || {
    echo "Cluster environment is missing: $CLUSTER_PYTHON" >&2
    exit 1
}
[[ -d "$DATA_DIR/train" && -d "$DATA_DIR/val" ]] || {
    echo "Tiny-ImageNet is missing or incomplete beneath $DATA_DIR." >&2
    exit 1
}

cancel_active_agents() {
    if [[ -n "$AGENT_JOB_ID" ]] && squeue -h -j "$AGENT_JOB_ID" 2>/dev/null | grep -q .; then
        scancel "$AGENT_JOB_ID"
    fi
}
trap 'cancel_active_agents; exit 143' TERM
trap 'cancel_active_agents; exit 130' INT

extract_value() {
    local key=$1
    local output=$2
    local value
    value=$(printf '%s\n' "$output" | sed -nE "s/^${key}=(.*)$/\\1/p" | tail -n 1)
    [[ -n "$value" ]] || {
        echo "Could not extract $key from sweep creation output." >&2
        return 1
    }
    printf '%s\n' "$value"
}

wait_for_job() {
    local job_id=$1
    local queue_snapshot
    while :; do
        queue_snapshot=$(squeue -h -j "$job_id" -o '%T' 2>/dev/null || true)
        [[ -z "$queue_snapshot" ]] && break
        echo "$(date -Is) AGAM-Lion array $job_id: $(printf '%s\n' "$queue_snapshot" | sort | uniq -c | xargs)"
        sleep 60
    done
    sacct -n -X -j "$job_id" --format=JobID,State,ExitCode,Elapsed -P 2>/dev/null || true
}

SWEEP_OUTPUT=$("$CLUSTER_PYTHON" sweeps/agam_lion_mae_extended_sweep.py \
    tasks/mae_pretrain.py \
    --sweep_name "agam-lion-mae-hparam-expansion-${SLURM_JOB_ID}" \
    --project_name "$PROJECT_NAME" \
    --data_dir "$DATA_DIR" \
    --output_dir "$MEMORY_ALIGN_PROJECT/outputs/agam-lion-mae-hparam-expansion")
printf '%s\n' "$SWEEP_OUTPUT"
SWEEP_PATH=$(extract_value SWEEP_PATH "$SWEEP_OUTPUT")
OBSERVED_EXPECTED_RUNS=$(extract_value EXPECTED_RUNS "$SWEEP_OUTPUT")

case "$SWEEP_PATH" in
    "$ENTITY_NAME/$PROJECT_NAME/"*) ;;
    *)
        echo "Refusing unexpected sweep path: $SWEEP_PATH" >&2
        exit 2
        ;;
esac
[[ "$OBSERVED_EXPECTED_RUNS" == "$EXPECTED_RUNS" ]] || {
    echo "Expected $EXPECTED_RUNS runs, sweep reports $OBSERVED_EXPECTED_RUNS." >&2
    exit 1
}

SWEEP_RECORD="$MEMORY_ALIGN_PROJECT/logs/agam-lion-mae-ext-sweep-${SLURM_JOB_ID}.env"
printf 'SWEEP_PATH=%q\nEXPECTED_RUNS=%q\n' "$SWEEP_PATH" "$EXPECTED_RUNS" >"$SWEEP_RECORD"

SUBMISSION=$(sbatch \
    --parsable \
    --array="1-${AGENT_COUNT}" \
    --job-name=agam-lion-mae-ext \
    "$MEMORY_ALIGN_PROJECT/wb-agents.sh" \
    "$SWEEP_PATH")
AGENT_JOB_ID=${SUBMISSION%%;*}
printf 'AGENT_JOB_ID=%q\n' "$AGENT_JOB_ID" >>"$SWEEP_RECORD"
echo "Submitted all 15 GPU agents as array $AGENT_JOB_ID for $SWEEP_PATH."

wait_for_job "$AGENT_JOB_ID"
AGENT_JOB_ID=""
for _attempt in {1..12}; do
    if "$CLUSTER_PYTHON" sweeps/validate_sweep.py "$SWEEP_PATH" --expected_runs "$EXPECTED_RUNS"; then
        echo "All $EXPECTED_RUNS AGAM-Lion runs completed. Receipt: $SWEEP_RECORD"
        exit 0
    fi
    sleep 10
done

echo "Sweep $SWEEP_PATH did not finish with exactly $EXPECTED_RUNS successful runs." >&2
exit 1
