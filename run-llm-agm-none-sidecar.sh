#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=180:00:00
#SBATCH --job-name=agm-none-sidecar
#SBATCH --output=/shared/b00090279/memory_align/logs/llm-agm-none-master-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/llm-agm-none-master-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
ENTITY_NAME=osuwaidi-khalifa-university
PROJECT_NAME=MAL_benchmark
CLUSTER_PYTHON="$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python"
TOKEN_DATA_DIR="$MEMORY_ALIGN_PROJECT/data/fineweb-edu-smollm2"
RUN_OUTPUT_DIR="$MEMORY_ALIGN_PROJECT/outputs/llm-pretrain"
AGENT_COUNT=12
ACTIVE_AGENT_JOB_ID=""

. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"
mkdir -p "$MEMORY_ALIGN_PROJECT/logs" "$RUN_OUTPUT_DIR"

[[ -x "$CLUSTER_PYTHON" ]] || {
    echo "Cluster environment is missing: $CLUSTER_PYTHON" >&2
    exit 1
}

cancel_active_agents() {
    local signal_name=${1:-TERM}
    if [[ -n "$ACTIVE_AGENT_JOB_ID" ]] && squeue -h -j "$ACTIVE_AGENT_JOB_ID" 2>/dev/null | grep -q .; then
        echo "Sidecar master received $signal_name; cancelling GPU array $ACTIVE_AGENT_JOB_ID" >&2
        scancel "$ACTIVE_AGENT_JOB_ID"
    fi
}
trap 'cancel_active_agents TERM; exit 143' TERM
trap 'cancel_active_agents INT; exit 130' INT

sweep_output=$("$CLUSTER_PYTHON" sweeps/llm_pretrain_sweep.py \
    tasks/llm_pretrain.py \
    --stage agm_none_screen \
    --sweep_name "agm-smollm2-360m-fineweb-scale-none-${SLURM_JOB_ID}" \
    --project_name "$PROJECT_NAME" \
    --data_dir "$TOKEN_DATA_DIR" \
    --output_dir "$RUN_OUTPUT_DIR" \
    --micro_batch_size 4 \
    --gradient_accumulation_steps 32)
printf '%s\n' "$sweep_output"

sweep_path=$(printf '%s\n' "$sweep_output" | sed -nE 's/^SWEEP_PATH=(.+)$/\1/p' | tail -n 1)
expected_runs=$(printf '%s\n' "$sweep_output" | sed -nE 's/^EXPECTED_RUNS=(.+)$/\1/p' | tail -n 1)
case "$sweep_path" in
    "$ENTITY_NAME/$PROJECT_NAME/"*) ;;
    *) echo "Invalid sweep path: $sweep_path" >&2; exit 2 ;;
esac
[[ "$expected_runs" == "$AGENT_COUNT" ]] || {
    echo "Expected $AGENT_COUNT paired runs, received $expected_runs" >&2
    exit 2
}

receipt="$MEMORY_ALIGN_PROJECT/logs/llm-agm-none-sidecar-${SLURM_JOB_ID}.env"
printf 'SWEEP_PATH=%q\nEXPECTED_RUNS=%q\n' "$sweep_path" "$expected_runs" >"$receipt"

submission=$(sbatch \
    --parsable \
    --array="1-${AGENT_COUNT}" \
    --job-name=agm-llm-none \
    "$MEMORY_ALIGN_PROJECT/wb-llm-pretrain-agent.sh" \
    "$sweep_path")
ACTIVE_AGENT_JOB_ID=${submission%%;*}
printf 'AGENT_JOB_ID=%q\n' "$ACTIVE_AGENT_JOB_ID" >>"$receipt"
echo "Submitted scale-none sweep $sweep_path as GPU array $ACTIVE_AGENT_JOB_ID"

while squeue -h -j "$ACTIVE_AGENT_JOB_ID" 2>/dev/null | grep -q .; do
    states=$(squeue -h -j "$ACTIVE_AGENT_JOB_ID" -o '%T' | sort | uniq -c | xargs)
    echo "$(date -Is) job $ACTIVE_AGENT_JOB_ID: $states"
    sleep 60
done
ACTIVE_AGENT_JOB_ID=""

for _attempt in {1..12}; do
    if "$CLUSTER_PYTHON" sweeps/validate_sweep.py "$sweep_path" --expected_runs "$expected_runs"; then
        echo "Paired AGM-AdamW scale comparison completed. Receipt: $receipt"
        exit 0
    fi
    sleep 10
done

echo "Scale-none sweep did not finish with exactly $expected_runs successful runs." >&2
exit 1
