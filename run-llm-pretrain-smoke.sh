#!/usr/bin/env bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=01:00:00
#SBATCH --job-name=agm-llm-smoke
#SBATCH --output=/shared/b00090279/memory_align/logs/llm-pretrain-smoke-%j.out
#SBATCH --error=/shared/b00090279/memory_align/logs/llm-pretrain-smoke-%j.err

set -euo pipefail

MEMORY_ALIGN_PROJECT=/shared/b00090279/memory_align
MICRO_BATCH_SIZE=${1:-4}
TOKEN_DATA_DIR=${2:-"$MEMORY_ALIGN_PROJECT/data/fineweb-edu-smollm2"}

. "$MEMORY_ALIGN_PROJECT/cluster-env.sh"
cd "$MEMORY_ALIGN_PROJECT"

for optimizer_name in AdamW AM_AdamW AdaTAMW AGM_AdamW; do
    "$MEMORY_ALIGN_PROJECT/.cluster-venv/bin/python" tasks/llm_pretrain.py \
        --data_dir "$TOKEN_DATA_DIR" \
        --optimizer_case "${optimizer_name}::0.00075::0.01" \
        --seed 42 \
        --sequence_length 2048 \
        --micro_batch_size "$MICRO_BATCH_SIZE" \
        --gradient_accumulation_steps 2 \
        --max_steps 1 \
        --warmup_ratio 0 \
        --eval_every 1 \
        --eval_max_sequences 2 \
        --num_workers 0 \
        --checkpoint_every 0 \
        --evaluate_test false \
        --wandb_mode disabled \
        --output_dir "$MEMORY_ALIGN_PROJECT/outputs/llm-pretrain-smoke"
done
