#!/usr/bin/env bash
set -euo pipefail

uv self update

git clone -b run_main --single-branch --depth 1 \
  https://github.com/OSuwaidi/memory_align.git

cd memory_align
mkdir -p data

uv sync --upgrade

echo "Downloading CIFAR-100 dataset..."

wget -q --show-progress \
  -O data/cifar-100-python.tar.gz \
  https://huggingface.co/datasets/nakroy/cifar100-python/resolve/main/cifar-100-python.tar.gz

tar -xzf data/cifar-100-python.tar.gz -C data/

echo "Setup and download complete."

read -rsp "Enter W&B API key: " WANDB_API_KEY </dev/tty
echo

tmux new-session -d -s sweep  -e "WANDB_API_KEY=$WANDB_API_KEY" "uv run wandb agent --forward-signals osuwaidi-khalifa-university/FINAL_MAL_CIFAR100/k9owtkim"

unset WANDB_API_KEY

echo "Sweep started."
echo "Attach with: tmux attach -t sweep"

# Run with: `$ curl -fsSL https://raw.githubusercontent.com/OSuwaidi/memory_align/run_main/bootstrap.sh | bash`
