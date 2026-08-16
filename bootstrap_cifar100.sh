#!/usr/bin/env bash
set -euo pipefail

readonly REPO_URL="https://github.com/OSuwaidi/memory_align.git"
readonly REPO_BRANCH="run_mal"
readonly REPO_DIR="memory_align"
readonly DATA_DIR="data"
readonly DATA_ARCHIVE="${DATA_DIR}/cifar-100-python.tar.gz"
readonly DATASET_URL="https://huggingface.co/datasets/nakroy/cifar100-python/resolve/main/cifar-100-python.tar.gz"
readonly TMUX_SESSION="sweep"
readonly SWEEP_PATH="osuwaidi-khalifa-university/FINAL_MAL_CIFAR100/9pj6h9ej"

# uv self update

git clone \
    --branch "${REPO_BRANCH}" \
    --single-branch \
    --depth 1 \
    "${REPO_URL}" \
    "${REPO_DIR}"

cd "${REPO_DIR}"
readonly REPO_ROOT="$(pwd -P)"
mkdir -p "${DATA_DIR}"

uv sync --upgrade

echo "Downloading CIFAR-100 dataset..."

wget -q --show-progress \
    -O "${DATA_ARCHIVE}" \
    "${DATASET_URL}"
tar -xzf "${DATA_ARCHIVE}" -C "${DATA_DIR}"

echo "Setup and download complete."

tmux new-session \
    -d \
    -s "${TMUX_SESSION}" \
    -c "${REPO_ROOT}" \
    -e "WANDB_API_KEY=${WANDB_API_KEY:?WANDB_API_KEY must be supplied by the provider}" \
    "uv run wandb agent --forward-signals ${SWEEP_PATH}"

echo "Sweep started."
echo "Attach with: tmux attach -t ${TMUX_SESSION}"

# Run with: `$ curl -fsSL https://raw.githubusercontent.com/OSuwaidi/memory_align/run_mal/bootstrap_cifar100.sh | bash`
