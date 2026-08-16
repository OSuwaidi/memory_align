#!/usr/bin/env bash
set -euo pipefail

readonly REPO_URL="https://github.com/OSuwaidi/memory_align.git"
readonly REPO_BRANCH="run_mal"
readonly REPO_DIR="memory_align"
readonly DATA_DIR="data"
readonly DATA_ARCHIVE="${DATA_DIR}/cifar-100-python.tar.gz"
readonly DATASET_URL="https://huggingface.co/datasets/nakroy/cifar100-python/resolve/main/cifar-100-python.tar.gz"
readonly UV_BIN_DIR="${HOME}/.local/bin"
readonly TMUX_SESSION="sweep"
readonly SWEEP_PATH="osuwaidi-khalifa-university/FINAL_MAL_CIFAR100/vp6dazyc"

if ! command -v tmux >/dev/null 2>&1 || ! command -v wget >/dev/null 2>&1; then
    apt-get update
    apt-get install -y tmux wget
fi

if [[ ! -x "${UV_BIN_DIR}/uv" ]]; then
    curl -LsSf https://astral.sh/uv/install.sh | \
        env UV_INSTALL_DIR="${UV_BIN_DIR}" UV_NO_MODIFY_PATH=1 sh
fi
export PATH="${UV_BIN_DIR}:${PATH}"

if [[ ! -x "${UV_BIN_DIR}/uv" ]]; then
    echo "uv installation failed: ${UV_BIN_DIR}/uv was not created." >&2
    exit 1
fi

git clone \
    --branch "${REPO_BRANCH}" \
    --single-branch \
    --depth 1 \
    "${REPO_URL}" \
    "${REPO_DIR}"

cd "${REPO_DIR}"
readonly REPO_ROOT="$(pwd -P)"
mkdir -p "${DATA_DIR}"

uv sync --no-dev --upgrade

echo "Downloading CIFAR-100 dataset..."

wget -q --show-progress \
    -O "${DATA_ARCHIVE}" \
    "${DATASET_URL}"
tar -xzf "${DATA_ARCHIVE}" -C "${DATA_DIR}"

echo "Setup and download complete."

readonly TTY_NAME="$(ps -o tty= -p "$$" | xargs)"
readonly TTY_PATH="/dev/${TTY_NAME}"

if [[ -z "${TTY_NAME}" || "${TTY_NAME}" == "?" || ! -c "${TTY_PATH}" ]]; then
    echo "A controlling terminal is required to enter the W&B key and attach to tmux." >&2
    exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    read -rsp "Enter W&B API key: " WANDB_API_KEY <"${TTY_PATH}"
    echo
fi
export WANDB_API_KEY

tmux new-session \
    -s "${TMUX_SESSION}" \
    -c "${REPO_ROOT}" \
    -e "WANDB_API_KEY=${WANDB_API_KEY}" \
    "uv run wandb agent --forward-signals ${SWEEP_PATH}"

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "Detached from sweep session."
    echo "Reattach with: tmux attach -t ${TMUX_SESSION}"
else
    echo "W&B agent exited; the tmux session has ended."
fi

# Run with: `$ curl -fsSLo /tmp/bootstrap_cifar100.sh https://raw.githubusercontent.com/OSuwaidi/memory_align/run_mal/bootstrap_cifar100.sh && bash /tmp/bootstrap_cifar100.sh`
