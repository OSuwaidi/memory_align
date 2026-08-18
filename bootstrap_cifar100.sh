#!/usr/bin/env bash
set -euo pipefail

readonly REPO_URL="https://github.com/OSuwaidi/memory_align.git"
readonly REPO_BRANCH="run_mal"
readonly REPO_DIR="memory_align"
readonly DATA_DIR="data"
readonly DATA_ARCHIVE="${DATA_DIR}/cifar-100-python.tar.gz"
readonly DATASET_DIR="${DATA_DIR}/cifar-100-python"
readonly DATASET_MD5="eb9058c3a382ffc7106e4002c42a8d85"
readonly DATASET_PRIMARY_URL="https://huggingface.co/datasets/nakroy/cifar100-python/resolve/main/cifar-100-python.tar.gz"
readonly DATASET_FALLBACK_URL="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
readonly UV_BIN_DIR="/usr/local/bin"
readonly UV_BIN="${UV_BIN_DIR}/uv"
readonly TMUX_SESSION="sweep"
readonly SWEEP_PATH="osuwaidi-khalifa-university/FINAL_MAL_CIFAR100/it79n63q"

if ! command -v tmux >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
    apt-get update
    apt-get install -y ca-certificates curl tmux
fi

if [[ ! -x "${UV_BIN}" ]]; then
    install -d -m 0755 "${UV_BIN_DIR}"
    curl -LsSf https://astral.sh/uv/install.sh | \
        env UV_INSTALL_DIR="${UV_BIN_DIR}" UV_NO_MODIFY_PATH=1 sh
fi

if [[ ! -x "${UV_BIN}" ]]; then
    echo "uv installation failed: ${UV_BIN} was not created." >&2
    exit 1
fi

if [[ -d "${REPO_DIR}/.git" ]]; then
    if git -C "${REPO_DIR}" show-ref --verify --quiet "refs/heads/${REPO_BRANCH}"; then
        git -C "${REPO_DIR}" switch "${REPO_BRANCH}"
    else
        git -C "${REPO_DIR}" fetch origin "${REPO_BRANCH}"
        git -C "${REPO_DIR}" switch --create "${REPO_BRANCH}" --track "origin/${REPO_BRANCH}"
    fi
    git -C "${REPO_DIR}" pull --ff-only origin "${REPO_BRANCH}"
elif [[ -e "${REPO_DIR}" ]]; then
    echo "${REPO_DIR} exists but is not a Git repository." >&2
    exit 1
else
    git clone \
        --branch "${REPO_BRANCH}" \
        --single-branch \
        --depth 1 \
        "${REPO_URL}" \
        "${REPO_DIR}"
fi

cd "${REPO_DIR}"
mkdir -p "${DATA_DIR}"

"${UV_BIN}" sync --no-dev --upgrade

cifar100_is_ready() {
    [[ -s "${DATASET_DIR}/train" && \
        -s "${DATASET_DIR}/test" && \
        -s "${DATASET_DIR}/meta" ]]
}

download_cifar100_archive() {
    local partial_archive="${DATA_ARCHIVE}.part"
    local dataset_url

    echo "Downloading CIFAR-100 dataset..."
    rm -f "${partial_archive}"

    for dataset_url in "${DATASET_PRIMARY_URL}" "${DATASET_FALLBACK_URL}"; do
        echo "Trying ${dataset_url}"
        if curl --fail --location --show-error \
            --retry 2 \
            --retry-all-errors \
            --retry-delay 2 \
            --connect-timeout 30 \
            --output "${partial_archive}" \
            "${dataset_url}"; then
            if echo "${DATASET_MD5}  ${partial_archive}" | md5sum --check --status; then
                mv -f "${partial_archive}" "${DATA_ARCHIVE}"
                return 0
            fi
            echo "Downloaded file from ${dataset_url} failed its checksum; trying another source." >&2
        else
            echo "Download from ${dataset_url} failed; trying another source." >&2
        fi
        rm -f "${partial_archive}"
    done

    echo "Unable to download a valid CIFAR-100 archive from any source." >&2
    return 1
}

cifar100_archive_is_valid() {
    [[ -s "${DATA_ARCHIVE}" ]] && \
        echo "${DATASET_MD5}  ${DATA_ARCHIVE}" | md5sum --check --status && \
        tar -tzf "${DATA_ARCHIVE}" >/dev/null 2>&1
}

if cifar100_is_ready; then
    echo "CIFAR-100 already exists in ${DATASET_DIR}; skipping download."
else
    if ! cifar100_archive_is_valid; then
        download_cifar100_archive
    else
        echo "Using existing CIFAR-100 archive: ${DATA_ARCHIVE}"
    fi

    tar -xzf "${DATA_ARCHIVE}" -C "${DATA_DIR}"

    if ! cifar100_is_ready; then
        echo "CIFAR-100 extraction failed: expected train, test, and meta files in ${DATASET_DIR}." >&2
        exit 1
    fi
fi

echo "Setup complete."

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
    -e "WANDB_API_KEY=${WANDB_API_KEY}" \
    "${UV_BIN} run wandb agent --forward-signals ${SWEEP_PATH}"

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "Detached from sweep session."
    echo "Reattach with: tmux attach -t ${TMUX_SESSION}"
else
    echo "W&B agent exited; the tmux session has ended."
fi

# Run with: `$ curl -fsSLo /tmp/bootstrap_cifar100.sh https://raw.githubusercontent.com/OSuwaidi/memory_align/run_mal/bootstrap_cifar100.sh && bash /tmp/bootstrap_cifar100.sh`
