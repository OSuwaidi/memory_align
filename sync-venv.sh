#!/bin/bash
# Build/update the shared venv WITHOUT ever corrupting a venv in use by jobs.
#
# Design: venvs are immutable and versioned by uv.lock hash.
#   .venv-<hash>/   one built venv per lockfile version (read-only once built)
#   .venv        -> symlink to the current one (what jobs resolve at startup)
#
# `uv sync` here builds a NEW .venv-<hash> (temp dir + atomic rename), then
# flips the .venv symlink. Running jobs keep their already-resolved venv.
# Built venvs are chmod'ed read-only so a stray `uv sync`/`uv pip install`
# on the login node fails loudly (EACCES) instead of silently corrupting —
# this is what previously injected a mismatched CUDA stack and broke cuDNN
# on every node (CUDNN_STATUS_NOT_INITIALIZED).
#
# Usage:  ./sync-venv.sh             build current lock if needed, flip symlink
#         ./sync-venv.sh --rebuild   force rebuild of the current-lock venv
#         ./sync-venv.sh --prune     also delete old, unreferenced venvs

set -euo pipefail

PROJECT=/shared/b00090279/memory_align
cd "$PROJECT"
[ -f /shared/b00090279/shared-paths.sh ] && . /shared/b00090279/shared-paths.sh

REBUILD=0; PRUNE=0
for a in "$@"; do
    case "$a" in
        --rebuild) REBUILD=1 ;;
        --prune)   PRUNE=1 ;;
        *) echo "usage: $0 [--rebuild] [--prune]" >&2; exit 2 ;;
    esac
done

# uv commands run with UV_NO_SYNC cleared: shared-paths.sh sets it globally
# so plain `uv run` never auto-syncs, but THIS script must be able to sync.
uvx() { env -u UV_NO_SYNC uv "$@"; }

# 1. Refresh the lock from pyproject (resolve only, no install).
uvx lock

# 2. Sanity-check the lock BEFORE building: this project must resolve torch
#    against the cu128 index (driver 570.86 == CUDA 12.8). A lock containing
#    CUDA-13 wheels means you're on a branch without the AWS fix — building
#    from it would produce a venv whose cuDNN cannot initialize on any node.
if grep -qE '^name = "nvidia-[a-z-]*-cu13"|^name = "nvidia-(cublas|cudnn|cufft|curand|cusolver|cusparse|nvjitlink|nvtx)"$' uv.lock; then
    echo "[sync-venv] REFUSING: uv.lock contains CUDA-13 wheels (PyPI torch)." >&2
    echo "            This branch is missing the cu128 pin in pyproject.toml." >&2
    echo "            Merge the AWS branch fix first." >&2
    exit 1
fi
grep -q 'version = "2\..*+cu128"' uv.lock || {
    echo "[sync-venv] REFUSING: no +cu128 torch in uv.lock — wrong lock for this cluster." >&2
    exit 1
}

HASH=$(sha256sum uv.lock | cut -c1-12)
TARGET=".venv-$HASH"
CURRENT=$(readlink .venv 2>/dev/null || true)

verify() {  # $1 = venv dir; real cuDNN conv, not just import — a poisoned
            # venv can import torch and report cuda.is_available()=True while
            # every convolution fails.
    "$1/bin/python" - <<'PY'
import torch, torch.nn as nn
assert torch.cuda.is_available(), "CUDA not available"
nn.Conv2d(3, 16, 3, padding=1).cuda()(torch.randn(2, 3, 32, 32, device="cuda"))
torch.cuda.synchronize()
print(f"[sync-venv] OK: torch {torch.__version__}, cudnn {torch.backends.cudnn.version()}, conv works")
PY
}

# 3. Reuse the target only if it VERIFIES; otherwise (re)build it.
if [ "$REBUILD" = "0" ] && [ -x "$TARGET/bin/python" ] && verify "$TARGET" 2>/dev/null; then
    echo "[sync-venv] $TARGET already built and healthy"
else
    if [ -d "$TARGET" ]; then
        echo "[sync-venv] $TARGET exists but is broken or --rebuild given — rebuilding"
        chmod -R u+w "$TARGET"; rm -rf "$TARGET"
    fi
    TMP=".venv-$HASH.tmp.$$"
    chmod -R u+w "$TMP" 2>/dev/null || true; rm -rf "$TMP"
    echo "[sync-venv] building $TARGET ..."
    UV_PROJECT_ENVIRONMENT="$TMP" uvx sync --all-groups --compile-bytecode
    # Rewrite self-references from the temp name to the final path so
    # pyvenv.cfg and console-script shebangs stay correct after rename.
    sed -i "s|$PROJECT/$TMP|$PROJECT/$TARGET|g" "$TMP/pyvenv.cfg" "$TMP"/bin/* 2>/dev/null || true
    mv -T "$TMP" "$TARGET"
    verify "$TARGET"
    # Freeze it: nothing may mutate a live venv, ever.
    chmod -R a-w "$TARGET"
    echo "[sync-venv] built and froze $TARGET (read-only)"
fi

# 4. Atomically flip .venv -> target.
if [ "$CURRENT" = "$TARGET" ]; then
    echo "[sync-venv] .venv already -> $TARGET"
else
    ln -sfn "$TARGET" .venv.new && mv -T .venv.new .venv
    echo "[sync-venv] .venv -> $TARGET (was: ${CURRENT:-none})"
fi

# 5. Optional prune of old venvs. Only versions no longer pointed to by
#    .venv and with no local reader. We can't see other nodes' processes,
#    so prune only when 'squeue --me' is empty.
if [ "$PRUNE" = "1" ]; then
    if command -v squeue >/dev/null 2>&1 && [ "$(squeue -h -u "$USER" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[sync-venv] skipping prune: SLURM jobs are active (they may still use old venvs)"
    else
        keep=$(readlink .venv)
        for d in .venv-*/; do
            d=${d%/}
            [ "$d" = "$keep" ] && continue
            echo "[sync-venv] pruning $d"
            chmod -R u+w "$d"; rm -rf "$d"
        done
    fi
fi

echo "[sync-venv] done."
