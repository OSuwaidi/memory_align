#!/bin/bash
# Update the shared venv WITHOUT ever corrupting a venv that jobs are using.
#
# Background: the venv lives on shared NFS and every SLURM node runs off it.
# If `uv sync` mutates the live .venv while a job reads it, uv unlinks in-use
# .so files, NFS silly-renames them to .nfsXXXX, the write never completes,
# and torch dies everywhere with "libcudnn.so.9: cannot open shared object
# file".
#
# Fix: venvs are immutable and versioned by uv.lock hash.
#   .venv-<hash>/   one built venv per lockfile version
#   .venv        -> symlink to the current one (this is what jobs use)
#
# `uv sync` here builds a NEW .venv-<hash> (into a temp dir, then atomic
# rename) and only flips the .venv symlink at the end. Running jobs keep
# reading their old .venv-<oldhash>, which is never touched -> no corruption,
# no need to stop jobs, no reader detection. Old venvs are pruned only when
# no job/symlink references them.
#
# Usage:  ./sync-venv.sh            # sync current uv.lock, flip symlink
#         ./sync-venv.sh --prune    # also delete unreferenced old venvs

set -euo pipefail

PROJECT=/shared/b00090279/memory_align
cd "$PROJECT"
[ -f /shared/b00090279/shared-paths.sh ] && . /shared/b00090279/shared-paths.sh

PRUNE=0
[ "${1:-}" = "--prune" ] && PRUNE=1

# 1. If uv.lock is stale, refresh it first (resolve only, no install).
uv lock

HASH=$(sha256sum uv.lock | cut -c1-12)
TARGET=".venv-$HASH"
CURRENT=$(readlink .venv 2>/dev/null || true)

# 2. Build the target venv if it doesn't already exist. Build into a temp
#    dir on the SAME filesystem, then atomically rename into place, so a
#    half-built venv can never be seen as ready.
if [ -d "$TARGET" ] && [ -x "$TARGET/bin/python" ]; then
    echo "[sync-venv] $TARGET already built"
else
    TMP=".venv-$HASH.tmp.$$"
    rm -rf "$TMP"
    echo "[sync-venv] building $TARGET ..."
    # UV_PROJECT_ENVIRONMENT points uv at the temp env instead of ./.venv.
    UV_PROJECT_ENVIRONMENT="$TMP" uv sync --all-groups
    # Make the env self-referential to its FINAL path (not the temp name)
    # so console scripts + pyvenv.cfg resolve correctly after rename.
    sed -i "s|$PROJECT/$TMP|$PROJECT/$TARGET|g" "$TMP/pyvenv.cfg" "$TMP"/bin/* 2>/dev/null || true
    mv -T "$TMP" "$TARGET"
    echo "[sync-venv] built $TARGET"
fi

# 3. Verify the target imports torch + sees CUDA before anyone depends on it.
"$TARGET/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('[sync-venv] OK:', torch.__version__)"

# 4. Atomically flip .venv -> target (ln -sfn + mv is atomic on same dir).
if [ "$CURRENT" = "$TARGET" ]; then
    echo "[sync-venv] .venv already -> $TARGET"
else
    ln -sfn "$TARGET" .venv.new && mv -T .venv.new .venv
    echo "[sync-venv] .venv -> $TARGET (was: ${CURRENT:-none})"
fi

# 5. Optionally prune old venvs that nothing references. "Referenced" = the
#    live .venv symlink, or any local process whose interpreter lives inside
#    it. We cannot see other nodes' processes, so only prune versions older
#    than the current one and skip any with a live local reader.
if [ "$PRUNE" = "1" ]; then
    keep=$(readlink .venv)
    for d in .venv-*/; do
        d=${d%/}
        [ "$d" = "$keep" ] && continue
        busy=""
        for e in /proc/[0-9]*/exe; do
            t=$(readlink "$e" 2>/dev/null) || continue
            case "$t" in "$PROJECT/$d"/*) busy=1; break;; esac
        done
        if [ -n "$busy" ]; then
            echo "[sync-venv] keeping $d (in use locally)"
        else
            echo "[sync-venv] pruning $d"
            rm -rf "$d"
        fi
    done
    echo "[sync-venv] NOTE: only local readers are visible; a venv still in"
    echo "            use by a job on another node was NOT pruned if it was"
    echo "            the current .venv when that job started. Prune again"
    echo "            after 'squeue --me' is empty to be fully safe."
fi

echo "[sync-venv] done."
