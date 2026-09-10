"""Run a small, reproducible MAL-AdamW MAE sweep on Modal.

Create the associated W&B sweep with::

    uv run sweeps/mal_adamw_mae_in_place_matched_sweep.py \
        tasks/mae_pretrain.py --phase controls \
        --sweep-name mal-adamw-mae-in-place-modal-controls

Then launch one detached two-GPU function with the printed sweep path::

    MAL_MODAL_SOURCE_REVISION=COMMIT MAL_MODAL_APP_NAME=APP_NAME \
        uv run modal run --detach modal_run.py \
        --sweep-path osuwaidi-khalifa-university/MAL_benchmark/SWEEP_ID \
        --runs-per-agent 1 --max-hours 2.25

By default, the image checks out the current committed revision rather than
mounting the working tree. Set ``MAL_MODAL_SOURCE_REVISION`` to reproduce a
previous experiment exactly and ``MAL_MODAL_APP_NAME`` when launching multiple
independent detached jobs concurrently.
"""

from __future__ import annotations

import math
import os
import re
import signal
import subprocess
import sys
import time

import modal

APP_NAME = os.environ.get("MAL_MODAL_APP_NAME", "mal-adamw-mae-modal")
ALLOWED_SWEEP_PREFIX = "osuwaidi-khalifa-university/MAL_benchmark/"
SOURCE_REPOSITORY = "https://github.com/OSuwaidi/memory_align.git"
SOURCE_REVISION = os.environ.get("MAL_MODAL_SOURCE_REVISION")
if SOURCE_REVISION is None:
    SOURCE_REVISION = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
if re.fullmatch(r"[0-9a-f]{40}", SOURCE_REVISION) is None:
    raise ValueError("MAL_MODAL_SOURCE_REVISION must be a full 40-character lowercase Git commit hash.")

REMOTE_PROJECT_DIR = "/workspace"
TINY_IMAGENET_DIR = f"{REMOTE_PROJECT_DIR}/data/tiny-imagenet-200"

# Modal and the AUS nodes both expose an NVIDIA A10-class GPU. Holding the GPU
# generation fixed is preferable for this paired optimizer comparison. Both
# agents live in one two-GPU function because detached Modal entrypoints only
# guarantee the lifetime of their last triggered function call.
GPU_TYPE = "A10"
GPU_COUNT = 2
GPU_REQUEST = f"{GPU_TYPE}:{GPU_COUNT}"
CPU_CORES = 8.0
MEMORY_MIB = 32 * 1024
AGENT_COUNT = 2

# Public Modal rates checked on 2026-09-10. This defines the absolute
# per-function ceiling; each launch should use a much smaller explicit
# ``--max-hours`` value based on the live billing balance.
A10_RATE = 0.000306
CPU_CORE_RATE = 0.0000131
MEMORY_GIB_RATE = 0.00000222
STARTER_CREDIT_USD = 30.0
BUILD_RESERVE_USD = 1.0
TOTAL_FUNCTION_RATE = GPU_COUNT * A10_RATE + CPU_CORES * CPU_CORE_RATE + (MEMORY_MIB / 1024) * MEMORY_GIB_RATE
DEFAULT_FUNCTION_SECONDS = math.floor((STARTER_CREDIT_USD - BUILD_RESERVE_USD) / TOTAL_FUNCTION_RATE)
CREDIT_SHUTDOWN_GRACE_SECONDS = 4 * 60
PREEMPTION_SHUTDOWN_GRACE_SECONDS = 20
FORCED_SHUTDOWN_GRACE_SECONDS = 5

PYPI_PACKAGES = (
    "numpy==2.5.2",
    "pillow==12.3.0",
    "scikit-learn==1.9.0",
    "timm==1.0.29",
    "tqdm==4.70.0",
    "wandb==0.29.0",
)
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu128"

image = (
    modal.Image.debian_slim(python_version="3.14")
    # The mounted runner is imported again inside the container, where local
    # shell variables are not inherited. Persist the resolved revision in the
    # image environment so both imports agree on the pinned training source.
    .env({"MAL_MODAL_SOURCE_REVISION": SOURCE_REVISION})
    .apt_install("ca-certificates", "git")
    .uv_pip_install("torch==2.11.0", "torchvision==0.26.0", index_url=PYTORCH_INDEX)
    .uv_pip_install(*PYPI_PACKAGES)
    .run_commands(
        f"git clone --filter=blob:none --no-checkout {SOURCE_REPOSITORY} {REMOTE_PROJECT_DIR}",
        f"git -C {REMOTE_PROJECT_DIR} fetch --depth 1 origin {SOURCE_REVISION}",
        f"git -C {REMOTE_PROJECT_DIR} checkout --detach {SOURCE_REVISION}",
        f'test "$(git -C {REMOTE_PROJECT_DIR} rev-parse HEAD)" = "{SOURCE_REVISION}"',
        f"python {REMOTE_PROJECT_DIR}/download_datasets.py --task tiny-imagenet --tiny-imagenet-dir {REMOTE_PROJECT_DIR}/data",
        f"test -s {TINY_IMAGENET_DIR}/val/val_annotations.txt",
    )
    # Keep the Modal runner's import root separate from the pinned repository.
    # Training subprocesses still use REMOTE_PROJECT_DIR explicitly. Without
    # this separation, an older checked-out modal_run.py can shadow this file.
    .workdir("/root")
)

app = modal.App(APP_NAME, image=image)
wandb_secret = modal.Secret.from_name("wandb-secret")


def _terminate_agent(agent: subprocess.Popen[bytes], *, reason: str, graceful_timeout: int) -> None:
    """Stop W&B cleanly so the active run is not left indefinitely running."""
    if agent.poll() is not None:
        return

    print(f"{reason}; stopping the W&B agent cleanly.", flush=True)
    try:
        agent.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        agent.wait(timeout=graceful_timeout)
    except subprocess.TimeoutExpired:
        agent.terminate()
        try:
            agent.wait(timeout=FORCED_SHUTDOWN_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            agent.kill()
            agent.wait()


@app.function(
    gpu=GPU_REQUEST,
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    timeout=DEFAULT_FUNCTION_SECONDS + CREDIT_SHUTDOWN_GRACE_SECONDS + 60,
    retries=modal.Retries(initial_delay=5.0, max_retries=1),
    single_use_containers=True,
    max_containers=1,
    secrets=[wandb_secret],
)
def run_sweep_agents(sweep_path: str, deadline_unix_seconds: float, runs_per_agent: int) -> list[int]:
    """Run one isolated bounded W&B agent on each allocated GPU."""
    if not sweep_path.startswith(ALLOWED_SWEEP_PREFIX):
        raise ValueError(f"sweep_path must start with {ALLOWED_SWEEP_PREFIX}")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Modal secret 'wandb-secret' must contain WANDB_API_KEY")
    if not 1 <= runs_per_agent <= 8:
        raise ValueError("runs_per_agent must be in [1, 8]")

    remaining_seconds = deadline_unix_seconds - time.time()
    if remaining_seconds <= 0:
        print("Shared credit deadline already reached.", flush=True)
        return []
    if remaining_seconds > DEFAULT_FUNCTION_SECONDS + 60:
        raise ValueError("deadline exceeds the shared Starter-credit guard")

    gpu_names = subprocess.run(
        ("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    if len(gpu_names) != GPU_COUNT:
        raise RuntimeError(f"Expected {GPU_COUNT} GPUs, found {len(gpu_names)}: {gpu_names}")
    torch_check = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "import torch, torchvision; "
                "assert torch.cuda.is_available(); "
                f"assert torch.cuda.device_count() == {GPU_COUNT}; "
                "print(f'torch={torch.__version__}, torchvision={torchvision.__version__}, devices={torch.cuda.device_count()}')"
            ),
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    checked_out_revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REMOTE_PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if checked_out_revision != SOURCE_REVISION:
        raise RuntimeError(f"source revision mismatch: {checked_out_revision} != {SOURCE_REVISION}")

    deadline_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(deadline_unix_seconds))
    print(
        f"GPUs={gpu_names}; {torch_check}; source={SOURCE_REVISION}; "
        f"{AGENT_COUNT} isolated W&B agents with {runs_per_agent} run(s) each; "
        f"shared deadline={deadline_utc}.",
        flush=True,
    )

    command = [
        sys.executable,
        "-m",
        "wandb",
        "agent",
        "--forward-signals",
        "--count",
        str(runs_per_agent),
        sweep_path,
    ]
    agents: list[subprocess.Popen[bytes]] = []
    for agent_index in range(AGENT_COUNT):
        agent_environment = dict(os.environ)
        agent_environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(agent_index),
                "OMP_NUM_THREADS": "4",
                "WANDB_DIR": f"/tmp/wandb-agent-{agent_index + 1}",
            }
        )
        os.makedirs(agent_environment["WANDB_DIR"], exist_ok=True)
        agents.append(subprocess.Popen(command, cwd=REMOTE_PROJECT_DIR, env=agent_environment))

    try:
        while any(agent.poll() is None for agent in agents):
            remaining_seconds = deadline_unix_seconds - time.time()
            if remaining_seconds <= 0:
                for agent_index, agent in enumerate(agents, start=1):
                    _terminate_agent(
                        agent,
                        reason=f"Agent {agent_index}: shared credit-aware runtime reached",
                        graceful_timeout=CREDIT_SHUTDOWN_GRACE_SECONDS,
                    )
                break
            time.sleep(min(30, remaining_seconds))
    except KeyboardInterrupt:
        for agent_index, agent in enumerate(agents, start=1):
            _terminate_agent(
                agent,
                reason=f"Agent {agent_index}: Modal interrupted or preempted the function",
                graceful_timeout=PREEMPTION_SHUTDOWN_GRACE_SECONDS,
            )
        raise
    except BaseException:
        for agent_index, agent in enumerate(agents, start=1):
            _terminate_agent(
                agent,
                reason=f"Agent {agent_index}: wrapper failed unexpectedly",
                graceful_timeout=PREEMPTION_SHUTDOWN_GRACE_SECONDS,
            )
        raise

    return_codes = [agent.wait() for agent in agents]
    for return_code in return_codes:
        if return_code not in (0, 130, -signal.SIGINT):
            raise subprocess.CalledProcessError(return_code, command)
    return return_codes


@app.local_entrypoint()
def main(
    sweep_path: str,
    runs_per_agent: int = 1,
    max_hours: float = DEFAULT_FUNCTION_SECONDS / 3600,
) -> None:
    """Launch one detached two-GPU function for an exact-size W&B sweep."""
    if not sweep_path.startswith(ALLOWED_SWEEP_PREFIX):
        raise ValueError(f"--sweep-path must start with {ALLOWED_SWEEP_PREFIX}")
    maximum_hours = DEFAULT_FUNCTION_SECONDS / 3600
    if not 0 < max_hours <= maximum_hours:
        raise ValueError(f"--max-hours must be in (0, {maximum_hours:.2f}]")
    if not 1 <= runs_per_agent <= 8:
        raise ValueError("--runs-per-agent must be in [1, 8]")

    deadline_unix_seconds = time.time() + math.floor(max_hours * 3600)
    estimated_maximum_cost = max_hours * TOTAL_FUNCTION_RATE * 3600
    print(
        f"Launching one Modal {GPU_REQUEST} function with {AGENT_COUNT} W&B agents and "
        f"{runs_per_agent} run(s) per agent from source {SOURCE_REVISION}. "
        f"The shared runtime guard is {max_hours:.2f} h (~${estimated_maximum_cost:.2f} total maximum)."
    )
    call = run_sweep_agents.spawn(sweep_path, deadline_unix_seconds, runs_per_agent)
    print(f"FUNCTION_CALL_ID={call.object_id}")
