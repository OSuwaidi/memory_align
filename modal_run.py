"""Run the paired MAL-AdamW in-place MAE pilot on Modal.

Create the associated W&B sweep with::

    uv run sweeps/mal_adamw_mae_in_place_sweep.py tasks/mae_pretrain.py \
        --sweep-name mal-adamw-mae-in-place-modal-pilot

Then launch two detached, single-run agents with the printed sweep path::

    uv run modal run --detach modal_run.py \
        --sweep-path osuwaidi-khalifa-university/MAL_benchmark/SWEEP_ID

The image checks out the current committed revision rather than mounting the
working tree. This makes Modal use the same task and optimizer implementation
as the AUS cluster even when the local repository has unrelated uncommitted
changes.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time

import modal

APP_NAME = "mal-adamw-mae-in-place-pilot"
ALLOWED_SWEEP_PREFIX = "osuwaidi-khalifa-university/MAL_benchmark/"
SOURCE_REPOSITORY = "https://github.com/OSuwaidi/memory_align.git"
SOURCE_REVISION = subprocess.run(
    ("git", "rev-parse", "HEAD"),
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()

REMOTE_PROJECT_DIR = "/workspace"
TINY_IMAGENET_DIR = f"{REMOTE_PROJECT_DIR}/data/tiny-imagenet-200"

# Modal and the AUS nodes both expose an NVIDIA A10-class GPU. Holding the GPU
# generation fixed is preferable for this paired optimizer comparison.
GPU = "A10"
CPU_CORES = 8.0
MEMORY_MIB = 16 * 1024
AGENT_COUNT = 2
RUNS_PER_AGENT = 1

# Public Modal rates checked on 2026-09-10. The two functions share the old
# script's $30 Starter-credit assumption and leave $1 for image-build overhead.
A10_RATE = 0.000306
CPU_CORE_RATE = 0.0000131
MEMORY_GIB_RATE = 0.00000222
STARTER_CREDIT_USD = 30.0
BUILD_RESERVE_USD = 1.0
TOTAL_RATE_PER_AGENT = A10_RATE + CPU_CORES * CPU_CORE_RATE + (MEMORY_MIB / 1024) * MEMORY_GIB_RATE
DEFAULT_AGENT_SECONDS = math.floor((STARTER_CREDIT_USD - BUILD_RESERVE_USD) / (AGENT_COUNT * TOTAL_RATE_PER_AGENT))
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
    .workdir(REMOTE_PROJECT_DIR)
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
    gpu=GPU,
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    timeout=DEFAULT_AGENT_SECONDS + CREDIT_SHUTDOWN_GRACE_SECONDS + 60,
    retries=modal.Retries(initial_delay=5.0, max_retries=1),
    single_use_containers=True,
    max_containers=AGENT_COUNT,
    secrets=[wandb_secret],
)
def run_sweep_agent(agent_index: int, sweep_path: str, deadline_unix_seconds: float) -> int:
    """Claim and execute exactly one W&B run before the shared deadline."""
    if not 1 <= agent_index <= AGENT_COUNT:
        raise ValueError(f"agent_index must be in [1, {AGENT_COUNT}]")
    if not sweep_path.startswith(ALLOWED_SWEEP_PREFIX):
        raise ValueError(f"sweep_path must start with {ALLOWED_SWEEP_PREFIX}")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Modal secret 'wandb-secret' must contain WANDB_API_KEY")

    remaining_seconds = deadline_unix_seconds - time.time()
    if remaining_seconds <= 0:
        print(f"Agent {agent_index}: shared credit deadline already reached.", flush=True)
        return 0
    if remaining_seconds > DEFAULT_AGENT_SECONDS + 60:
        raise ValueError("deadline exceeds the shared Starter-credit guard")

    gpu_name = subprocess.run(
        ("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    torch_check = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "import torch, torchvision; "
                "assert torch.cuda.is_available(); "
                "print(f'torch={torch.__version__}, torchvision={torchvision.__version__}')"
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
        f"Agent {agent_index}/{AGENT_COUNT}: {gpu_name}; {torch_check}; source={SOURCE_REVISION}; "
        f"one W&B run; shared deadline={deadline_utc}.",
        flush=True,
    )

    command = [
        sys.executable,
        "-m",
        "wandb",
        "agent",
        "--forward-signals",
        "--count",
        str(RUNS_PER_AGENT),
        sweep_path,
    ]
    agent = subprocess.Popen(command, cwd=REMOTE_PROJECT_DIR)
    try:
        while agent.poll() is None:
            remaining_seconds = deadline_unix_seconds - time.time()
            if remaining_seconds <= 0:
                _terminate_agent(
                    agent,
                    reason=f"Agent {agent_index}: shared credit-aware runtime reached",
                    graceful_timeout=CREDIT_SHUTDOWN_GRACE_SECONDS,
                )
                break
            time.sleep(min(30, remaining_seconds))
    except KeyboardInterrupt:
        _terminate_agent(
            agent,
            reason=f"Agent {agent_index}: Modal interrupted or preempted the function",
            graceful_timeout=PREEMPTION_SHUTDOWN_GRACE_SECONDS,
        )
        raise
    except BaseException:
        _terminate_agent(
            agent,
            reason=f"Agent {agent_index}: wrapper failed unexpectedly",
            graceful_timeout=PREEMPTION_SHUTDOWN_GRACE_SECONDS,
        )
        raise

    return_code = agent.wait()
    if return_code not in (0, 130, -signal.SIGINT):
        raise subprocess.CalledProcessError(return_code, command)
    return return_code


@app.local_entrypoint()
def main(sweep_path: str, max_hours: float = DEFAULT_AGENT_SECONDS / 3600) -> None:
    """Launch two detached one-run agents for the paired W&B pilot."""
    if not sweep_path.startswith(ALLOWED_SWEEP_PREFIX):
        raise ValueError(f"--sweep-path must start with {ALLOWED_SWEEP_PREFIX}")
    maximum_hours = DEFAULT_AGENT_SECONDS / 3600
    if not 0 < max_hours <= maximum_hours:
        raise ValueError(f"--max-hours must be in (0, {maximum_hours:.2f}]")

    deadline_unix_seconds = time.time() + math.floor(max_hours * 3600)
    estimated_maximum_cost = max_hours * AGENT_COUNT * TOTAL_RATE_PER_AGENT * 3600
    print(
        f"Launching {AGENT_COUNT} Modal {GPU} agents, one W&B run each, from source {SOURCE_REVISION}. "
        f"The shared runtime guard is {max_hours:.2f} h/agent (~${estimated_maximum_cost:.2f} total maximum)."
    )
    for agent_index in range(1, AGENT_COUNT + 1):
        call = run_sweep_agent.spawn(agent_index, sweep_path, deadline_unix_seconds)
        print(f"AGENT_{agent_index}_CALL_ID={call.object_id}")
