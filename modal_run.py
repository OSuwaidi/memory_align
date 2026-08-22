"""Run the active CIFAR-10 W&B sweep on a Modal L40S.

One-time setup::

    modal setup
    modal secret create wandb-secret WANDB_API_KEY=<your-wandb-api-key>

Launch the agent (``--detach`` lets it survive terminal disconnects)::

    modal run --detach modal_run.py

By default the W&B agent keeps requesting runs until the sweep finishes or the
credit-aware runtime limit is reached. To intentionally run fewer trials::

    modal run --detach modal_run.py --count 5

The runtime limit is a guardrail based on Modal's public August 2026 prices; it
is not an account-level spending cap. It assumes the Starter tier's full $30
monthly credit is available and leaves $1 for image builds and other overhead.
The absolute deadline is passed as a Function input so it survives Modal
preemptions and retries instead of granting a fresh budget to every attempt.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time

import modal

APP_NAME = "cifar10-wandb-sweep-d3vh2qqw"
SWEEP_ID = "osuwaidi-khalifa-university/FINAL_MAL_CIFAR10/d3vh2qqw"

# L40S is the best practical speed/credit trade-off for this small CNN workload:
# substantially faster than L4/A10, without paying H100/B200 rates for compute
# that ResNet18 on 32x32 inputs is unlikely to saturate.
GPU = "L40S"
CPU_CORES = 8.0  # Modal physical cores (1 physical CPU = 2 vCPUs); feeds torchvision transforms.
MEMORY_MIB = 16 * 1024

# Current public Modal rates (USD/s), as checked 2026-08-20.
L40S_RATE = 0.000542
CPU_CORE_RATE = 0.0000131
MEMORY_GIB_RATE = 0.00000222
STARTER_CREDIT = 30.0
BUILD_AND_OVERHEAD_RESERVE = 1.0
TOTAL_RATE = L40S_RATE + CPU_CORES * CPU_CORE_RATE + (MEMORY_MIB / 1024) * MEMORY_GIB_RATE
DEFAULT_AGENT_SECONDS = math.floor((STARTER_CREDIT - BUILD_AND_OVERHEAD_RESERVE) / TOTAL_RATE)
CREDIT_SHUTDOWN_GRACE_SECONDS = 4 * 60
# Modal gives a preempted container about 30 seconds to exit before killing it.
PREEMPTION_SHUTDOWN_GRACE_SECONDS = 20
FORCED_SHUTDOWN_GRACE_SECONDS = 5
MAX_FAILURE_RETRIES = 10

REMOTE_PROJECT_DIR = "/workspace"
CIFAR10_URL = "https://huggingface.co/datasets/liangnanying/cifar-10-python/resolve/main/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_ARCHIVE = "/tmp/cifar-10-python.tar.gz"
CIFAR10_DATA_DIR = f"{REMOTE_PROJECT_DIR}/data"

image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("ca-certificates", "curl")
    # Install the exact locked third-party environment, including CUDA PyTorch.
    .uv_sync(".", extra_options="--no-dev")
    # Download during the cached, CPU-only image build instead of GPU runtime.
    # The extracted path is exactly what torchvision.datasets.CIFAR10 expects.
    .run_commands(
        f"mkdir -p {CIFAR10_DATA_DIR}",
        (
            "curl --fail --location --show-error --retry 5 --retry-all-errors "
            f"--retry-delay 2 --connect-timeout 30 --output {CIFAR10_ARCHIVE} {CIFAR10_URL}"
        ),
        f"echo '{CIFAR10_MD5}  {CIFAR10_ARCHIVE}' | md5sum --check --status",
        f"tar -xzf {CIFAR10_ARCHIVE} -C {CIFAR10_DATA_DIR}",
        f"test -s {CIFAR10_DATA_DIR}/cifar-10-batches-py/data_batch_1",
        f"test -s {CIFAR10_DATA_DIR}/cifar-10-batches-py/test_batch",
        f"rm -f {CIFAR10_ARCHIVE}",
    )
    # All image build steps must precede add_local_* runtime mounts.
    .workdir(REMOTE_PROJECT_DIR)
    .add_local_file("main.py", f"{REMOTE_PROJECT_DIR}/main.py")
    .add_local_file("mal_opt.py", f"{REMOTE_PROJECT_DIR}/mal_opt.py")
    .add_local_file("cautious_opt.py", f"{REMOTE_PROJECT_DIR}/cautious_opt.py")
    .add_local_dir("sweeps", f"{REMOTE_PROJECT_DIR}/sweeps")
)

app = modal.App(APP_NAME, image=image)
wandb_secret = modal.Secret.from_name("wandb-secret")


def _terminate_agent(
    agent: subprocess.Popen[bytes],
    *,
    reason: str,
    graceful_timeout: int,
) -> None:
    """Stop W&B cleanly so its child training run is marked preempted."""
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
        print("W&B did not stop during the grace period; terminating it.", flush=True)
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
    retries=modal.Retries(initial_delay=0.0, max_retries=MAX_FAILURE_RETRIES),
    single_use_containers=True,
    secrets=[wandb_secret],
)
def run_sweep_agent(count: int, deadline_unix_seconds: float) -> int:
    """Run a retryable W&B agent until sweep completion or the fixed deadline."""
    if count < 0:
        raise ValueError("count must be zero (unlimited) or a positive integer")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Modal secret 'wandb-secret' must contain WANDB_API_KEY")

    remaining_seconds = deadline_unix_seconds - time.time()
    if remaining_seconds <= 0:
        print("Overall credit-aware deadline already reached; not restarting W&B.", flush=True)
        return 0
    if remaining_seconds > DEFAULT_AGENT_SECONDS + 60:
        raise ValueError("deadline exceeds the maximum credit-aware runtime")

    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    budget_hours = remaining_seconds / 3600
    estimated_cost = remaining_seconds * TOTAL_RATE
    deadline_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(deadline_unix_seconds))
    print(
        f"Starting/restarting W&B sweep agent on {gpu_name}. "
        f"Overall deadline: {deadline_utc}; remaining guard: {budget_hours:.2f} h (~${estimated_cost:.2f}).",
        flush=True,
    )

    command = [sys.executable, "-m", "wandb", "agent", "--forward-signals"]
    if count:
        command.extend(["--count", str(count)])
    command.append(SWEEP_ID)

    agent = subprocess.Popen(command, cwd=REMOTE_PROJECT_DIR)
    try:
        while agent.poll() is None:
            remaining = deadline_unix_seconds - time.time()
            if remaining <= 0:
                _terminate_agent(
                    agent,
                    reason="Overall credit-aware runtime reached",
                    graceful_timeout=CREDIT_SHUTDOWN_GRACE_SECONDS,
                )
                break
            time.sleep(min(30, remaining))
    except KeyboardInterrupt:
        # Modal uses an interrupt for preemption. Re-raise it after W&B marks
        # the active run preempted; Modal will restart this same Function input.
        _terminate_agent(
            agent,
            reason="Modal interrupted/preempted this Function attempt",
            graceful_timeout=PREEMPTION_SHUTDOWN_GRACE_SECONDS,
        )
        print("The Function input will be retried with the original overall deadline.", flush=True)
        raise
    except BaseException:
        _terminate_agent(
            agent,
            reason="The sweep-agent wrapper failed unexpectedly",
            graceful_timeout=PREEMPTION_SHUTDOWN_GRACE_SECONDS,
        )
        raise

    return_code = agent.wait()
    if return_code not in (0, 130, -signal.SIGINT):
        raise subprocess.CalledProcessError(return_code, command)
    return return_code


@app.local_entrypoint()
def main(count: int = 0, max_hours: float = DEFAULT_AGENT_SECONDS / 3600) -> None:
    """Launch the remote sweep agent via `modal run --detach modal_run.py`."""
    if count < 0:
        raise ValueError("--count must be zero (unlimited) or a positive integer")
    if not 0 < max_hours <= DEFAULT_AGENT_SECONDS / 3600:
        raise ValueError(f"--max-hours must be in (0, {DEFAULT_AGENT_SECONDS / 3600:.2f}]")

    max_runtime_seconds = max(1, math.floor(max_hours * 3600))
    # Modal retries receive the same serialized arguments, so this absolute
    # deadline prevents a preemption from resetting the credit guard.
    deadline_unix_seconds = time.time() + max_runtime_seconds
    return_code = run_sweep_agent.remote(count=count, deadline_unix_seconds=deadline_unix_seconds)
    print(f"W&B sweep agent exited with status {return_code}.")
