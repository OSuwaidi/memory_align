"""Create the paired MAL-AdamW in-place MAE pilot for Modal.

The pilot changes only the AdamW alignment geometry. Its MAE recipe matches
the focused out-of-place structure screen, while one paired seed keeps the
side experiment within the Modal Starter credit guard in ``modal_run.py``.
Promising configurations should subsequently be confirmed with all seeds.
"""

from __future__ import annotations

import argparse
import subprocess
from typing import Any

import wandb

ENTITY_NAME = "osuwaidi-khalifa-university"
PROJECT_NAME = "MAL_benchmark"

MODEL = "vit_tiny_patch16_224"
IMAGE_SIZE = 64
PATCH_SIZE = 8
EPOCHS = 300
WARMUP_EPOCHS = 15
PROBE_EVERY = 50

PILOT_SEED = 42
REPRESENTATIVE_BATCH_SIZE = 1024
REPRESENTATIVE_BASE_LR = 1e-3
REPRESENTATIVE_WEIGHT_DECAY = 5e-2
ALIGNMENTS = ("update", "metric")
MAL_CONFIGS = tuple(f"True,1.0,none,attenuate,{align},complement" for align in ALIGNMENTS)


def current_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_sweep_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "program": args.program,
        "name": args.sweep_name,
        "method": "grid",
        "metric": {"name": "final_probe_val_acc", "goal": "maximize"},
        "parameters": {
            "optimizer": {"values": ("MAL_AdamW",)},
            "MAL_config": {"values": MAL_CONFIGS},
            "batch_size": {"values": (REPRESENTATIVE_BATCH_SIZE,)},
            "base_lr": {"values": (REPRESENTATIVE_BASE_LR,)},
            "weight_decay": {"values": (REPRESENTATIVE_WEIGHT_DECAY,)},
            "seed": {"values": (PILOT_SEED,)},
            "use_scheduler": {"values": (True,)},
            "compute_platform": {"values": ("modal",)},
            "experiment_stage": {"values": ("mal_adamw_in_place_paired_pilot",)},
            "source_revision": {"values": (args.source_revision,)},
        },
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "--data_dir",
            args.data_dir,
            "--arch",
            MODEL,
            "--image_size",
            str(IMAGE_SIZE),
            "--patch_size",
            str(PATCH_SIZE),
            "--epochs",
            str(args.epochs),
            "--warmup_epochs",
            str(args.warmup_epochs),
            "--probe_every",
            str(args.probe_every),
            "--num_workers",
            "4",
            "--amp_dtype",
            "bfloat16",
            "--float32_precision",
            "tf32",
            "--output_dir",
            args.output_dir,
            "--save_every",
            "0",
            "--beta2",
            "0.95",
            "${args}",
        ],
    }


def expected_run_count(configuration: dict[str, Any]) -> int:
    count = 1
    for parameter in configuration["parameters"].values():
        count *= len(parameter["values"])
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", help="MAE entry point (normally tasks/mae_pretrain.py)")
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", default=PROJECT_NAME)
    parser.add_argument("--data_dir", "--data-dir", default="/workspace/data/tiny-imagenet-200")
    parser.add_argument("--output_dir", "--output-dir", default="/tmp/memory-align/mae-in-place")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--warmup_epochs", "--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--probe_every", "--probe-every", type=int, default=PROBE_EVERY)
    parser.add_argument("--source_revision", "--source-revision", default=current_commit())
    args = parser.parse_args()

    configuration = build_sweep_configuration(args)
    expected_runs = expected_run_count(configuration)
    if expected_runs != 2:
        raise RuntimeError(f"The paired Modal pilot must contain exactly 2 runs, found {expected_runs}.")

    sweep_id = wandb.sweep(
        entity=ENTITY_NAME,
        project=args.project_name,
        sweep=configuration,
    )
    sweep_path = f"{ENTITY_NAME}/{args.project_name}/{sweep_id}"
    print(f"SWEEP_PATH={sweep_path}")
    print(f"EXPECTED_RUNS={expected_runs}")
    print(f"Run with:\n$ uv run modal run --detach modal_run.py --sweep-path {sweep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
