"""Create the post-benchmark MAL-AdamW MAE structural extension sweep.

This is a matched, 15-run screen around the shipped transient/unscaled
configuration.  It deliberately excludes the already completed three-seed
controls

``False,1.0,none,attenuate,update,complement`` and
``True,1.0,none,attenuate,update,complement``.

The new cells isolate (1) power under recursive memory, (2) step-norm versus
moment-norm matching under transient memory, and (3) whether step-norm matching
changes the interaction between recursive memory and power.  Every other
training and optimizer setting is held fixed.
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

SEEDS = (42, 1337, 2026)
REPRESENTATIVE_BATCH_SIZE = 1024
REPRESENTATIVE_BASE_LR = 1e-3
REPRESENTATIVE_WEIGHT_DECAY = 5e-2
COMPARISON_GROUP = "mal_adamw_mae_structural_extensions_v1"

# ``scale=True`` is normalized by MAL_AdamW to ``scale="step"``.  The explicit
# spelling here keeps the W&B grouping field unambiguous.  ``moment`` is a
# distinct raw-first-moment norm match and therefore merits its own control.
MAL_CONFIGS = (
    "True,0.5,none,attenuate,update,complement",
    "False,1.0,step,attenuate,update,complement",
    "False,1.0,moment,attenuate,update,complement",
    "True,1.0,step,attenuate,update,complement",
    "True,0.5,step,attenuate,update,complement",
)


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
            "seed": {"values": SEEDS},
            "use_scheduler": {"values": (True,)},
            "compute_platform": {"values": ("aus_aws_slurm",)},
            "comparison_group": {"values": (COMPARISON_GROUP,)},
            "experiment_stage": {"values": ("mal_adamw_mae_postbenchmark_structure",)},
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
            "--amp_dtype",
            args.amp_dtype,
            "--float32_precision",
            args.float32_precision,
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
    parser.add_argument("--data_dir", "--data-dir", default="./data/tiny-imagenet-200")
    parser.add_argument("--output_dir", "--output-dir", default="./outputs/mae-structural-extensions")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--warmup_epochs", "--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--probe_every", "--probe-every", type=int, default=PROBE_EVERY)
    parser.add_argument("--amp_dtype", "--amp-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--float32_precision", "--float32-precision", choices=("tf32", "ieee"), default="tf32")
    parser.add_argument("--source_revision", "--source-revision", default=current_commit())
    args = parser.parse_args()

    configuration = build_sweep_configuration(args)
    expected_runs = expected_run_count(configuration)
    if expected_runs != 15:
        raise RuntimeError(f"Structural extension sweep must contain exactly 15 runs, found {expected_runs}.")

    sweep_id = wandb.sweep(
        entity=ENTITY_NAME,
        project=args.project_name,
        sweep=configuration,
    )
    sweep_path = f"{ENTITY_NAME}/{args.project_name}/{sweep_id}"
    print(f"SWEEP_PATH={sweep_path}")
    print(f"EXPECTED_RUNS={expected_runs}")
    print(f"Run with:\n$ uv run wandb agent --forward-signals {sweep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
