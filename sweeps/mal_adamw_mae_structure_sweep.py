"""Create the focused MAL-AdamW MAE structure-selection sweep.

The screen holds the MAE recipe fixed and changes only the fresh-gradient
weighting, alignment geometry, and norm-source choices requested for MAL.
It is intentionally a development experiment; the selected structure must be
confirmed on the remaining learning-rate, batch-size, and weight-decay cells.
"""

from __future__ import annotations

import argparse
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

FIXED_CONTROL = "False,1.0,none,attenuate,update,fixed"
COMPLEMENT_CONTROL = "False,1.0,none,attenuate,update,complement"
COMPLEMENT_SCALED_CONFIGS = tuple(
    f"False,1.0,{scale},attenuate,{align},complement"
    for scale in ("step", "moment")
    for align in ("update", "metric", "moment")
)
MAL_CONFIGS = (FIXED_CONTROL, COMPLEMENT_CONTROL, *COMPLEMENT_SCALED_CONFIGS)


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
    parser.add_argument("program", help="MAE training entry point (normally tasks/mae_pretrain.py)")
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", default=PROJECT_NAME)
    parser.add_argument("--data_dir", "--data-dir", default="./data/tiny-imagenet-200")
    parser.add_argument("--output_dir", "--output-dir", default="./outputs/mae-structure-screen")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--warmup_epochs", "--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--probe_every", "--probe-every", type=int, default=PROBE_EVERY)
    parser.add_argument("--amp_dtype", "--amp-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--float32_precision", "--float32-precision", choices=("tf32", "ieee"), default="tf32")
    args = parser.parse_args()

    configuration = build_sweep_configuration(args)
    sweep_id = wandb.sweep(
        entity=ENTITY_NAME,
        project=args.project_name,
        sweep=configuration,
    )
    print(f"EXPECTED_RUNS={expected_run_count(configuration)}")
    print(f"Run with:\n$ uv run wandb agent --forward-signals {ENTITY_NAME}/{args.project_name}/{sweep_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
