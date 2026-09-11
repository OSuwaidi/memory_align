"""Create matched MAE screens for AdaMAL and MAL-AdamW controls.

Both screens use the same representative ViT-Tiny/Tiny-ImageNet recipe. The
original AdaMAL grid changes recursion, second-moment bias correction, and
alignment geometry. The focused in-place follow-up drops raw-moment alignment,
which is not faithful to the geometry of an adaptively preconditioned update,
and compares direct update alignment against the induced-metric cosine. The
MAL-AdamW screen is one fixed-gradient control at the best matched
transient/update geometry. The complement LR bracket holds that structure and
the full recipe fixed while testing one point above its current best learning
rate.
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
COMPLEMENT_BRACKET_BASE_LR = 2e-3

FIXED_MAL_CONFIG = "False,1.0,none,attenuate,update,fixed"
COMPLEMENT_MAL_CONFIG = "False,1.0,none,attenuate,update,complement"
ADAMAL_CONFIGS = tuple(f"{in_place},1.0,none,attenuate,{align},{unbias}" for in_place in (False, True) for unbias in (False, True) for align in ("moment", "update"))
ADAMAL_IN_PLACE_CONFIGS = tuple(f"True,1.0,none,attenuate,{align},{unbias}" for unbias in (False, True) for align in ("update", "metric"))


def build_sweep_configuration(args: argparse.Namespace) -> dict[str, Any]:
    default_base_lr = COMPLEMENT_BRACKET_BASE_LR if args.screen == "complement-lr-bracket" else REPRESENTATIVE_BASE_LR
    batch_size = args.batch_size if args.batch_size is not None else REPRESENTATIVE_BATCH_SIZE
    base_lr = args.base_lr if args.base_lr is not None else default_base_lr
    weight_decay = args.weight_decay if args.weight_decay is not None else REPRESENTATIVE_WEIGHT_DECAY
    parameters: dict[str, Any] = {
        "batch_size": {"values": (batch_size,)},
        "base_lr": {"values": (base_lr,)},
        "weight_decay": {"values": (weight_decay,)},
        "seed": {"values": SEEDS},
        "use_scheduler": {"values": (True,)},
    }
    if args.screen in {"adamal", "adamal-in-place"}:
        configs = ADAMAL_CONFIGS if args.screen == "adamal" else ADAMAL_IN_PLACE_CONFIGS
        parameters.update(
            {
                "optimizer": {"values": ("AdaMAL",)},
                "AdaMAL_config": {"values": configs},
            }
        )
    elif args.screen == "fixed-control":
        parameters.update(
            {
                "optimizer": {"values": ("MAL_AdamW",)},
                "MAL_config": {"values": (FIXED_MAL_CONFIG,)},
            }
        )
    else:
        parameters.update(
            {
                "optimizer": {"values": ("MAL_AdamW",)},
                "MAL_config": {"values": (COMPLEMENT_MAL_CONFIG,)},
            }
        )

    return {
        "program": args.program,
        "name": args.sweep_name,
        "method": "grid",
        "metric": {"name": "final_probe_val_acc", "goal": "maximize"},
        "parameters": parameters,
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
    parser.add_argument(
        "--screen",
        choices=("adamal", "adamal-in-place", "fixed-control", "complement-control", "complement-lr-bracket"),
        required=True,
    )
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", default=PROJECT_NAME)
    parser.add_argument("--data_dir", "--data-dir", default="./data/tiny-imagenet-200")
    parser.add_argument("--output_dir", "--output-dir", default="./outputs/mae-structure-screen")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--warmup_epochs", "--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--probe_every", "--probe-every", type=int, default=PROBE_EVERY)
    parser.add_argument("--batch_size", "--batch-size", type=int)
    parser.add_argument("--base_lr", "--base-lr", type=float)
    parser.add_argument("--weight_decay", "--weight-decay", type=float)
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
