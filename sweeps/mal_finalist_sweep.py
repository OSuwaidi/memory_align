"""Create the bounded MAL structural-finalist sweeps.

These experiments fill only cells that are missing from the completed
``MAL_benchmark`` sweeps.  They are deliberately not a new Cartesian search.
"""

from __future__ import annotations

import argparse
from typing import Any

import wandb

ENTITY_NAME = "osuwaidi-khalifa-university"
PROJECT_NAME = "MAL_benchmark"
SEEDS = (42, 1337, 2026)

# Shared cross-family labels:
# T-Att/U = transient attenuation without norm preservation
# T-Rep/U = transient replacement without norm preservation
# T-Rep/N = transient replacement with norm preservation
T_ATT_U_SGDM = "False,1.0,False,attenuate,False"
T_REP_U_SGDM = "False,1.0,False,replace,False"
T_REP_N_SGDM = "False,1.0,True,replace,False"
T_ATT_U_ADAMW = "False,1.0,none,attenuate,False,metric"
T_REP_U_ADAMW = "False,1.0,none,replace,False,metric"
T_REP_N_ADAMW = "False,1.0,step,replace,False,metric"
T_REP_M_ADAMW = "False,1.0,moment,replace,False,moment"


def _sgdm_command(args: argparse.Namespace) -> list[str]:
    return [
        "${env}",
        "${interpreter}",
        "${program}",
        "--data",
        "cifar100",
        "--data_dir",
        args.cifar100_dir,
        "--arch",
        "resnet50",
        "--epochs",
        "200",
        "--val_acc_target",
        "70",
        "--amp_dtype",
        args.amp_dtype,
        "--float32_precision",
        args.float32_precision,
        "${args}",
    ]


def _mae_command(args: argparse.Namespace) -> list[str]:
    return [
        "${env}",
        "${interpreter}",
        "${program}",
        "--data_dir",
        args.tiny_imagenet_dir,
        "--arch",
        "vit_tiny_patch16_224",
        "--image_size",
        "64",
        "--patch_size",
        "8",
        "--epochs",
        "300",
        "--warmup_epochs",
        "15",
        "--probe_every",
        "50",
        "--amp_dtype",
        args.amp_dtype,
        "--float32_precision",
        args.float32_precision,
        "--output_dir",
        args.mae_output_dir,
        "--save_every",
        "0",
        "--beta2",
        "0.95",
        "${args}",
    ]


def build_sweep_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Return one controlled missing-cell experiment."""
    common: dict[str, Any] = {
        "program": args.program,
        "name": args.sweep_name,
        "method": "grid",
    }

    if args.experiment == "sgdm-scale":
        return {
            **common,
            "metric": {"name": "best_val_acc", "goal": "maximize"},
            "parameters": {
                "optimizer": {"values": ("MAL_SGDM",)},
                # The only missing SGDM scale cell among the finalists.
                "MAL_config": {"values": (T_REP_N_SGDM,)},
                "nesterov": {"values": (False,)},
                "batch_size": {"values": (256,)},
                "lr": {"values": (0.1, 0.2)},
                "weight_decay": {"values": (5e-4,)},
                "seed": {"values": SEEDS},
                "use_scheduler": {"values": (True, False)},
            },
            "command": _sgdm_command(args),
        }

    if args.experiment == "adamw-scale":
        return {
            **common,
            "metric": {"name": "final_probe_val_acc", "goal": "maximize"},
            "parameters": {
                "optimizer": {"values": ("MAL_AdamW",)},
                # Complete the two untested coherent replacement bundles:
                # unscaled metric geometry and raw-moment geometry/scaling.
                "MAL_config": {"values": (T_REP_U_ADAMW, T_REP_M_ADAMW)},
                "batch_size": {"values": (1024,)},
                "base_lr": {"values": (1.5e-4, 1e-3)},
                "weight_decay": {"values": (0.05,)},
                "seed": {"values": SEEDS},
                "use_scheduler": {"values": (True,)},
            },
            "command": _mae_command(args),
        }

    if args.experiment == "adamw-stress":
        return {
            **common,
            "metric": {"name": "final_probe_val_acc", "goal": "maximize"},
            "parameters": {
                "optimizer": {"values": ("MAL_AdamW",)},
                # One standard-LR scheduler-free screen of all four finalists.
                "MAL_config": {"values": (T_ATT_U_ADAMW, T_REP_U_ADAMW, T_REP_N_ADAMW, T_REP_M_ADAMW)},
                "batch_size": {"values": (1024,)},
                "base_lr": {"values": (1.5e-4,)},
                "weight_decay": {"values": (0.05,)},
                "seed": {"values": SEEDS},
                "use_scheduler": {"values": (False,)},
            },
            "command": _mae_command(args),
        }

    raise ValueError(f"Unknown experiment: {args.experiment}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", help="Training entry point for the selected experiment")
    parser.add_argument("--experiment", choices=("sgdm-scale", "adamw-scale", "adamw-stress"), required=True)
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", default=PROJECT_NAME)
    parser.add_argument("--cifar100_dir", "--cifar100-dir", default="./data")
    parser.add_argument("--tiny_imagenet_dir", "--tiny-imagenet-dir", default="./data/tiny-imagenet-200")
    parser.add_argument("--mae_output_dir", "--mae-output-dir", default="./outputs/mae-finalists")
    parser.add_argument("--amp_dtype", "--amp-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--float32_precision", "--float32-precision", choices=("tf32", "ieee"), default="tf32")
    args = parser.parse_args()

    sweep_id = wandb.sweep(
        entity=ENTITY_NAME,
        project=args.project_name,
        sweep=build_sweep_configuration(args),
    )
    print(f"Run with:\n$ uv run wandb agent --forward-signals {ENTITY_NAME}/{args.project_name}/{sweep_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
