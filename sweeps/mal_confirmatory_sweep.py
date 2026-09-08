"""Create compact, external-validation MAL structure sweeps.

These are not broader hyperparameter searches.  They carry forward only the
structures that remained competitive after CIFAR-100/ResNet-50 and
Tiny-ImageNet/MAE selection, and evaluate them on supervised Tiny-ImageNet with
new seeds and held-out test accuracy. The AdamW structure experiment isolates
alignment geometry and fresh-gradient weighting before the full MAE/LLM
benchmarks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import wandb

ENTITY_NAME = "osuwaidi-khalifa-university"
PROJECT_NAME = "MAL_benchmark"
SEEDS = (17, 73, 211, 997, 4099)

T_REP_U_SGDM = "False,1.0,False,replace"
T_REP_N_SGDM = "False,1.0,True,replace"

ADAMW_ALIGNMENTS = ("metric", "update", "moment")
ADAMW_GRADIENT_WEIGHT_MODES = ("fixed", "complement")
ADAMW_STRUCTURE_CONFIGS = tuple(
    f"False,1.0,none,attenuate,{align},{gradient_weight_mode}" for align in ADAMW_ALIGNMENTS for gradient_weight_mode in ADAMW_GRADIENT_WEIGHT_MODES
)


def build_recursive_configs(selection_file: Path) -> tuple[str, ...]:
    source_configs = tuple(line.strip() for line in selection_file.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(source_configs) != 2 or len(set(source_configs)) != 2:
        raise ValueError("The recursive follow-up requires exactly two distinct selected source configurations.")
    recursive_configs: list[str] = []
    for raw_config in source_configs:
        fields = raw_config.split(",")
        if len(fields) != 6 or tuple(fields[:4]) != ("False", "1.0", "none", "attenuate"):
            raise ValueError(f"Unexpected selected source MAL_config: {raw_config}")
        align, gradient_weight_mode = fields[4:]
        if align not in ADAMW_ALIGNMENTS or gradient_weight_mode not in ADAMW_GRADIENT_WEIGHT_MODES:
            raise ValueError(f"Unsupported selected source MAL_config: {raw_config}")
        recursive_configs.extend(f"True,{pwr},none,attenuate,{align},{gradient_weight_mode}" for pwr in ("0.5", "1.0"))
    return tuple(recursive_configs)


def _common_command(args: argparse.Namespace) -> list[str]:
    return [
        "${env}",
        "${interpreter}",
        "${program}",
        "--data_dir",
        args.tiny_imagenet_dir,
        "--warmup_epochs",
        "5",
        "--split_seed",
        "20260901",
        "--amp_dtype",
        args.amp_dtype,
        "--float32_precision",
        args.float32_precision,
    ]


def build_sweep_configuration(args: argparse.Namespace) -> dict[str, Any]:
    common: dict[str, Any] = {
        "program": args.program,
        "name": args.sweep_name,
        "method": "grid",
        "metric": {"name": "test_acc", "goal": "maximize"},
    }

    if args.experiment == "sgdm-resnet50":
        return {
            **common,
            "parameters": {
                "optimizer": {"values": ("MAL_SGDM",)},
                "MAL_config": {"values": (T_REP_U_SGDM, T_REP_N_SGDM)},
                "nesterov": {"values": (False,)},
                "batch_size": {"values": (256,)},
                "lr": {"values": (0.1,)},
                "weight_decay": {"values": (5e-4,)},
                "seed": {"values": SEEDS},
                "use_scheduler": {"values": (True, False)},
            },
            "command": [
                *_common_command(args),
                "--arch",
                "resnet50",
                "--pretrained",
                "False",
                "--image_size",
                "64",
                "--epochs",
                "100",
                "--val_acc_target",
                "50",
                "--max_micro_batch_size",
                "256",
                "--eval_batch_size",
                "512",
                "--beta2",
                "0.999",
                "${args}",
            ],
        }

    if args.experiment == "adamw-vit":
        return {
            **common,
            # Select structure on the held-out training split; test accuracy is
            # logged once from the validation-selected checkpoint.
            "metric": {"name": "selection_val_acc", "goal": "maximize"},
            "parameters": {
                "optimizer": {"values": ("MAL_AdamW",)},
                "MAL_config": {"values": ADAMW_STRUCTURE_CONFIGS},
                "batch_size": {"values": (256,)},
                "base_lr": {"values": (5e-4, 1e-3)},
                "weight_decay": {"values": (0.05,)},
                "seed": {"values": SEEDS},
                "use_scheduler": {"values": (True, False)},
            },
            "command": [
                *_common_command(args),
                "--arch",
                "vit_tiny_patch16_224",
                "--pretrained",
                "True",
                "--image_size",
                "224",
                "--epochs",
                "30",
                "--val_acc_target",
                "65",
                "--max_micro_batch_size",
                "64",
                "--eval_batch_size",
                "256",
                "--beta2",
                "0.999",
                "${args}",
            ],
        }

    if args.experiment == "adamw-gradient-weight":
        return {
            **common,
            # Select structure on the held-out training split; test accuracy is
            # still logged once from the validation-selected checkpoint.
            "metric": {"name": "selection_val_acc", "goal": "maximize"},
            "parameters": {
                "optimizer": {"values": ("MAL_AdamW",)},
                # Factorial matched comparison: three defensible alignment
                # geometries x historical/adaptive fresh-gradient weighting.
                # Scaling and replacement are intentionally excluded.
                "MAL_config": {"values": ADAMW_STRUCTURE_CONFIGS},
                "batch_size": {"values": (256,)},
                "base_lr": {"values": (5e-4, 1e-3)},
                "weight_decay": {"values": (0.05,)},
                "seed": {"values": SEEDS},
                "use_scheduler": {"values": (True, False)},
            },
            "command": [
                *_common_command(args),
                "--arch",
                "vit_tiny_patch16_224",
                "--pretrained",
                "True",
                "--image_size",
                "224",
                "--epochs",
                "30",
                "--val_acc_target",
                "65",
                "--max_micro_batch_size",
                "64",
                "--eval_batch_size",
                "256",
                "--beta2",
                "0.999",
                "${args}",
            ],
        }

    if args.experiment == "adamw-in-place":
        if args.selection_file is None or args.source_sweep is None:
            raise ValueError("adamw-in-place requires --selection_file and --source_sweep.")
        return {
            **common,
            "metric": {"name": "selection_val_acc", "goal": "maximize"},
            "parameters": {
                "optimizer": {"values": ("MAL_AdamW",)},
                "MAL_config": {"values": build_recursive_configs(args.selection_file)},
                "batch_size": {"values": (256,)},
                "base_lr": {"values": (5e-4, 1e-3)},
                "weight_decay": {"values": (0.05,)},
                "seed": {"values": SEEDS},
                "use_scheduler": {"values": (True, False)},
                "selection_source_sweep": {"values": (args.source_sweep,)},
            },
            "command": [
                *_common_command(args),
                "--arch",
                "vit_tiny_patch16_224",
                "--pretrained",
                "True",
                "--image_size",
                "224",
                "--epochs",
                "30",
                "--val_acc_target",
                "65",
                "--max_micro_batch_size",
                "64",
                "--eval_batch_size",
                "256",
                "--beta2",
                "0.999",
                "${args}",
            ],
        }

    raise ValueError(f"Unknown experiment: {args.experiment}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program")
    parser.add_argument(
        "--experiment",
        choices=("sgdm-resnet50", "adamw-vit", "adamw-gradient-weight", "adamw-in-place"),
        required=True,
    )
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", default=PROJECT_NAME)
    parser.add_argument("--tiny_imagenet_dir", "--tiny-imagenet-dir", default="./data/tiny-imagenet-200")
    parser.add_argument("--amp_dtype", "--amp-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--float32_precision", "--float32-precision", choices=("tf32", "ieee"), default="tf32")
    parser.add_argument("--selection_file", "--selection-file", type=Path)
    parser.add_argument("--source_sweep", "--source-sweep")
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
