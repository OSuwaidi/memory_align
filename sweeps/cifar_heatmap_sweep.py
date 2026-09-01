"""Create the two post-selection CIFAR LR/batch-size heatmap sweeps.

The optimizer and MAL structure are encoded in one ``optimizer_case`` sweep
parameter.  This avoids the silent Cartesian duplication that would result
from sweeping ``optimizer`` and ``MAL_config`` independently.
"""

from __future__ import annotations

import argparse
from typing import Any

import wandb

ENTITY_NAME = "osuwaidi-khalifa-university"
PROJECT_NAME = "MAL_benchmark"
SEEDS = (42, 1337, 2026)
WEIGHT_DECAY = 5e-4

T_ATT_U = "False,1.0,False,attenuate,False"
T_REP_U = "False,1.0,False,replace,False"
T_REP_N = "False,1.0,True,replace,False"

CIFAR10_BATCH_SIZES = (64, 128, 256, 512, 1024, 2048, 4096)
CIFAR10_LRS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
CIFAR100_BATCH_SIZES = (128, 256, 512, 1024, 2048, 4096)
CIFAR100_LRS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8)


def mal_case(label: str, config: str) -> str:
    return f"MAL_SGDM::{label}::{config}"


def validate_sgdm_mal_config(value: str) -> str:
    fields = value.split(",")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            "must be 'in_place,pwr,scale,gate_mode,descent_safeguard'"
        )
    if fields[0] not in {"True", "False"} or fields[2] not in {"True", "False"}:
        raise argparse.ArgumentTypeError("in_place and scale must be True or False")
    if fields[1] not in {"0.5", "1.0"}:
        raise argparse.ArgumentTypeError("pwr must be 0.5 or 1.0")
    if fields[3] not in {"attenuate", "replace", "cap"}:
        raise argparse.ArgumentTypeError("gate_mode must be attenuate, replace, or cap")
    if fields[4] not in {"True", "False"}:
        raise argparse.ArgumentTypeError("descent_safeguard must be True or False")
    return value


def build_configuration(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.experiment == "cifar10-screen":
        data = "cifar10"
        arch = "resnet18"
        batch_sizes = CIFAR10_BATCH_SIZES
        learning_rates = CIFAR10_LRS
        target = 90.0
        optimizer_cases = (
            "AM_MSGD",
            "TAM_SGDM",
            mal_case("T-Att/U", T_ATT_U),
            mal_case("T-Rep/U", T_REP_U),
            mal_case("T-Rep/N", T_REP_N),
        )
    else:
        if not args.mal_sgdm_config:
            raise ValueError("--mal_sgdm_config is required for cifar100-benchmark")
        data = "cifar100"
        arch = "resnet50"
        batch_sizes = CIFAR100_BATCH_SIZES
        learning_rates = CIFAR100_LRS
        target = 70.0
        optimizer_cases = (
            "SGDM",
            "AM_MSGD",
            "CAUTIOUS_SGDM",
            "TAM_SGDM",
            mal_case("MAL-selected", args.mal_sgdm_config),
        )

    expected_runs = len(optimizer_cases) * len(batch_sizes) * len(learning_rates) * len(SEEDS)
    configuration: dict[str, Any] = {
        "program": args.program,
        "name": args.sweep_name,
        "method": "grid",
        "metric": {"name": "test_acc", "goal": "maximize"},
        "parameters": {
            "optimizer_case": {"values": optimizer_cases},
            "nesterov": {"values": (False,)},
            "batch_size": {"values": batch_sizes},
            "lr": {"values": learning_rates},
            "weight_decay": {"values": (WEIGHT_DECAY,)},
            "seed": {"values": SEEDS},
            "use_scheduler": {"values": (True,)},
        },
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "--data",
            data,
            "--data_dir",
            args.data_dir,
            "--arch",
            arch,
            "--epochs",
            str(args.epochs),
            "--val_acc_target",
            str(target),
            "--amp_dtype",
            args.amp_dtype,
            "--float32_precision",
            args.float32_precision,
            "${args}",
        ],
    }
    return configuration, expected_runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program")
    parser.add_argument("--experiment", choices=("cifar10-screen", "cifar100-benchmark"), required=True)
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", default=PROJECT_NAME)
    parser.add_argument("--data_dir", "--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--amp_dtype", "--amp-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--float32_precision", "--float32-precision", choices=("tf32", "ieee"), default="tf32")
    parser.add_argument("--mal_sgdm_config", "--mal-sgdm-config", type=validate_sgdm_mal_config)
    args = parser.parse_args()

    configuration, expected_runs = build_configuration(args)
    sweep_id = wandb.sweep(entity=ENTITY_NAME, project=args.project_name, sweep=configuration)
    sweep_path = f"{ENTITY_NAME}/{args.project_name}/{sweep_id}"
    print(f"EXPECTED_RUNS={expected_runs}")
    print(f"Run with:\n$ uv run wandb agent --forward-signals {sweep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
