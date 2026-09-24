"""Create the matched AGAM-SGD component ablation for the paper.

The grid changes exactly one structural choice around canonical AGAM-SGD and
uses the established ResNet-50/CIFAR-100 recipe.  Five variants times three
seeds produces exactly fifteen runs, so one sweep fully occupies fifteen GPUs.
"""

from __future__ import annotations

import argparse
from typing import Any

import wandb


ENTITY = "osuwaidi-khalifa-university"
PROJECT = "MAL_benchmark"
SEEDS = (42, 1337, 2026)
VARIANTS = (
    "canonical",
    "previous_memory",
    "global_gate",
    "writeback",
    "hard_reset",
)
CANONICAL_CONFIG = "False,1.0,False,attenuate,moment,fixed,none"


def build_configuration(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    expected_runs = len(VARIANTS) * len(SEEDS)
    configuration: dict[str, Any] = {
        "program": args.program,
        "name": args.sweep_name,
        "method": "grid",
        # Validation remains the selection/monitoring metric. Test metrics are
        # reported only after training at the final and validation-selected models.
        "metric": {"name": "best_val_acc", "goal": "maximize"},
        "parameters": {
            "optimizer_case": {
                "value": f"MAL_SGDM::AGAM-component-ablation::{CANONICAL_CONFIG}"
            },
            "AGAM_variant": {"values": VARIANTS},
            "nesterov": {"value": False},
            "batch_size": {"value": 256},
            "lr": {"value": 0.1},
            "weight_decay": {"value": 5e-4},
            "seed": {"values": SEEDS},
            "use_scheduler": {"value": True},
            "experiment_group": {"value": "agam_component_ablation_resnet50_cifar100"},
            "comparison_scheduled_sweep": {"value": "c72berzj"},
            "comparison_constant_sweep": {"value": "52y7g41m"},
        },
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "--data",
            "cifar100",
            "--data_dir",
            args.data_dir,
            "--arch",
            "resnet50",
            "--epochs",
            "200",
            "--split_seed",
            "20260901",
            "--val_acc_target",
            "70",
            "--amp_dtype",
            "bfloat16",
            "--float32_precision",
            "tf32",
            "${args}",
        ],
    }
    return configuration, expected_runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program")
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", default=PROJECT)
    parser.add_argument("--data_dir", "--data-dir", default="./data")
    args = parser.parse_args()

    configuration, expected_runs = build_configuration(args)
    sweep_id = wandb.sweep(
        entity=ENTITY,
        project=args.project_name,
        sweep=configuration,
    )
    path = f"{ENTITY}/{args.project_name}/{sweep_id}"
    print(f"SWEEP_PATH={path}")
    print(f"EXPECTED_RUNS={expected_runs}")
    print(f"Run with:\n$ wandb agent --forward-signals {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
