"""Create the paired AGAM-AdamW standard-bias-correction MAE ablation."""

from __future__ import annotations

import argparse
import subprocess
from typing import Any

import wandb

ENTITY_NAME = "osuwaidi-khalifa-university"
PROJECT_NAME = "MAL_benchmark"
SEEDS = (42, 1337, 2026)
MAL_CONFIG = "False,1.0,none,attenuate,update,complement"
BATCH_SIZE = 1024
BASE_LR = 1.5e-3
WEIGHT_DECAY = 5e-2
EXPECTED_RUNS = 3


def current_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_sweep(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "program": args.program,
        "name": args.sweep_name,
        "method": "grid",
        "metric": {"name": "final/probe_val_acc", "goal": "maximize"},
        "parameters": {
            "optimizer": {"values": ("MAL_AdamW",)},
            "MAL_config": {"values": (MAL_CONFIG,)},
            "agam_first_moment_correction": {"values": ("standard",)},
            "batch_size": {"values": (BATCH_SIZE,)},
            "base_lr": {"values": (BASE_LR,)},
            "weight_decay": {"values": (WEIGHT_DECAY,)},
            "seed": {"values": SEEDS},
            "use_scheduler": {"values": (True,)},
            "compute_platform": {"values": ("aus_aws_slurm",)},
            "comparison_group": {"values": ("agam_adamw_mae_first_moment_correction_v1",)},
            "experiment_stage": {"values": ("standard_first_moment_correction_pretraining",)},
            "source_revision": {"values": (args.source_revision,)},
        },
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "--data_dir",
            args.data_dir,
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
            "--beta2",
            "0.95",
            "--amp_dtype",
            "bfloat16",
            "--float32_precision",
            "tf32",
            "--output_dir",
            args.output_dir,
            "--save_every",
            "0",
            "${args}",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program")
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", default=PROJECT_NAME)
    parser.add_argument("--data_dir", "--data-dir", required=True)
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--source_revision", "--source-revision", default=current_commit())
    args = parser.parse_args()

    sweep = build_sweep(args)
    cardinality = 1
    for parameter in sweep["parameters"].values():
        cardinality *= len(parameter["values"])
    if cardinality != EXPECTED_RUNS:
        raise RuntimeError(f"Expected {EXPECTED_RUNS} runs, found {cardinality}.")

    sweep_id = wandb.sweep(entity=ENTITY_NAME, project=args.project_name, sweep=sweep)
    print(f"SWEEP_PATH={ENTITY_NAME}/{args.project_name}/{sweep_id}")
    print(f"EXPECTED_RUNS={EXPECTED_RUNS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
