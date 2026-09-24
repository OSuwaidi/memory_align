"""Create the matched AGAM-AdamW component ablation on MAE/Tiny-ImageNet."""

from __future__ import annotations

import argparse
import subprocess
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
AGAM_CONFIG = "False,1.0,none,attenuate,update,complement"
EXPECTED_RUNS = len(SEEDS) * len(VARIANTS)


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
            "MAL_config": {"values": (AGAM_CONFIG,)},
            "AGAM_variant": {"values": VARIANTS},
            "agam_first_moment_correction": {"values": ("adaptive",)},
            "batch_size": {"values": (1024,)},
            "base_lr": {"values": (1e-3,)},
            "weight_decay": {"values": (5e-2,)},
            "seed": {"values": SEEDS},
            "use_scheduler": {"values": (True,)},
            "experiment_group": {"values": ("agam_adamw_component_ablation_mae_tiny_imagenet",)},
            "compute_platform": {"values": ("aus_aws_slurm",)},
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
            "--probe_epochs",
            "90",
            "--max_micro_batch_size",
            "256",
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
    parser.add_argument("--project_name", "--project-name", default=PROJECT)
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

    sweep_id = wandb.sweep(entity=ENTITY, project=args.project_name, sweep=sweep)
    print(f"SWEEP_PATH={ENTITY}/{args.project_name}/{sweep_id}")
    print(f"EXPECTED_RUNS={EXPECTED_RUNS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
