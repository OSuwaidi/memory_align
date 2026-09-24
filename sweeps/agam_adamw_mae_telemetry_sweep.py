"""Create the single canonical AGAM-AdamW MAE telemetry run."""

from __future__ import annotations

import argparse

import wandb

ENTITY = "osuwaidi-khalifa-university"
PROJECT = "MAL_benchmark"
AGAM_CONFIG = "False,1.0,none,attenuate,update,complement"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", help="MAE training entry point")
    parser.add_argument("--data_dir", "--data-dir", default="./data/tiny-imagenet-200")
    parser.add_argument("--output_dir", "--output-dir", default="./outputs/mae")
    parser.add_argument(
        "--telemetry_output_dir",
        "--telemetry-output-dir",
        default="./outputs/agam-adamw-mae-telemetry",
    )
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", default=PROJECT)
    args = parser.parse_args()

    sweep = {
        "program": args.program,
        "name": args.sweep_name,
        "method": "grid",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {
            "optimizer": {"value": "MAL_AdamW"},
            "MAL_config": {"value": AGAM_CONFIG},
            "batch_size": {"value": 1024},
            "base_lr": {"value": 1e-3},
            "weight_decay": {"value": 5e-2},
            "seed": {"value": 42},
            "use_scheduler": {"value": True},
            "telemetry_profile": {"value": "canonical_agam_adamw_mae_v1"},
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
            "--save_every",
            "0",
            "--output_dir",
            args.output_dir,
            "--telemetry_output_dir",
            args.telemetry_output_dir,
            "--telemetry_flush_steps",
            "128",
            "${args}",
        ],
    }
    sweep_id = wandb.sweep(entity=ENTITY, project=args.project_name, sweep=sweep)
    sweep_path = f"{ENTITY}/{args.project_name}/{sweep_id}"
    print("EXPECTED_RUNS=1")
    print(f"SWEEP_PATH={sweep_path}")
    print(f"Run with: wandb agent --count 1 --forward-signals {sweep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
