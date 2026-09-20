"""Create the 18-run AGAM-Lion MAE hyperparameter expansion sweep."""

from __future__ import annotations

import argparse
import subprocess
from typing import Any

import wandb

ENTITY_NAME = "osuwaidi-khalifa-university"
PROJECT_NAME = "MAL_benchmark"

BASE_LRS = (2e-4, 1.5e-4, 5e-5)
WEIGHT_DECAYS = (0.5, 0.25, 0.15)
SEEDS = (42, 1337)
BATCH_SIZE = 1024

MODEL = "vit_tiny_patch16_224"
IMAGE_SIZE = 64
PATCH_SIZE = 8
EPOCHS = 300
WARMUP_EPOCHS = 15
PROBE_EVERY = 50
LION_BETAS = (0.9, 0.99)

FINETUNE_EPOCHS = 100
FINETUNE_BATCH_SIZE = 1024
FINETUNE_BASE_LR = 5e-4
FINETUNE_MIN_LR = 1e-6
FINETUNE_WARMUP_EPOCHS = 5
FINETUNE_WEIGHT_DECAY = 0.05
FINETUNE_LAYER_DECAY = 0.65
FINETUNE_DROP_PATH = 0.1
FINETUNE_MIXUP = 0.8
FINETUNE_CUTMIX = 1.0
FINETUNE_LABEL_SMOOTHING = 0.1

COMPARISON_GROUP = "agam_lion_mae_hparam_expansion_v1"
EXPECTED_RUNS = len(BASE_LRS) * len(WEIGHT_DECAYS) * len(SEEDS)


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
        "metric": {"name": "finetune/final_val_top1_pct", "goal": "maximize"},
        "parameters": {
            "optimizer": {"values": ("AGAM_Lion",)},
            "batch_size": {"values": (BATCH_SIZE,)},
            "base_lr": {"values": BASE_LRS},
            "weight_decay": {"values": WEIGHT_DECAYS},
            "seed": {"values": SEEDS},
            "use_scheduler": {"values": (True,)},
            "comparison_group": {"values": (COMPARISON_GROUP,)},
            "study_stage": {"values": ("agam_lion_mae_hyperparameter_expansion",)},
            "selection_metric": {"values": ("finetune/final_val_top1_pct",)},
            "evaluation_protocol": {"values": ("periodic_linear_probe_plus_final_encoder_end_to_end_finetune",)},
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
            "--probe_epochs",
            "90",
            "--probe_batch_size",
            "4096",
            "--probe_base_lr",
            "0.1",
            "--probe_warmup_epochs",
            "10",
            "--run_finetune",
            "True",
            "--finetune_epochs",
            str(FINETUNE_EPOCHS),
            "--finetune_batch_size",
            str(FINETUNE_BATCH_SIZE),
            "--finetune_max_micro_batch_size",
            "256",
            "--finetune_base_lr",
            str(FINETUNE_BASE_LR),
            "--finetune_min_lr",
            str(FINETUNE_MIN_LR),
            "--finetune_warmup_epochs",
            str(FINETUNE_WARMUP_EPOCHS),
            "--finetune_weight_decay",
            str(FINETUNE_WEIGHT_DECAY),
            "--finetune_layer_decay",
            str(FINETUNE_LAYER_DECAY),
            "--finetune_drop_path",
            str(FINETUNE_DROP_PATH),
            "--finetune_mixup",
            str(FINETUNE_MIXUP),
            "--finetune_cutmix",
            str(FINETUNE_CUTMIX),
            "--finetune_label_smoothing",
            str(FINETUNE_LABEL_SMOOTHING),
            "--max_micro_batch_size",
            "256",
            "--num_workers",
            "4",
            "--amp_dtype",
            "bfloat16",
            "--float32_precision",
            "tf32",
            "--momentum",
            str(LION_BETAS[0]),
            "--beta2",
            str(LION_BETAS[1]),
            "--output_dir",
            args.output_dir,
            "--save_every",
            "0",
            "${args}",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", help="MAE entry point (normally tasks/mae_pretrain.py)")
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", default=PROJECT_NAME)
    parser.add_argument("--data_dir", "--data-dir", required=True)
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--warmup_epochs", "--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--probe_every", "--probe-every", type=int, default=PROBE_EVERY)
    parser.add_argument("--source_revision", "--source-revision", default=current_commit())
    args = parser.parse_args()

    sweep_id = wandb.sweep(
        entity=ENTITY_NAME,
        project=args.project_name,
        sweep=build_sweep(args),
    )
    print(f"SWEEP_PATH={ENTITY_NAME}/{args.project_name}/{sweep_id}")
    print(f"EXPECTED_RUNS={EXPECTED_RUNS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
