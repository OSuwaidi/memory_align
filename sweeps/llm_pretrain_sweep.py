"""Create the screened or confirmatory FineWeb-Edu pre-training sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import wandb

ENTITY_NAME = "osuwaidi-khalifa-university"
PROJECT_NAME = "MAL_benchmark"
OPTIMIZERS = ("AdamW", "AM_AdamW", "AdaTAMW", "AGM_AdamW")
LEARNING_RATES = (3e-4, 7.5e-4, 1.5e-3)
WEIGHT_DECAYS = (1e-2, 1e-1)
SCREEN_SEEDS = (42, 1337)
CONFIRMATION_SEEDS = (42, 1337, 2026)
SCREEN_STEPS = 1_144
CONFIRMATION_STEPS = 5_720


def optimizer_case(optimizer: str, learning_rate: float, weight_decay: float) -> str:
    return f"{optimizer}::{learning_rate:.12g}::{weight_decay:.12g}"


def read_selected_configs(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    selected = payload.get("selected", {})
    if set(selected) != set(OPTIMIZERS):
        raise ValueError(f"Selected-config receipt must contain exactly {OPTIMIZERS}; found {tuple(selected)}")
    cases: list[str] = []
    for optimizer in OPTIMIZERS:
        config = selected[optimizer]
        cases.append(optimizer_case(optimizer, float(config["learning_rate"]), float(config["weight_decay"])))
    return tuple(cases)


def build_sweep(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.stage == "screen":
        cases = tuple(
            optimizer_case(optimizer, learning_rate, weight_decay) for optimizer in OPTIMIZERS for learning_rate in LEARNING_RATES for weight_decay in WEIGHT_DECAYS
        )
        seeds = SCREEN_SEEDS
        max_steps = args.max_steps or SCREEN_STEPS
        evaluate_test = False
        eval_every = min(args.eval_every or 250, max_steps)
        checkpoint_every = min(args.checkpoint_every or 250, max_steps)
        stage_metadata = {
            "study_stage": {"value": "hyperparameter_screen"},
            "selection_rule": {"value": "minimum mean final dev loss across screen seeds"},
        }
    else:
        if args.selected_configs is None:
            raise ValueError("--selected_configs is required for the confirmation stage")
        cases = read_selected_configs(args.selected_configs)
        seeds = CONFIRMATION_SEEDS
        max_steps = args.max_steps or CONFIRMATION_STEPS
        evaluate_test = True
        eval_every = min(args.eval_every or 250, max_steps)
        checkpoint_every = min(args.checkpoint_every or 500, max_steps)
        stage_metadata = {
            "study_stage": {"value": "full_budget_confirmation"},
            "hyperparameter_selection_receipt": {"value": str(args.selected_configs.resolve())},
        }

    expected_runs = len(cases) * len(seeds)
    sweep = {
        "program": args.program,
        "name": args.sweep_name,
        "method": "grid",
        "metric": {"name": "final/val_loss", "goal": "minimize"},
        "parameters": {
            "optimizer_case": {"values": cases},
            "seed": {"values": seeds},
            **stage_metadata,
        },
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "--data_dir",
            args.data_dir,
            "--sequence_length",
            str(args.sequence_length),
            "--micro_batch_size",
            str(args.micro_batch_size),
            "--gradient_accumulation_steps",
            str(args.gradient_accumulation_steps),
            "--max_steps",
            str(max_steps),
            "--warmup_ratio",
            str(args.warmup_ratio),
            "--minimum_lr_ratio",
            str(args.minimum_lr_ratio),
            "--beta1",
            "0.9",
            "--beta2",
            "0.95",
            "--epsilon",
            "1e-8",
            "--max_grad_norm",
            "1.0",
            "--num_workers",
            "0",
            "--eval_every",
            str(eval_every),
            "--checkpoint_every",
            str(checkpoint_every),
            "--evaluate_test",
            str(evaluate_test),
            "--output_dir",
            args.output_dir,
            "--amp_dtype",
            "bfloat16",
            "--float32_precision",
            args.float32_precision,
            "${args}",
        ],
    }
    return sweep, expected_runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program")
    parser.add_argument("--stage", choices=("screen", "confirmation"), required=True)
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", default=PROJECT_NAME)
    parser.add_argument("--data_dir", "--data-dir", required=True)
    parser.add_argument("--output_dir", "--output-dir", default="./outputs/llm-pretrain")
    parser.add_argument("--selected_configs", "--selected-configs", type=Path)
    parser.add_argument("--sequence_length", "--sequence-length", type=int, default=2048)
    parser.add_argument("--micro_batch_size", "--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", "--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max_steps", "--max-steps", type=int)
    parser.add_argument("--warmup_ratio", "--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--minimum_lr_ratio", "--minimum-lr-ratio", type=float, default=0.1)
    parser.add_argument("--eval_every", "--eval-every", type=int)
    parser.add_argument("--checkpoint_every", "--checkpoint-every", type=int)
    parser.add_argument("--float32_precision", "--float32-precision", choices=("tf32", "ieee"), default="tf32")
    args = parser.parse_args()
    if args.sequence_length <= 1 or args.micro_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        parser.error("sequence length and batch components must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max_steps must be positive")

    sweep, expected_runs = build_sweep(args)
    sweep_id = wandb.sweep(entity=ENTITY_NAME, project=args.project_name, sweep=sweep)
    print(f"SWEEP_PATH={ENTITY_NAME}/{args.project_name}/{sweep_id}")
    print(f"EXPECTED_RUNS={expected_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
