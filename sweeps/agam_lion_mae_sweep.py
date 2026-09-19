"""Create the paired Lion/AGAM-Lion MAE screen and confirmation sweeps."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import wandb

ENTITY_NAME = "osuwaidi-khalifa-university"
PROJECT_NAME = "MAL_benchmark"

OPTIMIZERS = ("Lion", "AGAM_Lion")
BASE_LRS = (1e-4, 3e-4)
WEIGHT_DECAYS = (0.15, 0.5)
SCREEN_SEEDS = (42,)
CONFIRMATION_SEEDS = (1337, 2026)

MODEL = "vit_tiny_patch16_224"
IMAGE_SIZE = 64
PATCH_SIZE = 8
BATCH_SIZE = 1024
EPOCHS = 300
WARMUP_EPOCHS = 15
PROBE_EVERY = 50
LION_BETAS = (0.9, 0.99)
COMPARISON_GROUP = "agam_lion_mae_tiny_imagenet_v1"


def current_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def sweep_parameters(
    *,
    optimizers: tuple[str, ...],
    base_lrs: tuple[float, ...],
    weight_decays: tuple[float, ...],
    seeds: tuple[int, ...],
    study_stage: str,
    source_revision: str,
    selected_from_sweep: str = "",
) -> dict[str, Any]:
    return {
        "optimizer": {"values": optimizers},
        "batch_size": {"values": (BATCH_SIZE,)},
        "base_lr": {"values": base_lrs},
        "weight_decay": {"values": weight_decays},
        "seed": {"values": seeds},
        "use_scheduler": {"values": (True,)},
        "comparison_group": {"values": (COMPARISON_GROUP,)},
        "study_stage": {"values": (study_stage,)},
        "source_revision": {"values": (source_revision,)},
        "selected_from_sweep": {"values": (selected_from_sweep,)},
        "selection_metric": {"values": ("final/probe_val_acc",)},
    }


def build_sweep(
    args: argparse.Namespace,
    *,
    optimizers: tuple[str, ...],
    base_lrs: tuple[float, ...],
    weight_decays: tuple[float, ...],
    seeds: tuple[int, ...],
    study_stage: str,
    selected_from_sweep: str = "",
) -> tuple[dict[str, Any], int]:
    parameters = sweep_parameters(
        optimizers=optimizers,
        base_lrs=base_lrs,
        weight_decays=weight_decays,
        seeds=seeds,
        study_stage=study_stage,
        source_revision=args.source_revision,
        selected_from_sweep=selected_from_sweep,
    )
    expected_runs = 1
    for parameter in parameters.values():
        expected_runs *= len(parameter["values"])

    sweep = {
        "program": args.program,
        "name": args.sweep_name,
        "method": "grid",
        "metric": {"name": "final/probe_val_acc", "goal": "maximize"},
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
    return sweep, expected_runs


def summary_float(run: Any, *keys: str, default: float) -> float:
    summary = dict(run.summary)
    for key in keys:
        value = summary.get(key)
        if value is not None:
            return float(value)
    return default


def select_screen_winners(source_sweep: str) -> dict[str, dict[str, Any]]:
    expected_prefix = f"{ENTITY_NAME}/{PROJECT_NAME}/"
    if not source_sweep.startswith(expected_prefix):
        raise ValueError(f"Source sweep must start with {expected_prefix!r}.")

    runs = list(wandb.Api(timeout=180).sweep(source_sweep).runs)
    expected_signatures = {
        (optimizer, base_lr, weight_decay, seed)
        for optimizer in OPTIMIZERS
        for base_lr in BASE_LRS
        for weight_decay in WEIGHT_DECAYS
        for seed in SCREEN_SEEDS
    }
    finished_by_signature: dict[tuple[str, float, float, int], Any] = {}
    for run in runs:
        if run.state != "finished":
            continue
        config = dict(run.config)
        signature = (
            str(config.get("optimizer")),
            float(config.get("base_lr")),
            float(config.get("weight_decay")),
            int(config.get("seed")),
        )
        if signature in finished_by_signature:
            raise RuntimeError(f"Duplicate finished screen cell: {signature}")
        finished_by_signature[signature] = run

    observed_signatures = set(finished_by_signature)
    if observed_signatures != expected_signatures:
        missing = sorted(expected_signatures - observed_signatures)
        unexpected = sorted(observed_signatures - expected_signatures)
        raise RuntimeError(f"Screen is not complete: missing={missing}, unexpected={unexpected}")

    selected: dict[str, dict[str, Any]] = {}
    for optimizer in OPTIMIZERS:
        candidates = [
            (signature, run)
            for signature, run in finished_by_signature.items()
            if signature[0] == optimizer
        ]
        signature, winner = max(
            candidates,
            key=lambda item: (
                summary_float(item[1], "final/probe_val_acc", "final_probe_val_acc", default=float("-inf")),
                summary_float(item[1], "best/probe_val_acc", "best_probe_val_acc", default=float("-inf")),
                -summary_float(item[1], "best/val_loss", "best_val_loss", default=float("inf")),
                item[1].id,
            ),
        )
        selected[optimizer] = {
            "base_lr": signature[1],
            "actual_lr": signature[1] * BATCH_SIZE / 256.0,
            "weight_decay": signature[2],
            "screen_seed": signature[3],
            "screen_run_id": winner.id,
            "final_probe_val_acc": summary_float(winner, "final/probe_val_acc", "final_probe_val_acc", default=float("nan")),
            "best_probe_val_acc": summary_float(winner, "best/probe_val_acc", "best_probe_val_acc", default=float("nan")),
            "best_val_loss": summary_float(winner, "best/val_loss", "best_val_loss", default=float("nan")),
        }
    return selected


def create_sweep(project_name: str, sweep: dict[str, Any]) -> str:
    sweep_id = wandb.sweep(entity=ENTITY_NAME, project=project_name, sweep=sweep)
    return f"{ENTITY_NAME}/{project_name}/{sweep_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", help="MAE entry point (normally tasks/mae_pretrain.py)")
    parser.add_argument("--stage", choices=("screen", "confirmation"), required=True)
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", default=PROJECT_NAME)
    parser.add_argument("--data_dir", "--data-dir", required=True)
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--source_sweep", "--source-sweep")
    parser.add_argument("--selection_receipt", "--selection-receipt", type=Path)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--warmup_epochs", "--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--probe_every", "--probe-every", type=int, default=PROBE_EVERY)
    parser.add_argument("--source_revision", "--source-revision", default=current_commit())
    args = parser.parse_args()

    if args.stage == "screen":
        if args.source_sweep is not None:
            parser.error("--source_sweep is only valid for confirmation")
        sweep, expected_runs = build_sweep(
            args,
            optimizers=OPTIMIZERS,
            base_lrs=BASE_LRS,
            weight_decays=WEIGHT_DECAYS,
            seeds=SCREEN_SEEDS,
            study_stage="lion_agam_lion_hyperparameter_screen",
        )
        sweep_path = create_sweep(args.project_name, sweep)
        print(f"SWEEP_PATH={sweep_path}")
        print(f"EXPECTED_RUNS={expected_runs}")
        return 0

    if args.source_sweep is None or args.selection_receipt is None:
        parser.error("confirmation requires --source_sweep and --selection_receipt")
    selected = select_screen_winners(args.source_sweep)
    args.selection_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.selection_receipt.write_text(
        json.dumps(
            {
                "comparison_group": COMPARISON_GROUP,
                "source_sweep": args.source_sweep,
                "selection_metric": "final/probe_val_acc",
                "tie_breakers": ("best/probe_val_acc", "best/val_loss"),
                "selected": selected,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"SELECTION_RECEIPT={args.selection_receipt.resolve()}")

    for optimizer in OPTIMIZERS:
        winner = selected[optimizer]
        optimizer_args = argparse.Namespace(**vars(args))
        optimizer_args.sweep_name = f"{args.sweep_name}-{optimizer.lower().replace('_', '-')}"
        optimizer_args.output_dir = str(Path(args.output_dir) / optimizer.lower())
        sweep, expected_runs = build_sweep(
            optimizer_args,
            optimizers=(optimizer,),
            base_lrs=(float(winner["base_lr"]),),
            weight_decays=(float(winner["weight_decay"]),),
            seeds=CONFIRMATION_SEEDS,
            study_stage="lion_agam_lion_paired_seed_confirmation",
            selected_from_sweep=args.source_sweep,
        )
        sweep_path = create_sweep(args.project_name, sweep)
        key = optimizer.upper()
        print(f"SWEEP_PATH_{key}={sweep_path}")
        print(f"EXPECTED_RUNS_{key}={expected_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
