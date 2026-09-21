"""Create a fixed-recipe MAE fine-tune sweep from explicit selected checkpoints."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import wandb

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sweeps.mae_lion_finetune_confirmation_sweep import (
    ENTITY_NAME,
    PROJECT_NAME,
    build_sweep,
    current_commit,
    final_probe_accuracy,
)
from tasks.mae_finetune_eval import SUPPORTED_SOURCE_OPTIMIZERS

EXPECTED_SEEDS = (42, 1337, 2026)


def structure_label(config: dict[str, Any]) -> str:
    for key in ("MAL_config", "optimizer_config", "AdaMAL_config"):
        value = config.get(key)
        if value not in (None, ""):
            return str(value)
    return "base"


def validate_sources(
    api: wandb.Api,
    *,
    source_run_ids: list[str],
    expected_optimizers: tuple[str, ...],
    expected_seeds: tuple[int, ...],
    expected_batch_size: int,
    project_name: str,
) -> list[Any]:
    if len(source_run_ids) != len(set(source_run_ids)):
        raise ValueError("Every source run id must be distinct.")
    sources = [api.run(f"{ENTITY_NAME}/{project_name}/{run_id}") for run_id in source_run_ids]
    grouped: dict[str, list[Any]] = defaultdict(list)
    for run in sources:
        config = dict(run.config)
        optimizer = str(config.get("optimizer"))
        if optimizer not in SUPPORTED_SOURCE_OPTIMIZERS:
            raise ValueError(f"Unsupported source optimizer {optimizer!r} in run {run.id}.")
        if run.state != "finished":
            raise ValueError(f"Source run {run.id} is not finished: {run.state}.")
        if str(config.get("task")) != "tiny_imagenet_mae_pretraining":
            raise ValueError(f"Source run {run.id} is not a Tiny-ImageNet MAE pretraining run.")
        if int(config.get("epochs", 0)) != 300:
            raise ValueError(f"Source run {run.id} was not trained for 300 epochs.")
        observed_batch_size = int(config.get("batch_size", 0))
        if observed_batch_size != expected_batch_size:
            raise ValueError(
                f"Source run {run.id} has pretraining batch size {observed_batch_size}; "
                f"the comparison requires batch size {expected_batch_size}."
            )
        if int(config.get("image_size", 0)) != 64 or int(config.get("patch_size", 0)) != 8:
            raise ValueError(f"Source run {run.id} does not use the common 64px/patch-8 protocol.")
        checkpoint = Path(str(run.summary.get("checkpoint", ""))).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Source checkpoint is unavailable for run {run.id}: {checkpoint}")
        if int(run.summary.get("epoch", 0)) != 300:
            raise ValueError(f"Source run {run.id} did not finish at epoch 300.")
        final_probe_accuracy(run)
        grouped[optimizer].append(run)

    if tuple(sorted(grouped)) != tuple(sorted(expected_optimizers)):
        raise ValueError(
            f"Observed optimizers {sorted(grouped)} do not match expected {sorted(expected_optimizers)}."
        )

    selected: list[Any] = []
    for optimizer in sorted(grouped):
        runs = grouped[optimizer]
        seeds = tuple(sorted(int(run.config["seed"]) for run in runs))
        if seeds != expected_seeds:
            raise ValueError(f"{optimizer} has seeds {seeds}; expected {expected_seeds}.")
        fingerprints = {
            (
                int(run.config["batch_size"]),
                float(run.config["base_lr"]),
                float(run.config["weight_decay"]),
                structure_label(dict(run.config)),
            )
            for run in runs
        }
        if len(fingerprints) != 1:
            raise ValueError(f"{optimizer} sources do not share one selected configuration: {fingerprints}.")
        accuracies = [final_probe_accuracy(run) for run in runs]
        fingerprint = next(iter(fingerprints))
        print(
            f"SELECTED_{optimizer}="
            f"batch_size={fingerprint[0]},base_lr={fingerprint[1]},weight_decay={fingerprint[2]},"
            f"structure={fingerprint[3]},mean_final_probe={statistics.fmean(accuracies):.6f},"
            f"min_final_probe={min(accuracies):.6f},std_final_probe={statistics.pstdev(accuracies):.6f}"
        )
        selected.extend(sorted(runs, key=lambda run: int(run.config["seed"])))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", help="Fine-tune entry point (normally tasks/mae_finetune_eval.py)")
    parser.add_argument("--source_run_id", "--source-run-id", action="append", required=True)
    parser.add_argument("--expected_optimizer", "--expected-optimizer", action="append", required=True)
    parser.add_argument("--expected_seeds", "--expected-seeds", type=int, nargs="+", default=EXPECTED_SEEDS)
    parser.add_argument("--expected_batch_size", "--expected-batch-size", type=int, default=1024)
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", default=PROJECT_NAME)
    parser.add_argument("--data_dir", "--data-dir", required=True)
    parser.add_argument("--source_revision", "--source-revision", default=current_commit())
    parser.add_argument(
        "--comparison_group",
        "--comparison-group",
        default="mae_selected_optimizer_end_to_end_finetune_v1",
    )
    parser.add_argument(
        "--study_stage",
        "--study-stage",
        default="selected_checkpoint_end_to_end_finetune",
    )
    args = parser.parse_args()

    expected_seeds = tuple(sorted(args.expected_seeds))
    if len(expected_seeds) != len(set(expected_seeds)):
        parser.error("--expected_seeds must contain distinct values.")
    if args.expected_batch_size <= 0:
        parser.error("--expected_batch_size must be positive.")
    expected_optimizers = tuple(sorted(set(args.expected_optimizer)))
    expected_runs = len(expected_seeds) * len(expected_optimizers)
    if len(args.source_run_id) != expected_runs:
        parser.error(f"Expected {expected_runs} source runs, received {len(args.source_run_id)}.")

    sources = validate_sources(
        wandb.Api(timeout=180),
        source_run_ids=args.source_run_id,
        expected_optimizers=expected_optimizers,
        expected_seeds=expected_seeds,
        expected_batch_size=args.expected_batch_size,
        project_name=args.project_name,
    )
    print(f"EXPECTED_PRETRAIN_BATCH_SIZE={args.expected_batch_size}")
    source_ids = [run.id for run in sources]
    print(f"SOURCE_RUN_IDS={','.join(source_ids)}")
    sweep_id = wandb.sweep(
        entity=ENTITY_NAME,
        project=args.project_name,
        sweep=build_sweep(args, source_ids),
    )
    print(f"SWEEP_PATH={ENTITY_NAME}/{args.project_name}/{sweep_id}")
    print(f"EXPECTED_RUNS={expected_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
