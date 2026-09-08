"""Validate and rank the completed MAL-AdamW structural screen.

The source experiment is a matched 6 x 2 x 2 x 5 grid. Fitness is computed
within each learning-rate/scheduler/seed block so no regime can dominate merely
because its raw accuracy is larger. The two highest-fitness structures advance
to the recursive-state experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import wandb

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tasks.wandb_metadata import task_metadata

ALIGNMENTS = ("metric", "update", "moment")
GRADIENT_WEIGHT_MODES = ("fixed", "complement")
BASE_LRS = (5e-4, 1e-3)
SCHEDULER_VALUES = (False, True)
SEEDS = (17, 73, 211, 997, 4099)
SOURCE_CONFIGS = tuple(f"False,1.0,none,attenuate,{align},{gradient_weight_mode}" for align in ALIGNMENTS for gradient_weight_mode in GRADIENT_WEIGHT_MODES)
EXPECTED_RUNS = len(SOURCE_CONFIGS) * len(BASE_LRS) * len(SCHEDULER_VALUES) * len(SEEDS)
TOP_K = 2
FITNESS_WEIGHTS = {
    "selection_val_acc": 0.40,
    "val_auc": 0.25,
    "test_acc": 0.25,
    "convergence_speed": 0.10,
}
SOURCE_METADATA = task_metadata(
    task="tiny_imagenet_image_classification",
    task_type="supervised_image_classification",
    model_name="vit_tiny_patch16_224",
    model_source="timm",
    dataset_name="tiny-imagenet-200",
    dataset_config="train_90k_validation_10k_official_validation_test",
    dataset_source="official_tiny_imagenet",
    training_regime="pretrained_finetuning",
)


def scalar(value: Any, *, default: float = math.nan) -> float:
    try:
        result = float(value)
    except TypeError, ValueError:
        return default
    return result if math.isfinite(result) else default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"Expected a boolean W&B value, got {value!r}.")


def parse_structure(raw_config: str) -> tuple[str, str]:
    fields = raw_config.split(",")
    if len(fields) != 6:
        raise ValueError(f"Unexpected MAL_config: {raw_config}")
    in_place, pwr, scale, gate_mode, align, gradient_weight_mode = fields
    if (in_place, pwr, scale, gate_mode) != ("False", "1.0", "none", "attenuate"):
        raise ValueError(f"Source MAL_config is outside the predeclared screen: {raw_config}")
    if align not in ALIGNMENTS or gradient_weight_mode not in GRADIENT_WEIGHT_MODES:
        raise ValueError(f"Unknown source structure: {raw_config}")
    return align, gradient_weight_mode


def rank_scores(values: dict[str, float], *, higher_is_better: bool) -> dict[str, float]:
    """Return tie-aware ranks scaled from zero (worst) to one (best)."""
    if len(values) < 2:
        return {key: 1.0 for key in values}
    scores: dict[str, float] = {}
    for key, value in values.items():
        better_than = sum(value > other if higher_is_better else value < other for other in values.values())
        ties = sum(value == other for other in values.values()) - 1
        scores[key] = (better_than + 0.5 * ties) / (len(values) - 1)
    return scores


def bootstrap_interval(values: np.ndarray, *, iterations: int = 20_000) -> tuple[float, float]:
    """Bootstrap independent seed-level estimates, not correlated grid cells."""
    generator = np.random.default_rng(20260908)
    indices = generator.integers(0, len(values), size=(iterations, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def collect(sweep_path: str, *, backfill_metadata: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sweep = wandb.Api(timeout=180).sweep(sweep_path)
    runs = list(sweep.runs)
    state_counts = Counter(run.state for run in runs)
    if len(runs) != EXPECTED_RUNS or state_counts != Counter({"finished": EXPECTED_RUNS}):
        raise RuntimeError(f"Source sweep must contain exactly {EXPECTED_RUNS} finished runs; found {len(runs)} with states {dict(state_counts)}.")

    rows: list[dict[str, Any]] = []
    for run in runs:
        config = dict(run.config)
        summary = dict(run.summary)
        raw_config = str(config.get("MAL_config", ""))
        align, gradient_weight_mode = parse_structure(raw_config)
        if backfill_metadata and any(config.get(key) != value for key, value in SOURCE_METADATA.items()):
            run.config.update(SOURCE_METADATA)
            run.update()
        diverged = int(summary.get("diverged", 0))
        selection_val_acc = scalar(summary.get("selection_val_acc", summary.get("best_val_acc")))
        if diverged:
            selection_val_acc = 0.0
        rows.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "run_url": run.url,
                "MAL_config": raw_config,
                "structure": f"{align}+{gradient_weight_mode}",
                "align": align,
                "gradient_weight_mode": gradient_weight_mode,
                "base_lr": float(config["base_lr"]),
                "use_scheduler": parse_bool(config["use_scheduler"]),
                "seed": int(config["seed"]),
                "selection_val_acc": selection_val_acc,
                "best_val_acc": scalar(summary.get("best_val_acc")),
                "val_auc": scalar(summary.get("val_auc")),
                "test_acc": scalar(summary.get("test_acc")),
                "best_val_loss": scalar(summary.get("best_val_loss")),
                "epoch_to_target": scalar(summary.get("epoch_to_target"), default=31.0),
                "target_reached": int(summary.get("target_reached", 0)),
                "diverged": diverged,
            }
        )

    expected_cells = set(product(SOURCE_CONFIGS, BASE_LRS, SCHEDULER_VALUES, SEEDS))
    observed = Counter((row["MAL_config"], row["base_lr"], row["use_scheduler"], row["seed"]) for row in rows)
    missing = expected_cells - set(observed)
    extra = set(observed) - expected_cells
    duplicates = {key: count for key, count in observed.items() if count != 1}
    if missing or extra or duplicates:
        raise RuntimeError(f"Source grid mismatch: missing={len(missing)}, extra={len(extra)}, non_unit_cells={len(duplicates)}.")
    required_metrics = ("selection_val_acc", "val_auc", "test_acc", "best_val_loss")
    missing_metrics = {metric: sum(not math.isfinite(float(row[metric])) for row in rows) for metric in required_metrics}
    if any(missing_metrics.values()):
        raise RuntimeError(f"Source runs are missing required metrics: {missing_metrics}")
    metadata = {
        "source_sweep_path": sweep_path,
        "source_sweep_id": sweep.id,
        "source_sweep_name": sweep.name,
        "source_sweep_state": sweep.state,
        "run_count": len(rows),
        "state_counts": dict(state_counts),
    }
    return rows, metadata


def add_fitness(rows: list[dict[str, Any]]) -> None:
    by_cell: dict[tuple[float, bool, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[(row["base_lr"], row["use_scheduler"], row["seed"])].append(row)
    for key, cell_rows in by_cell.items():
        if len(cell_rows) != len(SOURCE_CONFIGS):
            raise RuntimeError(f"Matched cell {key} has {len(cell_rows)} rather than {len(SOURCE_CONFIGS)} runs.")
        structures = [str(row["structure"]) for row in cell_rows]
        metric_scores = {
            "selection_val_acc": rank_scores(
                {structure: float(row["selection_val_acc"]) for structure, row in zip(structures, cell_rows, strict=True)},
                higher_is_better=True,
            ),
            "val_auc": rank_scores(
                {structure: float(row["val_auc"]) for structure, row in zip(structures, cell_rows, strict=True)},
                higher_is_better=True,
            ),
            "test_acc": rank_scores(
                {structure: float(row["test_acc"]) for structure, row in zip(structures, cell_rows, strict=True)},
                higher_is_better=True,
            ),
            "convergence_speed": rank_scores(
                {structure: float(row["epoch_to_target"]) for structure, row in zip(structures, cell_rows, strict=True)},
                higher_is_better=False,
            ),
        }
        for row in cell_rows:
            structure = str(row["structure"])
            row["fitness"] = sum(FITNESS_WEIGHTS[metric] * metric_scores[metric][structure] for metric in FITNESS_WEIGHTS)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["structure"])].append(row)
    summaries: list[dict[str, Any]] = []
    for structure, structure_rows in grouped.items():
        align = str(structure_rows[0]["align"])
        gradient_weight_mode = str(structure_rows[0]["gradient_weight_mode"])
        scheduled_fitness = [float(row["fitness"]) for row in structure_rows if row["use_scheduler"]]
        constant_fitness = [float(row["fitness"]) for row in structure_rows if not row["use_scheduler"]]
        summaries.append(
            {
                "structure": structure,
                "align": align,
                "gradient_weight_mode": gradient_weight_mode,
                "source_MAL_config": structure_rows[0]["MAL_config"],
                "runs": len(structure_rows),
                "mean_fitness": float(np.mean([row["fitness"] for row in structure_rows])),
                "scheduled_fitness": float(np.mean(scheduled_fitness)),
                "constant_fitness": float(np.mean(constant_fitness)),
                "worst_policy_fitness": min(float(np.mean(scheduled_fitness)), float(np.mean(constant_fitness))),
                "mean_selection_val_acc": float(np.mean([row["selection_val_acc"] for row in structure_rows])),
                "mean_val_auc": float(np.mean([row["val_auc"] for row in structure_rows])),
                "mean_test_acc": float(np.mean([row["test_acc"] for row in structure_rows])),
                "mean_best_val_loss": float(np.mean([row["best_val_loss"] for row in structure_rows])),
                "target_reach_rate": float(np.mean([row["target_reached"] for row in structure_rows])),
                "mean_epoch_to_target": float(np.mean([row["epoch_to_target"] for row in structure_rows])),
                "divergence_rate": float(np.mean([row["diverged"] for row in structure_rows])),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            row["divergence_rate"],
            -row["mean_fitness"],
            -row["worst_policy_fitness"],
            -row["mean_selection_val_acc"],
            -row["mean_val_auc"],
            -row["mean_test_acc"],
            row["mean_best_val_loss"],
        ),
    )


def pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(row["structure"], row["base_lr"], row["use_scheduler"], row["seed"]): row for row in rows}
    structures = sorted({str(row["structure"]) for row in rows})
    regimes = tuple(product(BASE_LRS, SCHEDULER_VALUES))
    output: list[dict[str, Any]] = []
    for left, right in combinations(structures, 2):
        record: dict[str, Any] = {"left": left, "right": right}
        for metric in ("fitness", "selection_val_acc", "val_auc", "test_acc"):
            cell_differences = np.asarray(
                [float(index[(left, *regime, seed)][metric]) - float(index[(right, *regime, seed)][metric]) for seed in SEEDS for regime in regimes]
            ).reshape(len(SEEDS), len(regimes))
            seed_differences = cell_differences.mean(axis=1)
            low, high = bootstrap_interval(seed_differences)
            record[f"{metric}_mean_difference"] = float(seed_differences.mean())
            record[f"{metric}_seed_cluster_bootstrap_95_low"] = low
            record[f"{metric}_seed_cluster_bootstrap_95_high"] = high
            record[f"{metric}_matched_cell_wins"] = int((cell_differences > 0).sum())
            record[f"{metric}_matched_cell_losses"] = int((cell_differences < 0).sum())
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table to {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_path")
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    parser.add_argument("--backfill_metadata", "--backfill-metadata", action="store_true")
    args = parser.parse_args()

    rows, metadata = collect(args.sweep_path, backfill_metadata=args.backfill_metadata)
    add_fitness(rows)
    summaries = summarize(rows)
    comparisons = pairwise(rows)
    selected = summaries[:TOP_K]
    selected_configs = [str(row["source_MAL_config"]) for row in selected]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "runs.csv", rows)
    write_csv(args.output_dir / "structure_summary.csv", summaries)
    (args.output_dir / "pairwise_bootstrap.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "selected_configs.txt").write_text(
        "\n".join(selected_configs) + "\n",
        encoding="utf-8",
    )
    result = {
        **metadata,
        "selection_protocol": {
            "matched_unit": "base_lr x scheduler x seed",
            "fitness_weights": FITNESS_WEIGHTS,
            "ranking": "tie-aware within-cell ranks averaged equally over all 20 matched cells",
            "uncertainty": "paired bootstrap over the five seed-level mean differences; LR/scheduler cells stay clustered within seed",
            "stability_priority": "divergence rate precedes mean fitness",
            "selected_count": TOP_K,
            "test_policy": (
                "Official validation-as-test accuracy contributes to development fitness; this task must not subsequently be presented as an untouched final evaluation."
            ),
        },
        "selected": selected,
        "selected_source_configs": selected_configs,
        "structure_summary": summaries,
        "pairwise_bootstrap": comparisons,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
