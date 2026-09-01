"""Validate the CIFAR-10 heatmap and select the MAL-SGDM shipping variant.

Selection is predeclared and cell-balanced.  Test accuracy is primary, with a
0.25 percentage-point practical-equivalence band.  A close result is resolved
by typical-cell accuracy, lower-tail robustness, validation AUC, target reach,
then convergence speed.  Only a complete 7 x 7 x 3 grid is eligible.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb

BATCH_SIZES = (64, 128, 256, 512, 1024, 2048, 4096)
LEARNING_RATES = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
SEEDS = (42, 1337, 2026)
PRACTICAL_MARGIN = 0.25
MAL_CASES = {
    "T-Att/U": "False,1.0,False,attenuate,False",
    "T-Rep/U": "False,1.0,False,replace,False",
    "T-Rep/N": "False,1.0,True,replace,False",
}
SIMPLICITY_ORDER = {"T-Rep/U": 0, "T-Att/U": 1, "T-Rep/N": 2}


def scalar(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def label_from_case(value: str) -> str | None:
    fields = value.split("::", maxsplit=2)
    if len(fields) == 3 and fields[0] == "MAL_SGDM":
        return fields[1]
    return None


def bootstrap_interval(values: np.ndarray, *, iterations: int = 20_000) -> tuple[float, float]:
    generator = np.random.default_rng(20260901)
    indices = generator.integers(0, len(values), size=(iterations, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def collect_sweep(sweep_path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    api = wandb.Api(timeout=180)
    sweep = api.sweep(sweep_path)
    rows: list[dict[str, Any]] = []
    state_counts: dict[str, int] = {}
    for run in sweep.runs:
        state_counts[run.state] = state_counts.get(run.state, 0) + 1
        config = dict(run.config)
        summary = dict(run.summary)
        optimizer_case = str(config.get("optimizer_case", ""))
        rows.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "url": run.url,
                "optimizer_case": optimizer_case,
                "variant": label_from_case(optimizer_case),
                "MAL_config": config.get("MAL_config"),
                "batch_size": int(config["batch_size"]),
                "lr": float(config["lr"]),
                "seed": int(config["seed"]),
                "test_acc": scalar(summary.get("test_acc")),
                "best_val_acc": scalar(summary.get("best_val_acc")),
                "val_auc": scalar(summary.get("val_auc", summary.get("AUC"))),
                "epochs_to_target": scalar(summary.get("epochs_2_target"), 201.0),
                "target_reached": int(summary.get("target_reached", 0)),
                "diverged": int(summary.get("diverged", 0)),
            }
        )
    metadata = {
        "sweep_path": sweep_path,
        "sweep_id": sweep.id,
        "sweep_name": sweep.name,
        "sweep_state": sweep.state,
        "run_count": len(rows),
        "state_counts": state_counts,
    }
    return pd.DataFrame(rows), metadata


def validate_grid(frame: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    expected_total = 5 * len(BATCH_SIZES) * len(LEARNING_RATES) * len(SEEDS)
    if metadata["run_count"] != expected_total:
        raise RuntimeError(f"Expected {expected_total} total runs, found {metadata['run_count']}.")
    if metadata["state_counts"] != {"finished": expected_total}:
        raise RuntimeError(f"Sweep is not fully finished: {metadata['state_counts']}.")
    if frame["run_id"].duplicated().any():
        raise RuntimeError("Duplicate W&B run IDs were returned.")

    mal = frame.loc[frame["variant"].notna()].copy()
    expected_keys = set(product(MAL_CASES, BATCH_SIZES, LEARNING_RATES, SEEDS))
    observed_counts = mal.groupby(["variant", "batch_size", "lr", "seed"], observed=True).size()
    observed_keys = set(observed_counts.index)
    missing = expected_keys - observed_keys
    extra = observed_keys - expected_keys
    duplicates = observed_counts.loc[observed_counts != 1]
    if missing or extra or not duplicates.empty:
        raise RuntimeError(
            f"Invalid MAL grid: missing={len(missing)}, extra={len(extra)}, "
            f"non_unit_cells={len(duplicates)}."
        )
    required_metrics = ("test_acc", "best_val_acc", "val_auc")
    nulls = {metric: int(mal[metric].isna().sum()) for metric in required_metrics}
    if any(nulls.values()):
        raise RuntimeError(f"MAL runs have missing required metrics: {nulls}.")
    return mal


def select(mal: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], str, str]:
    cells = (
        mal.groupby(["variant", "batch_size", "lr"], observed=True)
        .agg(
            test_acc=("test_acc", "mean"),
            best_val_acc=("best_val_acc", "mean"),
            val_auc=("val_auc", "mean"),
            target_reach_rate=("target_reached", "mean"),
            epochs_to_target=("epochs_to_target", "mean"),
            divergence_rate=("diverged", "mean"),
        )
        .reset_index()
    )
    summary = (
        cells.groupby("variant", observed=True)
        .agg(
            cells=("test_acc", "size"),
            mean_cell_test_acc=("test_acc", "mean"),
            median_cell_test_acc=("test_acc", "median"),
            q25_cell_test_acc=("test_acc", lambda values: values.quantile(0.25)),
            peak_cell_test_acc=("test_acc", "max"),
            mean_best_val_acc=("best_val_acc", "mean"),
            mean_val_auc=("val_auc", "mean"),
            target_reach_rate=("target_reach_rate", "mean"),
            mean_epochs_to_target=("epochs_to_target", "mean"),
            divergence_rate=("divergence_rate", "mean"),
        )
        .reset_index()
    )
    if set(summary["cells"]) != {49}:
        raise RuntimeError(f"Each MAL variant must have 49 cell means: {summary[['variant', 'cells']].to_dict('records')}")

    comparisons: list[dict[str, Any]] = []
    wide = cells.pivot(index=["batch_size", "lr"], columns="variant", values="test_acc")
    for left, right in combinations(sorted(MAL_CASES), 2):
        differences = (wide[left] - wide[right]).to_numpy()
        low, high = bootstrap_interval(differences)
        comparisons.append(
            {
                "left": left,
                "right": right,
                "mean_test_difference": float(differences.mean()),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "cell_wins": int((differences > 0).sum()),
                "cell_losses": int((differences < 0).sum()),
                "cell_ties": int((differences == 0).sum()),
            }
        )

    best_mean = float(summary["mean_cell_test_acc"].max())
    contenders = summary.loc[summary["mean_cell_test_acc"] >= best_mean - PRACTICAL_MARGIN].copy()
    if len(contenders) == 1:
        winner = str(contenders.iloc[0]["variant"])
        reason = "highest equal-cell mean test accuracy by more than the 0.25 pp equivalence margin"
    else:
        contenders = contenders.sort_values(
            [
                "median_cell_test_acc",
                "q25_cell_test_acc",
                "mean_val_auc",
                "target_reach_rate",
                "mean_epochs_to_target",
            ],
            ascending=[False, False, False, False, True],
        )
        tied = contenders.iloc[0]
        exact = contenders.loc[
            (contenders["median_cell_test_acc"] == tied["median_cell_test_acc"])
            & (contenders["q25_cell_test_acc"] == tied["q25_cell_test_acc"])
            & (contenders["mean_val_auc"] == tied["mean_val_auc"])
            & (contenders["target_reach_rate"] == tied["target_reach_rate"])
            & (contenders["mean_epochs_to_target"] == tied["mean_epochs_to_target"])
        ].copy()
        if len(exact) > 1:
            exact["simplicity_order"] = exact["variant"].map(SIMPLICITY_ORDER)
            winner = str(exact.sort_values("simplicity_order").iloc[0]["variant"])
        else:
            winner = str(tied["variant"])
        reason = (
            "inside the 0.25 pp mean-test equivalence band; selected by median-cell accuracy, "
            "lower-quartile accuracy, validation AUC, target reach, and convergence speed"
        )

    summary["selected"] = summary["variant"].eq(winner)
    summary["MAL_config"] = summary["variant"].map(MAL_CASES)
    return summary.sort_values("mean_cell_test_acc", ascending=False), comparisons, winner, reason


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_path")
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame, metadata = collect_sweep(args.sweep_path)
    mal = validate_grid(frame, metadata)
    summary, comparisons, winner, reason = select(mal)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "runs.csv", index=False)
    summary.to_csv(args.output_dir / "variant_summary.csv", index=False)
    (args.output_dir / "pairwise_bootstrap.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        **metadata,
        "selection_protocol": {
            "unit": "7x7 LR/batch-size cell mean over three seeds",
            "primary": "equal-cell mean test accuracy",
            "practical_equivalence_pp": PRACTICAL_MARGIN,
            "tie_breakers": [
                "median cell test accuracy",
                "25th-percentile cell test accuracy",
                "mean validation AUC",
                "target-reach rate",
                "mean epochs to target",
                "predeclared simplicity order",
            ],
        },
        "winner": winner,
        "winner_config": MAL_CASES[winner],
        "reason": reason,
        "variant_summary": summary.to_dict(orient="records"),
        "pairwise_bootstrap": comparisons,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "selection.env").write_text(
        f"MAL_SGDM_VARIANT={shlex.quote(winner)}\n"
        f"MAL_SGDM_CONFIG={shlex.quote(MAL_CASES[winner])}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
