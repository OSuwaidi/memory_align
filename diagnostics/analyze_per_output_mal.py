"""Validate and combine paired per-output MAL benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean value: {value}")


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for key in (
                "adaptation_rate",
                "learning_rate",
                "best_validation_accuracy",
                "final_validation_accuracy",
                "selected_test_accuracy",
                "mean_validation_accuracy",
                "final_validation_loss",
                "mean_validation_loss",
                "elapsed_seconds",
                "beta_mean",
                "beta_std",
                "mean_within_tensor_beta_std",
                "pct_beta_below_0_5",
                "pct_beta_above_0_9",
            ):
                row[key] = float(row[key])
            for key in ("seed", "best_epoch", "steps"):
                row[key] = int(row[key])
            row["diverged"] = parse_bool(row["diverged"])
            rows.append(row)
    return rows


def t_critical_95(degrees_of_freedom: int) -> float:
    if degrees_of_freedom in T_CRITICAL_95:
        return T_CRITICAL_95[degrees_of_freedom]
    lower = max(key for key in T_CRITICAL_95 if key < degrees_of_freedom)
    upper_candidates = [key for key in T_CRITICAL_95 if key > degrees_of_freedom]
    if not upper_candidates:
        return 1.96
    upper = min(upper_candidates)
    fraction = (degrees_of_freedom - lower) / (upper - lower)
    return T_CRITICAL_95[lower] + fraction * (T_CRITICAL_95[upper] - T_CRITICAL_95[lower])


def describe(values: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    if len(values) > 1:
        half_width = t_critical_95(len(values) - 1) * std / math.sqrt(len(values))
        confidence_interval: list[float | None] = [
            mean - half_width,
            mean + half_width,
        ]
    else:
        confidence_interval = [None, None]
    return {
        "n": len(values),
        "mean": mean,
        "sample_std": std,
        "min": min(values),
        "max": max(values),
        "t_95_ci": confidence_interval,
    }


def validate_and_pair(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    issues = []
    indexed = {}
    for row in rows:
        key = (
            row["condition"],
            row["adaptation_rate"],
            row["learning_rate"],
            row["seed"],
            row["scope"],
        )
        if key in indexed:
            issues.append(f"duplicate run {key}")
        indexed[key] = row
        numeric_values = [value for value in row.values() if isinstance(value, float)]
        if not all(math.isfinite(value) for value in numeric_values):
            issues.append(f"non-finite metric in {key}")

    pairs = []
    base_keys = {key[:-1] for key in indexed}
    for key in sorted(base_keys):
        tensor = indexed.get((*key, "tensor"))
        output = indexed.get((*key, "output"))
        if tensor is None or output is None:
            issues.append(f"incomplete scope pair {key}")
            continue
        if tensor["initial_state_hash"] != output["initial_state_hash"]:
            issues.append(f"initial-state mismatch {key}")
        if tensor["epoch_order_hash"] != output["epoch_order_hash"]:
            issues.append(f"epoch-order mismatch {key}")
        if tensor["steps"] != output["steps"]:
            issues.append(f"optimizer-step mismatch {key}")
        pairs.append(
            {
                "condition": key[0],
                "adaptation_rate": key[1],
                "learning_rate": key[2],
                "seed": key[3],
                "delta_best_validation_accuracy": output["best_validation_accuracy"] - tensor["best_validation_accuracy"],
                "delta_selected_test_accuracy": output["selected_test_accuracy"] - tensor["selected_test_accuracy"],
                "delta_mean_validation_accuracy": output["mean_validation_accuracy"] - tensor["mean_validation_accuracy"],
                "elapsed_ratio": output["elapsed_seconds"] / tensor["elapsed_seconds"],
            }
        )
    return pairs, issues


def summarize(
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    cells: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        cells[
            (
                pair["condition"],
                pair["adaptation_rate"],
                pair["learning_rate"],
            )
        ].append(pair)

    cell_summary = {}
    for key, cell_pairs in sorted(cells.items()):
        name = f"{key[0]}|c={key[1]}|lr={key[2]}"
        cell_summary[name] = {
            metric: describe([pair[metric] for pair in cell_pairs])
            for metric in (
                "delta_best_validation_accuracy",
                "delta_selected_test_accuracy",
                "delta_mean_validation_accuracy",
                "elapsed_ratio",
            )
        }
        cell_summary[name]["output_win_count"] = sum(pair["delta_best_validation_accuracy"] > 0.0 for pair in cell_pairs)

    by_condition_rate_seed: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_condition_rate_seed[(pair["condition"], pair["adaptation_rate"], pair["seed"])].append(pair)
    clustered: dict[tuple[str, float], list[dict[str, float]]] = defaultdict(list)
    for (condition, rate, _seed), seed_pairs in by_condition_rate_seed.items():
        clustered[(condition, rate)].append(
            {
                metric: statistics.fmean(pair[metric] for pair in seed_pairs)
                for metric in (
                    "delta_best_validation_accuracy",
                    "delta_selected_test_accuracy",
                    "delta_mean_validation_accuracy",
                )
            }
        )

    seed_clustered_summary = {}
    for (condition, rate), seed_rows in sorted(clustered.items()):
        name = f"{condition}|c={rate}"
        seed_clustered_summary[name] = {metric: describe([row[metric] for row in seed_rows]) for metric in seed_rows[0]}

    by_condition_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_condition_seed[(pair["condition"], pair["seed"])].append(pair)
    condition_seed_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (condition, _seed), seed_pairs in by_condition_seed.items():
        condition_seed_rows[condition].append(
            {
                metric: statistics.fmean(pair[metric] for pair in seed_pairs)
                for metric in (
                    "delta_best_validation_accuracy",
                    "delta_selected_test_accuracy",
                    "delta_mean_validation_accuracy",
                )
            }
        )
    condition_summary = {}
    for condition, seed_rows in sorted(condition_seed_rows.items()):
        condition_pairs = [pair for pair in pairs if pair["condition"] == condition]
        condition_summary[condition] = {
            metric: describe([row[metric] for row in seed_rows])
            for metric in seed_rows[0]
        }
        condition_summary[condition].update(
            {
                "pair_count": len(condition_pairs),
                "output_win_count": sum(
                    pair["delta_best_validation_accuracy"] > 0.0
                    for pair in condition_pairs
                ),
                "tie_count": sum(
                    pair["delta_best_validation_accuracy"] == 0.0
                    for pair in condition_pairs
                ),
            }
        )

    scopes: dict[tuple[str, float, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scopes[(row["condition"], row["adaptation_rate"], row["seed"], row["scope"])].append(row)
    worst_lr_pairs: dict[tuple[str, float, int], dict[str, float]] = defaultdict(dict)
    for (condition, rate, seed, scope), scope_rows in scopes.items():
        worst_lr_pairs[(condition, rate, seed)][scope] = min(row["best_validation_accuracy"] for row in scope_rows)
    worst_lr_summary = {}
    grouped_worst_deltas: dict[tuple[str, float], list[float]] = defaultdict(list)
    for (condition, rate, _seed), scope_values in worst_lr_pairs.items():
        if set(scope_values) == {"tensor", "output"}:
            grouped_worst_deltas[(condition, rate)].append(scope_values["output"] - scope_values["tensor"])
    for (condition, rate), values in sorted(grouped_worst_deltas.items()):
        worst_lr_summary[f"{condition}|c={rate}"] = describe(values)

    microbenchmark_ratios = [item["optimizer_microbenchmark"]["output_over_tensor_step_time_ratio"] for item in metadata]
    output_rows = [row for row in rows if row["scope"] == "output"]
    telemetry_by_rate: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        telemetry_by_rate[row["adaptation_rate"]].append(row)

    return {
        "run_count": len(rows),
        "pair_count": len(pairs),
        "cell_summary": cell_summary,
        "condition_seed_clustered_summary": condition_summary,
        "seed_clustered_summary": seed_clustered_summary,
        "worst_lr_delta_best_validation": worst_lr_summary,
        "optimizer_microbenchmark_ratio": describe(microbenchmark_ratios),
        "per_output_beta_telemetry": {
            f"c={rate}": {
                "mean_beta": describe([row["beta_mean"] for row in rate_rows]),
                "within_tensor_beta_std": describe([row["mean_within_tensor_beta_std"] for row in rate_rows]),
                "pct_beta_below_0_5": describe(
                    [row["pct_beta_below_0_5"] for row in rate_rows]
                ),
                "pct_beta_above_0_9": describe(
                    [row["pct_beta_above_0_9"] for row in rate_rows]
                ),
            }
            for rate, rate_rows in sorted(telemetry_by_rate.items())
        },
    }


def markdown_report(summary: dict[str, Any], issues: list[str]) -> str:
    lines = [
        "# Per-output MAL validation summary",
        "",
        f"- Runs: {summary['run_count']} ({summary['pair_count']} exact scope pairs)",
        f"- Pairing/data-integrity issues: {len(issues)}",
        "- Deltas are `per-output - per-tensor` in percentage points.",
        "- Cell entries are paired means; condition-level 95% intervals cluster the hyperparameter grid within seed.",
        "",
        "| condition | c | lr | best-val delta | selected-test delta | output wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, cell in summary["cell_summary"].items():
        condition, rate, learning_rate = name.split("|")
        best = cell["delta_best_validation_accuracy"]
        test = cell["delta_selected_test_accuracy"]
        lines.append(
            f"| {condition} | {rate.removeprefix('c=')} | "
            f"{learning_rate.removeprefix('lr=')} | {best['mean']:+.2f} | "
            f"{test['mean']:+.2f} | {cell['output_win_count']}/{best['n']} |"
        )
    lines.extend(
        [
            "",
            "## Grid-averaged result by condition",
            "",
            "| condition | seeds | best-val delta (95% CI) | selected-test delta (95% CI) | output wins |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition, result in summary["condition_seed_clustered_summary"].items():
        best = result["delta_best_validation_accuracy"]
        test = result["delta_selected_test_accuracy"]
        best_ci = best["t_95_ci"]
        test_ci = test["t_95_ci"]
        lines.append(
            f"| {condition} | {best['n']} | {best['mean']:+.2f} "
            f"[{best_ci[0]:+.2f}, {best_ci[1]:+.2f}] | "
            f"{test['mean']:+.2f} [{test_ci[0]:+.2f}, {test_ci[1]:+.2f}] | "
            f"{result['output_win_count']}/{result['pair_count']} "
            f"({result['tie_count']} ties) |"
        )
    runtime = summary["optimizer_microbenchmark_ratio"]
    lines.extend(
        [
            "",
            f"Optimizer-only output/tensor runtime ratio across suites: {runtime['mean']:.3f} (range {runtime['min']:.3f}–{runtime['max']:.3f}).",
        ]
    )
    if issues:
        lines.extend(("", "## Validation issues", ""))
        lines.extend(f"- {issue}" for issue in issues)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    metadata = []
    for directory in args.result_dirs:
        rows.extend(read_rows(directory / "run_metrics.csv"))
        metadata.append(json.loads((directory / "metadata_and_summary.json").read_text()))
    pairs, issues = validate_and_pair(rows)
    summary = summarize(rows, pairs, metadata)
    payload = {
        "validation_status": "pass" if not issues else "fail",
        "issues": issues,
        "summary": summary,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_markdown.write_text(
        markdown_report(summary, issues),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
