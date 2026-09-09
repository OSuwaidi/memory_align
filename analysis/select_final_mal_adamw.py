"""Select MAL-AdamW after the scheduled recursive-state confirmation.

The source sweep supplies two transient (``in_place=False``) finalists.  The
scheduled follow-up supplies their recursive-state counterparts at pwr 0.5 and
1.0.  A recursive candidate advances to the scheduler-free stress test only if
it wins the matched scheduled comparison.  When that stress test is supplied,
the final configuration is selected over both scheduler policies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import wandb

BASE_LRS = (5e-4, 1e-3)
SEEDS = (17, 73, 211, 997, 4099)
FITNESS_WEIGHTS = {
    "selection_val_acc": 0.40,
    "val_auc": 0.25,
    "test_acc": 0.25,
    "convergence_speed": 0.10,
}


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
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def scalar(value: Any, *, default: float = math.nan) -> float:
    try:
        result = float(value)
    except TypeError, ValueError:
        return default
    return result if math.isfinite(result) else default


def read_source_configs(path: Path) -> tuple[str, str]:
    configs = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(configs) != 2 or len(set(configs)) != 2:
        raise ValueError("The source selection must contain exactly two distinct configurations.")
    for config in configs:
        fields = config.split(",")
        if len(fields) != 6 or fields[:4] != ["False", "1.0", "none", "attenuate"]:
            raise ValueError(f"Unexpected source configuration: {config}")
    return configs


def recursive_configs(source_configs: tuple[str, str]) -> tuple[str, ...]:
    output: list[str] = []
    for config in source_configs:
        fields = config.split(",")
        output.extend(f"True,{pwr},none,attenuate,{fields[4]},{fields[5]}" for pwr in ("0.5", "1.0"))
    return tuple(output)


def collect_sweep(
    sweep_path: str,
    *,
    allowed_configs: tuple[str, ...],
    scheduler_values: tuple[bool, ...],
    allow_extra_configs: bool = False,
) -> list[dict[str, Any]]:
    sweep = wandb.Api(timeout=180).sweep(sweep_path)
    all_runs = list(sweep.runs)
    runs = [run for run in all_runs if str(dict(run.config).get("MAL_config", "")) in allowed_configs]
    expected = len(allowed_configs) * len(BASE_LRS) * len(scheduler_values) * len(SEEDS)
    states = Counter(run.state for run in runs)
    if len(runs) != expected or states != Counter({"finished": expected}):
        raise RuntimeError(f"{sweep_path} must have {expected} finished runs; found {len(runs)} with {dict(states)}.")
    if not allow_extra_configs and len(all_runs) != len(runs):
        raise RuntimeError(f"{sweep_path} contains {len(all_runs) - len(runs)} runs outside the expected configurations.")

    rows: list[dict[str, Any]] = []
    for run in runs:
        config = dict(run.config)
        summary = dict(run.summary)
        raw_config = str(config.get("MAL_config", ""))
        if raw_config not in allowed_configs:
            raise RuntimeError(f"Run {run.id} has an unexpected MAL_config: {raw_config}")
        use_scheduler = parse_bool(config["use_scheduler"])
        diverged = int(summary.get("diverged", 0))
        row = {
            "sweep_path": sweep_path,
            "run_id": run.id,
            "run_name": run.name,
            "MAL_config": raw_config,
            "in_place": raw_config.startswith("True,"),
            "pwr": float(raw_config.split(",")[1]),
            "align": raw_config.split(",")[4],
            "gradient_weight_mode": raw_config.split(",")[5],
            "base_lr": float(config["base_lr"]),
            "use_scheduler": use_scheduler,
            "seed": int(config["seed"]),
            "selection_val_acc": 0.0 if diverged else scalar(summary.get("selection_val_acc", summary.get("best_val_acc"))),
            "val_auc": scalar(summary.get("val_auc"), default=0.0) if not diverged else 0.0,
            "test_acc": scalar(summary.get("test_acc_at_best_val", summary.get("test_acc")), default=0.0) if not diverged else 0.0,
            "epoch_to_target": scalar(summary.get("epoch_to_target"), default=31.0),
            "diverged": diverged,
        }
        rows.append(row)

    expected_cells = set(product(allowed_configs, BASE_LRS, scheduler_values, SEEDS))
    observed = Counter((row["MAL_config"], row["base_lr"], row["use_scheduler"], row["seed"]) for row in rows)
    if set(observed) != expected_cells or any(count != 1 for count in observed.values()):
        raise RuntimeError(f"Sweep grid mismatch for {sweep_path}.")
    return rows


def rank_scores(values: dict[str, float], *, higher_is_better: bool) -> dict[str, float]:
    if len(values) == 1:
        return {key: 1.0 for key in values}
    output: dict[str, float] = {}
    for key, value in values.items():
        better_than = sum(value > other if higher_is_better else value < other for other in values.values())
        ties = sum(value == other for other in values.values()) - 1
        output[key] = (better_than + 0.5 * ties) / (len(values) - 1)
    return output


def add_fitness(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[float, bool, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["base_lr"], row["use_scheduler"], row["seed"])].append(row)
    candidate_count = len({row["MAL_config"] for row in rows})
    for key, cell in groups.items():
        if len(cell) != candidate_count:
            raise RuntimeError(f"Matched cell {key} has {len(cell)} of {candidate_count} candidates.")
        scores = {
            "selection_val_acc": rank_scores({row["MAL_config"]: row["selection_val_acc"] for row in cell}, higher_is_better=True),
            "val_auc": rank_scores({row["MAL_config"]: row["val_auc"] for row in cell}, higher_is_better=True),
            "test_acc": rank_scores({row["MAL_config"]: row["test_acc"] for row in cell}, higher_is_better=True),
            "convergence_speed": rank_scores({row["MAL_config"]: row["epoch_to_target"] for row in cell}, higher_is_better=False),
        }
        for row in cell:
            config = row["MAL_config"]
            row["fitness"] = sum(FITNESS_WEIGHTS[metric] * scores[metric][config] for metric in FITNESS_WEIGHTS)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["MAL_config"]].append(row)
    output: list[dict[str, Any]] = []
    for config, group in groups.items():
        output.append(
            {
                "MAL_config": config,
                "in_place": group[0]["in_place"],
                "pwr": group[0]["pwr"],
                "align": group[0]["align"],
                "gradient_weight_mode": group[0]["gradient_weight_mode"],
                "runs": len(group),
                "mean_fitness": float(np.mean([row["fitness"] for row in group])),
                "mean_selection_val_acc": float(np.mean([row["selection_val_acc"] for row in group])),
                "mean_val_auc": float(np.mean([row["val_auc"] for row in group])),
                "mean_test_acc": float(np.mean([row["test_acc"] for row in group])),
                "mean_epoch_to_target": float(np.mean([row["epoch_to_target"] for row in group])),
                "divergence_rate": float(np.mean([row["diverged"] for row in group])),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            row["divergence_rate"],
            -row["mean_fitness"],
            -row["mean_selection_val_acc"],
            -row["mean_val_auc"],
            -row["mean_test_acc"],
            row["mean_epoch_to_target"],
        ),
    )


def bootstrap_pair(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    index = {(row["MAL_config"], row["base_lr"], row["seed"]): row for row in rows}
    generator = np.random.default_rng(20260909)
    output: dict[str, Any] = {"left": left, "right": right}
    for metric in ("fitness", "selection_val_acc", "val_auc", "test_acc"):
        differences = np.asarray([[index[(left, lr, seed)][metric] - index[(right, lr, seed)][metric] for lr in BASE_LRS] for seed in SEEDS])
        seed_means = differences.mean(axis=1)
        indices = generator.integers(0, len(SEEDS), size=(20_000, len(SEEDS)))
        bootstrap_means = seed_means[indices].mean(axis=1)
        low, high = np.quantile(bootstrap_means, (0.025, 0.975))
        output[f"{metric}_mean_difference"] = float(seed_means.mean())
        output[f"{metric}_seed_cluster_bootstrap_95_low"] = float(low)
        output[f"{metric}_seed_cluster_bootstrap_95_high"] = float(high)
        output[f"{metric}_matched_cell_wins"] = int((differences > 0).sum())
        output[f"{metric}_matched_cell_losses"] = int((differences < 0).sum())
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_sweep_path")
    parser.add_argument("scheduled_sweep_path")
    parser.add_argument("--source_selection_file", "--source-selection-file", type=Path, required=True)
    parser.add_argument("--unscheduled_sweep_path", "--unscheduled-sweep-path")
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_configs = read_source_configs(args.source_selection_file)
    recursive = recursive_configs(source_configs)
    source_rows = collect_sweep(
        args.source_sweep_path,
        allowed_configs=source_configs,
        scheduler_values=(False, True),
        allow_extra_configs=True,
    )
    scheduled_rows = [row for row in source_rows if row["use_scheduler"]]
    scheduled_rows.extend(collect_sweep(args.scheduled_sweep_path, allowed_configs=recursive, scheduler_values=(True,)))
    add_fitness(scheduled_rows)
    scheduled_summary = summarize(scheduled_rows)
    best_out_of_place = next(row for row in scheduled_summary if not row["in_place"])
    best_in_place = next(row for row in scheduled_summary if row["in_place"])
    scheduled_pair = bootstrap_pair(scheduled_rows, best_in_place["MAL_config"], best_out_of_place["MAL_config"])
    positive_core_metrics = sum(scheduled_pair[f"{metric}_mean_difference"] > 0.0 for metric in ("selection_val_acc", "val_auc", "test_acc"))
    run_unscheduled = bool(
        best_in_place["divergence_rate"] == 0.0
        and scheduled_pair["fitness_mean_difference"] > 0.0
        and scheduled_pair["fitness_matched_cell_wins"] >= 6
        and positive_core_metrics >= 2
    )

    final_rows: list[dict[str, Any]] | None = None
    final_summary: list[dict[str, Any]] | None = None
    final_pair: dict[str, Any] | None = None
    if args.unscheduled_sweep_path:
        if not run_unscheduled:
            raise RuntimeError("An unscheduled sweep was supplied even though recursive state did not pass the scheduled gate.")
        candidate = str(best_in_place["MAL_config"])
        unscheduled_candidate = collect_sweep(
            args.unscheduled_sweep_path,
            allowed_configs=(candidate,),
            scheduler_values=(False,),
        )
        final_rows = [row for row in source_rows if row["MAL_config"] in source_configs]
        final_rows.extend(row for row in scheduled_rows if row["in_place"] and row["MAL_config"] == candidate)
        final_rows.extend(unscheduled_candidate)
        add_fitness(final_rows)
        final_summary = summarize(final_rows)
        best_final_out_of_place = next(row for row in final_summary if not row["in_place"])
        final_candidate = next(row for row in final_summary if row["in_place"])
        final_pair = {
            "left": final_candidate["MAL_config"],
            "right": best_final_out_of_place["MAL_config"],
            "fitness_mean_difference": final_candidate["mean_fitness"] - best_final_out_of_place["mean_fitness"],
            "selection_val_acc_mean_difference": final_candidate["mean_selection_val_acc"] - best_final_out_of_place["mean_selection_val_acc"],
            "val_auc_mean_difference": final_candidate["mean_val_auc"] - best_final_out_of_place["mean_val_auc"],
            "test_acc_mean_difference": final_candidate["mean_test_acc"] - best_final_out_of_place["mean_test_acc"],
        }
        final_config = str(final_summary[0]["MAL_config"])
    elif run_unscheduled:
        final_config = ""
    else:
        final_config = str(best_out_of_place["MAL_config"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "scheduled_runs.csv", scheduled_rows)
    write_csv(args.output_dir / "scheduled_summary.csv", scheduled_summary)
    if final_rows is not None and final_summary is not None:
        write_csv(args.output_dir / "final_runs.csv", final_rows)
        write_csv(args.output_dir / "final_summary.csv", final_summary)
    (args.output_dir / "unscheduled_candidate.txt").write_text(
        f"{best_in_place['MAL_config']}\n" if run_unscheduled else "",
        encoding="utf-8",
    )
    if final_config:
        (args.output_dir / "final_config.txt").write_text(f"{final_config}\n", encoding="utf-8")
    decision = {
        "source_sweep_path": args.source_sweep_path,
        "scheduled_sweep_path": args.scheduled_sweep_path,
        "unscheduled_sweep_path": args.unscheduled_sweep_path,
        "fitness_weights": FITNESS_WEIGHTS,
        "scheduled_gate": {
            "rule": (
                "zero divergence; positive mean matched fitness; at least 6/10 matched fitness wins; "
                "positive mean differences on at least two of validation accuracy, validation AUC, and test accuracy"
            ),
            "run_unscheduled": run_unscheduled,
            "best_out_of_place": best_out_of_place,
            "best_in_place": best_in_place,
            "comparison": scheduled_pair,
        },
        "scheduled_summary": scheduled_summary,
        "final_summary": final_summary,
        "final_comparison": final_pair,
        "final_config": final_config or None,
        "test_policy": (
            "The official Tiny-ImageNet validation partition is treated as test and contributes to development selection; "
            "the result is optimizer-configuration evidence, not an untouched final test estimate."
        ),
    }
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "decision.env").write_text(
        f"RUN_UNSCHEDULED={int(run_unscheduled)}\nUNSCHEDULED_CANDIDATE={shlex.quote(str(best_in_place['MAL_config']))}\nFINAL_MAL_CONFIG={shlex.quote(final_config)}\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
