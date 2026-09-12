"""Analyze matched CIFAR scheduler and TAM fixed-gate ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import wandb

ENTITY = "osuwaidi-khalifa-university"
PROJECT = "MAL_benchmark"
SEEDS = (42, 1337, 2026)
METRICS = (
    "best_val_acc",
    "val_auc",
    "epochs_to_target",
    "test_acc_at_best_val",
    "test_acc_at_final_epoch",
)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as bool")


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def first(summary: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = finite(summary.get(name))
        if value is not None:
            return value
    return None


def canonical_path(raw: str) -> str:
    path = raw if raw.count("/") == 2 else f"{ENTITY}/{PROJECT}/{raw}"
    if not path.startswith(f"{ENTITY}/{PROJECT}/"):
        raise ValueError(f"Unexpected W&B path {path!r}")
    return path


def canonical_mal(value: Any) -> str:
    fields = [field.strip() for field in str(value).split(",")]
    # Historical SGDM runs inserted the removed safeguard flag as field five.
    return ",".join(fields[:4])


def load_runs(api: wandb.Api, paths: list[str]) -> list[tuple[str, Any]]:
    loaded: list[tuple[str, Any]] = []
    for raw in paths:
        path = canonical_path(raw)
        entity, project, sweep_id = path.split("/")
        loaded.extend(
            (path, run)
            for run in api.runs(
                f"{entity}/{project}",
                filters={"sweep": sweep_id},
                per_page=1_000,
                lazy=False,
            )
        )
    return loaded


def collect(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    api = wandb.Api(timeout=180)
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    roles = ((True, args.scheduled_sweeps), (False, args.unscheduled_sweeps))
    for expected_scheduler, paths in roles:
        for path, run in load_runs(api, paths):
            config = dict(run.config)
            optimizer = str(config.get("optimizer", ""))
            reason = None
            try:
                scheduled = as_bool(config.get("use_scheduler"))
            except ValueError:
                scheduled = not expected_scheduler
                reason = "invalid scheduler flag"
            if run.state != "finished":
                reason = f"state={run.state}"
            elif optimizer not in args.optimizers:
                reason = "optimizer outside analysis set"
            elif scheduled is not expected_scheduler:
                reason = "wrong scheduler role"
            elif finite(config.get("batch_size")) != args.batch_size:
                reason = "wrong batch size"
            elif finite(config.get("lr")) != args.learning_rate:
                reason = "wrong learning rate"
            elif finite(config.get("weight_decay")) != args.weight_decay:
                reason = "wrong weight decay"
            elif optimizer == "MAL_SGDM" and canonical_mal(config.get("MAL_config")) != canonical_mal(args.mal_config):
                reason = "non-selected MAL structure"
            if reason:
                excluded.append({"sweep_path": path, "run_id": run.id, "optimizer": optimizer, "reason": reason})
                continue

            summary = dict(run.summary)
            row: dict[str, Any] = {
                "sweep_path": path,
                "run_id": run.id,
                "optimizer": optimizer,
                "use_scheduler": scheduled,
                "seed": int(config["seed"]),
                "best_val_acc": first(summary, "best_val_acc", "best/val_acc"),
                "val_auc": first(summary, "val_auc", "val/auc", "AUC"),
                "epochs_to_target": first(summary, "epochs_2_target"),
                "test_acc_at_best_val": first(summary, "test_acc_at_best_val", "test/acc_at_best_val", "test_acc"),
                "test_acc_at_final_epoch": first(summary, "test_acc_at_final_epoch", "test/acc_at_final_epoch"),
            }
            required = ("best_val_acc", "val_auc", "test_acc_at_best_val")
            missing_metrics = [name for name in required if row[name] is None]
            if missing_metrics:
                raise RuntimeError(f"{path}/{run.id} lacks {missing_metrics}")
            selected.append(row)

    expected = {
        (optimizer, scheduled, seed)
        for optimizer in args.optimizers
        for scheduled in (True, False)
        for seed in SEEDS
    }
    observed: dict[tuple[str, bool, int], str] = {}
    for row in selected:
        key = (row["optimizer"], row["use_scheduler"], row["seed"])
        if key in observed:
            raise RuntimeError(f"Duplicate {key}: {observed[key]} and {row['run_id']}")
        observed[key] = row["run_id"]
    if set(observed) != expected:
        raise RuntimeError(
            f"Paired grid mismatch; missing={sorted(expected - set(observed))}, "
            f"extra={sorted(set(observed) - expected)}"
        )
    return selected, excluded


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate(rows: list[dict[str, Any]], optimizers: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["optimizer"], row["use_scheduler"])].append(row)
    output: list[dict[str, Any]] = []
    for (optimizer, scheduled), group in groups.items():
        result: dict[str, Any] = {"optimizer": optimizer, "use_scheduler": scheduled, "n": len(group)}
        for metric in METRICS:
            values = [float(row[metric]) for row in group if row[metric] is not None]
            result[f"{metric}_mean"], result[f"{metric}_sd"] = mean_sd(values) if values else (None, None)
        output.append(result)
    return sorted(output, key=lambda row: (optimizers.index(row["optimizer"]), not row["use_scheduler"]))


def paired_scheduler_deltas(rows: list[dict[str, Any]], optimizers: list[str]) -> list[dict[str, Any]]:
    indexed = {(row["optimizer"], row["seed"], row["use_scheduler"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for optimizer in optimizers:
        seed_rows = []
        for seed in SEEDS:
            scheduled = indexed[(optimizer, seed, True)]
            constant = indexed[(optimizer, seed, False)]
            seed_rows.append(
                {
                    metric: constant[metric] - scheduled[metric]
                    for metric in METRICS
                    if constant[metric] is not None and scheduled[metric] is not None
                }
            )
        result: dict[str, Any] = {"optimizer": optimizer, "n": len(seed_rows)}
        for metric in METRICS:
            values = [row[metric] for row in seed_rows if metric in row]
            result[f"{metric}_delta_mean"], result[f"{metric}_delta_sd"] = mean_sd(values) if values else (None, None)
        output.append(result)
    return output


def tam_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["optimizer"], row["seed"], row["use_scheduler"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for scheduled in (True, False):
        result: dict[str, Any] = {"use_scheduler": scheduled, "contrast": "TAM_SGDM - TAM_baseline", "n": len(SEEDS)}
        for metric in METRICS:
            values = []
            for seed in SEEDS:
                adaptive = indexed[("TAM_SGDM", seed, scheduled)][metric]
                baseline = indexed[("TAM_baseline", seed, scheduled)][metric]
                if adaptive is not None and baseline is not None:
                    values.append(adaptive - baseline)
            result[f"{metric}_delta_mean"], result[f"{metric}_delta_sd"] = mean_sd(values) if values else (None, None)
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_label", required=True)
    parser.add_argument("--scheduled_sweeps", nargs="+", required=True)
    parser.add_argument("--unscheduled_sweeps", nargs="+", required=True)
    parser.add_argument("--optimizers", nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--mal_config", default="False,1.0,False,attenuate")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    rows, excluded = collect(args)
    aggregates = aggregate(rows, args.optimizers)
    scheduler = paired_scheduler_deltas(rows, args.optimizers)
    tam = tam_deltas(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "runs.csv", rows)
    write_csv(args.output_dir / "aggregate.csv", aggregates)
    write_csv(args.output_dir / "scheduler_deltas.csv", scheduler)
    write_csv(args.output_dir / "tam_adaptive_minus_fixed.csv", tam)
    write_csv(args.output_dir / "excluded.csv", excluded)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "task": args.task_label,
                "scheduled_sweeps": [canonical_path(path) for path in args.scheduled_sweeps],
                "unscheduled_sweeps": [canonical_path(path) for path in args.unscheduled_sweeps],
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "mal_config": args.mal_config,
                "seeds": SEEDS,
            },
            indent=2,
        )
        + "\n"
    )

    lines = [
        f"# {args.task_label}: scheduler and TAM ablation",
        "",
        f"Fixed cell: BS={args.batch_size}, LR={args.learning_rate}, WD={args.weight_decay}; three matched seeds.",
        "Negative scheduler deltas mean that removing warmup+cosine hurt the metric.",
        "Positive TAM deltas mean adaptive TAM beat the fixed 0.5-gradient control.",
        "",
        "## Aggregate results",
        "",
        "| Optimizer | Scheduler | Best val | Val AUC | Test @ best val | Test @ final |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['optimizer']} | {'on' if row['use_scheduler'] else 'off'} | "
            f"{fmt(row['best_val_acc_mean'])} | {fmt(row['val_auc_mean'])} | "
            f"{fmt(row['test_acc_at_best_val_mean'])} | {fmt(row['test_acc_at_final_epoch_mean'])} |"
        )
    lines.extend(["", "## Adaptive TAM minus fixed-gate control", ""])
    for row in tam:
        lines.append(
            f"- Scheduler {'on' if row['use_scheduler'] else 'off'}: "
            f"best-val {fmt(row['best_val_acc_delta_mean'])} pp; "
            f"AUC {fmt(row['val_auc_delta_mean'])} pp; "
            f"test-at-best-val {fmt(row['test_acc_at_best_val_delta_mean'])} pp."
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
