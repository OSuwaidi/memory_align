"""Analyze the paired ResNet-50/CIFAR-100 scheduler-removal stress test.

The scheduled reference is the pre-existing heatmap cell at batch size 256,
learning rate 0.1, and weight decay 5e-4.  The new runs change only the
warmup-plus-cosine schedule to a constant learning rate.  Test accuracy never
selects a model or optimizer; each test score is evaluated after validation
checkpoint selection, with the final-model score retained separately whenever
the originating run logged it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import wandb

ENTITY = "osuwaidi-khalifa-university"
PROJECT = "MAL_benchmark"
OPTIMIZERS = ("SGDM", "AM_MSGD", "TAM_SGDM", "MAL_SGDM")
SEEDS = (42, 1337, 2026)
BATCH_SIZE = 256
LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot interpret boolean value {value!r}.")


def finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def first_metric(summary: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = finite_float(summary.get(key))
        if value is not None:
            return value
    return None


def canonical_sweep_path(value: str) -> str:
    return value if value.count("/") == 2 else f"{ENTITY}/{PROJECT}/{value}"


def runs_for_sweep(api: wandb.Api, raw_path: str) -> tuple[str, list[Any]]:
    path = canonical_sweep_path(raw_path)
    entity, project, sweep_id = path.split("/")
    if (entity, project) != (ENTITY, PROJECT):
        raise ValueError(f"Unexpected W&B project in {path!r}.")
    # Eager full-data pagination avoids one config/summary request per run.
    runs = list(
        api.runs(
            f"{entity}/{project}",
            filters={"sweep": sweep_id},
            per_page=1_000,
            lazy=False,
        )
    )
    return path, runs


def collect(
    api: wandb.Api,
    scheduled_path: str,
    unscheduled_path: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for expected_scheduler, raw_path in ((True, scheduled_path), (False, unscheduled_path)):
        path, runs = runs_for_sweep(api, raw_path)
        for run in runs:
            config = dict(run.config)
            optimizer = str(config.get("optimizer", ""))
            reason = None
            try:
                scheduled = parse_bool(config.get("use_scheduler"))
            except ValueError:
                scheduled = not expected_scheduler
                reason = "missing/invalid scheduler flag"
            batch_size = finite_float(config.get("batch_size"))
            learning_rate = finite_float(config.get("lr"))
            weight_decay = finite_float(config.get("weight_decay"))
            if run.state != "finished":
                reason = f"state={run.state}"
            elif optimizer not in OPTIMIZERS:
                reason = "optimizer outside the four-method ablation"
            elif scheduled is not expected_scheduler:
                reason = "scheduler condition does not match source role"
            elif batch_size != BATCH_SIZE or learning_rate != LEARNING_RATE or weight_decay != WEIGHT_DECAY:
                reason = "outside the pre-registered BS/LR/WD cell"
            if reason is not None:
                excluded.append(
                    {
                        "sweep_path": path,
                        "run_id": run.id,
                        "optimizer": optimizer,
                        "reason": reason,
                    }
                )
                continue

            summary = dict(run.summary)
            row = {
                "sweep_path": path,
                "run_id": run.id,
                "run_name": run.name,
                "optimizer": optimizer,
                "use_scheduler": scheduled,
                "scheduler": "linear_warmup_cosine" if scheduled else "constant",
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "seed": int(config["seed"]),
                "best_val_acc": first_metric(summary, "best_val_acc", "best/val_acc"),
                "val_auc": first_metric(summary, "val_auc", "val/auc", "AUC"),
                "epochs_to_target": first_metric(summary, "epochs_2_target"),
                "target_reached": first_metric(summary, "target_reached"),
                # Historical c72berzj defines test_acc as the test score at the
                # validation-selected checkpoint, exactly matching this alias.
                "test_acc_at_best_val": first_metric(
                    summary,
                    "test_acc_at_best_val",
                    "test/acc_at_best_val",
                    "test_acc",
                ),
                # Never reinterpret historical test_acc as a final-model score.
                "test_acc_at_final_epoch": first_metric(
                    summary,
                    "test_acc_at_final_epoch",
                    "test/acc_at_final_epoch",
                ),
            }
            missing = [
                key
                for key in ("best_val_acc", "val_auc", "test_acc_at_best_val")
                if row[key] is None
            ]
            if missing:
                raise RuntimeError(f"Run {path}/{run.id} lacks required metric(s): {', '.join(missing)}")
            if not scheduled and row["test_acc_at_final_epoch"] is None:
                raise RuntimeError(f"New scheduler-free run {path}/{run.id} lacks final-model test accuracy.")
            selected.append(row)

    expected_signatures = {
        (optimizer, scheduled, seed)
        for optimizer in OPTIMIZERS
        for scheduled in (True, False)
        for seed in SEEDS
    }
    observed: dict[tuple[str, bool, int], str] = {}
    for row in selected:
        signature = (str(row["optimizer"]), bool(row["use_scheduler"]), int(row["seed"]))
        if signature in observed:
            raise RuntimeError(f"Duplicate cell in {observed[signature]} and {row['run_id']}: {signature}")
        observed[signature] = str(row["run_id"])
    missing = sorted(expected_signatures - set(observed))
    extra = sorted(set(observed) - expected_signatures)
    if missing or extra:
        raise RuntimeError(f"Paired grid mismatch: missing={missing}, extra={extra}")
    return selected, excluded


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    materialized = list(values)
    return statistics.fmean(materialized), statistics.stdev(materialized) if len(materialized) > 1 else 0.0


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["optimizer"]), bool(row["use_scheduler"]))].append(row)
    output: list[dict[str, Any]] = []
    for (optimizer, scheduled), group in grouped.items():
        record: dict[str, Any] = {
            "optimizer": optimizer,
            "use_scheduler": scheduled,
            "scheduler": "linear_warmup_cosine" if scheduled else "constant",
            "n": len(group),
        }
        for metric in ("best_val_acc", "val_auc", "epochs_to_target", "test_acc_at_best_val", "test_acc_at_final_epoch"):
            values = [float(row[metric]) for row in group if row[metric] is not None]
            record[f"{metric}_mean"], record[f"{metric}_sd"] = mean_sd(values) if values else (None, None)
        output.append(record)
    return sorted(output, key=lambda row: (OPTIMIZERS.index(str(row["optimizer"])), not bool(row["use_scheduler"])))


def paired_deltas(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paired: dict[tuple[str, int], dict[bool, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(str(row["optimizer"]), int(row["seed"]))][bool(row["use_scheduler"])] = row
    run_deltas: list[dict[str, Any]] = []
    for (optimizer, seed), conditions in paired.items():
        scheduled, constant = conditions[True], conditions[False]
        run_deltas.append(
            {
                "optimizer": optimizer,
                "seed": seed,
                "best_val_acc_delta_pp": constant["best_val_acc"] - scheduled["best_val_acc"],
                "val_auc_delta_pp": constant["val_auc"] - scheduled["val_auc"],
                "test_acc_at_best_val_delta_pp": constant["test_acc_at_best_val"] - scheduled["test_acc_at_best_val"],
                # c72berzj predates final-model test logging, so no valid
                # scheduled denominator exists for a final-model test delta.
                "test_acc_at_final_epoch_delta_pp": None,
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_deltas:
        grouped[str(row["optimizer"])].append(row)
    aggregate_deltas: list[dict[str, Any]] = []
    for optimizer, group in grouped.items():
        record: dict[str, Any] = {"optimizer": optimizer, "n": len(group)}
        for metric in ("best_val_acc_delta_pp", "val_auc_delta_pp", "test_acc_at_best_val_delta_pp"):
            record[f"{metric}_mean"], record[f"{metric}_sd"] = mean_sd(float(row[metric]) for row in group)
        record["test_acc_at_final_epoch_delta_pp_mean"] = None
        record["test_acc_at_final_epoch_delta_pp_sd"] = None
        aggregate_deltas.append(record)
    aggregate_deltas.sort(key=lambda row: OPTIMIZERS.index(str(row["optimizer"])))
    return run_deltas, aggregate_deltas


def fmt(mean: float | None, sd: float | None) -> str:
    return "—" if mean is None or sd is None else f"{mean:.3f} ± {sd:.3f}"


def render_report(
    aggregate_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    scheduled_path: str,
    unscheduled_path: str,
) -> str:
    lookup = {(row["optimizer"], row["use_scheduler"]): row for row in aggregate_rows}
    deltas = {row["optimizer"]: row for row in delta_rows}
    lines = [
        "# ResNet-50/CIFAR-100 scheduler-removal report",
        "",
        f"Pre-registered common cell: batch size {BATCH_SIZE}, learning rate {LEARNING_RATE:g}, weight decay {WEIGHT_DECAY:g}, 200 epochs, three matched seeds. Negative deltas mean removing five-epoch linear warmup plus cosine decay made accuracy worse.",
        "",
        "| Optimizer | Scheduled best val | Constant best val | Δ best val (pp) | Scheduled test @ best val | Constant test @ best val | Δ test @ best val (pp) | Constant final-model test |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for optimizer in OPTIMIZERS:
        scheduled = lookup[(optimizer, True)]
        constant = lookup[(optimizer, False)]
        delta = deltas[optimizer]
        lines.append(
            f"| {optimizer} | {fmt(scheduled['best_val_acc_mean'], scheduled['best_val_acc_sd'])} | "
            f"{fmt(constant['best_val_acc_mean'], constant['best_val_acc_sd'])} | "
            f"{fmt(delta['best_val_acc_delta_pp_mean'], delta['best_val_acc_delta_pp_sd'])} | "
            f"{fmt(scheduled['test_acc_at_best_val_mean'], scheduled['test_acc_at_best_val_sd'])} | "
            f"{fmt(constant['test_acc_at_best_val_mean'], constant['test_acc_at_best_val_sd'])} | "
            f"{fmt(delta['test_acc_at_best_val_delta_pp_mean'], delta['test_acc_at_best_val_delta_pp_sd'])} | "
            f"{fmt(constant['test_acc_at_final_epoch_mean'], constant['test_acc_at_final_epoch_sd'])} |"
        )
    lines.extend(
        [
            "",
            "## Audit notes",
            "",
            f"- Scheduled source: {canonical_sweep_path(scheduled_path)}.",
            f"- Scheduler-free source: {canonical_sweep_path(unscheduled_path)}.",
            "- The experiment deliberately does not retune learning rate after scheduler removal; it measures robustness at one conventional shared setting.",
            "- Test accuracy is never used for selection. `test @ best val` is the generalization result from the validation-selected checkpoint; `final-model test` diagnoses late-training drift.",
            "- Historical scheduled sweep c72berzj logged test accuracy at the best-validation checkpoint but predates final-model test logging. A scheduled-versus-constant final-model test delta therefore remains unavailable rather than being fabricated.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduled_sweep", "--scheduled-sweep", required=True)
    parser.add_argument("--unscheduled_sweep", "--unscheduled-sweep", required=True)
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    args = parser.parse_args()

    api = wandb.Api(timeout=300)
    rows, excluded = collect(api, args.scheduled_sweep, args.unscheduled_sweep)
    aggregate_rows = aggregate(rows)
    run_deltas, aggregate_deltas = paired_deltas(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "runs.csv", rows)
    write_csv(args.output_dir / "aggregate.csv", aggregate_rows)
    write_csv(args.output_dir / "paired_run_deltas.csv", run_deltas)
    write_csv(args.output_dir / "paired_aggregate_deltas.csv", aggregate_deltas)
    write_csv(args.output_dir / "excluded_runs.csv", excluded)
    report = render_report(aggregate_rows, aggregate_deltas, args.scheduled_sweep, args.unscheduled_sweep)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "scheduled_sweep": canonical_sweep_path(args.scheduled_sweep),
                "unscheduled_sweep": canonical_sweep_path(args.unscheduled_sweep),
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "optimizers": OPTIMIZERS,
                "seeds": SEEDS,
                "selection_policy": "fixed pre-registered cell; test set not used for selection",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
