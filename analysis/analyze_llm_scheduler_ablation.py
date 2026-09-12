"""Aggregate the paired SmolLM2 scheduler ablation from W&B.

Learning rates are selected with validation loss only.  Test loss is reported
after selection and is never used to choose an optimizer or learning rate.
New completion and scheduler-free runs expose both the final-epoch checkpoint
and the checkpoint selected by validation loss. Older pilot runs remain
analyzable, with an unavailable validation-selected test field rather than a
fabricated alias.
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
OPTIMIZER_ORDER = ("AdamW", "AM_AdamW", "AdaTAMW", "MAL_AdamW")
DEFAULT_MAL_CONFIG = "False,1.0,none,attenuate,update,complement"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot interpret boolean value {value!r}.")


def sweep_path(value: str) -> str:
    return value if value.count("/") == 2 else f"{ENTITY}/{PROJECT}/{value}"


def finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def summary_metric(summary: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = finite_float(summary.get(name))
        if value is not None:
            return value
    return None


def history_mean_val_loss(run: Any) -> float | None:
    values_by_epoch: dict[int, float] = {}
    # The recipe logs six validation points (epoch 0 plus five epochs), so a
    # bounded history request is complete and avoids one paginated request per
    # epoch when auditing many sweeps.
    for row in run.history(samples=100, keys=["epoch", "val/loss"], pandas=False):
        epoch_value = finite_float(row.get("epoch"))
        loss = finite_float(row.get("val/loss"))
        if epoch_value is not None and epoch_value >= 1.0 and loss is not None:
            values_by_epoch[int(epoch_value)] = loss
    return statistics.fmean(values_by_epoch.values()) if values_by_epoch else None


def backfill_metadata(run: Any, config: dict[str, Any], scheduled: bool) -> None:
    metadata = {
        "logging_schema_version": 1,
        "task": "wikitext_causal_lm_finetuning",
        "task_type": "causal_language_modeling",
        "model_name": config.get("model_name", "HuggingFaceTB/SmolLM2-135M"),
        "model_source": "huggingface",
        "dataset_name": config.get("dataset_name", "Salesforce/wikitext"),
        "dataset_config": config.get("dataset_config", "wikitext-2-raw-v1"),
        "dataset_source": "huggingface",
        "training_regime": "full_parameter_finetuning",
        "scheduler": "linear_warmup_cosine" if scheduled else "constant",
        "scheduler_ablation": "scheduled" if scheduled else "scheduler_free",
        "effective_warmup_ratio": float(config.get("warmup_ratio", 0.1)) if scheduled else 0.0,
    }
    changed = False
    for key, value in metadata.items():
        if config.get(key) != value:
            run.config[key] = value
            changed = True
    if changed:
        run.update()


def collect_runs(
    api: wandb.Api,
    paths: Iterable[str],
    *,
    expected_mal_config: str,
    should_backfill: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for raw_path in paths:
        path = sweep_path(raw_path)
        entity, project, sweep_id = path.split("/")
        if (entity, project) != (ENTITY, PROJECT):
            raise ValueError(f"Unexpected W&B project in {path!r}.")
        runs = api.runs(
            f"{entity}/{project}",
            filters={"sweep": sweep_id},
            per_page=1_000,
            lazy=False,
        )
        for run in runs:
            config = dict(run.config)
            optimizer = str(config.get("optimizer", ""))
            mal_config = str(config.get("MAL_config", config.get("mal_config", "")))
            scheduled = parse_bool(config.get("use_scheduler", True))
            if should_backfill and run.state == "finished":
                backfill_metadata(run, config, scheduled)
                config = dict(run.config)
            reason = None
            if run.state != "finished":
                reason = f"state={run.state}"
            elif optimizer not in OPTIMIZER_ORDER:
                reason = "optimizer outside the four-method benchmark"
            elif optimizer == "MAL_AdamW" and mal_config != expected_mal_config:
                reason = "obsolete/non-shipping MAL configuration"
            if reason is not None:
                excluded.append({"sweep_path": path, "run_id": run.id, "optimizer": optimizer, "reason": reason})
                continue

            summary = dict(run.summary)
            mean_val_loss = summary_metric(summary, "mean_val_loss_over_epochs")
            if mean_val_loss is None:
                mean_val_loss = history_mean_val_loss(run)
            row = {
                "sweep_path": path,
                "run_id": run.id,
                "run_name": run.name,
                "optimizer": optimizer,
                "MAL_config": mal_config if optimizer == "MAL_AdamW" else "",
                "use_scheduler": scheduled,
                "scheduler": "linear_warmup_cosine" if scheduled else "constant",
                "batch_size": int(config["batch_size"]),
                "lr_multiplier": float(config["lr_multiplier"]),
                "learning_rate": finite_float(config.get("peak_lr", config.get("learning_rate"))),
                "weight_decay": float(config.get("weight_decay", 0.0)),
                "seed": int(config["seed"]),
                "best_val_loss": summary_metric(summary, "best_val_loss", "best/val_loss"),
                "best_val_epoch": summary_metric(summary, "best_val_epoch", "best/epoch"),
                "final_val_loss": summary_metric(summary, "final_val_loss", "final/val_loss"),
                "mean_val_loss_over_epochs": mean_val_loss,
                "test_loss_at_final_epoch": summary_metric(
                    summary,
                    "test_loss_at_final_epoch",
                    "test/loss_at_final_epoch",
                    "test_loss",
                    "test/loss",
                ),
                "test_loss_at_best_val": summary_metric(summary, "test_loss_at_best_val", "test/loss_at_best_val"),
            }
            required = ("best_val_loss", "final_val_loss", "mean_val_loss_over_epochs", "test_loss_at_final_epoch")
            missing = [name for name in required if row[name] is None]
            if missing:
                raise RuntimeError(f"Run {path}/{run.id} is missing required metric(s): {', '.join(missing)}")
            selected.append(row)

    signature_to_run: dict[tuple[Any, ...], str] = {}
    for row in selected:
        signature = (
            row["optimizer"],
            row["MAL_config"],
            row["batch_size"],
            row["lr_multiplier"],
            row["weight_decay"],
            row["seed"],
            row["use_scheduler"],
        )
        if signature in signature_to_run:
            raise RuntimeError(f"Duplicate experimental cell in {signature_to_run[signature]} and {row['run_id']}: {signature}")
        signature_to_run[signature] = str(row["run_id"])
    return selected, excluded


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    materialized = list(values)
    return statistics.fmean(materialized), statistics.stdev(materialized) if len(materialized) > 1 else 0.0


def aggregate_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["use_scheduler"], row["optimizer"], row["lr_multiplier"])].append(row)

    result: list[dict[str, Any]] = []
    metric_names = (
        "best_val_loss",
        "final_val_loss",
        "mean_val_loss_over_epochs",
        "test_loss_at_final_epoch",
        "test_loss_at_best_val",
    )
    for (scheduled, optimizer, lr_multiplier), group in grouped.items():
        aggregate: dict[str, Any] = {
            "use_scheduler": scheduled,
            "optimizer": optimizer,
            "lr_multiplier": lr_multiplier,
            "learning_rate": group[0]["learning_rate"],
            "n": len(group),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in group if row[metric] is not None]
            if values:
                mean, sd = mean_sd(values)
                aggregate[f"{metric}_mean"] = mean
                aggregate[f"{metric}_sd"] = sd
            else:
                aggregate[f"{metric}_mean"] = None
                aggregate[f"{metric}_sd"] = None
        result.append(aggregate)
    return sorted(result, key=lambda row: (not row["use_scheduler"], OPTIMIZER_ORDER.index(row["optimizer"]), row["lr_multiplier"]))


def paired_deltas(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_signature: dict[tuple[Any, ...], dict[bool, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        signature = (row["optimizer"], row["lr_multiplier"], row["seed"])
        by_signature[signature][bool(row["use_scheduler"])] = row

    pairs: list[dict[str, Any]] = []
    for (optimizer, lr_multiplier, seed), conditions in by_signature.items():
        if set(conditions) != {False, True}:
            continue
        scheduled, constant = conditions[True], conditions[False]
        pair = {
            "optimizer": optimizer,
            "lr_multiplier": lr_multiplier,
            "seed": seed,
            "best_val_loss_delta": constant["best_val_loss"] - scheduled["best_val_loss"],
            "final_val_loss_delta": constant["final_val_loss"] - scheduled["final_val_loss"],
            "mean_val_loss_over_epochs_delta": constant["mean_val_loss_over_epochs"] - scheduled["mean_val_loss_over_epochs"],
            "test_loss_at_final_epoch_delta": constant["test_loss_at_final_epoch"] - scheduled["test_loss_at_final_epoch"],
            "test_loss_at_best_val_delta": None,
        }
        if constant["test_loss_at_best_val"] is not None and scheduled["test_loss_at_best_val"] is not None:
            pair["test_loss_at_best_val_delta"] = constant["test_loss_at_best_val"] - scheduled["test_loss_at_best_val"]
        pairs.append(pair)

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[(row["optimizer"], row["lr_multiplier"])].append(row)
    aggregate: list[dict[str, Any]] = []
    for (optimizer, lr_multiplier), group in grouped.items():
        output: dict[str, Any] = {"optimizer": optimizer, "lr_multiplier": lr_multiplier, "n": len(group)}
        for metric in (
            "best_val_loss_delta",
            "final_val_loss_delta",
            "mean_val_loss_over_epochs_delta",
            "test_loss_at_final_epoch_delta",
            "test_loss_at_best_val_delta",
        ):
            values = [float(row[metric]) for row in group if row[metric] is not None]
            if values:
                mean, sd = mean_sd(values)
                output[f"{metric}_mean"] = mean
                output[f"{metric}_sd"] = sd
            else:
                output[f"{metric}_mean"] = None
                output[f"{metric}_sd"] = None
        aggregate.append(output)
    aggregate.sort(key=lambda row: (OPTIMIZER_ORDER.index(row["optimizer"]), row["lr_multiplier"]))
    return pairs, aggregate


def best_rows(aggregate: list[dict[str, Any]], scheduled: bool) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for optimizer in OPTIMIZER_ORDER:
        candidates = [row for row in aggregate if row["use_scheduler"] is scheduled and row["optimizer"] == optimizer]
        if candidates:
            selected[optimizer] = min(candidates, key=lambda row: (row["best_val_loss_mean"], row["lr_multiplier"]))
    return selected


def format_mean_sd(mean: float | None, sd: float | None) -> str:
    return "—" if mean is None or sd is None else f"{mean:.6f} ± {sd:.6f}"


def markdown_report(
    rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    paired_aggregate: list[dict[str, Any]],
    scheduled_paths: list[str],
    unscheduled_paths: list[str],
) -> str:
    scheduled_best = best_rows(aggregate, True)
    constant_best = best_rows(aggregate, False)
    paired_lookup = {(row["optimizer"], row["lr_multiplier"]): row for row in paired_aggregate}
    lines = [
        "# SmolLM2 scheduler-ablation report",
        "",
        "Learning rates are selected exclusively by the mean best validation loss across seeds. Test loss is a post-selection outcome. Positive deltas mean that removing the 10% linear-warmup + cosine schedule made loss worse.",
        "",
        "## Best scheduled configuration per optimizer",
        "",
        "| Optimizer | LR multiplier | Best validation loss | Test loss at best validation | Final-epoch test loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for optimizer in OPTIMIZER_ORDER:
        row = scheduled_best.get(optimizer)
        if row:
            lines.append(
                f"| {optimizer} | {row['lr_multiplier']:g} | "
                f"{format_mean_sd(row['best_val_loss_mean'], row['best_val_loss_sd'])} | "
                f"{format_mean_sd(row['test_loss_at_best_val_mean'], row['test_loss_at_best_val_sd'])} | "
                f"{format_mean_sd(row['test_loss_at_final_epoch_mean'], row['test_loss_at_final_epoch_sd'])} |"
            )

    lines.extend(
        [
            "",
            "## Removing the schedule at each optimizer's scheduled-optimal LR",
            "",
            "| Optimizer | Matched LR multiplier | Δ best validation loss | Δ validation-curve mean | Δ test loss at best validation | Δ final test loss |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for optimizer in OPTIMIZER_ORDER:
        best = scheduled_best.get(optimizer)
        if not best:
            continue
        delta = paired_lookup.get((optimizer, best["lr_multiplier"]))
        if delta:
            lines.append(
                f"| {optimizer} | {best['lr_multiplier']:g} | "
                f"{format_mean_sd(delta['best_val_loss_delta_mean'], delta['best_val_loss_delta_sd'])} | "
                f"{format_mean_sd(delta['mean_val_loss_over_epochs_delta_mean'], delta['mean_val_loss_over_epochs_delta_sd'])} | "
                f"{format_mean_sd(delta['test_loss_at_best_val_delta_mean'], delta['test_loss_at_best_val_delta_sd'])} | "
                f"{format_mean_sd(delta['test_loss_at_final_epoch_delta_mean'], delta['test_loss_at_final_epoch_delta_sd'])} |"
            )

    lines.extend(
        [
            "",
            "## Best scheduler-free configuration per optimizer",
            "",
            "| Optimizer | LR multiplier | Best validation loss | Validation-curve mean | Test loss at best validation | Final test loss |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for optimizer in OPTIMIZER_ORDER:
        row = constant_best.get(optimizer)
        if row:
            lines.append(
                f"| {optimizer} | {row['lr_multiplier']:g} | "
                f"{format_mean_sd(row['best_val_loss_mean'], row['best_val_loss_sd'])} | "
                f"{format_mean_sd(row['mean_val_loss_over_epochs_mean'], row['mean_val_loss_over_epochs_sd'])} | "
                f"{format_mean_sd(row['test_loss_at_best_val_mean'], row['test_loss_at_best_val_sd'])} | "
                f"{format_mean_sd(row['test_loss_at_final_epoch_mean'], row['test_loss_at_final_epoch_sd'])} |"
            )

    common_lrs = set.intersection(
        *(
            {float(row["lr_multiplier"]) for row in rows if row["optimizer"] == optimizer and not row["use_scheduler"]}
            for optimizer in OPTIMIZER_ORDER
        )
    )
    lines.extend(
        [
            "",
            "## Audit notes",
            "",
            f"- Scheduled sweeps: {', '.join(scheduled_paths)}.",
            f"- Scheduler-free sweeps: {', '.join(unscheduled_paths)}.",
            f"- Shared paired LR grid across all four optimizers: {', '.join(f'{value:g}' for value in sorted(common_lrs))}.",
            "- Learning-rate and optimizer selection use validation loss only. Test loss at the validation-selected checkpoint is the primary post-selection generalization metric; final-checkpoint test loss is retained to quantify late-training drift.",
            "- Historical pilot 9565gqxx logged only final-model test loss. Its unavailable test-at-best-validation entries and paired deltas are left blank rather than inferred; every newly launched run logs both test checkpoints.",
            "- This is a compact 135M-parameter, one-dataset continued-language-modeling benchmark. It supports claims about this setting, not model-scale invariance or instruction tuning.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduled_sweeps", "--scheduled-sweeps", nargs="+", required=True)
    parser.add_argument("--unscheduled_sweeps", "--unscheduled-sweeps", nargs="+", required=True)
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    parser.add_argument("--mal_config", "--mal-config", default=DEFAULT_MAL_CONFIG)
    parser.add_argument("--backfill_metadata", "--backfill-metadata", action="store_true")
    args = parser.parse_args()

    scheduled_paths = [sweep_path(value) for value in args.scheduled_sweeps]
    unscheduled_paths = [sweep_path(value) for value in args.unscheduled_sweeps]
    api = wandb.Api(timeout=180)
    rows, excluded = collect_runs(
        api,
        (*scheduled_paths, *unscheduled_paths),
        expected_mal_config=args.mal_config,
        should_backfill=args.backfill_metadata,
    )
    if not rows:
        raise RuntimeError("No matching finished runs were found.")
    aggregate = aggregate_runs(rows)
    pairs, paired_aggregate = paired_deltas(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "runs.csv", rows)
    write_csv(args.output_dir / "aggregate.csv", aggregate)
    write_csv(args.output_dir / "paired_run_deltas.csv", pairs)
    write_csv(args.output_dir / "paired_aggregate_deltas.csv", paired_aggregate)
    write_csv(args.output_dir / "excluded_runs.csv", excluded)
    (args.output_dir / "report.md").write_text(
        markdown_report(rows, aggregate, paired_aggregate, scheduled_paths, unscheduled_paths),
        encoding="utf-8",
    )
    manifest = {
        "scheduled_sweeps": scheduled_paths,
        "unscheduled_sweeps": unscheduled_paths,
        "MAL_config": args.mal_config,
        "selected_run_count": len(rows),
        "paired_run_count": len(pairs),
        "excluded_run_count": len(excluded),
        "selection_metric": "mean(best_val_loss across seeds)",
        "test_metric": "test_loss_at_best_val (primary), test_loss_at_final_epoch (late-training diagnostic)",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print((args.output_dir / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
