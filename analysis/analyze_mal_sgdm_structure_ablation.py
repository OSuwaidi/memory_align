"""Analyze the matched scheduler-free MAL-SGDM structure ablation."""

from __future__ import annotations

import argparse
import contextlib
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
CONFIGS = {
    "MAL-default": "False,1.0,False,attenuate",
    "MAL-pwr0.5": "False,0.5,False,attenuate",
    "MAL-in-place": "True,1.0,False,attenuate",
}
METRICS = (
    "best_val_acc",
    "val_auc",
    "epochs_to_target",
    "test_acc_at_best_val",
    "test_acc_at_final_epoch",
)


def canonical_path(raw: str) -> str:
    path = raw if raw.count("/") == 2 else f"{ENTITY}/{PROJECT}/{raw}"
    if not path.startswith(f"{ENTITY}/{PROJECT}/"):
        raise ValueError(f"Unexpected W&B path {path!r}")
    return path


def canonical_mal(value: Any) -> str:
    fields = [field.strip() for field in str(value).split(",")]
    return ",".join(fields[:4])


def finite(value: Any) -> float | None:
    with contextlib.suppress(TypeError, ValueError):
        result = float(value)
        if math.isfinite(result):
            return result
    return None


def first(summary: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = finite(summary.get(key))
        if value is not None:
            return value
    return None


def runs(api: wandb.Api, path: str) -> list[Any]:
    entity, project, sweep_id = canonical_path(path).split("/")
    return list(
        api.runs(
            f"{entity}/{project}",
            filters={"sweep": sweep_id},
            per_page=1_000,
            lazy=False,
        )
    )


def row_for(run: Any, source: str, variant: str) -> dict[str, Any]:
    config = dict(run.config)
    summary = dict(run.summary)
    return {
        "source": source,
        "run_id": run.id,
        "variant": variant,
        "MAL_config": canonical_mal(config.get("MAL_config")),
        "seed": int(config["seed"]),
        "best_val_acc": first(summary, "best_val_acc", "best/val_acc"),
        "val_auc": first(summary, "val_auc", "val/auc", "AUC"),
        "epochs_to_target": first(summary, "epochs_2_target"),
        "test_acc_at_best_val": first(summary, "test_acc_at_best_val", "test/acc_at_best_val", "test_acc"),
        "test_acc_at_final_epoch": first(summary, "test_acc_at_final_epoch", "test/acc_at_final_epoch"),
    }


def collect(api: wandb.Api, source_path: str, ablation_path: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    source_path = canonical_path(source_path)
    ablation_path = canonical_path(ablation_path)
    for run in runs(api, source_path):
        config = dict(run.config)
        if (
            run.state == "finished"
            and config.get("optimizer") == "MAL_SGDM"
            and canonical_mal(config.get("MAL_config")) == CONFIGS["MAL-default"]
            and finite(config.get("batch_size")) == 256
            and finite(config.get("lr")) == 0.1
            and finite(config.get("weight_decay")) == 5e-4
            and str(config.get("use_scheduler")).lower() == "false"
        ):
            selected.append(row_for(run, source_path, "MAL-default-source"))

    config_to_variant = {config: variant for variant, config in CONFIGS.items()}
    for run in runs(api, ablation_path):
        config = dict(run.config)
        mal_config = canonical_mal(config.get("MAL_config"))
        if run.state != "finished" or mal_config not in config_to_variant:
            continue
        if (
            config.get("optimizer") != "MAL_SGDM"
            or finite(config.get("batch_size")) != 256
            or finite(config.get("lr")) != 0.1
            or finite(config.get("weight_decay")) != 5e-4
            or str(config.get("use_scheduler")).lower() != "false"
        ):
            raise RuntimeError(f"Ablation run {run.id} does not match the registered recipe")
        selected.append(row_for(run, ablation_path, config_to_variant[mal_config]))

    expected = {
        (variant, seed)
        for variant in (*CONFIGS, "MAL-default-source")
        for seed in SEEDS
    }
    observed = [(row["variant"], row["seed"]) for row in selected]
    if len(observed) != len(set(observed)):
        raise RuntimeError("Duplicate variant/seed cells detected")
    if set(observed) != expected:
        raise RuntimeError(f"Incomplete grid: missing={sorted(expected - set(observed))}")
    for row in selected:
        required = ("best_val_acc", "val_auc", "test_acc_at_best_val")
        missing = [metric for metric in required if row[metric] is None]
        if missing:
            raise RuntimeError(f"Run {row['run_id']} lacks {missing}")
    return selected


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    output = []
    order = ("MAL-default-source", *CONFIGS)
    for variant in order:
        record: dict[str, Any] = {"variant": variant, "n": len(grouped[variant])}
        for metric in METRICS:
            values = [float(row[metric]) for row in grouped[variant] if row[metric] is not None]
            record[f"{metric}_mean"], record[f"{metric}_sd"] = mean_sd(values) if len(values) > 1 else (None, None)
        output.append(record)
    return output


def paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(row["variant"], row["seed"]): row for row in rows}
    output = []
    for variant in ("MAL-pwr0.5", "MAL-in-place", "MAL-default-source"):
        record: dict[str, Any] = {"contrast": f"{variant} - MAL-default", "n": len(SEEDS)}
        for metric in METRICS:
            values = []
            for seed in SEEDS:
                candidate = index[(variant, seed)][metric]
                default = index[("MAL-default", seed)][metric]
                if candidate is not None and default is not None:
                    values.append(candidate - default)
            record[f"{metric}_delta_mean"], record[f"{metric}_delta_sd"] = mean_sd(values) if len(values) > 1 else (None, None)
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_sweep", required=True)
    parser.add_argument("--ablation_sweep", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = collect(wandb.Api(timeout=180), args.source_sweep, args.ablation_sweep)
    aggregates = aggregate(rows)
    deltas = paired_deltas(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "runs.csv", rows)
    write_csv(args.output_dir / "aggregate.csv", aggregates)
    write_csv(args.output_dir / "paired_deltas.csv", deltas)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_sweep": canonical_path(args.source_sweep),
                "ablation_sweep": canonical_path(args.ablation_sweep),
                "recipe": {"batch_size": 256, "lr": 0.1, "weight_decay": 5e-4, "epochs": 200, "scheduler": False},
                "configs": CONFIGS,
                "seeds": SEEDS,
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# MAL-SGDM scheduler-free structure ablation",
        "",
        "ResNet-50/CIFAR-100; BS=256, LR=0.1, WD=5e-4, 200 epochs, no warmup or cosine; three matched seeds.",
        "",
        "| Variant | Best val | Val AUC | Test @ best val | Test @ final |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['variant']} | {fmt(row['best_val_acc_mean'])} | {fmt(row['val_auc_mean'])} | "
            f"{fmt(row['test_acc_at_best_val_mean'])} | {fmt(row['test_acc_at_final_epoch_mean'])} |"
        )
    lines.extend(["", "Model/configuration selection must use validation metrics; test metrics are reported only after selection."])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
