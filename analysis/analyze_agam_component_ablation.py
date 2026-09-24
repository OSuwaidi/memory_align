"""Validate and summarize the matched three-seed AGAM component ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import wandb


VARIANTS = (
    "canonical",
    "previous_memory",
    "global_gate",
    "writeback",
    "hard_reset",
)
SEEDS = (42, 1337, 2026)
METRICS = (
    "best_val_acc",
    "val_auc",
    "test_acc_at_final_epoch",
    "test_acc_at_best_val",
    "epochs_2_target",
)


def collect(sweep_path: str) -> tuple[pd.DataFrame, dict[str, object]]:
    sweep = wandb.Api(timeout=300).sweep(sweep_path)
    rows: list[dict[str, object]] = []
    for run in sweep.runs:
        config = dict(run.config)
        summary = dict(run.summary)
        row: dict[str, object] = {
            "run_id": run.id,
            "run_name": run.name,
            "url": run.url,
            "state": run.state,
            "variant": config.get("AGAM_variant"),
            "seed": config.get("seed"),
            "task": config.get("task"),
            "model": config.get("model"),
            "data": config.get("data"),
            "epochs": config.get("epochs"),
            "batch_size": config.get("batch_size"),
            "lr": config.get("lr"),
            "weight_decay": config.get("weight_decay"),
            "use_scheduler": config.get("use_scheduler"),
            "AGAM_config": config.get("AGAM_config"),
        }
        row.update({metric: summary.get(metric) for metric in METRICS})
        rows.append(row)

    frame = pd.DataFrame(rows)
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    observed = set(zip(frame["variant"], frame["seed"], strict=False))
    duplicates = frame.duplicated(["variant", "seed"], keep=False)
    recipe_columns = {
        "task": "cifar_image_classification",
        "model": "resnet50",
        "data": "cifar100",
        "epochs": 200,
        "batch_size": 256,
        "lr": 0.1,
        "weight_decay": 5e-4,
        "use_scheduler": True,
    }
    recipe_mismatches = {
        column: sorted(frame.loc[frame[column] != expected_value, column].dropna().astype(str).unique())
        for column, expected_value in recipe_columns.items()
        if not frame.loc[frame[column] != expected_value].empty
    }
    quality: dict[str, object] = {
        "sweep_path": sweep_path,
        "sweep_name": sweep.name,
        "sweep_state": sweep.state,
        "expected_runs": len(expected),
        "observed_runs": len(frame),
        "states": frame["state"].value_counts().to_dict(),
        "missing_cells": sorted(expected - observed),
        "unexpected_cells": sorted(observed - expected),
        "duplicate_cells": frame.loc[duplicates, ["variant", "seed", "run_id"]].to_dict("records"),
        "recipe_mismatches": recipe_mismatches,
        "missing_metrics": {metric: int(frame[metric].isna().sum()) for metric in METRICS},
    }
    failures = (
        sweep.state != "FINISHED"
        or len(frame) != len(expected)
        or set(frame["state"]) != {"finished"}
        or bool(quality["missing_cells"])
        or bool(quality["unexpected_cells"])
        or bool(quality["duplicate_cells"])
        or bool(recipe_mismatches)
        or any(quality["missing_metrics"].values())
    )
    if failures:
        raise RuntimeError(f"Ablation audit failed:\n{json.dumps(quality, indent=2, default=str)}")
    return frame, quality


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_metrics = list(METRICS)
    aggregate = frame.groupby("variant", sort=False)[numeric_metrics].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index()

    canonical = frame.loc[frame["variant"] == "canonical", ["seed", *numeric_metrics]].set_index("seed")
    paired_rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        seed = int(row["seed"])
        paired: dict[str, object] = {"variant": row["variant"], "seed": seed}
        for metric in numeric_metrics:
            paired[f"delta_{metric}"] = float(row[metric]) - float(canonical.loc[seed, metric])
        paired_rows.append(paired)
    paired_frame = pd.DataFrame(paired_rows)
    paired_summary = paired_frame.groupby("variant", sort=False).agg(
        {column: ["mean", "std"] for column in paired_frame.columns if column.startswith("delta_")}
    )
    paired_summary.columns = [f"{metric}_{stat}" for metric, stat in paired_summary.columns]
    paired_summary = paired_summary.reset_index()
    return aggregate, paired_summary


def report_markdown(aggregate: pd.DataFrame, paired: pd.DataFrame, quality: dict[str, object]) -> str:
    merged = aggregate.merge(paired, on="variant", how="left")
    lines = [
        "# AGAM-SGD component ablation",
        "",
        "Matched ResNet-50/CIFAR-100 runs: batch size 256, LR 0.1, weight decay 5e-4, "
        "200 epochs, five-epoch warmup plus cosine decay, seeds 42/1337/2026.",
        "",
        "| Variant | Final test, mean ± SD | Δ final test vs canonical | Best-val test, mean ± SD | Δ best-val test | Best val, mean ± SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in merged.iterrows():
        lines.append(
            "| {variant} | {final:.3f} ± {final_sd:.3f} | {delta_final:+.3f} pp | "
            "{selected:.3f} ± {selected_sd:.3f} | {delta_selected:+.3f} pp | "
            "{validation:.3f} ± {validation_sd:.3f} |".format(
                variant=row["variant"],
                final=row["test_acc_at_final_epoch_mean"],
                final_sd=row["test_acc_at_final_epoch_std"],
                delta_final=row["delta_test_acc_at_final_epoch_mean"],
                selected=row["test_acc_at_best_val_mean"],
                selected_sd=row["test_acc_at_best_val_std"],
                delta_selected=row["delta_test_acc_at_best_val_mean"],
                validation=row["best_val_acc_mean"],
                validation_sd=row["best_val_acc_std"],
            )
        )
    lines.extend(
        [
            "",
            "All deltas are paired by seed. The best-validation checkpoint is selected without test-set access; "
            "both final-model and validation-selected test accuracies are reported.",
            "",
            f"Source: W&B sweep `{quality['sweep_path']}` ({quality['observed_runs']} finished runs).",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame, quality = collect(args.sweep)
    aggregate, paired = summarize(frame)
    frame.sort_values(["variant", "seed"]).to_csv(args.output_dir / "runs.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate.csv", index=False)
    paired.to_csv(args.output_dir / "paired_deltas.csv", index=False)
    (args.output_dir / "data_quality.json").write_text(
        json.dumps(quality, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        report_markdown(aggregate, paired, quality),
        encoding="utf-8",
    )
    print(args.output_dir / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
