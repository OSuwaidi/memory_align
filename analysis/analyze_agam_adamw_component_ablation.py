"""Audit and summarize the matched AGAM-AdamW MAE component ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

VARIANTS = ("canonical", "previous_memory", "global_gate", "writeback", "hard_reset")
SEEDS = (42, 1337, 2026)
SUMMARY_METRICS = (
    "final/probe_val_acc",
    "best/probe_val_acc",
    "best/val_loss",
    "val/loss",
    "train/loss",
)


def normalized_auc(frame: pd.DataFrame, x: str, y: str) -> float:
    points = frame[[x, y]].dropna().drop_duplicates(x).sort_values(x)
    if len(points) < 2:
        return float("nan")
    span = float(points[x].iloc[-1] - points[x].iloc[0])
    if span <= 0:
        return float("nan")
    return float(np.trapezoid(points[y], points[x]) / span)


def collect(sweep_path: str) -> tuple[pd.DataFrame, dict[str, object]]:
    sweep = wandb.Api(timeout=300).sweep(sweep_path)
    rows: list[dict[str, object]] = []
    for run in sweep.runs:
        config = dict(run.config)
        summary = dict(run.summary)
        history = run.history(
            keys=("epoch", "train/loss", "val/loss", "probe/val_acc"),
            pandas=True,
            samples=1000,
        )
        row: dict[str, object] = {
            "run_id": run.id,
            "run_name": run.name,
            "url": run.url,
            "state": run.state,
            "variant": config.get("AGAM_variant"),
            "seed": config.get("seed"),
            "task": config.get("task"),
            "model_name": config.get("model_name"),
            "dataset_name": config.get("dataset_name"),
            "epochs": config.get("epochs"),
            "batch_size": config.get("batch_size"),
            "base_lr": config.get("base_lr"),
            "actual_lr": config.get("actual_lr"),
            "weight_decay": config.get("weight_decay"),
            "use_scheduler": config.get("use_scheduler"),
            "AGAM_config": config.get("AGAM_config"),
            "probe_auc": normalized_auc(history, "epoch", "probe/val_acc"),
            "val_loss_auc": normalized_auc(history, "epoch", "val/loss"),
        }
        row.update({metric: summary.get(metric) for metric in SUMMARY_METRICS})
        rows.append(row)

    frame = pd.DataFrame(rows)
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    observed = set(zip(frame.get("variant", ()), frame.get("seed", ()), strict=False))
    duplicates = frame.duplicated(["variant", "seed"], keep=False) if not frame.empty else pd.Series(dtype=bool)
    recipe = {
        "task": "tiny_imagenet_mae_pretraining",
        "model_name": "vit_tiny_patch16_224_mae_patch8",
        "dataset_name": "tiny-imagenet-200",
        "epochs": 300,
        "batch_size": 1024,
        "base_lr": 1e-3,
        "actual_lr": 4e-3,
        "weight_decay": 5e-2,
        "use_scheduler": True,
    }
    recipe_mismatches = {
        column: sorted(frame.loc[frame[column] != value, column].dropna().astype(str).unique())
        for column, value in recipe.items()
        if column not in frame or not frame.loc[frame[column] != value].empty
    }
    missing_metrics = {metric: int(frame[metric].isna().sum()) if metric in frame else len(expected) for metric in (*SUMMARY_METRICS, "probe_auc", "val_loss_auc")}
    quality: dict[str, object] = {
        "sweep_path": sweep_path,
        "sweep_name": sweep.name,
        "sweep_state": sweep.state,
        "expected_runs": len(expected),
        "observed_runs": len(frame),
        "states": frame["state"].value_counts().to_dict() if "state" in frame else {},
        "missing_cells": sorted(expected - observed),
        "unexpected_cells": sorted(observed - expected),
        "duplicate_cells": (frame.loc[duplicates, ["variant", "seed", "run_id"]].to_dict("records") if not frame.empty else []),
        "recipe_mismatches": recipe_mismatches,
        "missing_metrics": missing_metrics,
    }
    failed = (
        sweep.state != "FINISHED"
        or len(frame) != len(expected)
        or set(frame["state"]) != {"finished"}
        or bool(quality["missing_cells"])
        or bool(quality["unexpected_cells"])
        or bool(quality["duplicate_cells"])
        or bool(recipe_mismatches)
        or any(missing_metrics.values())
    )
    if failed:
        raise RuntimeError(f"Ablation audit failed:\n{json.dumps(quality, indent=2, default=str)}")
    return frame, quality


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [*SUMMARY_METRICS, "probe_auc", "val_loss_auc"]
    aggregate = frame.groupby("variant", sort=False)[metrics].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index()

    canonical = frame.loc[frame["variant"] == "canonical", ["seed", *metrics]].set_index("seed")
    paired_rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        seed = int(row["seed"])
        paired: dict[str, object] = {"variant": row["variant"], "seed": seed}
        for metric in metrics:
            paired[f"delta_{metric}"] = float(row[metric]) - float(canonical.loc[seed, metric])
        paired_rows.append(paired)
    paired = pd.DataFrame(paired_rows)
    paired_summary = paired.groupby("variant", sort=False).agg({column: ["mean", "std"] for column in paired if column.startswith("delta_")})
    paired_summary.columns = [f"{metric}_{stat}" for metric, stat in paired_summary.columns]
    return aggregate, paired_summary.reset_index()


def report(aggregate: pd.DataFrame, paired: pd.DataFrame, quality: dict[str, object]) -> str:
    merged = aggregate.merge(paired, on="variant", how="left")
    lines = [
        "# AGAM-AdamW component ablation",
        "",
        (
            "Matched ViT-Tiny MAE/Tiny-ImageNet runs: patch size 8, batch size 1024, "
            "base LR 0.001 (actual LR 0.004), WD 0.05, 300 epochs, 15-epoch warmup "
            "plus cosine decay, seeds 42/1337/2026."
        ),
        "",
        "| Variant | Final probe, mean ± SD | Δ final probe | Best probe, mean ± SD | Δ best probe | Probe AUC | Final val loss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in merged.iterrows():
        lines.append(
            "| {variant} | {final:.3f} ± {final_sd:.3f} | {delta_final:+.3f} pp | {best:.3f} ± {best_sd:.3f} | {delta_best:+.3f} pp | {auc:.3f} | {loss:.5f} |".format(
                variant=row["variant"],
                final=row["final/probe_val_acc_mean"],
                final_sd=row["final/probe_val_acc_std"],
                delta_final=row["delta_final/probe_val_acc_mean"],
                best=row["best/probe_val_acc_mean"],
                best_sd=row["best/probe_val_acc_std"],
                delta_best=row["delta_best/probe_val_acc_mean"],
                auc=row["probe_auc_mean"],
                loss=row["val/loss_mean"],
            )
        )
    lines.extend(
        [
            "",
            (
                "All deltas are paired by seed. Frozen linear probing on the official Tiny-ImageNet "
                "validation set is the generalization endpoint; reconstruction loss measures the SSL objective."
            ),
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
    (args.output_dir / "data_quality.json").write_text(json.dumps(quality, indent=2, default=str) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(report(aggregate, paired, quality), encoding="utf-8")
    print(args.output_dir / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
