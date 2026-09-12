"""Aggregate the matched MAL-SGDM gate-telemetry suite into paper figures.

The suite is expected to contain three matched seeds under ``scheduled/`` and
``constant/``.  Individual runs remain the lossless source of truth; this
script validates them, writes tagged source tables, and reports seed-level
variation without treating tensors or optimizer steps as independent runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

SEEDS = (42, 1337, 2026)
CONDITIONS = ("scheduled", "constant")
METRICS = ("gate_q", "beta_eff")
WEIGHTINGS = ("equal_tensor", "numel_weighted")
DEPTHS = ("stem", "layer1.0", "layer1.1", "layer2.0", "layer2.1", "layer3.0", "layer3.1", "layer4.0", "layer4.1", "head")
STAGES = ("stem", "layer1", "layer2", "layer3", "layer4", "head")
KINDS = ("conv_weight", "norm_weight", "norm_bias", "linear_weight", "linear_bias")
REFERENCE = 0.7
COLORS = {"scheduled": "#0072B2", "constant": "#D55E00"}
LINESTYLES = {"scheduled": "-", "constant": "--"}
DISPLAY = {"scheduled": "Warmup + cosine", "constant": "Constant LR"}
THRESHOLD_COLORS = ("#D55E00", "#E69F00", "#0072B2")
THRESHOLD_HATCHES = ("////", "....", "")


@dataclass(frozen=True)
class RunRecord:
    condition: str
    seed: int
    directory: Path
    run: dict[str, Any]
    analysis: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected a finite number, got {value!r}")
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty source table: {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_number(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def load_suite(root: Path, *, allow_synthetic: bool) -> list[RunRecord]:
    records: list[RunRecord] = []
    expected_schedule = {"scheduled": ("cosine", 5), "constant": ("constant", 0)}
    for condition in CONDITIONS:
        for seed in SEEDS:
            directory = root / condition / f"seed-{seed}"
            run_path = directory / "run.json"
            analysis_path = directory / "analysis" / "analysis.json"
            if not run_path.is_file() or not analysis_path.is_file():
                raise FileNotFoundError(f"Missing completed telemetry output under {directory}")
            run, analysis = read_json(run_path), read_json(analysis_path)
            arguments = run.get("arguments", {})
            if run.get("status") != "completed":
                raise ValueError(f"{directory} has status={run.get('status')!r}, not completed")
            if bool(run.get("synthetic")) and not allow_synthetic:
                raise ValueError(f"Synthetic output cannot be used as paper evidence: {directory}")
            if int(arguments.get("seed")) != seed:
                raise ValueError(f"Seed mismatch in {directory}")
            schedule, warmup = expected_schedule[condition]
            if arguments.get("schedule") != schedule:
                raise ValueError(f"Schedule mismatch in {directory}")
            if not allow_synthetic and int(arguments.get("warmup_epochs")) != warmup:
                raise ValueError(f"Warmup mismatch in {directory}")
            if int(run.get("completed_steps", -1)) != int(run.get("planned_steps", -2)):
                raise ValueError(f"Incomplete optimizer-step coverage in {directory}")
            if int(analysis.get("persisted_steps", -1)) != int(run["completed_steps"]):
                raise ValueError(f"Analysis/index step mismatch in {directory}")
            if not analysis.get("complete_tensor_step_coverage"):
                raise ValueError(f"Incomplete tensor-step coverage in {directory}")
            for required in (
                "summary.csv",
                "epoch_summary.csv",
                "histograms.csv",
                "epoch_model_histograms.csv",
                "epoch_model_distributions.csv",
                "step_model_summary.csv",
            ):
                if not (directory / "analysis" / required).is_file():
                    raise FileNotFoundError(directory / "analysis" / required)
            records.append(RunRecord(condition, seed, directory, run, analysis))

    reference = records[0]
    reference_args = reference.run["arguments"]
    invariant_arguments = (
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "min_lr",
        "norm",
        "augmentation",
        "split_seed",
        "late_fraction",
    )
    for record in records[1:]:
        for key in invariant_arguments:
            left, right = reference_args.get(key), record.run["arguments"].get(key)
            equivalent = same_number(left, right) if isinstance(left, (int, float)) else left == right
            if not equivalent:
                raise ValueError(f"Recipe mismatch for {key}: {reference.directory} vs {record.directory}")
        if record.run.get("optimizer_defaults") != reference.run.get("optimizer_defaults"):
            raise ValueError(f"Optimizer structure differs in {record.directory}")
        if record.run.get("architecture") != reference.run.get("architecture"):
            raise ValueError(f"Architecture differs in {record.directory}")
        if record.run.get("tensor_count") != reference.run.get("tensor_count"):
            raise ValueError(f"Tensor count differs in {record.directory}")
        if read_json(record.directory / "tensor_metadata.json") != read_json(reference.directory / "tensor_metadata.json"):
            raise ValueError(f"Tensor provenance differs in {record.directory}")
        split = record.directory / "split_indices.npz"
        reference_split = reference.directory / "split_indices.npz"
        if split.exists() and reference_split.exists() and sha256(split) != sha256(reference_split):
            raise ValueError(f"Train/validation split differs in {record.directory}")

    if not allow_synthetic:
        required_recipe = {
            "epochs": 200,
            "batch_size": 256,
            "lr": 0.1,
            "weight_decay": 5e-4,
            "norm": "group",
            "augmentation": "repo",
            "split_seed": 20260901,
        }
        for key, expected in required_recipe.items():
            actual = reference_args.get(key)
            equivalent = same_number(actual, expected) if isinstance(expected, (int, float)) else actual == expected
            if not equivalent:
                raise ValueError(f"Paper suite requires {key}={expected!r}, got {actual!r}")
        if int(reference.run.get("steps_per_epoch", -1)) != 166 or int(reference.run["planned_steps"]) != 33_200:
            raise ValueError("Unexpected CIFAR-10 optimizer-step count")
    return records


def tagged_rows(records: Iterable[RunRecord], relative_path: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        for row in read_csv(record.directory / relative_path):
            output.append(
                {
                    "condition": record.condition,
                    "seed": record.seed,
                    "source_directory": str(record.directory),
                    **row,
                }
            )
    return output


def grouped_values(rows: Iterable[dict[str, Any]], keys: tuple[str, ...], value: str) -> dict[tuple[Any, ...], list[float]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(finite(row[value]))
    return grouped


def seed_band(series: dict[int, dict[int, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    epochs = np.asarray(sorted(set.intersection(*(set(values) for values in series.values()))), dtype=int)
    matrix = np.asarray([[series[seed][int(epoch)] for epoch in epochs] for seed in SEEDS], dtype=float)
    return epochs, matrix.mean(axis=0), matrix.min(axis=0), matrix.max(axis=0)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, plots: Path, name: str) -> None:
    fig.savefig(plots / f"{name}.png", dpi=400, bbox_inches="tight")
    fig.savefig(plots / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(plots / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_model_evolution(distributions: list[dict[str, Any]], plots: Path, metric: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.5), sharex=True, sharey=True)
    for axis, weighting in zip(axes, WEIGHTINGS, strict=True):
        for condition in CONDITIONS:
            series: dict[int, dict[int, float]] = {seed: {} for seed in SEEDS}
            for row in distributions:
                if row["metric"] == metric and row["weighting"] == weighting and row["condition"] == condition:
                    series[int(row["seed"])][int(row["epoch"])] = finite(row["mean"])
            epochs, mean, low, high = seed_band(series)
            for seed in SEEDS:
                axis.plot(epochs, [series[seed][int(epoch)] for epoch in epochs], color=COLORS[condition], alpha=0.18, linewidth=0.65, linestyle=LINESTYLES[condition])
            axis.fill_between(epochs, low, high, color=COLORS[condition], alpha=0.10, linewidth=0)
            axis.plot(epochs, mean, color=COLORS[condition], linewidth=1.8, linestyle=LINESTYLES[condition], label=DISPLAY[condition])
        axis.axhline(REFERENCE, color="#4D4D4D", linestyle=(0, (2, 2)), linewidth=0.9, label="0.7 reference")
        axis.set_title("Equal tensor" if weighting == "equal_tensor" else "Parameter-count weighted")
        axis.set_xlabel("Epoch")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axes[0].set_ylabel("Applied memory coefficient $c_t$" if metric == "beta_eff" else "Raw alignment gate $q_t$")
    axes[0].set_ylim((0.42, 0.92) if metric == "beta_eff" else (0.46, 1.01))
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)
    fig.suptitle(f"MAL-SGDM {'applied memory' if metric == 'beta_eff' else 'raw gate'} evolution (n=3 seeds)", y=1.13, fontsize=12)
    fig.tight_layout()
    save_figure(fig, plots, f"01_{metric}_model_evolution")


def plot_depth_evolution(epoch_rows: list[dict[str, Any]], plots: Path) -> None:
    lookup = grouped_values(
        (row for row in epoch_rows if row["metric"] == "beta_eff" and row["weighting"] == "equal_tensor" and row["view"] == "depth" and row["group"] in DEPTHS),
        ("condition", "epoch", "group"),
        "mean",
    )
    epochs = sorted({int(key[1]) for key in lookup})
    matrices: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        matrices[condition] = np.asarray([[statistics.fmean(lookup[(condition, str(epoch), depth)]) for epoch in epochs] for depth in DEPTHS])
    combined = np.concatenate([matrices[condition].ravel() for condition in CONDITIONS])
    vmin = max(0.0, math.floor(float(np.nanmin(combined)) * 20) / 20)
    vmax = min(0.9, math.ceil(float(np.nanmax(combined)) * 20) / 20)
    difference = matrices["constant"] - matrices["scheduled"]
    bound = max(0.01, math.ceil(float(np.nanmax(np.abs(difference))) * 100) / 100)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.5), sharey=True, gridspec_kw={"width_ratios": (1, 1, 1.05)})
    images = []
    for axis, condition in zip(axes[:2], CONDITIONS, strict=True):
        image = axis.imshow(
            matrices[condition],
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=(min(epochs) - 0.5, max(epochs) + 0.5, len(DEPTHS) - 0.5, -0.5),
        )
        images.append(image)
        axis.set_title(DISPLAY[condition])
        axis.set_xlabel("Epoch")
    diff_image = axes[2].imshow(
        difference,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
        extent=(min(epochs) - 0.5, max(epochs) + 0.5, len(DEPTHS) - 0.5, -0.5),
    )
    axes[2].set_title("Constant − scheduled")
    axes[2].set_xlabel("Epoch")
    axes[0].set_yticks(range(len(DEPTHS)), labels=DEPTHS)
    axes[0].set_ylabel("Residual-block depth")
    fig.colorbar(images[0], ax=axes[:2], location="bottom", fraction=0.08, pad=0.17, label="Mean applied coefficient $c_t$")
    fig.colorbar(diff_image, ax=axes[2], location="bottom", fraction=0.08, pad=0.17, label="Coefficient difference")
    fig.suptitle("Depth-wise evolution of MAL-SGDM memory (equal-tensor view; n=3)", y=1.01, fontsize=12)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.89, bottom=0.24, wspace=0.12)
    save_figure(fig, plots, "02_beta_eff_depth_evolution")


def plot_kind_distributions(histograms: list[dict[str, Any]], plots: Path, *, weighting: str) -> None:
    selected = [
        row
        for row in histograms
        if row["period"] == "late" and row["metric"] == "beta_eff" and row["view"] == "kind" and row["weighting"] == weighting and row["group"] in KINDS
    ]
    counts: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    count_field = "weighted_count" if weighting == "numel_weighted" else "count"
    for row in selected:
        counts[(row["condition"], row["group"], int(row["seed"]))].append(((finite(row["left"]) + finite(row["right"])) / 2, finite(row[count_field])))
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.2), sharex=True, sharey=True)
    for axis, kind in zip(axes.flat, KINDS, strict=False):
        for condition in CONDITIONS:
            seed_curves = []
            centers = None
            for seed in SEEDS:
                pairs = sorted(counts[(condition, kind, seed)])
                centers = np.asarray([pair[0] for pair in pairs])
                values = np.asarray([pair[1] for pair in pairs], dtype=float)
                seed_curves.append(values / values.sum() * 100)
            matrix = np.asarray(seed_curves)
            axis.fill_between(centers, matrix.min(axis=0), matrix.max(axis=0), color=COLORS[condition], alpha=0.10, linewidth=0)
            axis.plot(centers, matrix.mean(axis=0), color=COLORS[condition], linestyle=LINESTYLES[condition], linewidth=1.55, label=DISPLAY[condition])
        axis.axvline(REFERENCE, color="#4D4D4D", linestyle=(0, (2, 2)), linewidth=0.8)
        axis.set_title(kind.replace("_", " "))
        axis.grid(axis="y", color="#E1E1E1", linewidth=0.45)
    axes.flat[-1].axis("off")
    for axis in axes[-1, :2]:
        axis.set_xlabel("Applied memory coefficient $c_t$")
    for axis in axes[:, 0]:
        axis.set_ylabel("Observations per 0.02 bin (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False)
    label = "equal tensor" if weighting == "equal_tensor" else "parameter-count weighted"
    fig.suptitle(f"Late-window coefficient distributions by parameter kind ({label}; n=3)", y=1.04, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, plots, f"03_beta_eff_kind_distributions_{weighting}")


def plot_stage_occupancy(summary_rows: list[dict[str, Any]], plots: Path) -> None:
    fields = ("pct_lt_0_5", "pct_0_5_to_0_7", "pct_gt_0_7")
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.8), sharex=True)
    x = np.arange(len(STAGES), dtype=float)
    width = 0.35
    for axis, weighting in zip(axes, WEIGHTINGS, strict=True):
        for condition_index, condition in enumerate(CONDITIONS):
            bottoms = np.zeros(len(STAGES))
            centers = x + (condition_index - 0.5) * width
            for field, color, hatch in zip(fields, THRESHOLD_COLORS, THRESHOLD_HATCHES, strict=True):
                values = []
                for stage in STAGES:
                    seeds = [
                        finite(row[field])
                        for row in summary_rows
                        if row["period"] == "late"
                        and row["metric"] == "beta_eff"
                        and row["view"] == "stage"
                        and row["weighting"] == weighting
                        and row["group"] == stage
                        and row["condition"] == condition
                    ]
                    if len(seeds) != len(SEEDS):
                        raise ValueError(f"Incomplete late occupancy for {condition}/{weighting}/{stage}")
                    values.append(statistics.fmean(seeds))
                axis.bar(
                    centers,
                    values,
                    width,
                    bottom=bottoms,
                    color=color,
                    edgecolor="white",
                    linewidth=0.45,
                    hatch=hatch,
                    label={"pct_lt_0_5": "$c_t<0.5$", "pct_0_5_to_0_7": "$0.5\\leq c_t\\leq0.7$", "pct_gt_0_7": "$c_t>0.7$"}[field] if condition_index == 0 else None,
                )
                bottoms += np.asarray(values)
            for center in centers:
                axis.text(center, 102.3, "S" if condition == "scheduled" else "C", ha="center", va="bottom", color=COLORS[condition], fontsize=8, fontweight="bold")
        axis.set_ylim(0, 108)
        axis.set_ylabel("Late-window observations (%)")
        axis.set_title("Equal tensor" if weighting == "equal_tensor" else "Parameter-count weighted")
        axis.grid(axis="y", color="#E1E1E1", linewidth=0.45)
    axes[-1].set_xticks(x, labels=STAGES)
    axes[-1].set_xlabel("Model stage (S = scheduled, C = constant LR)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.33), ncol=3, frameon=False)
    fig.suptitle("Where MAL-SGDM retains or suppresses memory (final 20%; n=3)", y=1.02, fontsize=12)
    fig.tight_layout()
    save_figure(fig, plots, "04_beta_eff_stage_threshold_occupancy")


def plot_tensor_map(summary_rows: list[dict[str, Any]], metadata: list[dict[str, Any]], plots: Path) -> None:
    names = [str(row["name"]) for row in sorted(metadata, key=lambda row: int(row["tensor_id"]))]
    matrices: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        values = []
        for name in names:
            seeds = [
                finite(row["mean"])
                for row in summary_rows
                if row["period"] == "late"
                and row["metric"] == "beta_eff"
                and row["view"] == "tensor"
                and row["weighting"] == "equal_tensor"
                and row["group"] == name
                and row["condition"] == condition
            ]
            if len(seeds) != len(SEEDS):
                raise ValueError(f"Incomplete tensor summary for {condition}/{name}")
            values.append(statistics.fmean(seeds))
        matrices[condition] = np.asarray(values)[:, None]
    difference = matrices["constant"] - matrices["scheduled"]
    values = np.concatenate((matrices["scheduled"], matrices["constant"])).ravel()
    vmin = max(0, math.floor(float(values.min()) * 20) / 20)
    vmax = min(0.9, math.ceil(float(values.max()) * 20) / 20)
    bound = max(0.01, math.ceil(float(np.abs(difference).max()) * 100) / 100)
    fig, axes = plt.subplots(1, 3, figsize=(8.7, 14.5), sharey=True, gridspec_kw={"width_ratios": (1, 1, 1.2)})
    images = []
    for axis, condition in zip(axes[:2], CONDITIONS, strict=True):
        image = axis.imshow(matrices[condition], aspect="auto", cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
        images.append(image)
        axis.set_title(DISPLAY[condition])
        axis.set_xticks([])
    diff_image = axes[2].imshow(difference, aspect="auto", cmap="RdBu_r", interpolation="nearest", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
    axes[2].set_title("Constant − scheduled")
    axes[2].set_xticks([])
    axes[0].set_yticks(range(len(names)), labels=names)
    axes[0].tick_params(axis="y", labelsize=5.5)
    # Explicit colorbar axes keep the final tensor labels clear in this tall
    # supplementary figure; automatic placement otherwise overlaps the head.
    coefficient_bar = fig.add_axes((0.37, 0.025, 0.35, 0.011))
    difference_bar = fig.add_axes((0.78, 0.025, 0.18, 0.011))
    fig.colorbar(images[0], cax=coefficient_bar, orientation="horizontal", label="Mean late $c_t$")
    fig.colorbar(diff_image, cax=difference_bar, orientation="horizontal", label="Difference")
    fig.suptitle("Tensor-level MAL-SGDM memory map (final 20%; n=3)", y=0.995, fontsize=12)
    fig.subplots_adjust(left=0.37, right=0.98, top=0.97, bottom=0.075, wspace=0.16)
    save_figure(fig, plots, "05_beta_eff_tensor_map")


def plot_performance(train_rows: list[dict[str, Any]], plots: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.45), sharex=True)
    for condition in CONDITIONS:
        for metric, axis, label in (("validation_accuracy", axes[0], "Validation accuracy (%)"), ("train_loss", axes[1], "Training loss")):
            series: dict[int, dict[int, float]] = {seed: {} for seed in SEEDS}
            for row in train_rows:
                if row["condition"] == condition and str(row.get("complete_epoch", "")).lower() in {"true", "1"}:
                    series[int(row["seed"])][int(row["epoch"])] = finite(row[metric])
            epochs, mean, low, high = seed_band(series)
            for seed in SEEDS:
                axis.plot(epochs, [series[seed][int(epoch)] for epoch in epochs], color=COLORS[condition], alpha=0.18, linewidth=0.6, linestyle=LINESTYLES[condition])
            axis.fill_between(epochs, low, high, color=COLORS[condition], alpha=0.10, linewidth=0)
            axis.plot(epochs, mean, color=COLORS[condition], linewidth=1.8, linestyle=LINESTYLES[condition], label=DISPLAY[condition])
            axis.set_ylabel(label)
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(axis="y", color="#E1E1E1", linewidth=0.45)
    axes[0].set_title("Generalization trajectory")
    axes[1].set_title("Optimization trajectory")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False)
    fig.suptitle("Performance context for the telemetry comparison (n=3)", y=1.12, fontsize=12)
    fig.tight_layout()
    save_figure(fig, plots, "06_training_performance_context")


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    materialized = list(values)
    return statistics.fmean(materialized), statistics.stdev(materialized) if len(materialized) > 1 else 0.0


def aggregate_late(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in summary_rows
        if row["period"] == "late" and row["view"] == "model" and row["group"] == "all" and row["metric"] in METRICS and row["weighting"] in WEIGHTINGS
    ]
    output = []
    for condition in CONDITIONS:
        for metric in METRICS:
            for weighting in WEIGHTINGS:
                rows = [row for row in selected if row["condition"] == condition and row["metric"] == metric and row["weighting"] == weighting]
                if len(rows) != len(SEEDS):
                    raise ValueError(f"Incomplete late model summary for {condition}/{metric}/{weighting}")
                record: dict[str, Any] = {"condition": condition, "metric": metric, "weighting": weighting, "n_seeds": len(rows)}
                for field in ("mean", "pct_lt_0_5", "pct_0_5_to_0_7", "pct_gt_0_7"):
                    mean, sd = mean_sd(finite(row[field]) for row in rows)
                    record[f"{field}_mean"] = mean
                    record[f"{field}_sd"] = sd
                output.append(record)
    return output


def paired_late_deltas(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (row["condition"], int(row["seed"]), row["metric"], row["weighting"]): finite(row["mean"])
        for row in summary_rows
        if row["period"] == "late" and row["view"] == "model" and row["group"] == "all" and row["metric"] in METRICS and row["weighting"] in WEIGHTINGS
    }
    output = []
    for metric in METRICS:
        for weighting in WEIGHTINGS:
            deltas = [index[("constant", seed, metric, weighting)] - index[("scheduled", seed, metric, weighting)] for seed in SEEDS]
            mean, sd = mean_sd(deltas)
            output.append(
                {
                    "contrast": "constant_minus_scheduled",
                    "metric": metric,
                    "weighting": weighting,
                    "n_paired_seeds": len(deltas),
                    "delta_mean": mean,
                    "delta_sd": sd,
                    **{f"seed_{seed}_delta": delta for seed, delta in zip(SEEDS, deltas, strict=True)},
                }
            )
    return output


def run_manifest(records: list[RunRecord]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        args, evaluation = record.run["arguments"], record.run.get("evaluation") or {}
        wandb_info = record.run.get("wandb") or {}
        output.append(
            {
                "condition": record.condition,
                "seed": record.seed,
                "source_directory": str(record.directory),
                "wandb_run_id": wandb_info.get("run_id", ""),
                "wandb_run_url": wandb_info.get("run_url", ""),
                "git_commit": record.run.get("git_commit", ""),
                "schedule": args["schedule"],
                "warmup_epochs": args["warmup_epochs"],
                "epochs": args["epochs"],
                "batch_size": args["batch_size"],
                "learning_rate": args["lr"],
                "weight_decay": args["weight_decay"],
                "completed_steps": record.run["completed_steps"],
                "tensor_count": record.run["tensor_count"],
                "tensor_step_observations": record.analysis["observed_tensor_step_observations"],
                "best_val_acc": evaluation.get("best_val_acc"),
                "best_val_epoch": evaluation.get("best_val_epoch"),
                "test_acc_at_best_val": evaluation.get("test_acc_at_best_val"),
                "test_acc_at_final_epoch": evaluation.get("test_acc_at_final_epoch"),
            }
        )
    return output


def extrema(summary_rows: list[dict[str, Any]], *, view: str, groups: tuple[str, ...], condition: str) -> tuple[tuple[str, float], tuple[str, float]]:
    values = []
    for group in groups:
        seeds = [
            finite(row["mean"])
            for row in summary_rows
            if row["period"] == "late"
            and row["metric"] == "beta_eff"
            and row["view"] == view
            and row["group"] == group
            and row["weighting"] == "equal_tensor"
            and row["condition"] == condition
        ]
        if seeds:
            values.append((group, statistics.fmean(seeds)))
    return min(values, key=lambda item: item[1]), max(values, key=lambda item: item[1])


def report_text(
    records: list[RunRecord], manifest: list[dict[str, Any]], late: list[dict[str, Any]], deltas: list[dict[str, Any]], summary_rows: list[dict[str, Any]]
) -> str:
    lookup = {(row["condition"], row["metric"], row["weighting"]): row for row in late}
    delta_lookup = {(row["metric"], row["weighting"]): row for row in deltas}
    lines = [
        "# MAL-SGDM gate telemetry: matched scheduler diagnostic",
        "",
        "ResNet18/CIFAR-10, batch size 256, LR 0.1, weight decay 5e-4, 200 epochs. Three matched seeds (42, 1337, 2026) are run once with five-epoch linear warmup plus cosine decay and once with constant LR. All intervals and bands show variation across the three independent seeds; tensor-step observations are not treated as independent experimental replicates.",
        "",
        "## Late-window model summary",
        "",
        "The late window is the final 20% of optimizer steps and is descriptive; no stationarity claim is made.",
        "",
        "| Condition | Quantity | Weighting | Mean ± seed SD | <0.5 | 0.5–0.7 | >0.7 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        for metric in METRICS:
            for weighting in WEIGHTINGS:
                row = lookup[(condition, metric, weighting)]
                lines.append(
                    f"| {DISPLAY[condition]} | {metric} | {weighting.replace('_', ' ')} | {row['mean_mean']:.4f} ± {row['mean_sd']:.4f} | {row['pct_lt_0_5_mean']:.2f}% | {row['pct_0_5_to_0_7_mean']:.2f}% | {row['pct_gt_0_7_mean']:.2f}% |"
                )
    lines.extend(
        [
            "",
            "## Matched scheduler contrast",
            "",
            "Positive values mean the coefficient is larger under constant LR for the same seed.",
            "",
            "| Quantity | Weighting | Constant − scheduled (mean ± seed SD) |",
            "|---|---|---:|",
        ]
    )
    for metric in METRICS:
        for weighting in WEIGHTINGS:
            row = delta_lookup[(metric, weighting)]
            lines.append(f"| {metric} | {weighting.replace('_', ' ')} | {row['delta_mean']:+.4f} ± {row['delta_sd']:.4f} |")
    lines.extend(["", "## Structural localization", ""])
    for condition in CONDITIONS:
        low_kind, high_kind = extrema(summary_rows, view="kind", groups=KINDS, condition=condition)
        low_depth, high_depth = extrema(summary_rows, view="depth", groups=DEPTHS, condition=condition)
        lines.append(
            f"- **{DISPLAY[condition]}:** lowest/highest late equal-tensor coefficient by parameter kind: `{low_kind[0]}` ({low_kind[1]:.4f}) / `{high_kind[0]}` ({high_kind[1]:.4f}); by depth: `{low_depth[0]}` ({low_depth[1]:.4f}) / `{high_depth[0]}` ({high_depth[1]:.4f})."
        )
    if all(row.get("best_val_acc") is not None for row in manifest):
        lines.extend(["", "## Performance context", "", "| Condition | Best validation accuracy | Test @ best validation | Test @ final epoch |", "|---|---:|---:|---:|"])
        for condition in CONDITIONS:
            selected = [row for row in manifest if row["condition"] == condition]
            formatted = []
            for field in ("best_val_acc", "test_acc_at_best_val", "test_acc_at_final_epoch"):
                mean, sd = mean_sd(finite(row[field]) for row in selected)
                formatted.append(f"{mean:.3f} ± {sd:.3f}")
            lines.append(f"| {DISPLAY[condition]} | {' | '.join(formatted)} |")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "This suite supports claims about the default tensor-level MAL-SGDM coefficient on one ResNet18/CIFAR-10 recipe and about its sensitivity to scheduler removal. It does not establish universal behavior across architectures, optimizer families, batch sizes, or learning rates. The parameter-count view repeats each tensor-level scalar by tensor size; it is an element-equivalent descriptive weighting, not additional independent evidence.",
            "",
            "The raw gate `gate_q` and applied coefficient `beta_eff = 0.9 × gate_q` have different scales except for the explicitly recorded zero-gradient fallback. The 0.7 line is a reference, not a fitted equilibrium.",
        ]
    )
    return "\n".join(lines) + "\n"


def log_to_wandb(output: Path, records: list[RunRecord], late: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if args.wandb_mode == "disabled":
        return
    import wandb

    source_ids = [record.run.get("wandb", {}).get("run_id") for record in records]
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        mode=args.wandb_mode,
        job_type="optimizer-diagnostics-analysis",
        tags=("optimizer-telemetry", "analysis", "cifar", "mal-sgdm"),
        name=args.wandb_name or f"MAL-SGDM telemetry suite analysis · {args.suite.name}",
        config={
            "logging_schema_version": 1,
            "task": "mal_sgdm_gate_telemetry_analysis",
            "task_type": "optimizer_diagnostics_analysis",
            "model_name": "resnet18",
            "model_source": "torchvision",
            "dataset_name": "cifar10",
            "dataset_config": "official_train_85_15_validation_official_test",
            "dataset_source": "torchvision",
            "training_regime": "supervised_optimizer_telemetry",
            "suite_directory": str(args.suite),
            "conditions": list(CONDITIONS),
            "seeds": list(SEEDS),
            "source_run_ids": source_ids,
        },
    )
    for row in late:
        prefix = f"telemetry/late/{row['metric']}/{row['weighting']}/{row['condition']}"
        run.summary[f"{prefix}/mean"] = row["mean_mean"]
        run.summary[f"{prefix}/seed_sd"] = row["mean_sd"]
    artifact = wandb.Artifact(
        f"mal-sgdm-gate-telemetry-suite-{run.id}",
        type="optimizer-telemetry-analysis",
        description="Matched three-seed scheduler/constant-LR MAL-SGDM telemetry source tables and publication figures.",
        metadata={"source_run_ids": source_ids, "n_seeds_per_condition": len(SEEDS)},
    )
    artifact.add_dir(str(output), name="aggregate")
    run.log_artifact(artifact, aliases=["latest", "resnet18-cifar10"])
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-synthetic", action="store_true", help="Pipeline validation only; never publication evidence")
    parser.add_argument("--wandb-mode", choices=("disabled", "online", "offline"), default="disabled")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-name")
    args = parser.parse_args()
    if args.wandb_mode != "disabled" and not args.wandb_project:
        parser.error("--wandb-project is required when W&B logging is enabled")

    args.suite = args.suite.resolve()
    output = (args.output or args.suite / "aggregate").resolve()
    output.mkdir(parents=True, exist_ok=True)
    plots = output / "figures"
    plots.mkdir(exist_ok=True)

    records = load_suite(args.suite, allow_synthetic=args.allow_synthetic)
    manifest = run_manifest(records)
    distributions = tagged_rows(records, "analysis/epoch_model_distributions.csv")
    epoch_rows = tagged_rows(records, "analysis/epoch_summary.csv")
    summary_rows = tagged_rows(records, "analysis/summary.csv")
    histograms = tagged_rows(records, "analysis/histograms.csv")
    train_rows = tagged_rows(records, "train_metrics.csv")
    late = aggregate_late(summary_rows)
    deltas = paired_late_deltas(summary_rows)
    metadata = read_json(records[0].directory / "tensor_metadata.json")

    write_csv(output / "run_manifest.csv", manifest)
    write_csv(output / "epoch_model_distributions.csv", distributions)
    write_csv(output / "epoch_group_summaries.csv", epoch_rows)
    write_csv(output / "full_and_late_group_summaries.csv", summary_rows)
    write_csv(output / "full_and_late_histograms.csv", histograms)
    write_csv(output / "training_curves.csv", train_rows)
    write_csv(output / "late_model_aggregate.csv", late)
    write_csv(output / "paired_late_deltas.csv", deltas)
    write_csv(output / "tensor_metadata.csv", [{key: value for key, value in row.items()} for row in metadata])

    style()
    for metric in METRICS:
        plot_model_evolution(distributions, plots, metric)
    plot_depth_evolution(epoch_rows, plots)
    for weighting in WEIGHTINGS:
        plot_kind_distributions(histograms, plots, weighting=weighting)
    plot_stage_occupancy(summary_rows, plots)
    plot_tensor_map(summary_rows, metadata, plots)
    if all(record.run.get("evaluation") for record in records):
        plot_performance(train_rows, plots)

    (output / "report.md").write_text(report_text(records, manifest, late, deltas, summary_rows))
    (output / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite": str(args.suite),
                "synthetic_pipeline_check": any(record.run.get("synthetic") for record in records),
                "conditions": list(CONDITIONS),
                "seeds": list(SEEDS),
                "n_independent_runs": len(records),
                "n_independent_runs_per_condition": len(SEEDS),
                "reference_coefficient": REFERENCE,
                "late_fraction": records[0].analysis["late_fraction"],
                "source_run_ids": [record.run.get("wandb", {}).get("run_id") for record in records],
                "figure_formats": ["png_400dpi", "pdf", "svg"],
                "uncertainty": "All bands are the observed min-max range across three seeds; faint lines show individual seeds.",
            },
            indent=2,
        )
        + "\n"
    )
    log_to_wandb(output, records, late, args)
    print(output / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
