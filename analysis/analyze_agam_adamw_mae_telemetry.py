"""Analyze lossless AGAM-AdamW gate telemetry from MAE ViT pre-training.

The primary estimand is the tensor-wise gate ``q_t``. Misalignment is defined
strictly as ``q_t < 0.5``; zero-gradient or undefined directions are excluded
from its denominator. Equal-tensor summaries are primary because AGAM applies
one gate per tensor. Parameter-count weighting is emitted as a sensitivity view.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from tasks.agam_adamw_mae_telemetry import FEATURE_INDEX, FEATURES, NORM_EPS

PERIODS = ("early", "middle", "late", "full")
COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#6F4E7C",
    "#8C6D31",
    "#4C78A8",
    "#B279A2",
    "#79706E",
    "#59A14F",
    "#F28E2B",
    "#9C755F",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, probabilities: Iterable[float]) -> np.ndarray:
    if not len(values):
        return np.full(len(tuple(probabilities)), np.nan)
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order].astype(np.float64, copy=False)
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(tuple(probabilities), cumulative, values, left=values[0], right=values[-1])


def summary_statistics(values: np.ndarray, weights: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[finite].astype(np.float64, copy=False), weights[finite].astype(np.float64, copy=False)
    if not len(values):
        return {
            "observations": 0,
            "weighted_observations": 0,
            "mean": math.nan,
            "std": math.nan,
            "q10": math.nan,
            "q25": math.nan,
            "median": math.nan,
            "q75": math.nan,
            "q90": math.nan,
            "misalignment_pct": math.nan,
            "severe_conflict_lt_0.25_pct": math.nan,
            "strong_alignment_gt_0.7_pct": math.nan,
            "near_full_alignment_gt_0.9_pct": math.nan,
        }
    total_weight = weights.sum()
    mean = float(np.dot(values, weights) / total_weight)
    variance = float(np.dot((values - mean) ** 2, weights) / total_weight)
    q10, q25, median, q75, q90 = weighted_quantiles(values, weights, (0.10, 0.25, 0.50, 0.75, 0.90))
    return {
        "observations": len(values),
        "weighted_observations": int(total_weight),
        "mean": mean,
        "std": math.sqrt(max(variance, 0.0)),
        "q10": float(q10),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "q90": float(q90),
        "misalignment_pct": float(100.0 * weights[values < 0.5].sum() / total_weight),
        "severe_conflict_lt_0.25_pct": float(100.0 * weights[values < 0.25].sum() / total_weight),
        "strong_alignment_gt_0.7_pct": float(100.0 * weights[values > 0.7].sum() / total_weight),
        "near_full_alignment_gt_0.9_pct": float(100.0 * weights[values > 0.9].sum() / total_weight),
    }


def load_run(run_directory: Path) -> dict[str, Any]:
    telemetry = run_directory / "telemetry"
    metadata = json.loads((run_directory / "tensor_metadata.json").read_text())
    index = json.loads((telemetry / "index.json").read_text())
    if tuple(index["features"]) != FEATURES:
        raise ValueError(f"Unexpected telemetry features: {index['features']}")

    values, observed, epochs, steps, losses = [], [], [], [], []
    expected_step = 1
    for chunk in index["chunks"]:
        path = telemetry / chunk["file"]
        with np.load(path) as shard:
            shard_steps = shard["step"].astype(np.int64)
            if int(shard_steps[0]) != expected_step or not np.array_equal(
                shard_steps, np.arange(expected_step, expected_step + len(shard_steps))
            ):
                raise ValueError(f"Non-contiguous telemetry shard: {path}")
            if shard["values"].shape[1:] != (len(metadata), len(FEATURES)):
                raise ValueError(f"Unexpected tensor/feature shape in {path}: {shard['values'].shape}")
            values.append(shard["values"].astype(np.float32))
            observed.append(shard["observed"].astype(bool))
            epochs.append(shard["epoch"].astype(np.int16))
            steps.append(shard_steps)
            losses.append(shard["loss"].astype(np.float32))
            expected_step += len(shard_steps)

    if expected_step - 1 != int(index["completed_steps"]):
        raise ValueError("The telemetry index and persisted shards disagree on completed steps.")
    return {
        "metadata": metadata,
        "index": index,
        "values": np.concatenate(values),
        "observed": np.concatenate(observed),
        "epoch": np.concatenate(epochs),
        "step": np.concatenate(steps),
        "loss": np.concatenate(losses),
    }


def group_indices(metadata: list[dict[str, Any]]) -> dict[tuple[str, str], np.ndarray]:
    groups: dict[tuple[str, str], list[int]] = {("model", "all"): list(range(len(metadata)))}
    for tensor_id, row in enumerate(metadata):
        for view in ("kind", "depth", "stage"):
            groups.setdefault((view, str(row[view])), []).append(tensor_id)
        groups[("tensor", str(row["name"]))] = [tensor_id]
    return {key: np.asarray(indices, dtype=np.int64) for key, indices in groups.items()}


def period_masks(epoch: np.ndarray) -> dict[str, np.ndarray]:
    maximum = int(epoch.max())
    early_end = max(1, math.ceil(0.10 * maximum))
    middle_start = max(1, math.floor(0.45 * maximum))
    middle_end = max(middle_start, math.ceil(0.55 * maximum))
    late_start = max(1, math.floor(0.80 * maximum) + 1)
    return {
        "early": epoch <= early_end,
        "middle": (epoch >= middle_start) & (epoch <= middle_end),
        "late": epoch >= late_start,
        "full": np.ones_like(epoch, dtype=bool),
    }


def summarize_groups(
    gate: np.ndarray,
    valid: np.ndarray,
    epoch: np.ndarray,
    metadata: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sizes = np.asarray([int(row["numel"]) for row in metadata], dtype=np.int64)
    groups = group_indices(metadata)
    epoch_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []

    def one(mask: np.ndarray, view: str, group: str, indices: np.ndarray, weighting: str) -> dict[str, Any]:
        block = gate[mask][:, indices]
        block_valid = valid[mask][:, indices]
        tensor_weights = np.ones(len(indices), dtype=np.int64) if weighting == "equal_tensor" else sizes[indices]
        weights = np.broadcast_to(tensor_weights, block.shape)[block_valid]
        stats = summary_statistics(block[block_valid], weights)
        return {
            "view": view,
            "group": group,
            "weighting": weighting,
            "tensor_count": len(indices),
            **stats,
        }

    for epoch_number in sorted(np.unique(epoch)):
        mask = epoch == epoch_number
        for (view, group), indices in groups.items():
            if view == "tensor":
                continue
            for weighting in ("equal_tensor", "numel_weighted"):
                epoch_rows.append({"epoch": int(epoch_number), **one(mask, view, group, indices, weighting)})

    for period, mask in period_masks(epoch).items():
        for (view, group), indices in groups.items():
            for weighting in ("equal_tensor",) if view == "tensor" else ("equal_tensor", "numel_weighted"):
                period_rows.append({"period": period, **one(mask, view, group, indices, weighting)})
    return epoch_rows, period_rows


def global_counterfactual(
    gate: np.ndarray,
    cosine: np.ndarray,
    gradient_norm: np.ndarray,
    probe_norm: np.ndarray,
    observed: np.ndarray,
    epoch: np.ndarray,
    step: np.ndarray,
    metadata: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sizes = np.asarray([int(row["numel"]) for row in metadata], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for index in range(len(step)):
        valid = (
            observed[index]
            & np.isfinite(gate[index])
            & np.isfinite(cosine[index])
            & np.isfinite(gradient_norm[index])
            & np.isfinite(probe_norm[index])
            & (gradient_norm[index] >= NORM_EPS)
            & (probe_norm[index] >= NORM_EPS)
        )
        if not valid.any():
            raise ValueError(f"No valid tensors at optimizer step {step[index]}")
        gnorm, pnorm, cos = gradient_norm[index, valid], probe_norm[index, valid], cosine[index, valid]
        dot = float(np.sum(cos * gnorm * pnorm, dtype=np.float64))
        denominator = float(np.sqrt(np.sum(gnorm**2, dtype=np.float64)) * np.sqrt(np.sum(pnorm**2, dtype=np.float64)))
        global_cosine = float(np.clip(dot / denominator, -1.0, 1.0))
        global_gate = 0.5 * (1.0 + global_cosine)
        tensor_gates = gate[index, valid].astype(np.float64)
        tensor_sizes = sizes[valid]
        tensor_conflict = tensor_gates < 0.5
        global_conflict = global_gate < 0.5
        disagreement = tensor_conflict != global_conflict
        q10, q25, median, q75, q90 = np.quantile(tensor_gates, (0.10, 0.25, 0.50, 0.75, 0.90))
        rows.append(
            {
                "step": int(step[index]),
                "epoch": int(epoch[index]),
                "valid_tensors": int(valid.sum()),
                "global_cosine": global_cosine,
                "global_gate": global_gate,
                "global_misaligned": int(global_conflict),
                "tensor_gate_mean": float(tensor_gates.mean()),
                "tensor_gate_std": float(tensor_gates.std()),
                "tensor_gate_q10": float(q10),
                "tensor_gate_q25": float(q25),
                "tensor_gate_median": float(median),
                "tensor_gate_q75": float(q75),
                "tensor_gate_q90": float(q90),
                "tensor_misalignment_pct": float(100 * tensor_conflict.mean()),
                "numel_weighted_tensor_gate_mean": float(np.average(tensor_gates, weights=tensor_sizes)),
                "numel_weighted_tensor_misalignment_pct": float(100 * np.average(tensor_conflict, weights=tensor_sizes)),
                "mean_abs_tensor_global_gap": float(np.abs(tensor_gates - global_gate).mean()),
                "numel_weighted_mean_abs_tensor_global_gap": float(
                    np.average(np.abs(tensor_gates - global_gate), weights=tensor_sizes)
                ),
                "tensor_global_decision_disagreement_pct": float(100 * disagreement.mean()),
                "numel_weighted_tensor_global_decision_disagreement_pct": float(
                    100 * np.average(disagreement, weights=tensor_sizes)
                ),
                "masked_conflict_pct": float(100 * tensor_conflict.mean()) if not global_conflict else 0.0,
                "overgeneralized_conflict_pct": float(100 * (~tensor_conflict).mean()) if global_conflict else 0.0,
                "mixed_alignment": int(tensor_conflict.any() and (~tensor_conflict).any()),
            }
        )

    epoch_rows: list[dict[str, Any]] = []
    numeric_fields = [key for key, value in rows[0].items() if key not in {"step", "epoch"} and isinstance(value, (int, float))]
    for epoch_number in sorted({int(row["epoch"]) for row in rows}):
        selected = [row for row in rows if row["epoch"] == epoch_number]
        epoch_rows.append(
            {
                "epoch": epoch_number,
                "steps": len(selected),
                **{field: float(np.mean([row[field] for row in selected])) for field in numeric_fields},
            }
        )
    return rows, epoch_rows


def configure_plots() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.7,
            "grid.color": "#E5E5E5",
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    return plt


def save_figure(figure: Any, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(directory / f"{stem}.png", dpi=400, bbox_inches="tight")


def make_figures(
    output: Path,
    epoch_rows: list[dict[str, Any]],
    period_rows: list[dict[str, Any]],
    counter_epoch: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
) -> None:
    plt = configure_plots()
    figures = output / "figures"

    model = [row for row in epoch_rows if row["view"] == "model" and row["weighting"] == "equal_tensor"]
    x = np.asarray([row["epoch"] for row in model])
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.7), sharex=True)
    axes[0].fill_between(x, [row["q10"] for row in model], [row["q90"] for row in model], color="#56B4E9", alpha=0.18, label="10th–90th percentile")
    axes[0].fill_between(x, [row["q25"] for row in model], [row["q75"] for row in model], color="#0072B2", alpha=0.28, label="25th–75th percentile")
    axes[0].plot(x, [row["median"] for row in model], color="#003B5C", linewidth=1.6, label="Tensor-step median")
    axes[0].plot(x, [row["mean"] for row in model], color="#D55E00", linewidth=1.2, label="Tensor-step mean")
    axes[0].axhline(0.5, color="#555555", linestyle="--", linewidth=1, label="Misalignment boundary")
    axes[0].set(ylabel="Alignment gate $q_t$", ylim=(0, 1), title="AGAM-AdamW gate evolution during MAE pre-training")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(x, [row["misalignment_pct"] for row in model], color="#CC79A7", linewidth=1.5, label="Per-tensor misalignment")
    axes[1].plot(x, [row["tensor_global_decision_disagreement_pct"] for row in counter_epoch], color="#009E73", linewidth=1.3, label="Tensor/global decision disagreement")
    axes[1].set(xlabel="Pre-training epoch", ylabel="Tensor-step observations (%)", ylim=(0, None))
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(True)
    fig.tight_layout()
    save_figure(fig, figures, "gate_evolution_overview")
    plt.close(fig)

    kinds = sorted({row["group"] for row in epoch_rows if row["view"] == "kind"})
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.1), sharex=True)
    for color, kind in zip(COLORS, kinds, strict=False):
        selected = [row for row in epoch_rows if row["view"] == "kind" and row["group"] == kind and row["weighting"] == "equal_tensor"]
        axes[0].plot([row["epoch"] for row in selected], [row["mean"] for row in selected], color=color, linewidth=1.05, label=kind.replace("_", " "))
        axes[1].plot([row["epoch"] for row in selected], [row["misalignment_pct"] for row in selected], color=color, linewidth=1.05, label=kind.replace("_", " "))
    axes[0].axhline(0.5, color="#555555", linestyle="--", linewidth=0.9)
    axes[0].set(ylabel="Mean gate", ylim=(0, 1), title="Gate dynamics by parameter kind")
    axes[1].set(xlabel="Pre-training epoch", ylabel="Misalignment (%)", ylim=(0, None))
    for axis in axes:
        axis.grid(True)
    axes[0].legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1)
    fig.tight_layout()
    save_figure(fig, figures, "gate_evolution_by_parameter_kind")
    plt.close(fig)

    depths = [
        depth
        for depth, _index in sorted(
            {(str(row["depth"]), int(row["depth_index"])) for row in metadata}, key=lambda item: item[1]
        )
    ]
    epochs = sorted({int(row["epoch"]) for row in epoch_rows})
    gate_lookup = {(int(row["epoch"]), str(row["group"])): float(row["mean"]) for row in epoch_rows if row["view"] == "depth" and row["weighting"] == "equal_tensor"}
    conflict_lookup = {(int(row["epoch"]), str(row["group"])): float(row["misalignment_pct"]) for row in epoch_rows if row["view"] == "depth" and row["weighting"] == "equal_tensor"}
    gate_matrix = np.asarray([[gate_lookup[(epoch_number, depth)] for epoch_number in epochs] for depth in depths])
    conflict_matrix = np.asarray([[conflict_lookup[(epoch_number, depth)] for epoch_number in epochs] for depth in depths])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.2), sharex=True, sharey=True)
    im0 = axes[0].imshow(gate_matrix, aspect="auto", vmin=0.45, vmax=0.9, cmap="cividis")
    upper = max(1.0, float(np.nanpercentile(conflict_matrix, 99)))
    im1 = axes[1].imshow(conflict_matrix, aspect="auto", vmin=0, vmax=upper, cmap="magma")
    ticks = np.unique(np.linspace(0, len(epochs) - 1, 7).astype(int))
    for axis in axes:
        axis.set_xticks(ticks, [epochs[index] for index in ticks])
        axis.set_yticks(range(len(depths)), [depth.replace("encoder.", "enc. ").replace("decoder.", "dec. ") for depth in depths])
        axis.set_xlabel("Pre-training epoch")
    axes[0].set_title("Mean alignment gate")
    axes[1].set_title("Misalignment frequency (%)")
    axes[0].set_ylabel("Model depth")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    fig.tight_layout()
    save_figure(fig, figures, "gate_depth_heatmaps")
    plt.close(fig)

    late = [row for row in period_rows if row["period"] == "late" and row["view"] == "kind" and row["weighting"] == "equal_tensor"]
    late.sort(key=lambda row: float(row["median"]))
    y = np.arange(len(late))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, max(4.2, 0.36 * len(late))), sharey=True)
    axes[0].hlines(y, [row["q10"] for row in late], [row["q90"] for row in late], color="#8DB9D9", linewidth=2.1)
    axes[0].hlines(y, [row["q25"] for row in late], [row["q75"] for row in late], color="#0072B2", linewidth=5)
    axes[0].scatter([row["median"] for row in late], y, color="#003B5C", s=18, zorder=3)
    axes[0].axvline(0.5, color="#555555", linestyle="--", linewidth=0.9)
    axes[0].set(xlabel="Gate distribution (10–90%, 25–75%, median)", xlim=(0, 1), yticks=y, yticklabels=[str(row["group"]).replace("_", " ") for row in late])
    axes[1].barh(y, [row["misalignment_pct"] for row in late], color="#CC79A7")
    axes[1].set(xlabel="Misalignment observations (%)")
    for axis in axes:
        axis.grid(axis="x")
    fig.suptitle("Late-training gate profile by parameter kind", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, figures, "late_gate_profile_by_parameter_kind")
    plt.close(fig)

    cx = np.asarray([row["epoch"] for row in counter_epoch])
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.7), sharex=True)
    axes[0].fill_between(cx, [row["tensor_gate_q25"] for row in counter_epoch], [row["tensor_gate_q75"] for row in counter_epoch], color="#0072B2", alpha=0.24, label="Per-tensor IQR")
    axes[0].plot(cx, [row["tensor_gate_median"] for row in counter_epoch], color="#003B5C", linewidth=1.5, label="Per-tensor median")
    axes[0].plot(cx, [row["global_gate"] for row in counter_epoch], color="#D55E00", linewidth=1.3, label="Counterfactual global gate")
    axes[0].axhline(0.5, color="#555555", linestyle="--", linewidth=0.9)
    axes[0].set(ylabel="Gate", ylim=(0, 1), title="A global gate conceals tensor-level heterogeneity")
    axes[0].legend(frameon=False)
    axes[1].plot(cx, [row["tensor_global_decision_disagreement_pct"] for row in counter_epoch], color="#009E73", linewidth=1.4, label="Decision disagreement")
    axes[1].plot(cx, [row["masked_conflict_pct"] for row in counter_epoch], color="#CC79A7", linewidth=1.2, label="Conflicts masked by global gate")
    axes[1].set(xlabel="Pre-training epoch", ylabel="Tensor-step observations (%)", ylim=(0, None))
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(True)
    fig.tight_layout()
    save_figure(fig, figures, "tensor_vs_global_gate_counterfactual")
    plt.close(fig)


def analyze(run_directory: Path, output: Path | None = None, *, plots: bool = True) -> Path:
    run_directory = run_directory.resolve()
    output = (output or run_directory / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    loaded = load_run(run_directory)
    metadata = loaded["metadata"]
    values = loaded["values"]
    observed = loaded["observed"]
    epoch, step = loaded["epoch"], loaded["step"]
    gate = values[:, :, FEATURE_INDEX["gate_q"]]
    cosine = values[:, :, FEATURE_INDEX["cosine"]]
    gradient_norm = values[:, :, FEATURE_INDEX["gradient_norm"]]
    probe_norm = values[:, :, FEATURE_INDEX["probe_norm"]]
    valid = (
        observed
        & np.isfinite(gate)
        & np.isfinite(cosine)
        & np.isfinite(gradient_norm)
        & np.isfinite(probe_norm)
        & (gradient_norm > 0)
        & (probe_norm > 0)
    )
    if not np.all((gate[valid] >= 0) & (gate[valid] <= 1)):
        raise ValueError("Observed gate outside [0, 1].")

    epoch_rows, period_rows = summarize_groups(gate, valid, epoch, metadata)
    counter_steps, counter_epoch = global_counterfactual(
        gate, cosine, gradient_norm, probe_norm, observed, epoch, step, metadata
    )
    write_csv(output / "tables" / "epoch_group_gate_statistics.csv", epoch_rows)
    write_csv(output / "tables" / "period_group_gate_statistics.csv", period_rows)
    write_csv(output / "tables" / "step_global_gate_counterfactual.csv", counter_steps)
    write_csv(output / "tables" / "epoch_global_gate_counterfactual.csv", counter_epoch)
    if plots:
        make_figures(output, epoch_rows, period_rows, counter_epoch, metadata)

    late_model = next(
        row
        for row in period_rows
        if row["period"] == "late" and row["view"] == "model" and row["group"] == "all" and row["weighting"] == "equal_tensor"
    )
    late_counter = [row for row in counter_steps if row["epoch"] >= math.floor(0.8 * int(epoch.max())) + 1]
    summary = {
        "status": "complete",
        "run_directory": str(run_directory),
        "completed_steps": int(loaded["index"]["completed_steps"]),
        "epochs": int(epoch.max()),
        "tensor_count": len(metadata),
        "valid_tensor_step_observations": int(valid.sum()),
        "missing_or_undefined_tensor_step_observations": int(valid.size - valid.sum()),
        "late_equal_tensor_gate_mean": float(late_model["mean"]),
        "late_equal_tensor_gate_median": float(late_model["median"]),
        "late_equal_tensor_gate_std": float(late_model["std"]),
        "late_equal_tensor_misalignment_pct": float(late_model["misalignment_pct"]),
        "late_steps_with_mixed_alignment_pct": float(100 * np.mean([row["mixed_alignment"] for row in late_counter])),
        "late_tensor_global_decision_disagreement_pct": float(
            np.mean([row["tensor_global_decision_disagreement_pct"] for row in late_counter])
        ),
        "late_conflicts_masked_by_global_gate_pct": float(np.mean([row["masked_conflict_pct"] for row in late_counter])),
        "primary_weighting": "equal_tensor",
        "sensitivity_weighting": "parameter_count",
        "misalignment_definition": "gate_q < 0.5 among defined nonzero alignment pairs",
    }
    write_json(output / "summary.json", summary)

    readme = "# AGAM-AdamW MAE telemetry\n\n"
    readme += f"Lossless observations: **{summary['completed_steps']:,} optimizer steps × {summary['tensor_count']} tensors**. "
    readme += "The primary aggregation gives every tensor-step one vote; parameter-count-weighted rows are a sensitivity analysis.\n\n"
    readme += "## Late-training profile (final 20% of epochs)\n\n"
    readme += f"- Mean gate: **{summary['late_equal_tensor_gate_mean']:.4f}**\n"
    readme += f"- Median gate: **{summary['late_equal_tensor_gate_median']:.4f}**\n"
    readme += f"- Gate standard deviation: **{summary['late_equal_tensor_gate_std']:.4f}**\n"
    readme += f"- Misalignment (`gate < 0.5`): **{summary['late_equal_tensor_misalignment_pct']:.3f}%**\n"
    readme += f"- Steps containing both aligned and misaligned tensors: **{summary['late_steps_with_mixed_alignment_pct']:.2f}%**\n"
    readme += f"- Tensor/global gate decision disagreement: **{summary['late_tensor_global_decision_disagreement_pct']:.3f}%**\n"
    readme += f"- Tensor conflicts masked by an aligned global gate: **{summary['late_conflicts_masked_by_global_gate_pct']:.3f}%**\n\n"
    readme += "These are descriptive telemetry statistics, not causal estimates. The global-gate series is a counterfactual computed from the same recorded gradient/probe pairs; it is not a separately trained model.\n"
    (output / "README.md").write_text(readme)
    manifest_files = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"
    )
    write_json(
        output / "manifest.json",
        {
            "analysis": "AGAM-AdamW MAE ViT gate telemetry",
            "source_steps": summary["completed_steps"],
            "source_tensors": summary["tensor_count"],
            "files": [{"path": str(path.relative_to(output)), "sha256": sha256(path)} for path in manifest_files],
        },
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    destination = analyze(args.run_directory, args.output, plots=not args.no_plots)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
