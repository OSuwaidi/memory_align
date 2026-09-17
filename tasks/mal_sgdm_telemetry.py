"""Exact per-step, per-tensor AGAM-SGD / SGDM telemetry on a CIFAR ResNet18.

Usage and file schema: tasks/mal_sgdm_telemetry.md. W&B logging is optional.
The train command automatically analyzes the completed measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets
from torchvision.models import resnet18
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optims.agam_opt import AGAM_SGD
from tasks.agam_sgd_diagnostics import ALIGNMENT_FEATURES, ObservedSGDM, alignment_scalars
from tasks.wandb_metadata import task_metadata

FEATURES = ("cosine", "gate_q", "beta_eff", "gradient_norm", "probe_norm")
METRICS = ("gate_q", "beta_eff")
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURES)}
LOW, REFERENCE = 0.5, 0.7
HIST_EDGES = np.linspace(0.0, 1.0, 51)
NORM_EPS = 1e-8  # get_norms_and_eff_beta's denominator floor, not an analysis cutoff.


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_model(norm: str = "group") -> nn.Module:
    """Match tasks/cifar_train.py's GroupNorm architecture by default."""
    norm_layer = (lambda channels: nn.GroupNorm(min(32, channels // 4), channels)) if norm == "group" else nn.BatchNorm2d
    model = resnet18(weights=None, norm_layer=norm_layer)
    model.conv1 = nn.Conv2d(3, model.conv1.out_channels, 3, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def tensor_metadata(model: nn.Module, optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    modules = dict(model.named_modules())
    groups = {id(p): (i, group) for i, group in enumerate(optimizer.param_groups) for p in group["params"]}
    rows = []
    norm_types = (nn.modules.batchnorm._BatchNorm, nn.GroupNorm, nn.LayerNorm, nn.modules.instancenorm._InstanceNorm)
    for name, parameter in model.named_parameters():
        if id(parameter) not in groups:
            continue
        module_name, _, role = name.rpartition(".")
        module = modules[module_name]
        family = (
            "conv"
            if isinstance(module, nn.modules.conv._ConvNd)
            else "linear"
            if isinstance(module, nn.Linear)
            else "norm"
            if isinstance(module, norm_types)
            else "other"
        )
        parts = name.split(".")
        if parts[0] in ("conv1", "bn1"):
            stage, stage_index, depth, depth_index, block_index = "stem", 0, "stem", 0, -1
        elif parts[0].startswith("layer") and parts[0][5:].isdigit():
            stage, stage_index, block_index = parts[0], int(parts[0][5:]), int(parts[1])
            depth = f"{stage}.{block_index}"
            depth_index = 1 + sum(len(getattr(model, f"layer{i}")) for i in range(1, stage_index)) + block_index
        elif parts[0] == "fc":
            stage, stage_index, depth, block_index = "head", 5, "head", -1
            depth_index = 1 + sum(len(getattr(model, f"layer{i}")) for i in range(1, 5))
        else:
            raise ValueError(f"Unclassified ResNet depth: {name}")
        group_id, group = groups[id(parameter)]
        rows.append(
            {
                "tensor_id": len(rows),
                "name": name,
                "module": module_name,
                "module_type": type(module).__name__,
                "family": family,
                "role": role,
                "kind": f"{family}_{role}",
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "dtype": str(parameter.dtype),
                "stage": stage,
                "stage_index": stage_index,
                "depth": depth,
                "depth_index": depth_index,
                "block_index": block_index,
                "branch": "shortcut" if "downsample" in parts else "main",
                "optimizer_group": group_id,
                "weight_decay": group["weight_decay"],
                "beta": group.get("beta", group.get("momentum")),
                "pwr": group.get("pwr", 1.0),
                "gate_mode": group.get("gate_mode", "disabled"),
            }
        )
    return rows


class GateRecorder:
    """Collect observer scalars without a device-to-host copy per parameter.

    Call begin_step before optimizer.step and end_step only after it succeeds.
    A shard preserves every completed step; compression is lossless.
    """

    def __init__(self, directory: Path, model: nn.Module, metadata: list[dict[str, Any]], flush_steps: int = 256,
                 *, alignment: bool = False):
        if flush_steps < 1:
            raise ValueError("flush_steps must be positive")
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "index.json").exists():
            raise ValueError(f"Refusing to overwrite existing telemetry: {directory}")
        self.flush_steps = flush_steps
        parameters = dict(model.named_parameters())
        self.ids = {id(parameters[row["name"]]): row["tensor_id"] for row in metadata}
        first = parameters[metadata[0]["name"]]
        if any(p.device != first.device or p.dtype != first.dtype for name, p in parameters.items() if id(p) in self.ids):
            raise ValueError("This recorder expects all observed parameters to share a device and dtype")
        self.nan = first.new_full((), float("nan"))
        self.tensor_count = len(metadata)
        self.numel_weights = np.asarray([row["numel"] for row in metadata], dtype=np.int64)
        self.tensor_steps = np.zeros(self.tensor_count, dtype=np.int64)
        self.active = False
        self.alignment = alignment
        self.pending: list[torch.Tensor] = []
        self.pending_alignment: list[torch.Tensor] = []
        self.seen = np.zeros(self.tensor_count, dtype=bool)
        self.buffer: list[tuple[torch.Tensor, dict[str, Any]]] = []
        self.epoch_statistics: dict[int, Statistics] = {}
        self.epoch_alignment: dict[int, dict[str, np.ndarray]] = {}
        self.index: dict[str, Any] = {"schema_version": 3 if alignment else 2, "features": list(FEATURES), "chunks": [], "completed_steps": 0}
        if alignment:
            self.index["alignment_features"] = list(ALIGNMENT_FEATURES)
        write_json(directory / "index.json", self.index)

    @property
    def completed_steps(self) -> int:
        return self.buffer[-1][1]["step"] if self.buffer else self.index["completed_steps"]

    def begin_step(self) -> None:
        if self.active:
            raise RuntimeError("Previous telemetry step was not completed or aborted")
        self.pending = [self.nan] * (self.tensor_count * len(FEATURES))
        self.pending_alignment = [self.nan] * (self.tensor_count * len(ALIGNMENT_FEATURES)) if self.alignment else []
        self.alignment_seen = np.zeros(self.tensor_count, dtype=bool)
        self.seen = np.zeros(self.tensor_count, dtype=bool)
        self.active = True

    def __call__(
        self,
        parameter: nn.Parameter,
        cosine: torch.Tensor,
        gate: torch.Tensor,
        coefficient: torch.Tensor,
        gradient_norm: torch.Tensor,
        probe_norm: torch.Tensor,
    ) -> None:
        if not self.active:
            raise RuntimeError("begin_step must precede optimizer.step")
        tensor_id = self.ids[id(parameter)]
        if self.seen[tensor_id]:
            raise RuntimeError("A tensor was observed twice in one optimizer step")
        self.seen[tensor_id] = True
        offset = tensor_id * len(FEATURES)
        self.pending[offset : offset + len(FEATURES)] = [value.detach() for value in (cosine, gate, coefficient, gradient_norm, probe_norm)]

    def observe_alignment(self, parameter, gradient, history, probe, applied) -> None:
        if not self.active or not self.alignment:
            raise RuntimeError("Extended alignment recording is not active")
        tensor_id = self.ids[id(parameter)]
        if self.alignment_seen[tensor_id]:
            raise RuntimeError("Duplicate alignment observation")
        self.alignment_seen[tensor_id] = True
        offset = tensor_id * len(ALIGNMENT_FEATURES)
        self.pending_alignment[offset:offset + len(ALIGNMENT_FEATURES)] = [
            value.detach() for value in alignment_scalars(parameter, gradient, history, probe, applied)
        ]

    def end_step(self, *, epoch: int, batch: int, batch_size: int, samples_seen: int, learning_rates: list[float], loss: torch.Tensor) -> None:
        if not self.active:
            raise RuntimeError("No telemetry step is active")
        if self.alignment and not np.array_equal(self.seen, self.alignment_seen):
            raise RuntimeError("Gate/alignment coverage differs within this step")
        packed = torch.stack([*self.pending, *self.pending_alignment, loss.detach()])
        record = {
            "step": self.completed_steps + 1,
            "epoch": epoch,
            "batch": batch,
            "batch_size": batch_size,
            "samples_seen": samples_seen,
            "learning_rates": learning_rates,
            "observed": self.seen.copy(),
            "tensor_step": self.tensor_steps + self.seen,
        }
        # Commit data and metadata together so an interruption cannot separate them.
        self.buffer.append((packed, record))
        self.tensor_steps = record["tensor_step"]
        self.abort_step()
        if len(self.buffer) >= self.flush_steps:
            self.flush()

    def abort_step(self) -> None:
        self.active = False
        self.pending = []
        self.pending_alignment = []

    def flush(self) -> None:
        if not self.buffer:
            return
        # One bulk transfer per shard, rather than .item() for every tensor.
        packed = torch.stack([value for value, _ in self.buffer]).cpu().numpy()
        records = [record for _, record in self.buffer]
        arrays = {key: np.asarray([record[key] for record in records]) for key in records[0]}
        boundary = self.tensor_count * len(FEATURES)
        arrays["values"] = packed[:, :boundary].reshape(len(self.buffer), self.tensor_count, len(FEATURES))
        if self.alignment:
            arrays["alignment_values"] = packed[:, boundary:-1].reshape(len(self.buffer), self.tensor_count, len(ALIGNMENT_FEATURES))
        arrays["loss"] = packed[:, -1]
        for epoch in np.unique(arrays["epoch"]):
            mask = arrays["epoch"] == epoch
            self.epoch_statistics.setdefault(int(epoch), Statistics(self.tensor_count)).add(arrays["values"][mask], arrays["observed"][mask])
            from analysis.agam_alignment_analysis import signals_from_arrays, valid_alignment
            signals = signals_from_arrays(arrays["values"][mask], arrays["alignment_values"][mask] if self.alignment else None)
            counts = self.epoch_alignment.setdefault(int(epoch), {})
            for signal, (cosine, gn, dn) in signals.items():
                valid = valid_alignment(cosine, gn, dn, arrays["observed"][mask])
                totals = counts.setdefault(signal, np.zeros((2, self.tensor_count), dtype=np.int64))
                totals[0] += valid.sum(axis=0)
                totals[1] += (valid & (cosine < 0)).sum(axis=0)
        first, last = int(arrays["step"][0]), int(arrays["step"][-1])
        filename = f"steps_{first:08d}_{last:08d}.npz"
        write_npz(self.directory / filename, **arrays)
        entry = {"file": filename, "first_step": first, "last_step": last, "steps": len(self.buffer)}
        # Idempotent if interrupted immediately after publishing the previous index.
        chunks = self.index["chunks"]
        if not chunks or chunks[-1]["file"] != filename:
            chunks = [*chunks, entry]
        updated = {**self.index, "chunks": chunks, "completed_steps": last}
        # The index only advertises completely written shards.
        write_json(self.directory / "index.json", updated)
        self.index = updated
        self.buffer.clear()

    def pop_epoch_aggregates(self, epoch: int) -> dict[str, float]:
        """Return compact model-wide aggregates for W&B, then release them."""
        stats = self.epoch_statistics.pop(epoch)
        groups = {"model": {"all": list(range(self.tensor_count))}}
        result: dict[str, float] = {}
        for weighting, weights in (
            ("equal_tensor", np.ones(self.tensor_count, dtype=np.int64)),
            ("numel_weighted", self.numel_weights),
        ):
            rows, _ = stats.rows(
                groups,
                tensor_weights=weights,
                weighting=weighting,
                include_histograms=False,
            )
            for row in rows:
                prefix = f"telemetry/{row['metric']}/{weighting}"
                result[f"{prefix}/mean"] = row["mean"]
                result[f"{prefix}/pct_lt_0_5"] = row["pct_lt_0_5"]
                result[f"{prefix}/pct_0_5_to_0_7"] = row["pct_0_5_to_0_7"]
                result[f"{prefix}/pct_gt_0_7"] = row["pct_gt_0_7"]
        for signal, (valid, negative) in self.epoch_alignment.pop(epoch).items():
            for weighting, weights in (("equal_tensor", np.ones(self.tensor_count, dtype=np.int64)), ("numel_weighted", self.numel_weights)):
                n, k = int(valid @ weights), int(negative @ weights)
                prefix = f"telemetry/alignment/{signal}/{weighting}"
                result[f"{prefix}/weighted_valid"] = n
                result[f"{prefix}/pct_negative"] = 100 * k / n if n else float("nan")
        return result


def alignment_masks(values: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = values[:, :, [FEATURE_INDEX["gradient_norm"], FEATURE_INDEX["probe_norm"]]]
    defined = observed & np.isfinite(norms).all(axis=2) & (norms > 0.0).all(axis=2)
    clamped = defined & (norms < NORM_EPS).any(axis=2)
    return defined, clamped


class Statistics:
    """Sufficient statistics per tensor; grouping never weights by tensor size."""

    def __init__(self, tensor_count: int):
        shape = (tensor_count, len(METRICS))
        self.count = np.zeros(shape, dtype=np.int64)
        self.total = np.zeros(shape)
        self.squares = np.zeros(shape)
        self.minimum = np.full(shape, np.inf)
        self.maximum = np.full(shape, -np.inf)
        self.bins = np.zeros((*shape, 3), dtype=np.int64)
        self.hist = np.zeros((*shape, len(HIST_EDGES) - 1), dtype=np.int64)
        self.observed = np.zeros(tensor_count, dtype=np.int64)
        self.undefined = np.zeros(tensor_count, dtype=np.int64)
        self.clamped = np.zeros(tensor_count, dtype=np.int64)
        self.zero_gradient = np.zeros(tensor_count, dtype=np.int64)
        self.nonfinite = np.zeros(shape, dtype=np.int64)
        self.steps = 0

    def add(self, values: np.ndarray, observed: np.ndarray) -> None:
        if not len(values):
            return
        defined, clamped = alignment_masks(values, observed)
        v = values[:, :, [FEATURE_INDEX[metric] for metric in METRICS]].astype(np.float64)
        valid = observed[:, :, None] & np.isfinite(v)
        self.nonfinite += (observed[:, :, None] & ~np.isfinite(v)).sum(axis=0)
        valid[:, :, 0] &= defined
        self.steps += len(v)
        self.observed += observed.sum(axis=0)
        self.undefined += (observed & ~defined).sum(axis=0)
        self.clamped += clamped.sum(axis=0)
        self.zero_gradient += (observed & (values[:, :, FEATURE_INDEX["gradient_norm"]] == 0.0)).sum(axis=0)
        self.count += valid.sum(axis=0)
        safe = np.where(valid, v, 0.0)
        self.total += safe.sum(axis=0)
        self.squares += (safe * safe).sum(axis=0)
        self.minimum = np.minimum(self.minimum, np.where(valid, v, np.inf).min(axis=0))
        self.maximum = np.maximum(self.maximum, np.where(valid, v, -np.inf).max(axis=0))
        # Compare the recorded values without decimal rounding; both boundaries belong to the middle bin.
        for i, mask in enumerate((v < LOW, (v >= LOW) & (v <= REFERENCE), v > REFERENCE)):
            self.bins[:, :, i] += (valid & mask).sum(axis=0)
        for metric in range(len(METRICS)):
            step_ids, tensor_ids = np.nonzero(valid[:, :, metric])
            raw = v[step_ids, tensor_ids, metric]
            if np.any((raw < 0.0) | (raw > 1.0)):
                raise ValueError("Gate outside [0, 1]; inspect the raw telemetry before summarizing")
            hist_ids = np.minimum(np.searchsorted(HIST_EDGES, raw, side="right") - 1, len(HIST_EDGES) - 2)
            np.add.at(self.hist[:, metric], (tensor_ids, hist_ids), 1)

    def rows(
        self,
        groups: dict[str, dict[str, list[int]]],
        *,
        tensor_weights: np.ndarray | None = None,
        weighting: str = "equal_tensor",
        include_histograms: bool = True,
        **labels: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if tensor_weights is None:
            tensor_weights = np.ones(self.count.shape[0], dtype=np.int64)
        tensor_weights = np.asarray(tensor_weights, dtype=np.int64)
        if tensor_weights.shape != (self.count.shape[0],) or np.any(tensor_weights <= 0):
            raise ValueError("tensor_weights must contain one positive integer per tensor")
        summaries, histograms = [], []
        for view, mapping in groups.items():
            for group, indices in mapping.items():
                weights = tensor_weights[indices]
                for metric_id, metric in enumerate(METRICS):
                    per_tensor_count = self.count[indices, metric_id]
                    n = int(per_tensor_count.sum())
                    weighted_n = int(np.dot(per_tensor_count, weights))
                    total = float(np.dot(self.total[indices, metric_id], weights))
                    mean = total / weighted_n if weighted_n else float("nan")
                    second_moment = float(np.dot(self.squares[indices, metric_id], weights)) / weighted_n if weighted_n else float("nan")
                    std = math.sqrt(max(0.0, second_moment - mean * mean)) if weighted_n else float("nan")
                    occupancy = (self.bins[indices, metric_id] * weights[:, None]).sum(axis=0)
                    base = dict(**labels, metric=metric, view=view, group=group, weighting=weighting)
                    summaries.append(
                        dict(
                            **base,
                            tensor_count=len(indices),
                            count=n,
                            weighted_count=weighted_n,
                            mean=mean,
                            std=std,
                            minimum=float(self.minimum[indices, metric_id].min()) if n else float("nan"),
                            maximum=float(self.maximum[indices, metric_id].max()) if n else float("nan"),
                            pct_lt_0_5=100.0 * int(occupancy[0]) / weighted_n if weighted_n else float("nan"),
                            pct_0_5_to_0_7=100.0 * int(occupancy[1]) / weighted_n if weighted_n else float("nan"),
                            pct_gt_0_7=100.0 * int(occupancy[2]) / weighted_n if weighted_n else float("nan"),
                            missing_grad=int(self.steps * len(indices) - self.observed[indices].sum()),
                            undefined_alignment=int(self.undefined[indices].sum()),
                            epsilon_clamped_alignment=int(self.clamped[indices].sum()),
                            zero_gradient_fallback=int(self.zero_gradient[indices].sum()),
                            nonfinite_coefficient=int(self.nonfinite[indices, metric_id].sum()),
                        )
                    )
                    if include_histograms:
                        raw_hist = self.hist[indices, metric_id].sum(axis=0)
                        weighted_hist = (self.hist[indices, metric_id] * weights[:, None]).sum(axis=0)
                        for i, (raw_count, weighted_count) in enumerate(zip(raw_hist, weighted_hist, strict=True)):
                            histograms.append(
                                dict(
                                    **base,
                                    left=float(HIST_EDGES[i]),
                                    right=float(HIST_EDGES[i + 1]),
                                    count=int(raw_count),
                                    weighted_count=int(weighted_count),
                                )
                            )
        return summaries, histograms


def group_indices(metadata: list[dict[str, Any]]) -> dict[str, dict[str, list[int]]]:
    groups: dict[str, dict[str, list[int]]] = {"model": {"all": list(range(len(metadata)))}}
    for view, key in (("tensor", "name"), ("kind", "kind"), ("role", "role"), ("depth", "depth"), ("stage", "stage")):
        groups[view] = {}
        for row in metadata:
            groups[view].setdefault(row[key], []).append(row["tensor_id"])
    return groups


def plot_results(
    destination: Path,
    metadata: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    epochs: list[dict[str, Any]],
    histograms: list[dict[str, Any]],
    epoch_distributions: list[dict[str, Any]],
    run_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    epoch_numbers = sorted({row["epoch"] for row in epochs})
    names = [row["name"] for row in metadata]
    labels = {"gate_q": r"Alignment gate $q_t$", "beta_eff": r"Applied memory coefficient $c_t$ (beta_eff)"}
    threshold_colors = ("#0072B2", "#E69F00", "#CC79A7")
    threshold_hatches = ("///", "", "\\\\")
    line_colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9")
    depth_order = [row["depth"] for row in sorted(metadata, key=lambda row: row["depth_index"])]
    depth_order = list(dict.fromkeys(depth_order))
    stage_order = [stage for stage in ("stem", "layer1", "layer2", "layer3", "layer4", "head") if any(row["stage"] == stage for row in metadata)]
    kind_order = list(dict.fromkeys(row["kind"] for row in metadata))

    def finish(fig, stem: str, title: str, *, use_tight_layout: bool = True) -> None:
        fig.suptitle(title, y=0.995, fontweight="bold")
        fig.text(0.5, 0.002, run_label, ha="center", va="bottom", fontsize=7, color="#555555")
        if use_tight_layout:
            fig.tight_layout(rect=(0, 0.025, 1, 0.965))
        for suffix, kwargs in (
            ("png", {"dpi": 400}),
            ("pdf", {}),
            ("svg", {}),
        ):
            fig.savefig(destination / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
        plt.close(fig)

    def occupancy(axis, selected: list[dict[str, Any]], title: str, order: list[str]) -> None:
        lookup = {row["group"]: row for row in selected}
        selected = [lookup[group] for group in order if group in lookup]
        left = np.zeros(len(selected))
        for field, label, color, hatch in zip(
            ("pct_lt_0_5", "pct_0_5_to_0_7", "pct_gt_0_7"),
            ("< 0.5", "0.5–0.7", "> 0.7"),
            threshold_colors,
            threshold_hatches,
            strict=True,
        ):
            widths = np.array([row[field] for row in selected])
            axis.barh(range(len(selected)), widths, left=left, label=label, color=color, hatch=hatch, edgecolor="white", linewidth=0.35)
            left += widths
        axis.set_yticks(range(len(selected)), [row["group"] for row in selected])
        axis.set_ylim(len(selected) - 0.5, -0.5)
        axis.set(xlim=(0, 100), xlabel="% of valid tensor-step observations", title=title)
        axis.grid(axis="x", color="#dddddd", linewidth=0.5, zorder=0)

    for metric in METRICS:
        erows = [row for row in epochs if row["metric"] == metric]
        srows = [row for row in summaries if row["metric"] == metric]

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True, sharey=True)
        for axis, weighting, panel in zip(axes, ("equal_tensor", "numel_weighted"), ("Equal tensor", "Parameter-count weighted"), strict=True):
            selected = sorted(
                (row for row in epoch_distributions if row["metric"] == metric and row["weighting"] == weighting),
                key=lambda row: row["epoch"],
            )
            x = np.array([row["epoch"] for row in selected])
            median = np.array([row["median"] for row in selected])
            q10, q25 = (np.array([row[key] for row in selected]) for key in ("q10", "q25"))
            q75, q90 = (np.array([row[key] for row in selected]) for key in ("q75", "q90"))
            axis.fill_between(x, q10, q90, color="#56B4E9", alpha=0.18, label="10th–90th percentile")
            axis.fill_between(x, q25, q75, color="#0072B2", alpha=0.28, label="25th–75th percentile")
            axis.plot(x, median, color="#003B5C", linewidth=1.6, label="Median")
            axis.axhline(REFERENCE, color="#D55E00", linestyle="--", linewidth=1.1, label="0.7 reference")
            axis.set(xlabel="Epoch", ylabel=labels[metric], ylim=(0, 1), title=panel)
            axis.grid(color="#e6e6e6", linewidth=0.5)
        axes[0].legend(frameon=False, ncol=2, loc="lower right")
        finish(fig, f"{metric}_model_evolution", f"Model-wide evolution of {labels[metric].lower()}")

        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharex=True, sharey=True)
        for axis, weighting, panel in zip(axes, ("equal_tensor", "numel_weighted"), ("Equal tensor", "Parameter-count weighted"), strict=True):
            lookup = {(row["epoch"], row["group"]): row["mean"] for row in erows if row["view"] == "depth" and row["weighting"] == weighting}
            matrix = np.asarray([[lookup[(epoch, depth)] for epoch in epoch_numbers] for depth in depth_order])
            im = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="cividis", origin="upper")
            ticks = np.unique(np.linspace(0, len(epoch_numbers) - 1, min(8, len(epoch_numbers))).astype(int))
            axis.set_xticks(ticks, [epoch_numbers[i] for i in ticks])
            axis.set_yticks(range(len(depth_order)), depth_order)
            axis.set(xlabel="Epoch", ylabel="Residual-block depth", title=panel)
        fig.subplots_adjust(left=0.08, right=0.88, bottom=0.15, top=0.86, wspace=0.16)
        color_axis = fig.add_axes((0.905, 0.18, 0.018, 0.63))
        fig.colorbar(im, cax=color_axis, label=labels[metric])
        finish(
            fig,
            f"{metric}_depth_progress",
            f"Depth-wise evolution of {labels[metric].lower()}",
            use_tight_layout=False,
        )

        fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), sharex=True, sharey=True)
        centers = (HIST_EDGES[:-1] + HIST_EDGES[1:]) / 2
        for axis, weighting, panel in zip(axes, ("equal_tensor", "numel_weighted"), ("Equal tensor", "Parameter-count weighted"), strict=True):
            for color, kind in zip(line_colors, kind_order, strict=False):
                selected = [
                    row
                    for row in histograms
                    if row["metric"] == metric and row["view"] == "kind" and row["group"] == kind and row.get("period") == "late" and row["weighting"] == weighting
                ]
                field = "count" if weighting == "equal_tensor" else "weighted_count"
                counts = np.asarray([row[field] for row in selected], dtype=float)
                axis.plot(centers, 100.0 * counts / max(1.0, counts.sum()), color=color, linewidth=1.5, label=kind.replace("_", " "))
            axis.axvline(REFERENCE, color="#555555", linestyle="--", linewidth=1)
            axis.set(xlabel=labels[metric], ylabel="% per 0.02-wide bin", xlim=(0, 1), title=panel)
            axis.grid(color="#e6e6e6", linewidth=0.5)
        axes[1].legend(frameon=False, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")
        finish(fig, f"{metric}_kind_late_distribution", "Final-20% distribution by parameter kind")

        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharex=True, sharey=True)
        for axis, weighting, panel in zip(axes, ("equal_tensor", "numel_weighted"), ("Equal tensor", "Parameter-count weighted"), strict=True):
            selected = [row for row in srows if row["view"] == "stage" and row["period"] == "late" and row["weighting"] == weighting]
            occupancy(axis, selected, panel, stage_order)
        axes[0].legend(frameon=False, ncol=3, bbox_to_anchor=(0, 1.16), loc="upper left")
        finish(fig, f"{metric}_stage_late_occupancy", "Final-20% coefficient regimes by network stage")

        fig, ax = plt.subplots(figsize=(max(10, len(epoch_numbers) / 15), max(8, len(names) * 0.22)))
        lookup = {(row["epoch"], row["group"]): row["mean"] for row in erows if row["view"] == "tensor" and row["weighting"] == "equal_tensor"}
        matrix = [[lookup[(epoch, name)] for epoch in epoch_numbers] for name in names]
        im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ticks = np.unique(np.linspace(0, len(epoch_numbers) - 1, min(12, len(epoch_numbers))).astype(int))
        ax.set_xticks(ticks, [epoch_numbers[i] for i in ticks])
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.set(xlabel="Epoch", title="Mean for every tensor and epoch")
        fig.colorbar(im, ax=ax, label="Mean")
        finish(fig, f"{metric}_tensor_heatmap", f"Tensor-wise evolution of {labels[metric].lower()}")

        fig, axes = plt.subplots(1, 3, figsize=(20, max(9, len(names) * 0.23)), sharey=True)
        for period, marker in (("full", "o"), ("late", "x")):
            selected = [row for row in srows if row["view"] == "tensor" and row["period"] == period and row["weighting"] == "equal_tensor"]
            axes[0].scatter([row["mean"] for row in selected], range(len(names)), marker=marker, s=16, label=period)
        axes[0].axvline(REFERENCE, color="gray", linestyle="--")
        axes[0].set(xlim=(0, 1), xlabel="Mean", title="Full run / late estimate")
        axes[0].legend()
        for axis, period in zip(axes[1:], ("full", "late"), strict=True):
            occupancy(
                axis,
                [row for row in srows if row["view"] == "tensor" and row["period"] == period and row["weighting"] == "equal_tensor"],
                f"{period}: threshold occupancy",
                names,
            )
        # Shared y-axis is inverted once, after setting all panels.
        axes[0].set_ylim(len(names) - 0.5, -0.5)
        axes[0].tick_params(axis="y", labelsize=7)
        axes[-1].legend(fontsize=7, loc="lower right")
        finish(fig, f"{metric}_tensor_summary", f"Tensor-level summary of {labels[metric].lower()}")


def analyze_run(directory: Path, late_fraction: float = 0.2, plots: bool = True, alignment_window: int = 100) -> Path:
    if not 0.0 < late_fraction <= 1.0:
        raise ValueError("late_fraction must be in (0, 1]")
    metadata = json.loads((directory / "tensor_metadata.json").read_text())
    run = json.loads((directory / "run.json").read_text())
    index = json.loads((directory / "telemetry" / "index.json").read_text())
    if index["features"] != list(FEATURES) or index["schema_version"] not in (2, 3):
        raise ValueError("Unsupported telemetry schema")
    total_steps = index["completed_steps"]
    if not total_steps:
        raise ValueError("No persisted optimizer steps to analyze")
    late_steps = max(1, math.ceil(total_steps * late_fraction))
    late_start = total_steps - late_steps + 1
    tensor_count = len(metadata)
    numel_weights = np.asarray([row["numel"] for row in metadata], dtype=np.int64)
    betas = np.asarray([row["beta"] for row in metadata], dtype=np.float64)
    powers = np.asarray([row["pwr"] for row in metadata], dtype=np.float64)
    groups = group_indices(metadata)
    aggregate_groups = {view: mapping for view, mapping in groups.items() if view != "tensor"}
    full, late = Statistics(tensor_count), Statistics(tensor_count)
    per_epoch: dict[int, Statistics] = {}
    epoch_samples: dict[tuple[int, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    step_rows = []
    previous_step = 0
    previous_tensor_steps = np.zeros(tensor_count, dtype=np.int64)
    for entry in index["chunks"]:
        with np.load(directory / "telemetry" / entry["file"], allow_pickle=False) as shard:
            steps, values, observed = shard["step"], shard["values"], shard["observed"]
            if not np.array_equal(steps, np.arange(previous_step + 1, previous_step + 1 + len(steps))):
                raise ValueError("Telemetry contains a missing, duplicated, or out-of-order step")
            if values.shape != (len(steps), tensor_count, len(FEATURES)) or observed.shape != (len(steps), tensor_count):
                raise ValueError("Telemetry shape does not match the tensor manifest")
            if not np.array_equal(shard["tensor_step"], previous_tensor_steps + observed.cumsum(axis=0)):
                raise ValueError("Per-tensor step counters disagree with the observed mask")
            cosine = values[:, :, FEATURE_INDEX["cosine"]]
            finite_cosine = observed & np.isfinite(cosine)
            if np.any((cosine[finite_cosine] < -1.0) | (cosine[finite_cosine] > 1.0)):
                raise ValueError("Recorded cosine lies outside [-1, 1]")
            gate = values[:, :, FEATURE_INDEX["gate_q"]]
            coefficient = values[:, :, FEATURE_INDEX["beta_eff"]]
            gradient_norm = values[:, :, FEATURE_INDEX["gradient_norm"]]
            relation_valid = observed & np.isfinite(cosine) & np.isfinite(gate) & np.isfinite(coefficient) & np.isfinite(gradient_norm)
            expected_gate = ((1.0 + cosine.astype(np.float64)) * 0.5) ** powers[None, :]
            expected_coefficient = expected_gate * betas[None, :]
            expected_coefficient = np.where(gradient_norm == 0.0, betas[None, :], expected_coefficient)
            if run.get("optimizer") == "SGDM":
                expected_coefficient = np.broadcast_to(betas[None, :], coefficient.shape)
            if not np.allclose(gate[relation_valid], expected_gate[relation_valid], rtol=2e-6, atol=2e-7):
                raise ValueError("Recorded q is inconsistent with the recorded cosine and pwr")
            if not np.allclose(coefficient[relation_valid], expected_coefficient[relation_valid], rtol=2e-6, atol=2e-7):
                raise ValueError("Recorded coefficient is inconsistent with the selected optimizer")
            previous_step, previous_tensor_steps = int(steps[-1]), shard["tensor_step"][-1].copy()
            full.add(values, observed)
            late.add(values[steps >= late_start], observed[steps >= late_start])
            for epoch in np.unique(shard["epoch"]):
                mask = shard["epoch"] == epoch
                per_epoch.setdefault(int(epoch), Statistics(tensor_count)).add(values[mask], observed[mask])
                defined, _ = alignment_masks(values[mask], observed[mask])
                for metric_id, metric in enumerate(METRICS):
                    metric_values = values[mask, :, FEATURE_INDEX[metric]].astype(np.float64)
                    valid = observed[mask] & np.isfinite(metric_values)
                    if metric == "gate_q":
                        valid &= defined
                    _step_ids, tensor_ids = np.nonzero(valid)
                    epoch_samples.setdefault((int(epoch), metric), []).append((metric_values[valid], tensor_ids))
            # Compact model-wide evolution at the original step resolution.
            defined, _ = alignment_masks(values, observed)
            for i, step in enumerate(steps):
                row = {
                    "step": int(step),
                    "epoch": int(shard["epoch"][i]),
                    "batch": int(shard["batch"][i]),
                    "batch_size": int(shard["batch_size"][i]),
                    "samples_seen": int(shard["samples_seen"][i]),
                    "loss": float(shard["loss"][i]),
                }
                row.update({f"lr_group_{j}": float(lr) for j, lr in enumerate(shard["learning_rates"][i])})
                for metric in METRICS:
                    v = values[i, :, FEATURE_INDEX[metric]].astype(np.float64)
                    valid = observed[i] & np.isfinite(v)
                    if metric == "gate_q":
                        valid &= defined[i]
                    tensor_ids = np.flatnonzero(valid)
                    v = v[tensor_ids]
                    weights = numel_weights[tensor_ids]
                    row.update(
                        {
                            f"{metric}_count": len(v),
                            f"{metric}_mean": float(v.mean()) if len(v) else float("nan"),
                            f"{metric}_std": float(v.std()) if len(v) else float("nan"),
                            f"{metric}_median": float(np.median(v)) if len(v) else float("nan"),
                            f"{metric}_q10": float(np.quantile(v, 0.10)) if len(v) else float("nan"),
                            f"{metric}_q25": float(np.quantile(v, 0.25)) if len(v) else float("nan"),
                            f"{metric}_q75": float(np.quantile(v, 0.75)) if len(v) else float("nan"),
                            f"{metric}_q90": float(np.quantile(v, 0.90)) if len(v) else float("nan"),
                            f"{metric}_numel_weighted_mean": float(np.average(v, weights=weights)) if len(v) else float("nan"),
                        }
                    )
                step_rows.append(row)
    if previous_step != total_steps:
        raise ValueError("Index step count does not match the persisted data")
    expected_observations = total_steps * tensor_count
    observed_observations = int(previous_tensor_steps.sum())
    if run["status"] == "completed" and not run["synthetic"] and observed_observations != expected_observations:
        raise ValueError(f"Completed ResNet run is missing {expected_observations - observed_observations} tensor-step observations")
    summaries, histograms, epoch_rows = [], [], []
    for period, stats in (("full", full), ("late", late)):
        rows, hist = stats.rows(groups, period=period, weighting="equal_tensor")
        summaries.extend(rows)
        histograms.extend(hist)
        rows, hist = stats.rows(
            aggregate_groups,
            period=period,
            tensor_weights=numel_weights,
            weighting="numel_weighted",
        )
        summaries.extend(rows)
        histograms.extend(hist)
    epoch_histograms = []
    for epoch, stats in sorted(per_epoch.items()):
        rows, _ = stats.rows(groups, include_histograms=False, epoch=epoch, weighting="equal_tensor")
        epoch_rows.extend(rows)
        rows, _ = stats.rows(
            aggregate_groups,
            include_histograms=False,
            epoch=epoch,
            tensor_weights=numel_weights,
            weighting="numel_weighted",
        )
        epoch_rows.extend(rows)
        _, hist = stats.rows({"model": groups["model"]}, epoch=epoch, weighting="equal_tensor")
        epoch_histograms.extend(hist)
        _, hist = stats.rows(
            {"model": groups["model"]},
            epoch=epoch,
            tensor_weights=numel_weights,
            weighting="numel_weighted",
        )
        epoch_histograms.extend(hist)

    def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]) -> list[float]:
        order = np.argsort(values, kind="stable")
        ordered_values, ordered_weights = values[order], weights[order].astype(np.float64)
        midpoints = np.cumsum(ordered_weights) - 0.5 * ordered_weights
        return [float(np.interp(q * ordered_weights.sum(), midpoints, ordered_values)) for q in quantiles]

    epoch_distribution_rows = []
    quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)
    for (epoch, metric), parts in sorted(epoch_samples.items()):
        values = np.concatenate([part[0] for part in parts])
        tensor_ids = np.concatenate([part[1] for part in parts])
        for weighting, weights in (
            ("equal_tensor", np.ones(len(values), dtype=np.int64)),
            ("numel_weighted", numel_weights[tensor_ids]),
        ):
            q10, q25, median, q75, q90 = weighted_quantile(values, weights, quantiles)
            epoch_distribution_rows.append(
                {
                    "epoch": epoch,
                    "metric": metric,
                    "weighting": weighting,
                    "count": len(values),
                    "weighted_count": int(weights.sum()),
                    "mean": float(np.average(values, weights=weights)),
                    "q10": q10,
                    "q25": q25,
                    "median": median,
                    "q75": q75,
                    "q90": q90,
                }
            )
    destination = directory / "analysis"
    destination.mkdir(exist_ok=True)
    write_csv(destination / "summary.csv", summaries)
    write_csv(destination / "epoch_summary.csv", epoch_rows)
    write_csv(destination / "histograms.csv", histograms)
    write_csv(destination / "epoch_model_histograms.csv", epoch_histograms)
    write_csv(destination / "epoch_model_distributions.csv", epoch_distribution_rows)
    write_csv(destination / "step_model_summary.csv", step_rows)
    write_json(
        destination / "analysis.json",
        {
            "analyzed_at": utc_now(),
            "source_status": run["status"],
            "persisted_steps": total_steps,
            "tensor_count": tensor_count,
            "expected_tensor_step_observations": expected_observations,
            "observed_tensor_step_observations": observed_observations,
            "complete_tensor_step_coverage": observed_observations == expected_observations,
            "late_fraction": late_fraction,
            "late_first_step": late_start,
            "late_last_step": total_steps,
            "late_steps": late_steps,
            "late_interpretation": "Descriptive final-window estimate; stationarity is not assumed or tested.",
            "weighting": {
                "equal_tensor": "One vote per finite observed tensor-step.",
                "numel_weighted": "Each tensor-step is weighted by the tensor's parameter count; this is an element-equivalent view, not independent samples.",
            },
            "gate_q_validity": "Exclude undefined alignment (nonfinite or zero gradient/probe norm); retain norm-floor effects.",
            "beta_eff_validity": "All finite observed applied coefficients, including zero-gradient fallback.",
            "thresholds": {"low": LOW, "reference": REFERENCE, "bins": ["x < 0.5", "0.5 <= x <= 0.7", "x > 0.7"]},
            "boundary_rule": "Compare stored floating-point values, promoted to float64, without rounding.",
            "plots_generated": plots,
        },
    )
    if plots:
        optimizer_label = "SGDM (q is hypothetical)" if run.get("optimizer") == "SGDM" else "AGAM-SGD"
        label = f"{optimizer_label} · {'SYNTHETIC CHECK' if run['synthetic'] else 'CIFAR-10'} · ResNet18 · {run['norm']} norm · {total_steps:,} steps · {run['status']}"
        plot_results(
            destination / "plots",
            metadata,
            summaries,
            epoch_rows,
            histograms + epoch_histograms,
            epoch_distribution_rows,
            label,
        )
    from analysis.agam_alignment_analysis import analyze_alignment
    analyze_alignment(directory, window_steps=alignment_window, late_fraction=late_fraction, plots=plots)
    return destination


def seed_worker(_: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loaders(args: argparse.Namespace, directory: Path, device: torch.device) -> tuple[DataLoader, DataLoader, DataLoader]:
    if args.synthetic:
        generator = torch.Generator().manual_seed(args.split_seed)
        count = args.batch_size * 4
        train = TensorDataset(torch.randn(count, 3, 32, 32, generator=generator), torch.randint(10, (count,), generator=generator))
        validation = TensorDataset(torch.randn(args.batch_size * 2, 3, 32, 32, generator=generator), torch.randint(10, (args.batch_size * 2,), generator=generator))
        test = TensorDataset(torch.randn(args.batch_size * 2, 3, 32, 32, generator=generator), torch.randint(10, (args.batch_size * 2,), generator=generator))
    else:
        from sklearn.model_selection import train_test_split

        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        transform = [v2.PILToTensor(), v2.RandomCrop(32, padding=4, padding_mode="reflect"), v2.RandomHorizontalFlip(0.5)]
        if args.augmentation == "repo":
            transform.append(v2.RandAugment(num_ops=2, magnitude=9))
        transform.extend([v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])
        if args.augmentation == "repo":
            transform.append(v2.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)))
        train_source = datasets.CIFAR10(args.data_dir, train=True, download=args.download, transform=v2.Compose(transform))
        val_source = datasets.CIFAR10(
            args.data_dir,
            train=True,
            download=False,
            transform=v2.Compose(
                [
                    v2.PILToTensor(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean, std),
                ]
            ),
        )
        test = datasets.CIFAR10(
            args.data_dir,
            train=False,
            download=args.download,
            transform=v2.Compose(
                [
                    v2.PILToTensor(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean, std),
                ]
            ),
        )
        train_ids, val_ids = train_test_split(np.arange(len(train_source)), train_size=0.85, stratify=train_source.targets, random_state=args.split_seed)
        write_npz(directory / "split_indices.npz", train=train_ids, validation=val_ids)
        train, validation = Subset(train_source, train_ids.tolist()), Subset(val_source, val_ids.tolist())
    common = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker}
    loaders = (
        DataLoader(train, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(args.seed), **common),
        DataLoader(validation, shuffle=False, generator=torch.Generator().manual_seed(args.seed + 1), **common),
        DataLoader(test, shuffle=False, generator=torch.Generator().manual_seed(args.seed + 2), **common),
    )
    return loaders


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.eval()
    totals = torch.zeros(2, device=device)
    count = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(inputs)
            batch_loss = nn.functional.cross_entropy(logits, targets, reduction="sum")
        totals[0] += batch_loss
        totals[1] += (logits.argmax(1) == targets).sum()
        count += len(targets)
    loss, correct = totals.cpu().tolist()
    return loss / count, 100.0 * correct / count


def learning_rate_at(
    step: int,
    total_steps: int,
    base: float,
    schedule: str,
    warmup_steps: int,
    minimum: float = 1e-5,
) -> float:
    """Zero-based step; the returned rate is used on that same optimizer step."""
    if warmup_steps and step < warmup_steps:
        return base * (0.01 + 0.99 * step / warmup_steps)
    if schedule == "constant":
        return base
    fraction = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return minimum + (base - minimum) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, fraction)))


def configure_precision(device: torch.device, amp_dtype_name: str, float32_precision: str) -> tuple[torch.dtype, bool]:
    amp_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[amp_dtype_name]
    amp_enabled = amp_dtype != torch.float32
    if device.type == "cuda" and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support bfloat16 AMP; use --amp-dtype float32")
    if amp_enabled and device.type == "cpu":
        raise RuntimeError("CPU telemetry runs require --amp-dtype float32")

    use_tf32 = float32_precision == "tf32" and device.type == "cuda"
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    # Support both the current precision API and older cluster PyTorch builds.
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32" if use_tf32 else "ieee"
        torch.backends.cudnn.conv.fp32_precision = "tf32" if use_tf32 else "ieee"  # type: ignore[attr-defined]
    except AttributeError, RuntimeError:
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
    return amp_dtype, amp_enabled


def train(args: argparse.Namespace) -> Path:
    directory = args.output.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Output directory must be empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    amp_dtype, amp_enabled = configure_precision(device, args.amp_dtype, args.float32_precision)
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    sources = {}
    for path in (Path(__file__), ROOT / "optims/agam_opt.py", ROOT / "tasks/agam_sgd_diagnostics.py",
                 ROOT / "analysis/agam_alignment_analysis.py", ROOT / "tasks/agam_sgd_telemetry.py"):
        content = path.read_bytes()
        relative_path = path.relative_to(ROOT)
        sources[str(relative_path)] = hashlib.sha256(content).hexdigest()
        snapshot = directory / "source" / relative_path
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(content)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except OSError, subprocess.CalledProcessError:
        commit = None
    run = {
        "schema_version": 3,
        "optimizer": "AGAM-SGD" if args.optimizer == "agam" else "SGDM",
        "gate_interpretation": "applied" if args.optimizer == "agam" else "hypothetical diagnostic only; SGDM never applies q",
        "started_at": utc_now(),
        "status": "initializing",
        "synthetic": args.synthetic,
        "norm": args.norm,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        "precision": f"{args.amp_dtype} autocast={'enabled' if amp_enabled else 'disabled'}; float32_precision={args.float32_precision}; no clipping or gradient accumulation",
        "seed": args.seed,
        "split_seed": args.split_seed,
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "git_commit": commit,
        "source_sha256": sources,
        "coefficient_features": list(FEATURES),
        "alignment_features": list(ALIGNMENT_FEATURES),
        "architecture": "torchvision ResNet18; 3x3 stride-1 stem; no maxpool; 10-class head",
        "depth_definition": "stem=0; residual blocks layer1.0 through layer4.1=1..8; head=9; shortcuts share block depth",
        "dataset": (
            "synthetic tensors (pipeline verification only)"
            if args.synthetic
            else "CIFAR-10 official train split, stratified 85%/15%; official test evaluated only after training"
        ),
        "reproducibility": "Seeded RNGs and data loaders; bitwise reproducibility across devices/backends is not promised.",
    }
    write_json(directory / "run.json", run)
    wb_run = None
    if args.wandb_mode != "disabled":
        import wandb

        wb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            mode=args.wandb_mode,
            job_type="optimizer-diagnostics",
            tags=("optimizer-telemetry", "cifar", args.optimizer),
            name=args.wandb_name or f"{run['optimizer']}_alignment_telemetry_seed:{args.seed}",
            config={
                **task_metadata(
                    task="agam_sgd_alignment_telemetry",
                    task_type="optimizer_diagnostics",
                    model_name="resnet18",
                    model_source="torchvision",
                    dataset_name="cifar10" if not args.synthetic else "synthetic",
                    dataset_config="official_train_85_15_validation_official_test" if not args.synthetic else "pipeline_check",
                    dataset_source="torchvision" if not args.synthetic else "generated",
                    training_regime="supervised_from_scratch",
                ),
                "optimizer": run["optimizer"],
                "AGAM_config": "False,1.0,False,attenuate" if args.optimizer == "agam" else None,
                "in_place": False,
                "pwr": 1.0,
                "scale": False,
                "gate_mode": "attenuate" if args.optimizer == "agam" else "disabled",
                "beta": 0.9,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "use_scheduler": args.schedule == "cosine",
                "warmup_epochs": args.warmup_epochs,
                "min_lr": args.min_lr,
                "epochs": args.epochs,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "amp_dtype": args.amp_dtype,
                "float32_precision": args.float32_precision,
                "telemetry_schema_version": 3,
                "alignment_features": list(ALIGNMENT_FEATURES),
                "telemetry_features": list(FEATURES),
                "telemetry_weightings": ["equal_tensor", "numel_weighted"],
            },
        )
        wb_run.define_metric("epoch")
        wb_run.define_metric("train/*", step_metric="epoch")
        wb_run.define_metric("val/*", step_metric="epoch")
        wb_run.define_metric("telemetry/*", step_metric="epoch")
        run["wandb"] = {"entity": wb_run.entity, "project": wb_run.project, "run_id": wb_run.id, "run_url": wb_run.url}
        write_json(directory / "run.json", run)
    recorder = None
    metrics = []
    started = time.monotonic()
    try:
        model = build_model(args.norm).to(device)
        # Record initialization so matched comparisons can verify actual equality.
        initial_hash = hashlib.sha256()
        for name, value in model.state_dict().items():
            initial_hash.update(name.encode())
            initial_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        run["initial_model_sha256"] = initial_hash.hexdigest()
        optimizer_class = AGAM_SGD if args.optimizer == "agam" else ObservedSGDM
        optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        metadata = tensor_metadata(model, optimizer)
        write_json(directory / "tensor_metadata.json", metadata)
        write_csv(directory / "tensor_metadata.csv", [{**row, "shape": json.dumps(row["shape"])} for row in metadata])
        recorder = GateRecorder(directory / "telemetry", model, metadata, args.flush_steps, alignment=True)
        optimizer.gate_observer = recorder
        optimizer.alignment_observer = recorder.observe_alignment
        train_loader, val_loader, test_loader = make_loaders(args, directory, device)
        if not len(train_loader):
            raise ValueError("Batch size exceeds the training split (drop_last=True)")
        total_steps = args.epochs * len(train_loader)
        warmup_steps = args.warmup_epochs * len(train_loader)
        if warmup_steps >= total_steps:
            raise ValueError("Warmup must be shorter than training")
        run.update(
            status="running",
            optimizer_defaults=optimizer.defaults,
            optimizer_groups=[{key: value for key, value in group.items() if key != "params"} for group in optimizer.param_groups],
            train_examples=len(train_loader.dataset),
            validation_examples=len(val_loader.dataset),
            test_examples=len(test_loader.dataset),
            steps_per_epoch=len(train_loader),
            planned_steps=total_steps,
            tensor_count=len(metadata),
            trainable_parameters=sum(row["numel"] for row in metadata),
        )
        if wb_run is not None:
            wb_run.config.update(
                {
                    "optimizer_defaults": dict(optimizer.defaults),
                    "tensor_count": len(metadata),
                    "trainable_parameters": sum(row["numel"] for row in metadata),
                    "steps_per_epoch": len(train_loader),
                    "planned_optimizer_steps": total_steps,
                },
                allow_val_change=True,
            )
        write_json(directory / "run.json", run)
        print(
            f"{'SYNTHETIC CHECK' if args.synthetic else 'CIFAR-10'}: ResNet18/{args.norm} norm on {device}; {len(metadata)} tensors; telemetry: {directory}", flush=True
        )
        samples_seen = 0
        best_val_accuracy = -math.inf
        best_val_epoch = 0
        best_model: dict[str, torch.Tensor] = {}
        for epoch in range(1, args.epochs + 1):
            model.train()
            totals = torch.zeros(2, device=device)
            epoch_examples = 0
            epoch_start = time.monotonic()
            for batch, (inputs, targets) in enumerate(train_loader, start=1):
                lr = learning_rate_at(recorder.completed_steps, total_steps, args.lr, args.schedule, warmup_steps, args.min_lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits = model(inputs)
                    loss = nn.functional.cross_entropy(logits, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss before optimizer step {recorder.completed_steps + 1}")
                loss.backward()
                recorder.begin_step()
                optimizer.step()
                samples_seen += len(targets)
                recorder.end_step(
                    epoch=epoch,
                    batch=batch,
                    batch_size=len(targets),
                    samples_seen=samples_seen,
                    learning_rates=[group["lr"] for group in optimizer.param_groups],
                    loss=loss,
                )
                totals[0] += loss.detach() * len(targets)
                totals[1] += (logits.detach().argmax(1) == targets).sum()
                epoch_examples += len(targets)
                if recorder.completed_steps % args.log_every == 0:
                    print(f"epoch {epoch}/{args.epochs}, step {recorder.completed_steps}/{total_steps}, loss={float(loss.detach()):.4f}, lr={lr:.6g}", flush=True)
                if args.max_steps and recorder.completed_steps >= args.max_steps:
                    break
            recorder.flush()
            train_loss, train_correct = totals.cpu().tolist()
            val_loss, val_accuracy = evaluate(model, val_loader, device, amp_dtype=amp_dtype, amp_enabled=amp_enabled)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_val_epoch = epoch
                best_model = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            gate_aggregates = recorder.pop_epoch_aggregates(epoch)
            epoch_metrics = {
                "epoch": epoch,
                "step": recorder.completed_steps,
                "train_examples": epoch_examples,
                "train_loss": train_loss / epoch_examples,
                "train_accuracy": 100.0 * train_correct / epoch_examples,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
                "last_lr": lr,
                "seconds": time.monotonic() - epoch_start,
                "complete_epoch": batch == len(train_loader),
            }
            metrics.append(epoch_metrics)
            write_csv(directory / "train_metrics.csv", metrics)
            if wb_run is not None:
                wb_run.log(
                    {
                        "epoch": epoch,
                        "optimizer_step": recorder.completed_steps,
                        "train/loss": epoch_metrics["train_loss"],
                        "train/acc": epoch_metrics["train_accuracy"],
                        "val/loss": val_loss,
                        "val/acc": val_accuracy,
                        "lr": lr,
                        **gate_aggregates,
                    }
                )
            print(
                f"epoch {epoch}: train loss {train_loss / epoch_examples:.4f}; validation accuracy {val_accuracy:.2f}%; saved through step {recorder.completed_steps}",
                flush=True,
            )
            if args.max_steps and recorder.completed_steps >= args.max_steps:
                break
        run["status"] = "completed" if recorder.completed_steps == total_steps else "step_limit"
        if not args.synthetic:
            if not best_model:
                raise RuntimeError("No validation-selected checkpoint was captured")
            final_model = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            test_loss_at_final_epoch, test_acc_at_final_epoch = evaluate(model, test_loader, device, amp_dtype=amp_dtype, amp_enabled=amp_enabled)
            model.load_state_dict(best_model)
            test_loss_at_best_val, test_acc_at_best_val = evaluate(model, test_loader, device, amp_dtype=amp_dtype, amp_enabled=amp_enabled)
            run["evaluation"] = {
                "best_val_acc": best_val_accuracy,
                "best_val_epoch": best_val_epoch,
                "test_loss_at_best_val": test_loss_at_best_val,
                "test_acc_at_best_val": test_acc_at_best_val,
                "test_loss_at_final_epoch": test_loss_at_final_epoch,
                "test_acc_at_final_epoch": test_acc_at_final_epoch,
            }
            if wb_run is not None:
                for key, value in run["evaluation"].items():
                    wb_run.summary[key] = value
                wb_run.summary["test/loss_at_best_val"] = test_loss_at_best_val
                wb_run.summary["test/acc_at_best_val"] = test_acc_at_best_val
                wb_run.summary["test/loss_at_final_epoch"] = test_loss_at_final_epoch
                wb_run.summary["test/acc_at_final_epoch"] = test_acc_at_final_epoch
            torch.save(
                {
                    "model": final_model,
                    "optimizer": optimizer.state_dict(),
                    "step": recorder.completed_steps,
                    "epoch": epoch,
                    "arguments": run["arguments"],
                    "test_loss": test_loss_at_final_epoch,
                    "test_accuracy": test_acc_at_final_epoch,
                },
                directory / "final_checkpoint.pt",
            )
            torch.save(
                {
                    "model": best_model,
                    "selected_epoch": best_val_epoch,
                    "validation_accuracy": best_val_accuracy,
                    "test_loss": test_loss_at_best_val,
                    "test_accuracy": test_acc_at_best_val,
                    "arguments": run["arguments"],
                },
                directory / "best_validation_checkpoint.pt",
            )
    except BaseException as error:
        run["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        run["error"] = f"{type(error).__name__}: {error}"
        if wb_run is not None:
            wb_run.summary["telemetry_status"] = run["status"]
            wb_run.summary["telemetry_error"] = run["error"]
            wb_run.finish(exit_code=1)
        raise
    finally:
        if recorder is not None:
            recorder.abort_step()  # Discard observations from an optimizer step that did not return successfully.
            recorder.flush()
            run["completed_steps"] = recorder.completed_steps
            run["persisted_steps"] = recorder.index["completed_steps"]
        run.update(finished_at=utc_now(), elapsed_seconds=time.monotonic() - started)
        write_json(directory / "run.json", run)
    analysis_directory = analyze_run(directory, args.late_fraction, not args.no_plots, args.alignment_window)
    if wb_run is not None:
        with (analysis_directory / "summary.csv").open(newline="") as handle:
            summary_rows = list(csv.DictReader(handle))
        for metric in METRICS:
            for weighting in ("equal_tensor", "numel_weighted"):
                row = next(
                    row
                    for row in summary_rows
                    if row["period"] == "late" and row["metric"] == metric and row["view"] == "model" and row["group"] == "all" and row["weighting"] == weighting
                )
                prefix = f"telemetry/late/{metric}/{weighting}"
                for field in ("mean", "std", "pct_lt_0_5", "pct_0_5_to_0_7", "pct_gt_0_7"):
                    wb_run.summary[f"{prefix}/{field}"] = float(row[field])
        wb_run.summary["telemetry_status"] = run["status"]
        wb_run.summary["telemetry_persisted_steps"] = run["persisted_steps"]
        artifact = wandb.Artifact(
            name=f"{args.optimizer}-sgd-alignment-telemetry-{wb_run.id}",
            type="optimizer-telemetry",
            description=f"Lossless per-step, per-tensor {run['optimizer']} alignment telemetry and reproducible analysis outputs.",
            metadata={
                "schema_version": 3,
                "features": list(FEATURES),
                "tensor_count": len(metadata),
                "optimizer_steps": run["persisted_steps"],
                "seed": args.seed,
            },
        )
        for relative in ("run.json", "tensor_metadata.json", "tensor_metadata.csv", "split_indices.npz", "train_metrics.csv"):
            path = directory / relative
            if path.exists():
                artifact.add_file(str(path), name=relative)
        artifact.add_dir(str(directory / "source"), name="source")
        artifact.add_dir(str(directory / "telemetry"), name="telemetry")
        artifact.add_dir(str(analysis_directory), name="analysis")
        wb_run.log_artifact(artifact, aliases=["latest", f"seed-{args.seed}"])
        wb_run.finish()
    return directory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train and record every tensor on every optimizer step")
    training.add_argument("--output", type=Path, default=ROOT / "analysis" / f"agam_sgd_telemetry_{datetime.now(UTC):%Y%m%d_%H%M%S}")
    training.add_argument("--optimizer", choices=("agam", "sgdm"), default="agam", help="AGAM-SGD or native PyTorch SGDM, both beta=0.9")
    training.add_argument("--data-dir", type=Path, default=ROOT / "data")
    training.add_argument("--download", action="store_true", help="Allow torchvision to download CIFAR-10")
    training.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    training.add_argument("--epochs", type=int, default=200)
    training.add_argument("--batch-size", type=int, default=256)
    training.add_argument("--lr", type=float, default=0.1)
    training.add_argument("--weight-decay", type=float, default=0.0)
    training.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    training.add_argument("--warmup-epochs", type=int, default=0)
    training.add_argument("--min-lr", type=float, default=1e-5, help="Cosine-decay floor; ignored for a constant schedule")
    training.add_argument("--norm", choices=("group", "batch"), default="group")
    training.add_argument("--augmentation", choices=("repo", "basic"), default="repo")
    training.add_argument("--amp-dtype", choices=("float32", "bfloat16"), default="float32")
    training.add_argument("--float32-precision", choices=("ieee", "tf32"), default="ieee")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--split-seed", type=int, default=20260901)
    training.add_argument("--workers", type=int, default=0, help="Zero is portable on macOS; raise for faster loading")
    training.add_argument("--cpu-threads", type=int, default=0, help="Zero preserves PyTorch's thread setting")
    training.add_argument("--flush-steps", type=int, default=64, help="Persist lossless shards this often; no step sampling")
    training.add_argument("--log-every", type=int, default=100)
    training.add_argument("--max-steps", type=int, help="Optional early stop; marked as a partial run")
    training.add_argument("--synthetic", action="store_true", help="Synthetic pipeline check, requires --max-steps; never CIFAR evidence")
    training.add_argument("--wandb-mode", choices=("disabled", "online", "offline"), default="disabled")
    training.add_argument("--wandb-entity")
    training.add_argument("--wandb-project")
    training.add_argument("--wandb-name")
    analysis = commands.add_parser("analyze", help="Reanalyze persisted telemetry, including interrupted runs")
    analysis.add_argument("run_dir", type=Path)
    for command in (training, analysis):
        command.add_argument("--alignment-window", type=int, default=100, help="Nonoverlapping step windows for grouped conflict summaries; every raw step is retained")
        command.add_argument("--late-fraction", type=float, default=0.2, help="Final fraction of observed steps used for a steady-state estimate")
        command.add_argument("--no-plots", action="store_true", help="Write numeric analysis only (matplotlib not required)")
    args = parser.parse_args(argv)
    if not 0 < args.late_fraction <= 1:
        parser.error("--late-fraction must be in (0, 1]")
    if args.alignment_window < 1:
        parser.error("--alignment-window must be positive")
    if args.command == "train":
        if any(getattr(args, name) < 1 for name in ("epochs", "batch_size", "flush_steps", "log_every")):
            parser.error("epochs, batch size, flush steps, and log interval must be positive")
        if args.max_steps is not None and args.max_steps < 1:
            parser.error("--max-steps must be positive")
        if min(args.workers, args.cpu_threads, args.warmup_epochs) < 0 or args.warmup_epochs >= args.epochs:
            parser.error("workers/threads/warmup must be nonnegative; warmup must be shorter than training")
        if not all(math.isfinite(value) for value in (args.lr, args.min_lr, args.weight_decay)) or min(args.lr, args.min_lr, args.weight_decay) < 0:
            parser.error("learning rate, minimum LR, and weight decay must be finite and nonnegative")
        if args.min_lr > args.lr:
            parser.error("--min-lr cannot exceed --lr")
        if not all(0 <= seed < 2**32 for seed in (args.seed, args.split_seed)):
            parser.error("seeds must be integers in [0, 2**32)")
        if args.synthetic and args.max_steps is None:
            parser.error("--synthetic requires an explicit --max-steps limit")
        if args.wandb_mode != "disabled" and not args.wandb_project:
            parser.error("--wandb-project is required when W&B logging is enabled")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "train":
        destination = train(args)
    else:
        destination = analyze_run(args.run_dir.resolve(), args.late_fraction, not args.no_plots, args.alignment_window)
    print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
