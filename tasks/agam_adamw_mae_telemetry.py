"""Telemetry helpers for canonical AGAM-AdamW on MAE ViT-Tiny.

The recorder stores every completed optimizer step and every trainable parameter
tensor without sampling.  This module owns the ViT-specific tensor taxonomy so
the generic training entry point does not embed architecture-name heuristics.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

SCHEMA_VERSION = 1
TAXONOMY_VERSION = "mae-vit-v1"
FEATURES = ("cosine", "gate_q", "beta_eff", "gradient_norm", "probe_norm")
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURES)}
NORM_EPS = 1e-8


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty telemetry table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _depth(name: str) -> tuple[str, int, str, int]:
    """Return ordered depth, depth index, stage, and block index."""
    if name == "cls_token" or name.startswith("patch_embed."):
        return "encoder.stem", 0, "encoder", -1
    match = re.match(r"blocks\.(\d+)\.", name)
    if match:
        block = int(match.group(1))
        return f"encoder.block.{block:02d}", 1 + block, "encoder", block
    if name.startswith("norm."):
        return "encoder.final_norm", 13, "encoder", -1
    if name == "mask_token" or name.startswith("decoder_embed."):
        return "decoder.stem", 14, "decoder", -1
    match = re.match(r"decoder_blocks\.(\d+)\.", name)
    if match:
        block = int(match.group(1))
        return f"decoder.block.{block:02d}", 15 + block, "decoder", block
    if name.startswith("decoder_norm."):
        return "decoder.final_norm", 19, "decoder", -1
    if name.startswith("decoder_pred."):
        return "decoder.reconstruction_head", 20, "reconstruction_head", -1
    raise ValueError(f"Unclassified MAE ViT depth for trainable tensor: {name}")


def _parameter_kind(name: str, module: nn.Module, role: str) -> tuple[str, str]:
    """Return a paper-facing coarse kind and a more specific subkind."""
    normalized_role = "scale" if isinstance(module, nn.LayerNorm) and role == "weight" else role
    if name in {"cls_token", "mask_token"}:
        return "special_token", name
    if isinstance(module, nn.modules.conv._ConvNd):
        return f"convolution_{normalized_role}", f"patch_embedding_{normalized_role}"
    if isinstance(module, nn.LayerNorm):
        return f"normalization_{normalized_role}", f"layer_norm_{normalized_role}"
    if ".attn.qkv." in name:
        return f"attention_qkv_{normalized_role}", f"fused_qkv_{normalized_role}"
    if ".attn.proj." in name:
        return f"attention_output_{normalized_role}", f"attention_projection_{normalized_role}"
    if ".mlp.fc" in name:
        return f"mlp_{normalized_role}", f"mlp_linear_{normalized_role}"
    if name.startswith("decoder_embed."):
        return f"decoder_projection_{normalized_role}", f"encoder_to_decoder_{normalized_role}"
    if name.startswith("decoder_pred."):
        return f"reconstruction_head_{normalized_role}", f"patch_prediction_{normalized_role}"
    if isinstance(module, nn.Linear):
        return f"linear_{normalized_role}", f"other_linear_{normalized_role}"
    return f"other_{normalized_role}", f"{type(module).__name__.lower()}_{normalized_role}"


def vit_tensor_metadata(model: nn.Module, optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    """Describe every observed tensor exactly once using ViT-aware groups."""
    modules = dict(model.named_modules())
    optimizer_groups = {
        id(parameter): (group_index, group)
        for group_index, group in enumerate(optimizer.param_groups)
        for parameter in group["params"]
    }
    rows: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if id(parameter) not in optimizer_groups:
            continue
        module_name, separator, role = name.rpartition(".")
        if not separator:
            module_name, role = "", name
        module = modules[module_name]
        depth, depth_index, stage, block_index = _depth(name)
        kind, subkind = _parameter_kind(name, module, role)
        optimizer_group, group = optimizer_groups[id(parameter)]
        rows.append(
            {
                "tensor_id": len(rows),
                "name": name,
                "module": module_name,
                "module_type": type(module).__name__,
                "role": role,
                "kind": kind,
                "subkind": subkind,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "dtype": str(parameter.dtype),
                "stage": stage,
                "depth": depth,
                "depth_index": depth_index,
                "block_index": block_index,
                "optimizer_group": optimizer_group,
                "weight_decay": float(group["weight_decay"]),
                "beta1": float(group["beta1"]),
                "beta2": float(group["beta2"]),
                "pwr": float(group["pwr"]),
                "align": group["align"],
                "scale": group["scale"],
                "gate_mode": group["gate_mode"],
                "gradient_weight_mode": group["gradient_weight_mode"],
            }
        )

    expected = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    observed = {id(dict(model.named_parameters())[row["name"]]) for row in rows}
    if observed != expected:
        raise RuntimeError("ViT telemetry taxonomy did not cover every optimizer tensor exactly once.")
    if [row["tensor_id"] for row in rows] != list(range(len(rows))):
        raise RuntimeError("Telemetry tensor IDs must be contiguous and deterministic.")
    return rows


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


class ViTGateRecorder:
    """Losslessly record every tensor gate with one bulk copy per shard."""

    def __init__(
        self,
        directory: Path,
        model: nn.Module,
        metadata: list[dict[str, Any]],
        flush_steps: int,
    ) -> None:
        if flush_steps < 1:
            raise ValueError("flush_steps must be positive")
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite telemetry directory: {directory}")
        directory.mkdir(parents=True)
        parameters = dict(model.named_parameters())
        self.ids = {id(parameters[row["name"]]): int(row["tensor_id"]) for row in metadata}
        first = parameters[metadata[0]["name"]]
        if any(
            parameter.device != first.device or parameter.dtype != first.dtype
            for parameter in parameters.values()
            if id(parameter) in self.ids
        ):
            raise ValueError("All observed parameters must share one device and dtype.")
        self.directory = directory
        self.metadata = metadata
        self.flush_steps = flush_steps
        self.tensor_count = len(metadata)
        self.numel_weights = np.asarray([int(row["numel"]) for row in metadata], dtype=np.int64)
        self.nan = first.new_full((), float("nan"))
        self.tensor_steps = np.zeros(self.tensor_count, dtype=np.int64)
        self.active = False
        self.pending: list[torch.Tensor] = []
        self.seen = np.zeros(self.tensor_count, dtype=bool)
        self.buffer: list[tuple[torch.Tensor, dict[str, Any]]] = []
        self.epoch_statistics: dict[int, dict[str, np.ndarray]] = {}
        self.index: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "features": list(FEATURES),
            "chunks": [],
            "completed_steps": 0,
        }
        write_json(directory / "index.json", self.index)

    @property
    def completed_steps(self) -> int:
        return int(self.buffer[-1][1]["step"]) if self.buffer else int(self.index["completed_steps"])

    def begin_step(self) -> None:
        if self.active:
            raise RuntimeError("Previous telemetry step was not completed or aborted.")
        self.pending = [self.nan] * (self.tensor_count * len(FEATURES))
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
            raise RuntimeError("begin_step must precede optimizer.step.")
        tensor_id = self.ids[id(parameter)]
        if self.seen[tensor_id]:
            raise RuntimeError("A tensor was observed twice in one optimizer step.")
        self.seen[tensor_id] = True
        offset = tensor_id * len(FEATURES)
        self.pending[offset : offset + len(FEATURES)] = [
            value.detach() for value in (cosine, gate, coefficient, gradient_norm, probe_norm)
        ]

    def end_step(
        self,
        *,
        epoch: int,
        batch: int,
        batch_size: int,
        samples_seen: int,
        learning_rates: list[float],
        loss: torch.Tensor,
    ) -> None:
        if not self.active:
            raise RuntimeError("No telemetry step is active.")
        if not self.seen.all():
            missing = [self.metadata[index]["name"] for index in np.flatnonzero(~self.seen)]
            raise RuntimeError(f"Missing gate observations for {len(missing)} tensors: {missing[:5]}")
        packed = torch.stack([*self.pending, loss.detach()])
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
        self.buffer.append((packed, record))
        self.tensor_steps = record["tensor_step"]
        self.abort_step()
        if len(self.buffer) >= self.flush_steps:
            self.flush()

    def abort_step(self) -> None:
        self.active = False
        self.pending = []

    def flush(self) -> None:
        if not self.buffer:
            return
        packed = torch.stack([values for values, _record in self.buffer]).cpu().numpy()
        records = [record for _values, record in self.buffer]
        arrays = {key: np.asarray([record[key] for record in records]) for key in records[0]}
        boundary = self.tensor_count * len(FEATURES)
        arrays["values"] = packed[:, :boundary].reshape(len(self.buffer), self.tensor_count, len(FEATURES))
        arrays["loss"] = packed[:, -1]

        gate = arrays["values"][:, :, FEATURE_INDEX["gate_q"]].astype(np.float64)
        cosine = arrays["values"][:, :, FEATURE_INDEX["cosine"]]
        gradient_norm = arrays["values"][:, :, FEATURE_INDEX["gradient_norm"]]
        probe_norm = arrays["values"][:, :, FEATURE_INDEX["probe_norm"]]
        valid = (
            arrays["observed"]
            & np.isfinite(gate)
            & np.isfinite(cosine)
            & np.isfinite(gradient_norm)
            & np.isfinite(probe_norm)
            & (gradient_norm > 0)
            & (probe_norm > 0)
        )
        for epoch in np.unique(arrays["epoch"]):
            mask = arrays["epoch"] == epoch
            epoch_gate, epoch_valid = gate[mask], valid[mask]
            stats = self.epoch_statistics.setdefault(
                int(epoch),
                {
                    "count": np.zeros(self.tensor_count, dtype=np.int64),
                    "sum": np.zeros(self.tensor_count, dtype=np.float64),
                    "squares": np.zeros(self.tensor_count, dtype=np.float64),
                    "misaligned": np.zeros(self.tensor_count, dtype=np.int64),
                    "middle": np.zeros(self.tensor_count, dtype=np.int64),
                    "strong": np.zeros(self.tensor_count, dtype=np.int64),
                },
            )
            safe = np.where(epoch_valid, epoch_gate, 0.0)
            stats["count"] += epoch_valid.sum(axis=0)
            stats["sum"] += safe.sum(axis=0)
            stats["squares"] += (safe * safe).sum(axis=0)
            stats["misaligned"] += (epoch_valid & (epoch_gate < 0.5)).sum(axis=0)
            stats["middle"] += (epoch_valid & (epoch_gate >= 0.5) & (epoch_gate <= 0.7)).sum(axis=0)
            stats["strong"] += (epoch_valid & (epoch_gate > 0.7)).sum(axis=0)

        first_step, last_step = int(arrays["step"][0]), int(arrays["step"][-1])
        filename = f"steps_{first_step:08d}_{last_step:08d}.npz"
        _write_npz(self.directory / filename, **arrays)
        entry = {
            "file": filename,
            "first_step": first_step,
            "last_step": last_step,
            "steps": len(self.buffer),
        }
        chunks = self.index["chunks"]
        if not chunks or chunks[-1]["file"] != filename:
            chunks = [*chunks, entry]
        self.index = {**self.index, "chunks": chunks, "completed_steps": last_step}
        write_json(self.directory / "index.json", self.index)
        self.buffer.clear()

    def pop_epoch_aggregates(self, epoch: int) -> dict[str, float]:
        stats = self.epoch_statistics.pop(epoch)
        groups: dict[str, dict[str, list[int]]] = {"model": {"all": list(range(self.tensor_count))}}
        for view in ("kind", "depth", "stage"):
            mapping: dict[str, list[int]] = {}
            for row in self.metadata:
                mapping.setdefault(str(row[view]), []).append(int(row["tensor_id"]))
            groups[view] = mapping

        result: dict[str, float] = {}
        for weighting, weights in (
            ("equal_tensor", np.ones(self.tensor_count, dtype=np.int64)),
            ("numel_weighted", self.numel_weights),
        ):
            for view, mapping in groups.items():
                for group, indices_list in mapping.items():
                    indices = np.asarray(indices_list, dtype=np.int64)
                    group_weights = weights[indices]
                    counts = stats["count"][indices]
                    denominator = int(np.dot(counts, group_weights))
                    total = float(np.dot(stats["sum"][indices], group_weights))
                    squares = float(np.dot(stats["squares"][indices], group_weights))
                    mean = total / denominator if denominator else float("nan")
                    variance = squares / denominator - mean * mean if denominator else float("nan")
                    prefix = f"telemetry/{view}/{_slug(str(group))}/{weighting}"
                    result[f"{prefix}/gate_mean"] = mean
                    result[f"{prefix}/gate_std"] = math.sqrt(max(variance, 0.0)) if denominator else float("nan")
                    for field, label in (
                        ("misaligned", "misalignment_pct"),
                        ("middle", "aligned_0.5_to_0.7_pct"),
                        ("strong", "strong_alignment_gt_0.7_pct"),
                    ):
                        numerator = int(np.dot(stats[field][indices], group_weights))
                        result[f"{prefix}/{label}"] = 100.0 * numerator / denominator if denominator else float("nan")
        return result


def telemetry_manifest(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "grain": "one observation per completed optimizer step and trainable parameter tensor",
        "gate_definition": "q_t=(1+cosine(g_t,u_probe_t))/2",
        "misalignment_definition": "q_t < 0.5 (equivalently cosine < 0 for pwr=1)",
        "tensor_count": len(metadata),
        "trainable_parameters": sum(int(row["numel"]) for row in metadata),
        "kinds": sorted({str(row["kind"]) for row in metadata}),
        "depths": [
            depth
            for depth, _index in sorted(
                {(str(row["depth"]), int(row["depth_index"])) for row in metadata},
                key=lambda item: item[1],
            )
        ],
    }
