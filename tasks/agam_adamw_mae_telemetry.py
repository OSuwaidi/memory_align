"""Telemetry helpers for canonical AGAM-AdamW on MAE ViT-Tiny.

The recorder stores every completed optimizer step and every trainable parameter
tensor without sampling.  This module owns the ViT-specific tensor taxonomy so
the generic training entry point does not embed architecture-name heuristics.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tasks.mal_sgdm_telemetry import GateRecorder

SCHEMA_VERSION = 1
TAXONOMY_VERSION = "mae-vit-v1"


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


class ViTGateRecorder(GateRecorder):
    """GateRecorder with compact per-kind/depth epoch metrics for W&B."""

    def __init__(
        self,
        directory: Path,
        model: nn.Module,
        metadata: list[dict[str, Any]],
        flush_steps: int,
    ) -> None:
        super().__init__(directory, model, metadata, flush_steps=flush_steps, alignment=False)
        self.metadata = metadata

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
            rows, _ = stats.rows(
                groups,
                tensor_weights=weights,
                weighting=weighting,
                include_histograms=False,
            )
            for row in rows:
                if row["metric"] != "gate_q":
                    continue
                prefix = f"telemetry/{row['view']}/{_slug(str(row['group']))}/{weighting}"
                result[f"{prefix}/gate_mean"] = float(row["mean"])
                result[f"{prefix}/gate_std"] = float(row["std"])
                result[f"{prefix}/misalignment_pct"] = float(row["pct_lt_0_5"])
                result[f"{prefix}/aligned_0.5_to_0.7_pct"] = float(row["pct_0_5_to_0_7"])
                result[f"{prefix}/strong_alignment_gt_0.7_pct"] = float(row["pct_gt_0_7"])

        # ``flush`` also builds probe-alignment counters from the raw feature
        # stream. They are redundant with gate<0.5 for pwr=1, but discarding
        # them here bounds host memory across a long pre-training run.
        self.epoch_alignment.pop(epoch, None)
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
