"""Paired CIFAR-10 benchmark for tensor-global versus per-output adaptive MAL.

The two arms use the same optimizer implementation and differ only in
``MAL_SGD(per_output=...)``.  Initial weights, data splits, epoch permutations,
augmentations, schedules, and all other hyperparameters are paired exactly.

Example:
    python diagnostics/benchmark_per_output_mal.py \
        --device mps --output-dir diagnostics/results/per_output_mps
"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import CIFAR10

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mal_sgd import MAL_SGD


BASE_BETA = 0.9
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(8, out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(8, out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        output = F.silu(self.norm1(self.conv1(inputs)))
        output = self.norm2(self.conv2(output))
        return F.silu(output + residual)


class CifarResidualCNN(nn.Module):
    """A compact five-block residual ConvNet with GroupNorm and a linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(16, 16),
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
        )
        self.head = nn.Linear(64, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.blocks(self.stem(inputs))
        return self.head(output.mean(dim=(-2, -1)))


@dataclass
class EpochRow:
    condition: str
    scope: str
    adaptation_rate: float
    learning_rate: float
    seed: int
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    beta_mean: float
    beta_std: float
    mean_within_tensor_beta_std: float
    pct_beta_below_0_5: float
    pct_beta_above_0_9: float


@dataclass
class RunRow:
    condition: str
    scope: str
    adaptation_rate: float
    learning_rate: float
    seed: int
    initial_state_hash: str
    epoch_order_hash: str
    best_validation_accuracy: float
    final_validation_accuracy: float
    selected_test_accuracy: float
    mean_validation_accuracy: float
    final_validation_loss: float
    mean_validation_loss: float
    best_epoch: int
    elapsed_seconds: float
    steps: int
    diverged: bool
    beta_mean: float
    beta_std: float
    mean_within_tensor_beta_std: float
    pct_beta_below_0_5: float
    pct_beta_above_0_9: float


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def tensor_hash(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(CIFAR_MEAN).reshape(1, 3, 1, 1)
    std = images.new_tensor(CIFAR_STD).reshape(1, 3, 1, 1)
    return (images - mean) / std


def as_image_tensor(data: np.ndarray, indices: np.ndarray | None = None) -> torch.Tensor:
    selected = data if indices is None else data[indices]
    images = torch.from_numpy(np.array(selected, copy=True)).permute(0, 3, 1, 2)
    return normalize(images.to(torch.float32).div_(255.0))


def stratified_split(
    targets: np.ndarray,
    *,
    train_per_class: int,
    validation_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []
    for label in range(10):
        candidates = np.flatnonzero(targets == label)
        generator.shuffle(candidates)
        end_train = train_per_class
        end_validation = end_train + validation_per_class
        if end_validation > len(candidates):
            raise ValueError("requested split exceeds the available class examples")
        train_indices.extend(candidates[:end_train])
        validation_indices.extend(candidates[end_train:end_validation])
    generator.shuffle(train_indices)
    generator.shuffle(validation_indices)
    return np.asarray(train_indices), np.asarray(validation_indices)


def corrupt_labels(
    labels: torch.Tensor,
    *,
    fraction: float,
    seed: int,
) -> torch.Tensor:
    output = labels.clone()
    count = round(len(output) * fraction)
    if count == 0:
        return output
    generator = np.random.default_rng(seed)
    selected = generator.choice(len(output), size=count, replace=False)
    offsets = generator.integers(1, 10, size=count)
    output[selected] = (output[selected] + torch.as_tensor(offsets, dtype=torch.long)) % 10
    return output


def make_epoch_orders(size: int, epochs: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randperm(size, generator=generator) for _ in range(epochs)]


def augment_batch(images: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Deterministic per-example reflect crop and horizontal flip on CPU."""
    generator = torch.Generator().manual_seed(seed)
    offsets = torch.randint(0, 9, (len(images), 2), generator=generator)
    flip = torch.rand(len(images), generator=generator) < 0.5
    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    output = torch.empty_like(images)
    for row_offset in range(9):
        for column_offset in range(9):
            selected = (offsets[:, 0] == row_offset) & (offsets[:, 1] == column_offset)
            if selected.any():
                output[selected] = padded[
                    selected,
                    :,
                    row_offset : row_offset + 32,
                    column_offset : column_offset + 32,
                ]
    output[flip] = output[flip].flip(-1)
    return output


def learning_rate_at_step(
    base_lr: float,
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> float:
    if step < warmup_steps:
        progress = (step + 1) / max(1, warmup_steps)
        return base_lr * (0.05 + 0.95 * progress)
    cosine_steps = max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, (step - warmup_steps) / cosine_steps)
    factor = 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * factor


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    for start in range(0, len(labels), batch_size):
        stop = min(start + batch_size, len(labels))
        batch_images = images[start:stop].to(device)
        batch_labels = labels[start:stop].to(device)
        logits = model(batch_images)
        loss_sum += float(F.cross_entropy(logits, batch_labels, reduction="sum").cpu())
        correct += int(logits.argmax(dim=1).eq(batch_labels).sum().cpu())
    return loss_sum / len(labels), 100.0 * correct / len(labels)


@torch.no_grad()
def beta_telemetry(optimizer: MAL_SGD) -> dict[str, float]:
    beta_values = []
    beta_weights = []
    within_tensor_stds = []
    for group in optimizer.param_groups:
        for parameter, beta in zip(group["params"], group["beta"]):
            values = beta.detach().to(torch.float32).reshape(-1)
            beta_values.append(values)
            beta_weights.append(torch.full_like(values, parameter.numel() / values.numel()))
            if values.numel() > 1:
                within_tensor_stds.append(values.std(unbiased=False))

    values = torch.cat(beta_values)
    weights = torch.cat(beta_weights)
    within_std = torch.stack(within_tensor_stds).mean() if within_tensor_stds else values.new_zeros(())
    summary = torch.stack(
        (
            (values * weights).sum() / weights.sum(),
            values.std(unbiased=False),
            within_std,
            (values < 0.5).to(torch.float32).mean(),
            (values > 0.9).to(torch.float32).mean(),
        )
    ).cpu()
    return {
        "beta_mean": float(summary[0]),
        "beta_std": float(summary[1]),
        "mean_within_tensor_beta_std": float(summary[2]),
        "pct_beta_below_0_5": 100.0 * float(summary[3]),
        "pct_beta_above_0_9": 100.0 * float(summary[4]),
    }


def average_telemetry(samples: list[dict[str, float]]) -> dict[str, float]:
    return {key: statistics.fmean(sample[key] for sample in samples) for key in samples[0]}


def train_one(
    *,
    condition: str,
    scope: str,
    adaptation_rate: float,
    learning_rate: float,
    seed: int,
    initial_state: dict[str, torch.Tensor],
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    validation_images: torch.Tensor,
    validation_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    epoch_orders: list[torch.Tensor],
    batch_size: int,
    weight_decay: float,
    telemetry_every: int,
    device: torch.device,
) -> tuple[list[EpochRow], RunRow]:
    model = CifarResidualCNN()
    model.load_state_dict(initial_state)
    model.to(device)
    optimizer = MAL_SGD(
        model.parameters(),
        lr=learning_rate,
        beta=BASE_BETA,
        weight_decay=weight_decay,
        adaptive=True,
        c=adaptation_rate,
        nesterov=False,
        per_output=scope == "output",
    )

    steps_per_epoch = math.ceil(len(train_labels) / batch_size)
    total_steps = len(epoch_orders) * steps_per_epoch
    warmup_steps = steps_per_epoch
    initial_hash = tensor_hash(tuple(initial_state.values()))
    order_hash = tensor_hash(tuple(epoch_orders))
    epoch_rows = []
    all_telemetry = []
    best_accuracy = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    optimizer_step = 0
    diverged = False
    sync(device)
    start_time = time.perf_counter()

    for epoch_index, order in enumerate(epoch_orders):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        epoch_telemetry = []
        for batch_index, start in enumerate(range(0, len(order), batch_size)):
            indices = order[start : start + batch_size]
            batch_images = augment_batch(
                train_images[indices],
                seed=seed * 1_000_003 + epoch_index * 10_007 + batch_index,
            ).to(device)
            batch_labels = train_labels[indices].to(device)
            current_lr = learning_rate_at_step(
                learning_rate,
                optimizer_step,
                total_steps,
                warmup_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images)
            loss = F.cross_entropy(logits, batch_labels)
            if not bool(torch.isfinite(loss).cpu()):
                diverged = True
                break
            loss.backward()
            optimizer.step()
            optimizer_step += 1

            batch_count = len(batch_labels)
            loss_sum += float(loss.detach().cpu()) * batch_count
            correct += int(logits.argmax(dim=1).eq(batch_labels).sum().cpu())
            count += batch_count
            if optimizer_step % telemetry_every == 0:
                sample = beta_telemetry(optimizer)
                epoch_telemetry.append(sample)
                all_telemetry.append(sample)

        if diverged:
            break

        if not epoch_telemetry:
            sample = beta_telemetry(optimizer)
            epoch_telemetry.append(sample)
            all_telemetry.append(sample)

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_images,
            validation_labels,
            batch_size=512,
            device=device,
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch_index + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        telemetry = average_telemetry(epoch_telemetry)
        epoch_rows.append(
            EpochRow(
                condition=condition,
                scope=scope,
                adaptation_rate=adaptation_rate,
                learning_rate=learning_rate,
                seed=seed,
                epoch=epoch_index + 1,
                train_loss=loss_sum / count,
                train_accuracy=100.0 * correct / count,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                **telemetry,
            )
        )
        print(
            f"{condition:>7} scope={scope:>6} c={adaptation_rate:.2f} "
            f"lr={learning_rate:.3f} seed={seed} "
            f"epoch={epoch_index + 1:02d}/{len(epoch_orders)} "
            f"val={validation_accuracy:5.2f}% loss={validation_loss:.4f}",
            flush=True,
        )

    sync(device)
    elapsed = time.perf_counter() - start_time
    if diverged or not epoch_rows or best_state is None:
        final_validation_accuracy = 0.0
        final_validation_loss = 1e6
        mean_validation_accuracy = 0.0
        mean_validation_loss = 1e6
        selected_test_accuracy = 0.0
        best_accuracy = 0.0
        aggregate_telemetry = (
            average_telemetry(all_telemetry)
            if all_telemetry
            else {
                "beta_mean": float("nan"),
                "beta_std": float("nan"),
                "mean_within_tensor_beta_std": float("nan"),
                "pct_beta_below_0_5": float("nan"),
                "pct_beta_above_0_9": float("nan"),
            }
        )
    else:
        final_validation_accuracy = epoch_rows[-1].validation_accuracy
        final_validation_loss = epoch_rows[-1].validation_loss
        mean_validation_accuracy = statistics.fmean(row.validation_accuracy for row in epoch_rows)
        mean_validation_loss = statistics.fmean(row.validation_loss for row in epoch_rows)
        model.load_state_dict(best_state)
        _, selected_test_accuracy = evaluate(
            model,
            test_images,
            test_labels,
            batch_size=512,
            device=device,
        )
        aggregate_telemetry = average_telemetry(all_telemetry)

    run_row = RunRow(
        condition=condition,
        scope=scope,
        adaptation_rate=adaptation_rate,
        learning_rate=learning_rate,
        seed=seed,
        initial_state_hash=initial_hash,
        epoch_order_hash=order_hash,
        best_validation_accuracy=best_accuracy,
        final_validation_accuracy=final_validation_accuracy,
        selected_test_accuracy=selected_test_accuracy,
        mean_validation_accuracy=mean_validation_accuracy,
        final_validation_loss=final_validation_loss,
        mean_validation_loss=mean_validation_loss,
        best_epoch=best_epoch,
        elapsed_seconds=elapsed,
        steps=optimizer_step,
        diverged=diverged,
        **aggregate_telemetry,
    )
    del model, optimizer
    if device.type == "mps":
        torch.mps.empty_cache()
    return epoch_rows, run_row


def run_optimizer_steps(
    model: nn.Module,
    optimizer: MAL_SGD,
    positive_gradients: list[torch.Tensor],
    negative_gradients: list[torch.Tensor],
    count: int,
    *,
    offset: int = 0,
) -> None:
    for step in range(offset, offset + count):
        gradients = positive_gradients if step % 2 == 0 else negative_gradients
        for parameter, gradient in zip(model.parameters(), gradients):
            parameter.grad = gradient
        optimizer.step()


def optimizer_microbenchmark(
    *,
    device: torch.device,
    steps: int,
) -> dict[str, Any]:
    timings: dict[str, list[float]] = {"tensor": [], "output": []}
    state_sizes = {}
    for repeat in range(6):
        scopes = ("tensor", "output") if repeat % 2 == 0 else ("output", "tensor")
        for scope in scopes:
            torch.manual_seed(9191 + repeat)
            model = CifarResidualCNN().to(device)
            positive_gradients = [torch.randn_like(parameter) for parameter in model.parameters()]
            negative_gradients = [-gradient for gradient in positive_gradients]
            optimizer = MAL_SGD(
                model.parameters(),
                lr=0.0,
                beta=BASE_BETA,
                adaptive=True,
                c=0.3,
                per_output=scope == "output",
            )
            run_optimizer_steps(
                model,
                optimizer,
                positive_gradients,
                negative_gradients,
                20,
            )
            sync(device)
            start = time.perf_counter()
            run_optimizer_steps(
                model,
                optimizer,
                positive_gradients,
                negative_gradients,
                steps,
                offset=20,
            )
            sync(device)
            timings[scope].append((time.perf_counter() - start) / steps * 1e3)

            momentum_bytes = 0
            beta_bytes = 0
            beta_scalars = 0
            for group in optimizer.param_groups:
                momentum_bytes += sum(value.numel() * value.element_size() for value in group["momentum"])
                beta_bytes += sum(value.numel() * value.element_size() for value in group["beta"])
                beta_scalars += sum(value.numel() for value in group["beta"])
            state_sizes[scope] = {
                "momentum_state_bytes": momentum_bytes,
                "beta_state_bytes": beta_bytes,
                "beta_scalar_count": beta_scalars,
            }
            del model, optimizer

    results = {
        scope: {
            "median_step_ms": statistics.median(timings[scope]),
            "step_ms_samples": timings[scope],
            **state_sizes[scope],
        }
        for scope in ("tensor", "output")
    }
    results["output_over_tensor_step_time_ratio"] = results["output"]["median_step_ms"] / results["tensor"]["median_step_ms"]
    return results


def mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize(run_rows: list[RunRow]) -> dict[str, Any]:
    cells: dict[tuple[str, float, float, str], list[RunRow]] = defaultdict(list)
    pairs: dict[tuple[str, float, float, int], dict[str, RunRow]] = defaultdict(dict)
    for row in run_rows:
        cells[(row.condition, row.adaptation_rate, row.learning_rate, row.scope)].append(row)
        pairs[(row.condition, row.adaptation_rate, row.learning_rate, row.seed)][row.scope] = row

    cell_summary = {}
    for key, rows in cells.items():
        condition, adaptation_rate, learning_rate, scope = key
        name = f"{condition}|c={adaptation_rate}|lr={learning_rate}|{scope}"
        cell_summary[name] = {
            "best_validation_accuracy": mean_std([row.best_validation_accuracy for row in rows]),
            "selected_test_accuracy": mean_std([row.selected_test_accuracy for row in rows]),
            "mean_validation_accuracy": mean_std([row.mean_validation_accuracy for row in rows]),
            "elapsed_seconds": mean_std([row.elapsed_seconds for row in rows]),
            "divergence_count": sum(row.diverged for row in rows),
            "beta_mean": mean_std([row.beta_mean for row in rows]),
            "mean_within_tensor_beta_std": mean_std([row.mean_within_tensor_beta_std for row in rows]),
        }

    paired_rows = []
    for key, scopes in sorted(pairs.items()):
        if set(scopes) != {"tensor", "output"}:
            continue
        tensor_row = scopes["tensor"]
        output_row = scopes["output"]
        if tensor_row.initial_state_hash != output_row.initial_state_hash:
            raise AssertionError(f"unpaired initial states for {key}")
        if tensor_row.epoch_order_hash != output_row.epoch_order_hash:
            raise AssertionError(f"unpaired epoch orders for {key}")
        paired_rows.append(
            {
                "condition": key[0],
                "adaptation_rate": key[1],
                "learning_rate": key[2],
                "seed": key[3],
                "delta_best_validation_accuracy": output_row.best_validation_accuracy - tensor_row.best_validation_accuracy,
                "delta_selected_test_accuracy": output_row.selected_test_accuracy - tensor_row.selected_test_accuracy,
                "delta_mean_validation_accuracy": output_row.mean_validation_accuracy - tensor_row.mean_validation_accuracy,
                "output_over_tensor_elapsed_ratio": output_row.elapsed_seconds / tensor_row.elapsed_seconds,
            }
        )

    aggregate_paired = {}
    for metric in (
        "delta_best_validation_accuracy",
        "delta_selected_test_accuracy",
        "delta_mean_validation_accuracy",
        "output_over_tensor_elapsed_ratio",
    ):
        aggregate_paired[metric] = mean_std([row[metric] for row in paired_rows])
    aggregate_paired["output_wins_best_validation"] = sum(row["delta_best_validation_accuracy"] > 0.0 for row in paired_rows)
    aggregate_paired["ties_best_validation"] = sum(row["delta_best_validation_accuracy"] == 0.0 for row in paired_rows)
    aggregate_paired["pair_count"] = len(paired_rows)

    return {
        "cells": cell_summary,
        "paired_rows": paired_rows,
        "aggregate_paired": aggregate_paired,
    }


def write_csv(path: Path, rows: list[Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=asdict(rows[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/omar/Python/datasets/CIFAR10"),
    )
    parser.add_argument("--seeds", type=parse_ints, default=[101, 202, 303])
    parser.add_argument(
        "--learning-rates",
        type=parse_floats,
        default=[0.05, 0.2, 0.6],
    )
    parser.add_argument(
        "--adaptation-rates",
        type=parse_floats,
        default=[0.3, 1.0],
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("clean", "noise20"),
        default=["clean", "noise20"],
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-per-class", type=int, default=1_000)
    parser.add_argument("--validation-per-class", type=int, default=200)
    parser.add_argument(
        "--test-per-class",
        type=int,
        default=1_000,
        help="Use 1000 for the full official test set; smaller values make a fixed stratified test subset.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--telemetry-every", type=int, default=5)
    parser.add_argument("--microbenchmark-steps", type=int, default=100)
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was explicitly requested but torch.backends.mps.is_available() is false; refusing to silently substitute CPU.")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_raw = CIFAR10(args.data_root, train=True, download=False)
    test_raw = CIFAR10(args.data_root, train=False, download=False)
    test_indices, _ = stratified_split(
        np.asarray(test_raw.targets),
        train_per_class=args.test_per_class,
        validation_per_class=0,
        seed=424_242,
    )
    test_images = as_image_tensor(test_raw.data, test_indices)
    test_labels = torch.as_tensor(
        np.asarray(test_raw.targets)[test_indices],
        dtype=torch.long,
    )
    train_targets = np.asarray(train_raw.targets)

    model_probe = CifarResidualCNN()
    parameter_count = sum(parameter.numel() for parameter in model_probe.parameters())
    del model_probe
    microbenchmark = optimizer_microbenchmark(
        device=device,
        steps=args.microbenchmark_steps,
    )

    all_epoch_rows: list[EpochRow] = []
    all_run_rows: list[RunRow] = []
    data_metadata = {}
    for seed in args.seeds:
        torch.manual_seed(seed)
        initial_model = CifarResidualCNN()
        initial_state = {key: value.detach().cpu().clone() for key, value in initial_model.state_dict().items()}
        del initial_model
        train_indices, validation_indices = stratified_split(
            train_targets,
            train_per_class=args.train_per_class,
            validation_per_class=args.validation_per_class,
            seed=seed,
        )
        train_images = as_image_tensor(train_raw.data, train_indices)
        clean_train_labels = torch.as_tensor(
            train_targets[train_indices],
            dtype=torch.long,
        )
        validation_images = as_image_tensor(train_raw.data, validation_indices)
        validation_labels = torch.as_tensor(
            train_targets[validation_indices],
            dtype=torch.long,
        )
        epoch_orders = make_epoch_orders(
            len(train_indices),
            args.epochs,
            seed + 700_000,
        )

        for condition in args.conditions:
            noise_fraction = 0.0 if condition == "clean" else 0.2
            train_labels = corrupt_labels(
                clean_train_labels,
                fraction=noise_fraction,
                seed=seed + 900_000,
            )
            data_metadata[f"{condition}:{seed}"] = {
                "train_examples": len(train_labels),
                "validation_examples": len(validation_labels),
                "test_examples": len(test_labels),
                "noise_fraction": noise_fraction,
                "train_indices_hash": tensor_hash((torch.as_tensor(train_indices),)),
                "validation_indices_hash": tensor_hash((torch.as_tensor(validation_indices),)),
                "train_labels_hash": tensor_hash((train_labels,)),
                "epoch_order_hash": tensor_hash(tuple(epoch_orders)),
            }

            for adaptation_rate in args.adaptation_rates:
                for learning_rate in args.learning_rates:
                    pair_index = len(all_run_rows) // 2
                    scopes = ("tensor", "output") if pair_index % 2 == 0 else ("output", "tensor")
                    for scope in scopes:
                        epoch_rows, run_row = train_one(
                            condition=condition,
                            scope=scope,
                            adaptation_rate=adaptation_rate,
                            learning_rate=learning_rate,
                            seed=seed,
                            initial_state=initial_state,
                            train_images=train_images,
                            train_labels=train_labels,
                            validation_images=validation_images,
                            validation_labels=validation_labels,
                            test_images=test_images,
                            test_labels=test_labels,
                            epoch_orders=epoch_orders,
                            batch_size=args.batch_size,
                            weight_decay=args.weight_decay,
                            telemetry_every=args.telemetry_every,
                            device=device,
                        )
                        all_epoch_rows.extend(epoch_rows)
                        all_run_rows.append(run_row)
                        write_csv(args.output_dir / "epoch_metrics.csv", all_epoch_rows)
                        write_csv(args.output_dir / "run_metrics.csv", all_run_rows)

    summary = summarize(all_run_rows)
    metadata = {
        "experiment": {
            "device": str(device),
            "dataset": "CIFAR-10",
            "model": "5-block residual CNN with GroupNorm",
            "parameter_count": parameter_count,
            "scopes": ["tensor", "output"],
            "adaptive": True,
            "base_beta": BASE_BETA,
            "adaptation_rates": args.adaptation_rates,
            "learning_rates": args.learning_rates,
            "seeds": args.seeds,
            "conditions": args.conditions,
            "epochs": args.epochs,
            "train_per_class": args.train_per_class,
            "validation_per_class": args.validation_per_class,
            "test_per_class": args.test_per_class,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "augmentation": "paired per-example reflect crop + horizontal flip",
            "scheduler": "one-epoch linear warmup then cosine decay to 2%",
            "selection": "test evaluated from each run's best-validation epoch",
        },
        "data": data_metadata,
        "optimizer_microbenchmark": microbenchmark,
        "summary": summary,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    (args.output_dir / "metadata_and_summary.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
