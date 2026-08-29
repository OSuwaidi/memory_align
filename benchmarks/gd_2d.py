"""Publication-oriented 2D convergence benchmarks for GD-family optimizers.

The benchmark intentionally uses exact gradients. Seed sensitivity comes from
small, paired perturbations of each objective's canonical starting point, so
every optimizer and learning rate sees the same start for a given seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from torch.optim import Optimizer

from optims.am_opt import AM_MSGD
from optims.cautious_opt import CAUTIOUS_SGD
from optims.mal_opt import MAL_SGDM
from optims.tam_opt import TAM_SGDM


DEFAULT_SEEDS = (42, 1337, 2026, 31415, 271828)
MOMENTUM = 0.9
GAP_FLOOR = 1e-12
DIVERGENCE_RELATIVE_GAP = 1e6
DIVERGENCE_LOG_AUC = 6.0
MAX_ABS_COORDINATE = 1e8
MAX_OBJECTIVE_VALUE = 1e30


TensorObjective = Callable[[torch.Tensor], torch.Tensor]
OptimizerFactory = Callable[[Iterable[torch.nn.Parameter], float], Optimizer]


@dataclass(frozen=True)
class ObjectiveSpec:
    key: str
    label: str
    category: str
    formula: str
    rationale: str
    value_fn: TensorObjective
    start: tuple[float, float]
    jitter_std: tuple[float, float]
    domain: tuple[float, float, float, float]
    steps: int
    learning_rates: tuple[float, ...]
    global_minimum: float
    minimizers: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class OptimizerSpec:
    key: str
    label: str
    short_label: str
    factory: OptimizerFactory
    config: dict[str, Any]
    color: str
    linestyle: Any
    marker: str
    linewidth: float = 1.8
    zorder: int = 3


@dataclass
class RunResult:
    values: np.ndarray
    positions: np.ndarray
    diverged: bool
    steps_completed: int


def _rotated_coordinates(point: torch.Tensor, angle_degrees: float) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = point.unbind(dim=-1)
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return cosine * x + sine * y, -sine * x + cosine * y


def _ill_conditioned_quadratic(point: torch.Tensor) -> torch.Tensor:
    z1, z2 = _rotated_coordinates(point, 35.0)
    return 0.5 * (z1.square() + 50.0 * z2.square())


def _rotated_quartic(point: torch.Tensor) -> torch.Tensor:
    z1, z2 = _rotated_coordinates(point, -25.0)
    return 0.25 * z1.pow(4) + 6.0 * z2.square()


def _rosenbrock(point: torch.Tensor) -> torch.Tensor:
    x, y = point.unbind(dim=-1)
    return (1.0 - x).square() + 100.0 * (y - x.square()).square()


def _himmelblau(point: torch.Tensor) -> torch.Tensor:
    x, y = point.unbind(dim=-1)
    return (x.square() + y - 11.0).square() + (x + y.square() - 7.0).square()


def _memory_trap(point: torch.Tensor) -> torch.Tensor:
    x, y = point.unbind(dim=-1)
    ripple = 1.0 - torch.cos(torch.pi * x)
    return 0.5 * (x.square() + y.square()) + 0.5 * ripple.square()


def _memory_trap_learning_rates() -> tuple[float, ...]:
    broad_grid = (
        2e-3,
        4e-3,
        8e-3,
        1e-2,
        1.5e-2,
        3e-2,
        6e-2,
        1e-1,
        1.6e-1,
        5.5e-1,
        7e-1,
        8.5e-1,
        1.0,
        1.2,
        1.5,
        2.0,
    )
    # The stale-memory transition is intentionally sharp. Resolve its useful
    # region densely so the GDM result cannot be attributed to a coarse grid.
    local_grid = tuple(float(value) for value in np.geomspace(0.2, 0.5, 121))
    return tuple(sorted(set((*broad_grid, *local_grid))))


def objective_specs() -> tuple[ObjectiveSpec, ...]:
    return (
        ObjectiveSpec(
            key="ill_conditioned_quadratic",
            label="Ill-conditioned rotated quadratic",
            category="Convex · smooth · strongly convex",
            formula="f(x,y) = ½(z₁² + 50z₂²),  z = R₃₅°(x,y)",
            rationale="A controlled condition-number-50 ravine isolates acceleration, oscillation, and stability on constant curvature.",
            value_fn=_ill_conditioned_quadratic,
            start=(-3.2, 2.4),
            jitter_std=(0.10, 0.10),
            domain=(-4.2, 4.2, -4.2, 4.2),
            steps=180,
            learning_rates=(1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 4e-2, 8e-2, 1.2e-1, 2e-1, 3e-1, 5e-1),
            global_minimum=0.0,
            minimizers=((0.0, 0.0),),
        ),
        ObjectiveSpec(
            key="rotated_quartic",
            label="Rotated quartic valley",
            category="Convex · smooth · not strongly convex",
            formula="f(x,y) = ¼z₁⁴ + 6z₂²,  z = R₋₂₅°(x,y)",
            rationale="Variable curvature and a flat optimum test whether accumulated memory remains useful as one direction loses curvature.",
            value_fn=_rotated_quartic,
            start=(-2.0, 1.8),
            jitter_std=(0.08, 0.08),
            domain=(-3.0, 3.0, -3.0, 3.0),
            steps=260,
            learning_rates=(2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 4e-2, 7e-2, 1.1e-1, 1.7e-1, 2.5e-1, 3.5e-1, 5e-1, 5.5e-1, 5.8e-1, 6e-1, 6.5e-1, 8e-1),
            global_minimum=0.0,
            minimizers=((0.0, 0.0),),
        ),
        ObjectiveSpec(
            key="rosenbrock",
            label="Rosenbrock function",
            category="Non-convex · curved narrow valley",
            formula="f(x,y) = (1−x)² + 100(y−x²)²",
            rationale="The classic banana valley tests delayed directions while the gradient rotates along a curved manifold.",
            value_fn=_rosenbrock,
            start=(-1.5, 1.5),
            jitter_std=(0.05, 0.06),
            domain=(-2.0, 2.2, -1.0, 3.2),
            steps=700,
            learning_rates=(1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 4e-4, 7e-4, 1e-3, 1.5e-3, 2.2e-3, 3.2e-3, 5e-3),
            global_minimum=0.0,
            minimizers=((1.0, 1.0),),
        ),
        ObjectiveSpec(
            key="himmelblau",
            label="Himmelblau function",
            category="Non-convex · multimodal",
            formula="f(x,y) = (x²+y−11)² + (x+y²−7)²",
            rationale="Four equal global minima expose basin selection, overshoot, and robustness on a standard multimodal landscape.",
            value_fn=_himmelblau,
            start=(-3.8, 0.0),
            jitter_std=(0.08, 0.08),
            domain=(-5.0, 5.0, -5.0, 5.0),
            steps=300,
            learning_rates=(2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 7e-3, 1e-2, 1.5e-2, 2.5e-2),
            global_minimum=0.0,
            minimizers=((3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)),
        ),
        ObjectiveSpec(
            key="memory_trap",
            label="Stale-memory trap (custom)",
            category="Non-convex · adversarial stale-memory case",
            formula="f(x,y) = ½(x²+y²) + ½[1−cos(πx)]²",
            rationale=(
                "A smooth coercive objective with a unique known global minimum. From (2.01,1.5), LR=1 sends GD essentially to the origin on step one; "
                "GDM takes the same first step, but its retained buffer immediately pushes it to roughly (−1.81,−1.35) on step two. The x-curvature "
                "equals 1−2π² at x=1, proving the objective is non-convex."
            ),
            value_fn=_memory_trap,
            start=(2.01, 1.5),
            jitter_std=(0.005, 0.10),
            domain=(-3.0, 3.0, -2.5, 2.5),
            steps=150,
            learning_rates=_memory_trap_learning_rates(),
            global_minimum=0.0,
            minimizers=((0.0, 0.0),),
        ),
    )


def optimizer_specs() -> tuple[OptimizerSpec, ...]:
    def vanilla_gd(params: Iterable[torch.nn.Parameter], lr: float) -> Optimizer:
        return torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=0.0, foreach=False)

    def gdm(params: Iterable[torch.nn.Parameter], lr: float) -> Optimizer:
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=MOMENTUM,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=False,
            foreach=False,
        )

    def cautious_gdm(params: Iterable[torch.nn.Parameter], lr: float) -> Optimizer:
        return CAUTIOUS_SGD(params, lr=lr, beta=MOMENTUM, weight_decay=0.0, nesterov=False)

    def tam_gdm(params: Iterable[torch.nn.Parameter], lr: float) -> Optimizer:
        return TAM_SGDM(params, lr=lr, beta=MOMENTUM, gamma=0.9, torque_eps=1e-8, weight_decay=0.0)

    def am_gdm(params: Iterable[torch.nn.Parameter], lr: float) -> Optimizer:
        return AM_MSGD(params, lr=lr, beta_max=MOMENTUM, model_lambda=0.1, weight_decay=0.0)

    def mal_unscaled(params: Iterable[torch.nn.Parameter], lr: float) -> Optimizer:
        return MAL_SGDM(
            params,
            lr=lr,
            beta=MOMENTUM,
            weight_decay=0.0,
            pwr=1.0,
            in_place=False,
            scale=False,
            nesterov=False,
            gate_mode="attenuate",
        )

    def mal_scaled(params: Iterable[torch.nn.Parameter], lr: float) -> Optimizer:
        return MAL_SGDM(
            params,
            lr=lr,
            beta=MOMENTUM,
            weight_decay=0.0,
            pwr=1.0,
            in_place=False,
            scale=True,
            nesterov=False,
            gate_mode="attenuate",
        )

    return (
        OptimizerSpec(
            key="gd",
            label="Vanilla GD",
            short_label="Vanilla GD",
            factory=vanilla_gd,
            config={"implementation": "torch.optim.SGD", "momentum": 0.0, "weight_decay": 0.0},
            color="#7A7F87",
            linestyle=(0, (4, 2)),
            marker="o",
            linewidth=1.6,
            zorder=2,
        ),
        OptimizerSpec(
            key="gdm",
            label="GDM",
            short_label="GDM",
            factory=gdm,
            config={"implementation": "torch.optim.SGD", "momentum": MOMENTUM, "dampening": 0.0, "nesterov": False, "weight_decay": 0.0},
            color="#202124",
            linestyle="-",
            marker="s",
            linewidth=2.0,
            zorder=4,
        ),
        OptimizerSpec(
            key="cautious_gdm",
            label="Cautious GDM",
            short_label="Cautious GDM",
            factory=cautious_gdm,
            config={"implementation": "CAUTIOUS_SGD", "beta": MOMENTUM, "nesterov": False, "weight_decay": 0.0},
            color="#D97706",
            linestyle=(0, (6, 2, 1, 2)),
            marker="D",
        ),
        OptimizerSpec(
            key="tam_gdm",
            label="TAM GDM",
            short_label="TAM GDM",
            factory=tam_gdm,
            config={"implementation": "TAM_SGDM", "beta": MOMENTUM, "gamma": 0.9, "torque_eps": 1e-8, "weight_decay": 0.0},
            color="#6B7D2A",
            linestyle=(0, (1, 1.6)),
            marker="^",
        ),
        OptimizerSpec(
            key="am_gdm",
            label="AM GDM",
            short_label="AM GDM",
            factory=am_gdm,
            config={"implementation": "AM_MSGD", "beta_max": MOMENTUM, "model_lambda": 0.1, "weight_decay": 0.0},
            color="#B64B7C",
            linestyle=(0, (5, 1, 1, 1)),
            marker="v",
        ),
        OptimizerSpec(
            key="mal_gdm_unscaled",
            label="MAL GDM (scale=False)",
            short_label="MAL GDM\nscale=False",
            factory=mal_unscaled,
            config={
                "implementation": "MAL_SGDM",
                "beta": MOMENTUM,
                "in_place": False,
                "pwr": 1.0,
                "scale": False,
                "gate_mode": "attenuate",
                "nesterov": False,
                "weight_decay": 0.0,
            },
            color="#2563EB",
            linestyle="-",
            marker="P",
            linewidth=2.5,
            zorder=6,
        ),
        OptimizerSpec(
            key="mal_gdm_scaled",
            label="MAL GDM (scale=True)",
            short_label="MAL GDM\nscale=True",
            factory=mal_scaled,
            config={
                "implementation": "MAL_SGDM",
                "beta": MOMENTUM,
                "in_place": False,
                "pwr": 1.0,
                "scale": True,
                "gate_mode": "attenuate",
                "nesterov": False,
                "weight_decay": 0.0,
            },
            color="#C28A0E",
            linestyle=(0, (7, 2)),
            marker="X",
            linewidth=2.5,
            zorder=7,
        ),
    )


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 12.0,
            "axes.labelsize": 10.0,
            "axes.edgecolor": "#34373B",
            "axes.linewidth": 0.8,
            "axes.facecolor": "#FCFCFB",
            "figure.facecolor": "white",
            "grid.color": "#D9DDE2",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _validate_configuration(objectives: Sequence[ObjectiveSpec], optimizers: Sequence[OptimizerSpec]) -> None:
    if len({objective.key for objective in objectives}) != len(objectives):
        raise ValueError("Objective keys must be unique")
    if len({optimizer.key for optimizer in optimizers}) != len(optimizers):
        raise ValueError("Optimizer keys must be unique")

    for objective in objectives:
        if tuple(sorted(objective.learning_rates)) != objective.learning_rates or min(objective.learning_rates) <= 0.0:
            raise ValueError(f"Learning rates for {objective.key} must be positive and sorted")
        x_min, x_max, y_min, y_max = objective.domain
        if not (x_min < objective.start[0] < x_max and y_min < objective.start[1] < y_max):
            raise ValueError(f"Start for {objective.key} must be strictly inside the contour domain")
        for minimizer in objective.minimizers:
            point = torch.tensor(minimizer, dtype=torch.float64, requires_grad=True)
            value = objective.value_fn(point)
            gradient = torch.autograd.grad(value, point)[0]
            if not torch.isclose(value, torch.tensor(objective.global_minimum, dtype=torch.float64), atol=2e-7, rtol=0.0):
                raise ValueError(f"Declared minimizer for {objective.key} has value {value.item():.6g}")
            if torch.linalg.vector_norm(gradient).item() > 2e-4:
                raise ValueError(f"Declared minimizer for {objective.key} is not stationary")


def _seeded_start(objective: ObjectiveSpec, seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    start = np.asarray(objective.start, dtype=np.float64)
    jitter = generator.normal(loc=0.0, scale=np.asarray(objective.jitter_std, dtype=np.float64), size=2)
    perturbed = start + jitter
    x_min, x_max, y_min, y_max = objective.domain
    margin_x = 0.01 * (x_max - x_min)
    margin_y = 0.01 * (y_max - y_min)
    perturbed[0] = np.clip(perturbed[0], x_min + margin_x, x_max - margin_x)
    perturbed[1] = np.clip(perturbed[1], y_min + margin_y, y_max - margin_y)
    return float(perturbed[0]), float(perturbed[1])


def _run_optimizer(
    objective: ObjectiveSpec,
    optimizer_spec: OptimizerSpec,
    learning_rate: float,
    start: tuple[float, float],
    steps: int,
) -> RunResult:
    point = torch.nn.Parameter(torch.tensor(start, dtype=torch.float64))
    optimizer = optimizer_spec.factory([point], learning_rate)
    values = np.full(steps + 1, np.nan, dtype=np.float64)
    positions = np.full((steps + 1, 2), np.nan, dtype=np.float64)

    with torch.no_grad():
        initial_value = objective.value_fn(point)
    values[0] = float(initial_value)
    positions[0] = point.detach().cpu().numpy()

    if not np.isfinite(values[0]):
        return RunResult(values=values, positions=positions, diverged=True, steps_completed=0)

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        value = objective.value_fn(point)
        if not bool(torch.isfinite(value)):
            return RunResult(values=values, positions=positions, diverged=True, steps_completed=step - 1)
        value.backward()
        if point.grad is None or not bool(torch.isfinite(point.grad).all()):
            return RunResult(values=values, positions=positions, diverged=True, steps_completed=step - 1)
        optimizer.step()

        with torch.no_grad():
            next_value = objective.value_fn(point)
            finite = bool(torch.isfinite(point).all()) and bool(torch.isfinite(next_value))
            bounded = finite and float(point.detach().abs().max()) <= MAX_ABS_COORDINATE and float(next_value) <= MAX_OBJECTIVE_VALUE
            if not bounded:
                return RunResult(values=values, positions=positions, diverged=True, steps_completed=step - 1)
            values[step] = float(next_value)
            positions[step] = point.detach().cpu().numpy()

    return RunResult(values=values, positions=positions, diverged=False, steps_completed=steps)


def _performance_metrics(result: RunResult, global_minimum: float) -> tuple[float, float]:
    if result.diverged or not np.isfinite(result.values).all():
        return DIVERGENCE_RELATIVE_GAP, DIVERGENCE_LOG_AUC
    gaps = np.maximum(result.values - global_minimum, 0.0)
    initial_gap = max(float(gaps[0]), GAP_FLOOR)
    relative_gap = np.clip(gaps / initial_gap, GAP_FLOOR, DIVERGENCE_RELATIVE_GAP)
    tail_count = max(1, math.ceil(0.10 * relative_gap.size))
    tail_mean_relative_gap = max(float(np.mean(relative_gap[-tail_count:])), GAP_FLOOR)
    log_gap_auc = float(np.mean(np.log10(relative_gap)))
    return tail_mean_relative_gap, log_gap_auc


def _sem(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return 0.0
    return float(array.std(ddof=1) / math.sqrt(array.size))


def _format_lr(value: float) -> str:
    if value < 1e-3 or value >= 100.0:
        return f"{value:.1e}"
    return f"{value:.4g}"


def _tested_learning_rates(objective: ObjectiveSpec, quick: bool) -> tuple[float, ...]:
    if not quick:
        return objective.learning_rates
    indices = np.linspace(0, len(objective.learning_rates) - 1, min(10, len(objective.learning_rates)), dtype=int)
    rates = [objective.learning_rates[index] for index in indices]
    for required in (1.0, objective.learning_rates[-1]):
        if required in objective.learning_rates and required not in rates:
            rates.append(required)
    return tuple(sorted(rates))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: plt.Figure, target_without_suffix: Path, dpi: int) -> None:
    fig.savefig(target_without_suffix.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(
        target_without_suffix.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={"Creator": "memory_align 2D benchmark", "CreationDate": None},
    )
    plt.close(fig)


def _aggregate_sweep(
    sweep_rows: Sequence[dict[str, Any]],
    objectives: Sequence[ObjectiveSpec],
    optimizers: Sequence[OptimizerSpec],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float]]:
    summaries: list[dict[str, Any]] = []
    selected: dict[tuple[str, str], float] = {}

    for objective in objectives:
        tested_learning_rates = sorted({float(row["learning_rate"]) for row in sweep_rows if row["objective"] == objective.key})
        for optimizer in optimizers:
            candidates: list[dict[str, Any]] = []
            for learning_rate in tested_learning_rates:
                matching = [
                    row
                    for row in sweep_rows
                    if row["objective"] == objective.key and row["optimizer"] == optimizer.key and row["learning_rate"] == learning_rate
                ]
                tail_gaps = [float(row["tail_mean_relative_gap"]) for row in matching]
                log_aucs = [float(row["log_gap_auc"]) for row in matching]
                divergence_rate = float(np.mean([bool(row["diverged"]) for row in matching]))
                summary = {
                    "objective": objective.key,
                    "optimizer": optimizer.key,
                    "learning_rate": learning_rate,
                    "mean_tail_relative_gap": float(np.mean(tail_gaps)),
                    "sem_tail_relative_gap": _sem(tail_gaps),
                    "mean_log_gap_auc": float(np.mean(log_aucs)),
                    "divergence_rate": divergence_rate,
                    "seed_count": len(tail_gaps),
                }
                summaries.append(summary)
                candidates.append(summary)

            stable = [candidate for candidate in candidates if candidate["divergence_rate"] == 0.0]
            selection_pool = stable if stable else candidates
            winner = min(
                selection_pool,
                key=lambda candidate: (
                    candidate["mean_tail_relative_gap"],
                    candidate["mean_log_gap_auc"],
                    candidate["sem_tail_relative_gap"],
                    candidate["learning_rate"],
                ),
            )
            selected[(objective.key, optimizer.key)] = float(winner["learning_rate"])

    return summaries, selected


def _plot_lr_sensitivity(
    objective: ObjectiveSpec,
    optimizers: Sequence[OptimizerSpec],
    summaries: Sequence[dict[str, Any]],
    selected: dict[tuple[str, str], float],
    target: Path,
    seed_count: int,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=False)
    for optimizer in optimizers:
        rows = sorted(
            (row for row in summaries if row["objective"] == objective.key and row["optimizer"] == optimizer.key),
            key=lambda row: row["learning_rate"],
        )
        learning_rates = np.asarray([row["learning_rate"] for row in rows], dtype=np.float64)
        means = np.asarray([row["mean_tail_relative_gap"] for row in rows], dtype=np.float64)
        sems = np.asarray([row["sem_tail_relative_gap"] for row in rows], dtype=np.float64)
        lower = np.clip(means - sems, GAP_FLOOR, None)
        upper = np.clip(means + sems, GAP_FLOOR, None)
        ax.fill_between(learning_rates, lower, upper, color=optimizer.color, alpha=0.10, linewidth=0, zorder=optimizer.zorder - 1)
        ax.plot(
            learning_rates,
            means,
            label=optimizer.label,
            color=optimizer.color,
            linestyle=optimizer.linestyle,
            marker=optimizer.marker,
            markersize=4.4,
            markerfacecolor="white",
            markeredgewidth=0.9,
            markevery=max(1, len(learning_rates) // 18),
            linewidth=optimizer.linewidth,
            zorder=optimizer.zorder,
        )
        chosen_lr = selected[(objective.key, optimizer.key)]
        chosen_index = int(np.flatnonzero(np.isclose(learning_rates, chosen_lr, rtol=0.0, atol=1e-15))[0])
        ax.scatter(
            [chosen_lr],
            [means[chosen_index]],
            marker="*",
            s=115,
            color=optimizer.color,
            edgecolor="white",
            linewidth=0.9,
            zorder=10,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Tail-mean relative objective gap  ↓")
    ax.grid(True, which="major", axis="both")
    ax.grid(True, which="minor", axis="x", alpha=0.24)
    ax.axhline(1.0, color="#8A8F98", linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
    ax.set_title(f"Learning-rate and seed sensitivity — {objective.label}", loc="left", fontweight="bold", pad=26)
    ax.text(
        0.0,
        1.015,
        f"{objective.category}  ·  mean ± 1 SEM across {seed_count} paired initialization seeds  ·  stars mark independently selected LRs",
        transform=ax.transAxes,
        color="#555B63",
        fontsize=8.8,
        va="bottom",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, columnspacing=1.5, handlelength=3.0)
    fig.text(
        0.01,
        0.01,
        "Per-seed metric: arithmetic mean of (f(xₜ)−f*)/(f(x₀)−f*) over the final 10% of steps; lower is better. Divergent runs are set to 10⁶.",
        fontsize=8.0,
        color="#60656D",
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.84, bottom=0.29)
    _save_figure(fig, target, dpi)


def _contour_grid(objective: ObjectiveSpec, resolution: int = 360) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = objective.domain
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(xs, ys)
    points = torch.from_numpy(np.stack((xx, yy), axis=-1)).to(dtype=torch.float64)
    with torch.no_grad():
        values = objective.value_fn(points).cpu().numpy()
    log_gap = np.log10(np.clip(values - objective.global_minimum, GAP_FLOOR, None))
    return xx, yy, log_gap


def _plot_trajectories(
    objective: ObjectiveSpec,
    optimizers: Sequence[OptimizerSpec],
    selected: dict[tuple[str, str], float],
    canonical: dict[tuple[str, str], RunResult],
    target: Path,
    dpi: int,
) -> None:
    xx, yy, log_gap = _contour_grid(objective)
    x_min, x_max, y_min, y_max = objective.domain
    finite = log_gap[np.isfinite(log_gap)]
    low = float(np.percentile(finite, 0.35))
    high = float(np.percentile(finite, 99.5))
    if high - low < 1e-6:
        high = low + 1.0
    levels = np.linspace(low, high, 19)

    fig, axes_array = plt.subplots(2, 4, figsize=(13.2, 7.0), sharex=True, sharey=True)
    axes = list(axes_array.flat)
    contour = None
    for axis, optimizer in zip(axes[:7], optimizers):
        contour = axis.contourf(xx, yy, log_gap, levels=levels, cmap="Greys", extend="both", alpha=0.86)
        axis.contour(xx, yy, log_gap, levels=levels[::3], colors="#6F747B", linewidths=0.35, alpha=0.55)
        result = canonical[(objective.key, optimizer.key)]
        valid = np.isfinite(result.positions).all(axis=1)
        path = result.positions[valid]
        axis.plot(path[:, 0], path[:, 1], color=optimizer.color, linewidth=2.1, linestyle=optimizer.linestyle, zorder=5)
        outside_domain = (path[:, 0] < x_min) | (path[:, 0] > x_max) | (path[:, 1] < y_min) | (path[:, 1] > y_max)
        axis.scatter(
            [path[0, 0]],
            [path[0, 1]],
            marker="o",
            s=34,
            facecolor="white",
            edgecolor="#202124",
            linewidth=1.0,
            zorder=8,
        )
        axis.scatter(
            [path[-1, 0]],
            [path[-1, 1]],
            marker="X",
            s=42,
            color=optimizer.color,
            edgecolor="white",
            linewidth=0.7,
            zorder=9,
        )
        for minimizer in objective.minimizers:
            axis.scatter(
                [minimizer[0]],
                [minimizer[1]],
                marker="*",
                s=70,
                color="#E8B43A",
                edgecolor="#202124",
                linewidth=0.55,
                zorder=7,
            )
        final_gap = max(float(result.values[result.steps_completed] - objective.global_minimum), 0.0)
        axis.set_title(
            f"{optimizer.short_label}\nη={_format_lr(selected[(objective.key, optimizer.key)])}  ·  final gap={final_gap:.1e}",
            fontsize=9.2,
            fontweight="bold",
            color="#25282D",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.grid(False)
        axis.tick_params(labelsize=7.5)
        if bool(outside_domain.any()):
            axis.text(
                0.025,
                0.025,
                f"↗ {int(outside_domain.sum())} steps leave contour domain",
                transform=axis.transAxes,
                fontsize=7.2,
                color="#555B63",
                bbox={"facecolor": "white", "edgecolor": "#B9BDC3", "alpha": 0.88, "boxstyle": "round,pad=0.22", "linewidth": 0.5},
                zorder=10,
            )

    axes[7].axis("off")
    axes[7].text(0.03, 0.95, "Trajectory key", transform=axes[7].transAxes, fontsize=11, fontweight="bold", va="top")
    symbol_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#202124", label="Canonical start"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#2563EB", markeredgecolor="white", label="Final iterate"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#E8B43A", markeredgecolor="#202124", markersize=10, label="Global minimizer"),
    ]
    axes[7].legend(handles=symbol_handles, loc="upper left", bbox_to_anchor=(0.0, 0.80), fontsize=9)
    axes[7].text(
        0.03,
        0.47,
        "Contours show log₁₀ objective gap.\nAll panels use the same domain, levels,\ncanonical start, and exact gradients.\n\nMAL variants: in_place=False, pwr=1,\ngate_mode=attenuate; only scale differs.",
        transform=axes[7].transAxes,
        fontsize=8.7,
        color="#555B63",
        linespacing=1.45,
        va="top",
    )

    for row in range(2):
        axes_array[row, 0].set_ylabel("y")
    for column in range(4):
        axes_array[1, column].set_xlabel("x")

    fig.suptitle(f"LR-tuned optimizer trajectories — {objective.label}", x=0.055, y=0.985, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.055, 0.942, f"{objective.category}  ·  {objective.formula}", fontsize=9.3, color="#555B63")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.13, wspace=0.22, hspace=0.34)
    if contour is not None:
        colorbar_axis = fig.add_axes((0.23, 0.055, 0.50, 0.022))
        colorbar = fig.colorbar(contour, cax=colorbar_axis, orientation="horizontal")
        colorbar.set_label("log₁₀[f(x,y) − f*]", fontsize=8.5)
        colorbar.ax.tick_params(labelsize=7.5)
    _save_figure(fig, target, dpi)


def _plot_convergence(
    objective: ObjectiveSpec,
    optimizers: Sequence[OptimizerSpec],
    selected: dict[tuple[str, str], float],
    canonical: dict[tuple[str, str], RunResult],
    target: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    maximum_step = max(result.steps_completed for (objective_key, _), result in canonical.items() if objective_key == objective.key)
    for optimizer in optimizers:
        result = canonical[(objective.key, optimizer.key)]
        valid = np.isfinite(result.values)
        steps = np.flatnonzero(valid)
        gaps = np.clip(result.values[valid] - objective.global_minimum, GAP_FLOOR, None)
        mark_every = max(1, maximum_step // 11)
        ax.semilogy(
            steps,
            gaps,
            label=f"{optimizer.label}  (η={_format_lr(selected[(objective.key, optimizer.key)])})",
            color=optimizer.color,
            linestyle=optimizer.linestyle,
            linewidth=optimizer.linewidth,
            marker=optimizer.marker,
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=0.8,
            markevery=mark_every,
            zorder=optimizer.zorder,
        )

    ax.set_xlim(0, maximum_step)
    ax.set_ylim(bottom=GAP_FLOOR)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Objective value  f(xₜ)  (global minimum = 0)")
    ax.grid(True, which="major", axis="both")
    ax.grid(True, which="minor", axis="y", alpha=0.22)
    ax.set_title(f"Convergence speed with independently tuned LRs — {objective.label}", loc="left", fontweight="bold", pad=26)
    ax.text(
        0.0,
        1.015,
        f"{objective.category}  ·  exact gradients from the shared canonical start  ·  floor at {GAP_FLOOR:.0e}",
        transform=ax.transAxes,
        color="#555B63",
        fontsize=8.8,
        va="bottom",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, columnspacing=1.8, handlelength=3.1)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.84, bottom=0.31)
    _save_figure(fig, target, dpi)


def _write_readme(
    path: Path,
    objectives: Sequence[ObjectiveSpec],
    optimizers: Sequence[OptimizerSpec],
    seeds: Sequence[int],
    selected: dict[tuple[str, str], float],
    quick: bool,
) -> None:
    lines = [
        "# GD-family 2D convergence benchmark",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} using {'quick smoke-test' if quick else 'full'} settings.",
        "",
        "## Output layout",
        "",
        "- `lr_seed_sensitivity/`: LR tuning curves with ±1 SEM bands across paired seeded starts.",
        "- `trajectory_contours/`: one small-multiple contour figure per objective using each method's independently selected LR.",
        "- `convergence_curves/`: objective value versus step from the canonical start using the same tuned LRs.",
        "- `lr_sweep.csv`, `lr_summary.csv`, `selected_lrs.csv`, and `canonical_runs.csv`: reproducible numerical results.",
        "- `manifest.json`: formulas, settings, optimizer arguments, protocol, and chart map.",
        "- `validation_report.json`: generated by the companion validator after numerical and file-integrity checks pass.",
        "",
        "Every plot is exported as both a high-resolution PNG and a vector PDF.",
        "",
        "## Protocol",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}.",
        "- Objectives and gradients are deterministic and evaluated in float64. Seeds control small Gaussian perturbations of the canonical start; the same per-seed start is reused for every optimizer and LR (common random numbers).",
        "- The primary LR metric is the arithmetic mean of `(f(x_t)-f*)/(f(x_0)-f*)` over the final 10% of steps, floored at 1e-12. Lower is better. A divergent run is assigned 1e6.",
        "- Selection is independent per objective and optimizer. The minimum across-seed mean tail gap is selected among zero-divergence LR candidates; full-run log-gap AUC breaks numerical ties.",
        "- Trajectory and convergence plots use exact gradients and the unperturbed canonical start. All optimizer state is recreated for every run.",
        "- Every method uses one 2-vector parameter, no weight decay, no Nesterov, and momentum/beta 0.9 where applicable.",
        "- MAL is shown twice with `in_place=False`, `pwr=1.0`, `gate_mode=\"attenuate\"`; only `scale=False` versus `scale=True` differs.",
        "",
        "## Objectives",
        "",
        "| Objective | Class | Formula | Why included |",
        "|---|---|---|---|",
    ]
    for objective in objectives:
        lines.append(f"| {objective.label} | {objective.category} | {objective.formula} | {objective.rationale} |")

    lines.extend(["", "## Selected learning rates", ""])
    header = "| Objective | " + " | ".join(optimizer.label for optimizer in optimizers) + " |"
    divider = "|---|" + "---|" * len(optimizers)
    lines.extend((header, divider))
    for objective in objectives:
        values = " | ".join(_format_lr(selected[(objective.key, optimizer.key)]) for optimizer in optimizers)
        lines.append(f"| {objective.label} | {values} |")

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "uv run python -m benchmarks.gd_2d --output-dir benchmark_outputs",
            "uv run python -m benchmarks.validate_gd_2d --output-dir benchmark_outputs",
            "```",
            "",
            "Use `--quick` for a short smoke test or `--objectives <key> ...` for a subset.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _chart_map(objectives: Sequence[ObjectiveSpec]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for objective in objectives:
        rows.extend(
            (
                {
                    "section": f"{objective.key}/lr_seed_sensitivity",
                    "analytical_question": "Which LR is robust and fastest for each optimizer across paired starts?",
                    "chart_family": "Uncertainty & Benchmark",
                    "chart_type": "Multi-series line with SEM ribbon",
                    "fields": "learning_rate, tail_mean_relative_gap, optimizer, seed",
                    "supported_takeaway": "Independent LR choice and sensitivity width",
                    "palette_policy": "Five restrained roots plus neutrals; marker and dash redundancy",
                    "artifact": f"lr_seed_sensitivity/{objective.key}.png|pdf",
                },
                {
                    "section": f"{objective.key}/trajectory_contours",
                    "analytical_question": "How does each tuned optimizer move through the same 2D geometry?",
                    "chart_family": "Relationship",
                    "chart_type": "Seven-panel log-gap contour with trajectory overlay",
                    "fields": "x, y, objective_gap, optimizer, step",
                    "supported_takeaway": "Path geometry, oscillation, basin selection, and final iterate",
                    "palette_policy": "Neutral contours with one explicit optimizer color per panel",
                    "artifact": f"trajectory_contours/{objective.key}.png|pdf",
                },
                {
                    "section": f"{objective.key}/convergence_curves",
                    "analytical_question": "How quickly does each independently tuned method reduce objective value?",
                    "chart_family": "Trend",
                    "chart_type": "Multi-series log-scale line",
                    "fields": "step, objective_value, optimizer, selected_learning_rate",
                    "supported_takeaway": "Convergence speed and asymptotic floor from a shared start",
                    "palette_policy": "Five restrained roots plus neutrals; marker and dash redundancy",
                    "artifact": f"convergence_curves/{objective.key}.png|pdf",
                },
            )
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_outputs"), help="Root directory for results and the three plot folders.")
    parser.add_argument("--objectives", nargs="*", default=None, help="Optional objective keys to run; default is the full suite.")
    parser.add_argument("--quick", action="store_true", help="Use three seeds and a shortened step budget for smoke testing.")
    parser.add_argument("--dpi", type=int, default=240, help="PNG export resolution.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    all_objectives = objective_specs()
    if args.objectives:
        requested = set(args.objectives)
        known = {objective.key for objective in all_objectives}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown objective keys: {', '.join(sorted(unknown))}. Known keys: {', '.join(sorted(known))}")
        objectives = tuple(objective for objective in all_objectives if objective.key in requested)
    else:
        objectives = all_objectives
    optimizers = optimizer_specs()
    _validate_configuration(objectives, optimizers)
    _configure_plot_style()
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    seeds = DEFAULT_SEEDS[:3] if args.quick else DEFAULT_SEEDS
    output_dir = args.output_dir.resolve()
    plot_dirs = {
        "lr_seed_sensitivity": output_dir / "lr_seed_sensitivity",
        "trajectory_contours": output_dir / "trajectory_contours",
        "convergence_curves": output_dir / "convergence_curves",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in plot_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    sweep_rows: list[dict[str, Any]] = []
    effective_steps: dict[str, int] = {}
    paired_starts: dict[str, dict[str, tuple[float, float]]] = {}
    for objective_index, objective in enumerate(objectives, start=1):
        steps = min(objective.steps, 90) if args.quick else objective.steps
        effective_steps[objective.key] = steps
        starts = {str(seed): _seeded_start(objective, seed) for seed in seeds}
        paired_starts[objective.key] = starts
        learning_rates = _tested_learning_rates(objective, args.quick)
        print(f"[{objective_index}/{len(objectives)}] LR sweep: {objective.label} ({steps} steps, {len(learning_rates)} LRs, {len(seeds)} seeds)", flush=True)
        for optimizer in optimizers:
            for learning_rate in learning_rates:
                for seed in seeds:
                    start = starts[str(seed)]
                    result = _run_optimizer(objective, optimizer, learning_rate, start, steps)
                    tail_mean_relative_gap, log_gap_auc = _performance_metrics(result, objective.global_minimum)
                    finite_values = result.values[np.isfinite(result.values)]
                    final_value = float(finite_values[-1]) if finite_values.size else math.inf
                    best_value = float(np.min(finite_values)) if finite_values.size else math.inf
                    sweep_rows.append(
                        {
                            "objective": objective.key,
                            "objective_label": objective.label,
                            "optimizer": optimizer.key,
                            "optimizer_label": optimizer.label,
                            "learning_rate": learning_rate,
                            "seed": seed,
                            "start_x": start[0],
                            "start_y": start[1],
                            "step_budget": steps,
                            "steps_completed": result.steps_completed,
                            "diverged": result.diverged,
                            "tail_mean_relative_gap": tail_mean_relative_gap,
                            "log_gap_auc": log_gap_auc,
                            "final_objective": final_value,
                            "final_gap": max(final_value - objective.global_minimum, 0.0),
                            "best_objective": best_value,
                            "best_gap": max(best_value - objective.global_minimum, 0.0),
                        }
                    )

    summaries, selected = _aggregate_sweep(sweep_rows, objectives, optimizers)
    canonical: dict[tuple[str, str], RunResult] = {}
    canonical_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for objective in objectives:
        print(f"Canonical tuned runs and plots: {objective.label}", flush=True)
        for optimizer in optimizers:
            learning_rate = selected[(objective.key, optimizer.key)]
            result = _run_optimizer(objective, optimizer, learning_rate, objective.start, effective_steps[objective.key])
            if result.diverged:
                raise RuntimeError(f"Selected LR {learning_rate:g} diverged for {optimizer.label} on the canonical {objective.label} run")
            canonical[(objective.key, optimizer.key)] = result
            summary = next(
                row
                for row in summaries
                if row["objective"] == objective.key and row["optimizer"] == optimizer.key and row["learning_rate"] == learning_rate
            )
            selected_rows.append(
                {
                    "objective": objective.key,
                    "objective_label": objective.label,
                    "optimizer": optimizer.key,
                    "optimizer_label": optimizer.label,
                    "selected_learning_rate": learning_rate,
                    "mean_tail_relative_gap": summary["mean_tail_relative_gap"],
                    "sem_tail_relative_gap": summary["sem_tail_relative_gap"],
                    "mean_log_gap_auc": summary["mean_log_gap_auc"],
                    "divergence_rate": summary["divergence_rate"],
                    "canonical_final_objective": float(result.values[-1]),
                    "canonical_final_gap": max(float(result.values[-1] - objective.global_minimum), 0.0),
                }
            )
            for step, (value, position) in enumerate(zip(result.values, result.positions)):
                canonical_rows.append(
                    {
                        "objective": objective.key,
                        "optimizer": optimizer.key,
                        "learning_rate": learning_rate,
                        "step": step,
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "objective_value": float(value),
                        "objective_gap": max(float(value - objective.global_minimum), 0.0),
                    }
                )

        _plot_lr_sensitivity(
            objective,
            optimizers,
            summaries,
            selected,
            plot_dirs["lr_seed_sensitivity"] / objective.key,
            len(seeds),
            args.dpi,
        )
        _plot_trajectories(
            objective,
            optimizers,
            selected,
            canonical,
            plot_dirs["trajectory_contours"] / objective.key,
            args.dpi,
        )
        _plot_convergence(
            objective,
            optimizers,
            selected,
            canonical,
            plot_dirs["convergence_curves"] / objective.key,
            args.dpi,
        )

    _write_csv(output_dir / "lr_sweep.csv", sweep_rows, tuple(sweep_rows[0].keys()))
    _write_csv(output_dir / "lr_summary.csv", summaries, tuple(summaries[0].keys()))
    _write_csv(output_dir / "selected_lrs.csv", selected_rows, tuple(selected_rows[0].keys()))
    _write_csv(output_dir / "canonical_runs.csv", canonical_rows, tuple(canonical_rows[0].keys()))

    chart_rows = _chart_map(objectives)
    _write_csv(output_dir / "chart_map.csv", chart_rows, tuple(chart_rows[0].keys()))

    boundary_warnings: list[str] = []
    for objective in objectives:
        tested_lrs = _tested_learning_rates(objective, args.quick)
        for optimizer in optimizers:
            chosen = selected[(objective.key, optimizer.key)]
            if math.isclose(chosen, tested_lrs[0]) or math.isclose(chosen, tested_lrs[-1]):
                boundary_warnings.append(f"{objective.key}/{optimizer.key}: selected LR {chosen:g} lies on the tested boundary")

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_scope": "Current live raw_run working tree only; figures/ and Git history were excluded from development and validation.",
        "quick_mode": args.quick,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "dtype": "torch.float64",
            "device": "cpu",
        },
        "protocol": {
            "seeds": list(seeds),
            "seed_role": "Paired Gaussian perturbations of the canonical initialization; exact gradients and deterministic optimizer updates.",
            "paired_starts": paired_starts,
            "tuning_metric": "Arithmetic mean of (f(x_t)-f*)/(f(x_0)-f*) over the final 10% of steps, clipped below at 1e-12.",
            "selection": "Lowest across-seed mean tail relative gap among zero-divergence candidates; mean full-run log-gap AUC breaks ties.",
            "uncertainty": "Pointwise ±1 standard error of the mean across paired seeded starts.",
            "divergence_relative_gap": DIVERGENCE_RELATIVE_GAP,
            "momentum": MOMENTUM,
            "weight_decay": 0.0,
            "nesterov": False,
            "parameterization": "One torch.nn.Parameter of shape (2,), preserving tensor-level MAL alignment.",
        },
        "objectives": [
            {
                **{key: value for key, value in asdict(objective).items() if key != "value_fn"},
                "effective_steps": effective_steps[objective.key],
            }
            for objective in objectives
        ],
        "optimizers": [{"key": optimizer.key, "label": optimizer.label, "config": optimizer.config} for optimizer in optimizers],
        "selected_learning_rates": {
            objective.key: {optimizer.key: selected[(objective.key, optimizer.key)] for optimizer in optimizers} for objective in objectives
        },
        "output_folders": {key: str(path.relative_to(output_dir)) for key, path in plot_dirs.items()},
        "chart_map": chart_rows,
        "validation_warnings": boundary_warnings,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_readme(output_dir / "README.md", objectives, optimizers, seeds, selected, args.quick)

    print("Selected LRs:", flush=True)
    for objective in objectives:
        values = ", ".join(f"{optimizer.short_label.replace(chr(10), ' ')}={selected[(objective.key, optimizer.key)]:g}" for optimizer in optimizers)
        print(f"  {objective.key}: {values}", flush=True)
    if boundary_warnings:
        print("Validation warnings:", flush=True)
        for warning in boundary_warnings:
            print(f"  - {warning}", flush=True)
    print(f"Wrote benchmark artifacts to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
