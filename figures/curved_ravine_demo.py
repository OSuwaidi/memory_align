"""A controlled 2-D failure case for momentum and a MAL comparison.

The smooth, non-convex objective is

    F(x, y) = axial/2 * x**2 + wall/2 * q(x, y)**2,
    q(x, y) = y - amplitude * sin(frequency * x).

For every fixed x, the curve q(x, y)=0 gives the unique y that minimizes F;
it is therefore the valley centerline, not a set of global minimizers.
Along this centerline F(x, y)=axial/2*x**2, which decreases toward the
origin. Both terms vanish together only at (0, 0), so the origin is the
unique global minimizer.

This single-valued ravine is preferable to a literal polar spiral for the
controlled demonstration: it is smooth everywhere, has one centerline rather
than two spiral arms, and has no angular-coordinate singularity at the
minimum.  It isolates directional lag and wall crossings without relying on
an unstable learning rate.

Run from the repository root:

    .venv/bin/python curved_ravine_demo.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mal-matplotlib")
)

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mal_opt import MAL_SGDM


@dataclass(frozen=True)
class ExperimentConfig:
    """A deterministic regime selected to expose momentum's directional lag."""

    axial: float = 5.0
    wall: float = 20.0
    amplitude: float = 0.4
    frequency: float = 2.5
    start_x: float = 4.0
    learning_rate: float = 0.02
    momentum: float = 0.9
    steps: int = 25


@dataclass
class OptimizationRun:
    name: str
    trajectory: np.ndarray
    objective: np.ndarray
    update_gradient_cosine: np.ndarray
    wall_coordinate: np.ndarray
    effective_momentum_coefficient: np.ndarray

    def summary(self, nominal_beta: float) -> dict[str, float | int]:
        steps = len(self.objective) - 1
        differences = np.diff(self.objective)
        summary: dict[str, float | int] = {
            "final_objective": float(self.objective[-1]),
            "best_objective": float(self.objective.min()),
            "best_step": int(self.objective.argmin()),
            "final_distance_to_minimum": float(np.linalg.norm(self.trajectory[-1])),
            "path_length": float(
                np.linalg.norm(np.diff(self.trajectory, axis=0), axis=1).sum()
            ),
            "uphill_steps": int(np.count_nonzero(differences > 1e-12)),
            "uphill_fraction": float(np.count_nonzero(differences > 1e-12) / steps),
            "first_order_ascent_steps": int(
                np.count_nonzero(self.update_gradient_cosine < 0.0)
            ),
            "minimum_update_gradient_cosine": float(self.update_gradient_cosine.min()),
            "rms_wall_displacement": float(
                np.sqrt(np.mean(np.square(self.wall_coordinate)))
            ),
            "maximum_wall_displacement": float(np.abs(self.wall_coordinate).max()),
        }
        finite_coefficients = self.effective_momentum_coefficient[
            np.isfinite(self.effective_momentum_coefficient)
        ]
        if finite_coefficients.size:
            summary.update(
                {
                    "minimum_effective_momentum": float(finite_coefficients.min()),
                    "median_effective_momentum": float(np.median(finite_coefficients)),
                    "strong_memory_decay_steps": int(
                        np.count_nonzero(finite_coefficients < 0.5 * nominal_beta)
                    ),
                }
            )
        return summary


def ravine_quantities_torch(
    weights: torch.Tensor, config: ExperimentConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return objective and signed displacement from the valley centerline."""

    x, y = weights.unbind(dim=-1)
    wall_coordinate = y - config.amplitude * torch.sin(config.frequency * x)
    objective = (
        0.5 * config.axial * x.square() + 0.5 * config.wall * wall_coordinate.square()
    )
    return objective, wall_coordinate


def ravine_quantities_numpy(
    weights: np.ndarray, config: ExperimentConfig
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy version used for dense plotting grids."""

    x = weights[..., 0]
    y = weights[..., 1]
    wall_coordinate = y - config.amplitude * np.sin(config.frequency * x)
    objective = (
        0.5 * config.axial * x * x
        + 0.5 * config.wall * wall_coordinate * wall_coordinate
    )
    return objective, wall_coordinate


def initial_point(config: ExperimentConfig) -> torch.Tensor:
    """Put the common start exactly on q(x, y)=0."""

    return torch.tensor(
        [
            config.start_x,
            config.amplitude * math.sin(config.frequency * config.start_x),
        ],
        dtype=torch.float64,
    )


def _mal_effective_coefficient(
    optimizer: torch.optim.Optimizer,
    gradient: torch.Tensor,
    config: ExperimentConfig,
) -> float:
    """Read the pre-update MAL state and reproduce its static gate."""

    momentum_buffer = optimizer.param_groups[0]["momentum"][0]
    candidate = gradient + config.momentum * momentum_buffer
    gradient_norm = torch.linalg.vector_norm(gradient)
    candidate_norm = torch.linalg.vector_norm(candidate)
    if gradient_norm == 0.0 or candidate_norm == 0.0:
        return config.momentum
    cosine = torch.dot(candidate, gradient) / (candidate_norm * gradient_norm)
    retention = 0.5 * (1.0 + cosine.clamp(-1.0, 1.0))
    return float(config.momentum * retention)


def run_optimizer(
    name: str,
    optimizer_factory: Callable[[list[torch.nn.Parameter]], torch.optim.Optimizer],
    start: torch.Tensor,
    config: ExperimentConfig,
) -> OptimizationRun:
    """Optimize from ``start`` and retain diagnostics for every update."""

    weights = torch.nn.Parameter(start.clone())
    optimizer = optimizer_factory([weights])

    trajectory = [weights.detach().cpu().numpy().copy()]
    initial_objective, initial_wall = ravine_quantities_torch(weights, config)
    objectives = [float(initial_objective.detach())]
    wall_coordinates = [float(initial_wall.detach())]
    update_gradient_cosines: list[float] = []
    effective_coefficients: list[float] = []

    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        objective, _ = ravine_quantities_torch(weights, config)
        objective.backward()

        gradient = weights.grad.detach().clone()
        old_weights = weights.detach().clone()
        if name == "MAL momentum":
            effective_coefficients.append(
                _mal_effective_coefficient(optimizer, gradient, config)
            )
        else:
            effective_coefficients.append(math.nan)

        optimizer.step()

        # All three methods use the same scalar learning rate.  This recovers
        # the actual direction applied by each optimizer, including memory.
        applied_direction = (old_weights - weights.detach()) / config.learning_rate
        denominator = torch.linalg.vector_norm(
            applied_direction
        ) * torch.linalg.vector_norm(gradient)
        if denominator > 0.0:
            cosine = torch.dot(applied_direction, gradient) / denominator
            update_gradient_cosines.append(float(cosine.clamp(-1.0, 1.0)))
        else:
            update_gradient_cosines.append(0.0)

        next_objective, next_wall = ravine_quantities_torch(weights, config)
        trajectory.append(weights.detach().cpu().numpy().copy())
        objectives.append(float(next_objective.detach()))
        wall_coordinates.append(float(next_wall.detach()))

    return OptimizationRun(
        name=name,
        trajectory=np.asarray(trajectory),
        objective=np.asarray(objectives),
        update_gradient_cosine=np.asarray(update_gradient_cosines),
        wall_coordinate=np.asarray(wall_coordinates),
        effective_momentum_coefficient=np.asarray(effective_coefficients),
    )


def run_experiment(config: ExperimentConfig) -> list[OptimizationRun]:
    """Run all optimizers under identical initialization and hyperparameters."""

    start = initial_point(config)
    return [
        run_optimizer(
            "GD",
            lambda parameters: torch.optim.SGD(parameters, lr=config.learning_rate),
            start,
            config,
        ),
        run_optimizer(
            "GDM",
            lambda parameters: torch.optim.SGD(
                parameters,
                lr=config.learning_rate,
                momentum=config.momentum,
            ),
            start,
            config,
        ),
        run_optimizer(
            "MAL momentum",
            lambda parameters: MAL_SGDM(
                parameters,
                lr=config.learning_rate,
                beta=config.momentum,
                nesterov=False,
            ),
            start,
            config,
        ),
    ]


def nonconvexity_certificate(config: ExperimentConfig) -> dict[str, object]:
    """Return a sampled negative-curvature witness using the analytic Hessian."""

    x_axis = np.linspace(-config.start_x, config.start_x, 321)
    y_axis = np.linspace(-1.2, 1.2, 241)
    x, y = np.meshgrid(x_axis, y_axis)

    sine = np.sin(config.frequency * x)
    cosine = np.cos(config.frequency * x)
    wall_coordinate = y - config.amplitude * sine
    q_x = -config.amplitude * config.frequency * cosine
    q_xx = config.amplitude * config.frequency**2 * sine

    h_xx = config.axial + config.wall * (q_x * q_x + wall_coordinate * q_xx)
    h_xy = config.wall * q_x
    h_yy = np.full_like(h_xx, config.wall)
    discriminant = np.sqrt((h_xx - h_yy) ** 2 + 4.0 * h_xy**2)
    eigenvalue_min = 0.5 * (h_xx + h_yy - discriminant)
    eigenvalue_max = 0.5 * (h_xx + h_yy + discriminant)

    flat_index = int(np.argmin(eigenvalue_min))
    index = np.unravel_index(flat_index, eigenvalue_min.shape)
    return {
        "minimum_sampled_hessian_eigenvalue": float(eigenvalue_min[index]),
        "negative_curvature_sample_fraction": float(np.mean(eigenvalue_min < -1e-9)),
        "witness_point": [float(x[index]), float(y[index])],
        "witness_hessian_eigenvalues": [
            float(eigenvalue_min[index]),
            float(eigenvalue_max[index]),
        ],
        "sample_count": int(eigenvalue_min.size),
    }


SERIES_STYLE = {
    "GD": {
        "color": "#2CA02C",
        "linestyle": (0, (5, 3)),
        "marker": "o",
        "markerfacecolor": "white",
    },
    "GDM": {
        "color": "#FF7F0E",
        "linestyle": (0, (7, 2, 1.5, 2)),
        "marker": "^",
        "markerfacecolor": "white",
    },
    "MAL momentum": {
        "color": "#1F77B4",
        "linestyle": "-",
        "marker": "s",
        "markerfacecolor": "white",
    },
}


def _apply_plot_defaults() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "axes.edgecolor": "#A8B0BC",
            "axes.linewidth": 0.8,
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "text.color": "#20262E",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _trajectory_bounds(
    runs: list[OptimizationRun], config: ExperimentConfig
) -> tuple[float, float, float, float]:
    all_points = np.concatenate([run.trajectory for run in runs])
    x_min = min(float(all_points[:, 0].min()), -0.5) - 0.3
    x_max = max(float(all_points[:, 0].max()), config.start_x) + 0.3
    y_min = min(float(all_points[:, 1].min()), -config.amplitude) - 0.22
    y_max = max(float(all_points[:, 1].max()), config.amplitude) + 0.22
    return x_min, x_max, y_min, y_max


def plot_trajectories(
    runs: list[OptimizationRun],
    config: ExperimentConfig,
    output_dir: Path,
) -> None:
    """Create a clean contour map with directly labeled trajectories."""

    _apply_plot_defaults()
    x_min, x_max, y_min, y_max = _trajectory_bounds(runs, config)
    x_axis = np.linspace(x_min, x_max, 760)
    y_axis = np.linspace(y_min, y_max, 360)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    grid = np.stack([grid_x, grid_y], axis=-1)
    grid_objective, _ = ravine_quantities_numpy(grid, config)

    # Sparse, light contours preserve the landscape without competing with
    # the paths. The levels cover the full objective range shown here.
    fill_levels = [0.0, 0.1, 0.5, 2.0, 8.0, 32.0, 100.0]
    fill_colors = [
        "#FFFFFF",
        "#F8FAFC",
        "#F1F5F9",
        "#E8EDF3",
        "#DDE4EC",
        "#D1DAE4",
    ]
    line_levels = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0]

    fig, ax = plt.subplots(figsize=(12.2, 5.0))
    ax.contourf(
        grid_x,
        grid_y,
        grid_objective,
        levels=fill_levels,
        colors=fill_colors,
        extend="max",
    )
    ax.contour(
        grid_x,
        grid_y,
        grid_objective,
        levels=line_levels,
        colors="#B8C2CE",
        linewidths=0.65,
        alpha=0.8,
    )

    centerline_y = config.amplitude * np.sin(config.frequency * x_axis)
    ax.plot(
        x_axis,
        centerline_y,
        color="#8995A3",
        linestyle=(0, (3, 3)),
        linewidth=1.3,
        alpha=0.95,
        zorder=3,
    )
    centerline_label_x = 2.55
    centerline_label_y = config.amplitude * math.sin(
        config.frequency * centerline_label_x
    )
    ax.annotate(
        "ravine floor",
        xy=(centerline_label_x, centerline_label_y),
        xytext=(8, 10),
        textcoords="offset points",
        color="#697586",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 1.5,
        },
        zorder=8,
    )

    for run in runs:
        style = SERIES_STYLE[run.name]
        ax.plot(
            run.trajectory[:, 0],
            run.trajectory[:, 1],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.1 if run.name != "MAL momentum" else 2.6,
            zorder=5 if run.name == "MAL momentum" else 4,
        )

        ax.scatter(
            run.trajectory[-1, 0],
            run.trajectory[-1, 1],
            s=76,
            marker="o",
            facecolor="white",
            edgecolor=style["color"],
            linewidth=2.0,
            zorder=9,
        )

    run_by_name = {run.name: run for run in runs}
    endpoint_labels = {
        "GD": (8, 8, "GD end"),
        "GDM": (-8, -20, "GDM end"),
        "MAL momentum": (24, -21, "MAL end"),
    }
    for name, (x_offset, y_offset, label) in endpoint_labels.items():
        run = run_by_name[name]
        style = SERIES_STYLE[name]
        ax.annotate(
            label,
            xy=run.trajectory[-1],
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="right" if x_offset < 0 else "left",
            color=style["color"],
            fontsize=9.5,
            fontweight="bold" if name == "MAL momentum" else "normal",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
                "pad": 1.5,
            },
            zorder=12,
        )

    start = runs[0].trajectory[0]
    start_artist = ax.scatter(
        start[0],
        start[1],
        marker="D",
        s=105,
        facecolor="white",
        edgecolor="#111827",
        linewidth=2.0,
        zorder=13,
    )
    minimum_artist = ax.scatter(
        0.0,
        0.0,
        marker="X",
        s=145,
        facecolor="#111827",
        edgecolor="white",
        linewidth=1.2,
        zorder=13,
    )
    ax.legend(
        handles=[start_artist, minimum_artist],
        labels=["Start", "Global minimum"],
        loc="upper right",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#D4DAE2",
        framealpha=0.95,
        fontsize=9.5,
        handletextpad=0.5,
        columnspacing=1.2,
    )

    ax.set(
        xlabel=r"$w_1$",
        ylabel=r"$w_2$",
        xlim=(x_min, x_max),
        ylim=(y_min, y_max),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#FFFFFF", linewidth=0.75, alpha=0.9)

    fig.suptitle(
        "Optimization trajectories in a curved valley",
        x=0.075,
        y=0.98,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    ax.set_title(
        (
            r"$F(w)=2.5w_1^2+10[w_2-0.4\sin(2.5w_1)]^2$"
            "\n"
            rf"Same start; $\eta={config.learning_rate:g}$, "
            rf"$\beta={config.momentum:g}$, {config.steps} steps."
            "\nThe dashed ravine floor minimizes $F$ over $w_2$ at fixed $w_1$; "
            r"along it, $F=2.5w_1^2$, so only the origin is globally minimizing."
        ),
        loc="left",
        color="#5B6472",
        pad=12,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    for extension in ("png", "svg"):
        fig.savefig(
            output_dir / f"curved_ravine_trajectories.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_objective_history(
    runs: list[OptimizationRun],
    config: ExperimentConfig,
    output_dir: Path,
) -> None:
    """Create the requested objective-versus-step figure."""

    _apply_plot_defaults()
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    steps = np.arange(config.steps + 1)
    marker_indices = np.unique(
        np.linspace(0, config.steps, min(config.steps + 1, 9), dtype=int)
    )

    for run in runs:
        style = SERIES_STYLE[run.name]
        ax.plot(
            steps,
            run.objective,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0 if run.name != "MAL momentum" else 2.5,
            label=run.name,
            zorder=4 if run.name == "MAL momentum" else 3,
        )
        ax.plot(
            marker_indices,
            run.objective[marker_indices],
            linestyle="none",
            marker=style["marker"],
            markersize=5.0,
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=1.0,
            zorder=5,
        )
        ax.text(
            config.steps + 0.7,
            run.objective[-1],
            f"{run.name}: {run.objective[-1]:.2e}",
            color=style["color"],
            fontsize=9.5,
            va="center",
            fontweight="bold" if run.name == "MAL momentum" else "normal",
        )

    ax.set_yscale("log")
    ax.set_xlim(0, config.steps + 8.5)
    positive_values = np.concatenate([run.objective[run.objective > 0] for run in runs])
    ax.set_ylim(positive_values.min() * 0.45, positive_values.max() * 1.45)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel(r"Objective $F(w_t)$ (log scale)")
    ax.axvline(
        config.steps,
        color="#CBD2DC",
        linewidth=0.8,
        linestyle=(0, (2, 3)),
        zorder=1,
    )
    ax.grid(which="major", color="#D8DEE7", linewidth=0.7, alpha=0.75)
    ax.grid(which="minor", color="#EEF1F5", linewidth=0.5, alpha=0.75)
    fig.suptitle(
        "Objective value across optimization steps",
        x=0.09,
        y=0.98,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    ax.set_title(
        (
            rf"Identical initialization and $\eta={config.learning_rate:g}$; "
            rf"momentum methods use $\beta={config.momentum:g}$"
        ),
        loc="left",
        color="#5B6472",
        pad=12,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    for extension in ("png", "svg"):
        fig.savefig(
            output_dir / f"curved_ravine_objective.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Directory for PNG, SVG, and diagnostics JSON outputs.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    config = ExperimentConfig()
    runs = run_experiment(config)
    diagnostics = {
        "config": asdict(config),
        "nonconvexity": nonconvexity_certificate(config),
        "optimizers": {run.name: run.summary(config.momentum) for run in runs},
    }

    plot_trajectories(runs, config, args.output_dir)
    plot_objective_history(runs, config, args.output_dir)

    diagnostics_path = args.output_dir / "curved_ravine_metrics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2))
    print(f"\nWrote figures and diagnostics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
