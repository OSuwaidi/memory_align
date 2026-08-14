"""Self-contained generator for the paper MAL-GDM benchmark figures.

Choose an objective by editing ``OBJECTIVE`` below or from the command line:

    .venv/bin/python diagnostics/paper_mal_plots_single_script.py --objective turning_ravine
    .venv/bin/python diagnostics/paper_mal_plots_single_script.py --objective all

The script contains the objectives, seeded-start protocol, four optimizers,
independent learning-rate selection, sensitivity evaluation, and all styling.
It requires only NumPy and Matplotlib; it does not import project modules.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mal-paper-plots"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# -----------------------------------------------------------------------------
# USER-EDITABLE SETTINGS
# -----------------------------------------------------------------------------

OBJECTIVE = "turning_ravine"  # turning_ravine | rosenbrock | anisotropic_quadratic | deep_linear_network | all
STEPS = 60
BASE_BETA = 0.9
# These are the exact disjoint seed ranges used for the validated paper plots.
TURNING_TUNING_SEEDS = tuple(range(190_700, 190_828))
TURNING_EVALUATION_SEEDS = tuple(range(290_700, 290_764))
STANDARD_TUNING_SEEDS = tuple(range(390_700, 390_828))
STANDARD_EVALUATION_SEEDS = tuple(range(490_700, 490_764))
DIVERGENCE_CAP = 1e12
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "paper_mal_single_script"

METHODS = ("GD", "GDM", "MAL-GDM", "C-GDM")
STYLE = {
    "GD": {"color": "#009E73", "linestyle": (0, (6, 2, 1.5, 2)), "marker": "o"},
    "GDM": {"color": "#E69F00", "linestyle": (0, (6, 2.5)), "marker": "s"},
    "MAL-GDM": {"color": "#0072B2", "linestyle": "-", "marker": "D"},
    "C-GDM": {"color": "#CC3311", "linestyle": (0, (1.2, 2.0)), "marker": "^"},
}
BETA_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "mal_effective_beta", ("#E8F5FB", "#9BD4EA", "#3B9BC4", "#0868AC", "#08306B")
)
BETA_NORM = mcolors.Normalize(0.0, 1.0)


@dataclass(frozen=True)
class Objective:
    name: str
    title: str
    start: tuple[float, float]
    minimum: tuple[float, float]
    learning_rates: tuple[float, ...]
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    contour_levels: tuple[float, float, int]
    line_levels: tuple[float, float, int]
    start_std: tuple[float, float]
    start_clip: tuple[float, float]
    secondary_minimum: tuple[float, float] | None = None


OBJECTIVES = {
    "turning_ravine": Objective(
        "turning_ravine", "Turning Ravine", (2.5, 0.2 * math.sin(7.5)), (0.0, 0.0),
        tuple(index / 100 for index in range(1, 13)), (-1.85, 2.70), (-0.62, 0.62),
        (-5.0, 1.45, 32), (-4.5, 1.3, 19), (0.06, 0.015), (0.12, 0.03),
    ),
    "rosenbrock": Objective(
        "rosenbrock", "Rosenbrock Function", (-1.2, 1.0), (1.0, 1.0),
        (0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.0010, 0.0012, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035),
        (-2.0, 2.0), (-1.0, 3.0), (-3.0, 3.5, 34), (-2.5, 3.2, 20), (0.04, 0.04), (0.10, 0.10),
    ),
    "anisotropic_quadratic": Objective(
        "anisotropic_quadratic", "Ill-Conditioned Rotated Quadratic", (2.5, 2.0), (0.0, 0.0),
        (0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025),
        (-3.0, 3.2), (-2.5, 2.6), (-4.0, 3.0, 38), (-3.5, 2.7, 22), (0.06, 0.06), (0.15, 0.15),
    ),
    "deep_linear_network": Objective(
        "deep_linear_network", "Two-Layer Deep Linear Network", (2.5, 0.15),
        (math.sqrt(0.95), math.sqrt(0.95)),
        (0.0025, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100, 0.120),
        (-2.8, 2.8), (-2.8, 2.8), (-2.0, 2.0, 38), (-1.25, 1.7, 22), (0.06, 0.04), (0.15, 0.10),
        (-math.sqrt(0.95), -math.sqrt(0.95)),
    ),
}


@dataclass
class Trace:
    method: str
    learning_rate: float
    trajectory: np.ndarray
    objective: np.ndarray
    effective_beta: np.ndarray


@dataclass(frozen=True)
class Sensitivity:
    method: str
    learning_rate: float
    median: float
    q25: float
    q75: float


def objective_and_gradient(spec: Objective, point: np.ndarray) -> tuple[float, np.ndarray]:
    w1, w2 = point
    if spec.name == "turning_ravine":
        wall = w2 - 0.2 * math.sin(3.0 * w1)
        return 2.0 * w1**2 + 10.0 * wall**2, np.array([4.0 * w1 - 12.0 * wall * math.cos(3.0 * w1), 20.0 * wall])
    if spec.name == "rosenbrock":
        residual = w2 - w1**2
        return (1.0 - w1) ** 2 + 100.0 * residual**2, np.array([2.0 * (w1 - 1.0) - 400.0 * w1 * residual, 200.0 * residual])
    if spec.name == "anisotropic_quadratic":
        cosine, sine = math.cos(math.pi / 6), math.sin(math.pi / 6)
        u, v = cosine * w1 + sine * w2, -sine * w1 + cosine * w2
        return 0.5 * (u**2 + 100.0 * v**2), np.array([cosine * u - 100.0 * sine * v, sine * u + 100.0 * cosine * v])
    residual = w1 * w2 - 1.0
    return 0.5 * residual**2 + 0.025 * (w1**2 + w2**2), np.array([residual * w2 + 0.05 * w1, residual * w1 + 0.05 * w2])


def objective_grid(spec: Objective, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    if spec.name == "turning_ravine":
        return 2.0 * w1**2 + 10.0 * (w2 - 0.2 * np.sin(3.0 * w1)) ** 2
    if spec.name == "rosenbrock":
        return (1.0 - w1) ** 2 + 100.0 * (w2 - w1**2) ** 2
    if spec.name == "anisotropic_quadratic":
        cosine, sine = math.cos(math.pi / 6), math.sin(math.pi / 6)
        u, v = cosine * w1 + sine * w2, -sine * w1 + cosine * w2
        return 0.5 * (u**2 + 100.0 * v**2)
    return 0.5 * (w1 * w2 - 1.0) ** 2 + 0.025 * (w1**2 + w2**2)


def seeded_start(spec: Objective, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    offsets = np.array([np.clip(rng.normal(0, sd), -clip, clip) for sd, clip in zip(spec.start_std, spec.start_clip)])
    if spec.name == "turning_ravine":
        w1 = spec.start[0] + offsets[0]
        return np.array([w1, 0.2 * math.sin(3.0 * w1) + offsets[1]])
    return np.asarray(spec.start) + offsets


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-8)
    if np.linalg.norm(right) == 0.0 or np.linalg.norm(left) == 0.0:
        return 1.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def optimizer_step(method: str, momentum: np.ndarray, gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if method == "GD":
        return gradient, momentum, math.nan
    if method == "MAL-GDM":
        # Fixed-beta, no-carry probe. Smooth MAL is applied at every step:
        # candidate = 0.9*m + g; c = (1 + cos(candidate,g))/2;
        # committed buffer = c*m + g. The previous c never enters the probe.
        candidate = BASE_BETA * momentum + gradient
        effective_beta = 0.5 * (1.0 + cosine(candidate, gradient))
        momentum = effective_beta * momentum + gradient
        return momentum, momentum, effective_beta
    momentum = BASE_BETA * momentum + gradient
    if method == "GDM":
        return momentum, momentum, math.nan
    mask = momentum * gradient > 0.0
    scale = float(mask.size / (mask.sum() + 1.0))
    return momentum * mask * scale, momentum, math.nan


def run(spec: Objective, method: str, learning_rate: float, start: np.ndarray | None = None) -> Trace:
    point = np.array(spec.start if start is None else start, dtype=float)
    momentum = np.zeros(2)
    trajectory, values, betas = [point.copy()], [objective_and_gradient(spec, point)[0]], []
    for _ in range(STEPS):
        _, gradient = objective_and_gradient(spec, point)
        direction, momentum, beta = optimizer_step(method, momentum, gradient)
        point -= learning_rate * direction
        value, _ = objective_and_gradient(spec, point)
        trajectory.append(point.copy())
        values.append(min(value, DIVERGENCE_CAP) if math.isfinite(value) else DIVERGENCE_CAP)
        if method == "MAL-GDM":
            betas.append(beta)
        if not math.isfinite(value) or value >= DIVERGENCE_CAP:
            break
    while len(values) < STEPS + 1:
        trajectory.append(trajectory[-1].copy())
        values.append(DIVERGENCE_CAP)
        if method == "MAL-GDM":
            betas.append(betas[-1])
    return Trace(method, learning_rate, np.asarray(trajectory), np.asarray(values), np.asarray(betas))


def tune_and_evaluate(spec: Objective) -> tuple[dict[str, float], list[Sensitivity]]:
    if spec.name == "turning_ravine":
        tuning_seeds = TURNING_TUNING_SEEDS
        evaluation_seeds = TURNING_EVALUATION_SEEDS
    else:
        tuning_seeds = STANDARD_TUNING_SEEDS
        evaluation_seeds = STANDARD_EVALUATION_SEEDS
    tuning_starts = [seeded_start(spec, seed) for seed in tuning_seeds]
    evaluation_starts = [seeded_start(spec, seed) for seed in evaluation_seeds]
    rates, rows = {}, []
    for method in METHODS:
        candidates = []
        for rate in spec.learning_rates:
            finals = np.array([run(spec, method, rate, start).objective[-1] for start in tuning_starts])
            candidates.append((float(np.median(finals)), float(np.quantile(finals, 0.75)), rate))
        rates[method] = min(candidates)[2]
        for rate in spec.learning_rates:
            finals = np.array([run(spec, method, rate, start).objective[-1] for start in evaluation_starts])
            rows.append(Sensitivity(method, rate, float(np.median(finals)), float(np.quantile(finals, 0.25)), float(np.quantile(finals, 0.75))))
    return rates, rows


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.titlesize": 12.2, "axes.titleweight": "bold",
        "axes.labelsize": 10.8, "xtick.labelsize": 9.4, "ytick.labelsize": 9.4,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def style_axis(axis: plt.Axes, minor_grid: bool = False) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#98A2B3")
    axis.grid(True, which="major", color="#D9E1EA", linewidth=0.65)
    if minor_grid:
        axis.grid(True, which="minor", color="#EEF2F6", linewidth=0.38)


def add_colored_line(axis: plt.Axes, x: np.ndarray, y: np.ndarray, beta: np.ndarray, width: float) -> None:
    points = np.column_stack((x, y))
    collection = LineCollection(np.stack((points[:-1], points[1:]), axis=1), cmap=BETA_CMAP, norm=BETA_NORM, linewidths=width, zorder=12)
    collection.set_array(np.clip(beta, 0, 1))
    collection.set_path_effects([path_effects.Stroke(linewidth=width + 1.05, foreground="white", alpha=0.94), path_effects.Normal()])
    axis.add_collection(collection)


def legend(figure: plt.Figure, rates: dict[str, float]) -> None:
    handles = []
    for method in METHODS:
        style, rate = STYLE[method], rates[method]
        precision = 4 if rate < 0.01 else (2 if math.isclose(rate, round(rate, 2)) else 3)
        handles.append(Line2D([0], [0], color=style["color"], linestyle=style["linestyle"], marker=style["marker"], markerfacecolor="white", linewidth=2, label=rf"{method} ($\eta={rate:.{precision}f}$)"))
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False, fontsize=8.6, columnspacing=1.25, handlelength=2.7)


def plot_main(spec: Objective, traces: list[Trace], rates: dict[str, float], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    axis = axes[0]
    w1, w2 = np.meshgrid(np.linspace(*spec.xlim, 720), np.linspace(*spec.ylim, 500))
    levels = np.logspace(*spec.contour_levels)
    base = plt.get_cmap("Greys")
    cmap = mcolors.LinearSegmentedColormap.from_list("light_greys", base(np.linspace(0, 0.40, 256)))
    filled = axis.contourf(w1, w2, np.maximum(objective_grid(spec, w1, w2), levels[0]), levels=levels, cmap=cmap, norm=mcolors.LogNorm(levels[0], levels[-1]), extend="both")
    axis.contour(w1, w2, np.maximum(objective_grid(spec, w1, w2), levels[0]), levels=np.logspace(*spec.line_levels), colors="#667085", linewidths=0.40, alpha=0.50)
    if spec.name == "turning_ravine":
        floor_x = np.linspace(*spec.xlim, 720)
        axis.plot(floor_x, 0.2 * np.sin(3 * floor_x), color="#778391", linestyle=(0, (3, 2.5)), linewidth=1)
    for trace in traces:
        style = STYLE[trace.method]
        line = axis.plot(trace.trajectory[:, 0], trace.trajectory[:, 1], color=style["color"], linestyle=style["linestyle"], linewidth=1.65 if trace.method == "MAL-GDM" else 1.35, zorder=5)[0]
        line.set_path_effects([path_effects.Stroke(linewidth=2.25, foreground="white", alpha=0.85), path_effects.Normal()])
    mal = next(trace for trace in traces if trace.method == "MAL-GDM")
    add_colored_line(axis, mal.trajectory[:, 0], mal.trajectory[:, 1], mal.effective_beta, 2.2)
    axis.scatter(*spec.start, marker="*", s=105, facecolor="white", edgecolor="#111827", zorder=20)
    for trace in traces:
        style = STYLE[trace.method]
        face = BETA_CMAP(BETA_NORM(trace.effective_beta[-1])) if trace.method == "MAL-GDM" else "white"
        axis.plot(*trace.trajectory[-1], linestyle="none", marker=style["marker"], markersize=8.8, markerfacecolor=face, markeredgecolor=style["color"], markeredgewidth=1.75, zorder=30)
    if spec.secondary_minimum:
        axis.scatter(*spec.secondary_minimum, marker="X", s=82, facecolor="#111827", edgecolor="white", linewidth=0.95, zorder=40)
    # Draw the global minimum last: it is the foremost layer in the contour panel.
    axis.scatter(*spec.minimum, marker="X", s=82, facecolor="#111827", edgecolor="white", linewidth=0.95, zorder=40)
    axis.set(title="Optimization Trajectories", xlabel=r"$w_1$", ylabel=r"$w_2$", xlim=spec.xlim, ylim=spec.ylim)
    axis.spines[["top", "right"]].set_visible(False)
    cax = axis.inset_axes([0.66, 0.075, 0.28, 0.034])
    exponents = sorted({math.ceil(spec.contour_levels[0]), 0, math.floor(spec.contour_levels[1])})
    figure.colorbar(filled, cax=cax, orientation="horizontal", ticks=[10.0**value for value in exponents]).ax.tick_params(labelsize=7.5, length=2, pad=1)

    axis = axes[1]
    steps, markers = np.arange(STEPS + 1), np.arange(0, STEPS + 1, 10)
    for trace in traces:
        style, displayed = STYLE[trace.method], np.clip(trace.objective, 1e-18, DIVERGENCE_CAP)
        axis.plot(steps, displayed, color=style["color"], linestyle=style["linestyle"], linewidth=2)
        axis.plot(markers, displayed[markers], linestyle="none", marker=style["marker"], markersize=3.8, markerfacecolor="white", markeredgecolor=style["color"])
    displayed = np.clip(mal.objective, 1e-18, DIVERGENCE_CAP)
    add_colored_line(axis, steps, displayed, mal.effective_beta, 2.9)
    axis.set_yscale("log")
    positive = np.concatenate([trace.objective for trace in traces]); positive = positive[(positive > 0) & np.isfinite(positive)]
    axis.set(title="Objective vs. Step", xlabel=r"Step $t$", ylabel=r"Objective $F(\mathbf{w}_t)$", xlim=(0, 72), ylim=(max(1e-18, positive.min() / 3), min(DIVERGENCE_CAP, positive.max() * 2)))
    axis.set_xticks(markers); style_axis(axis, True)
    legend(figure, rates)
    beta_axis = figure.add_axes([0.405, 0.895, 0.19, 0.017])
    colorbar = figure.colorbar(ScalarMappable(norm=BETA_NORM, cmap=BETA_CMAP), cax=beta_axis, orientation="horizontal", ticks=(0, 0.25, 0.5, 0.75, 1))
    colorbar.ax.set_title(r"MAL-GDM effective $\beta_t$", fontsize=8.5, pad=3); colorbar.ax.tick_params(labelsize=7.5, length=2.3, pad=1.2)
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.14, top=0.80, wspace=0.28)
    save(figure, output, "optimization_trajectories_and_objective_beta_colored")


def plot_sensitivity(spec: Objective, rows: list[Sensitivity], rates: dict[str, float], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(12.8, 6.8))
    for method in METHODS:
        data = [row for row in rows if row.method == method]
        x = np.array([row.learning_rate for row in data]); median = np.clip([row.median for row in data], 1e-18, DIVERGENCE_CAP)
        q25 = np.clip([row.q25 for row in data], 1e-18, DIVERGENCE_CAP); q75 = np.clip([row.q75 for row in data], 1e-18, DIVERGENCE_CAP)
        style = STYLE[method]
        axis.fill_between(x, q25, q75, color=style["color"], alpha=0.10, linewidth=0)
        axis.plot(x, median, color=style["color"], linestyle=style["linestyle"], marker=style["marker"], markersize=3.7, markerfacecolor="white", linewidth=1.8)
        selected = next(row for row in data if math.isclose(row.learning_rate, rates[method]))
        axis.scatter(selected.learning_rate, max(selected.median, 1e-18), marker="*", s=105, facecolor=style["color"], edgecolor="white", zorder=10)
    margin = 0.025 * (spec.learning_rates[-1] - spec.learning_rates[0])
    axis.set_yscale("log"); axis.set(title=f"{spec.title}: Step Size and Seed Sensitivity", xlabel=r"Learning Rate $\eta$", ylabel=r"Median Objective $F$", xlim=(spec.learning_rates[0] - margin, spec.learning_rates[-1] + margin))
    axis.xaxis.set_major_locator(MaxNLocator(11)); style_axis(axis, True); legend(figure, rates)
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.13, top=0.78)
    save(figure, output, "step_size_seed_sensitivity")


def save(figure: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"{stem}.{suffix}", dpi=400, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def generate(spec: Objective) -> None:
    output = OUTPUT_ROOT / spec.name
    rates, sensitivity = tune_and_evaluate(spec)
    traces = [run(spec, method, rates[method]) for method in METHODS]
    plot_main(spec, traces, rates, output)
    plot_sensitivity(spec, sensitivity, rates, output)
    with (output / "seed_sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "learning_rate", "median", "q25", "q75")); writer.writeheader()
        writer.writerows(vars(row) for row in sensitivity)
    print(spec.name, rates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=(*OBJECTIVES, "all"), default=OBJECTIVE)
    args = parser.parse_args()
    configure_matplotlib()
    selected = OBJECTIVES.values() if args.objective == "all" else (OBJECTIVES[args.objective],)
    for spec in selected:
        generate(spec)


if __name__ == "__main__":
    main()
