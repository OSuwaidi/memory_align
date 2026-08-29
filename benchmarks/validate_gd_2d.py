"""Validate numerical and artifact integrity for the 2D GD benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from benchmarks.gd_2d import _run_optimizer, objective_specs, optimizer_specs


PLOT_FOLDERS = ("lr_seed_sensitivity", "trajectory_contours", "convergence_curves")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _close(left: float, right: float, *, atol: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=atol)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_outputs"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    sweep_rows = _read_csv(output_dir / "lr_sweep.csv")
    summary_rows = _read_csv(output_dir / "lr_summary.csv")
    selected_rows = _read_csv(output_dir / "selected_lrs.csv")
    canonical_rows = _read_csv(output_dir / "canonical_runs.csv")

    objectives = {objective["key"]: objective for objective in manifest["objectives"]}
    optimizers = {optimizer["key"]: optimizer for optimizer in manifest["optimizers"]}
    seeds = tuple(int(seed) for seed in manifest["protocol"]["seeds"])
    checks: list[dict[str, Any]] = []

    expected_optimizer_keys = {optimizer.key for optimizer in optimizer_specs()}
    _require(set(optimizers) == expected_optimizer_keys, "Manifest optimizer set does not match the live benchmark implementation")
    checks.append({"name": "optimizer_inventory", "status": "passed", "count": len(optimizers)})

    mal_expected = {"in_place": False, "pwr": 1.0, "gate_mode": "attenuate"}
    for key, scale in (("mal_gdm_unscaled", False), ("mal_gdm_scaled", True)):
        config = optimizers[key]["config"]
        _require(all(config[name] == value for name, value in mal_expected.items()), f"{key} does not match the requested MAL configuration")
        _require(config["scale"] is scale, f"{key} has the wrong scale setting")
    checks.append({"name": "requested_mal_variants", "status": "passed"})

    sweep_keys = [
        (row["objective"], row["optimizer"], float(row["learning_rate"]), int(row["seed"]))
        for row in sweep_rows
    ]
    _require(len(sweep_keys) == len(set(sweep_keys)), "LR sweep contains duplicate objective/optimizer/LR/seed rows")
    grouped_seeds: dict[tuple[str, str, float], set[int]] = defaultdict(set)
    for objective_key, optimizer_key, learning_rate, seed in sweep_keys:
        _require(objective_key in objectives, f"Unknown objective in sweep: {objective_key}")
        _require(optimizer_key in optimizers, f"Unknown optimizer in sweep: {optimizer_key}")
        grouped_seeds[(objective_key, optimizer_key, learning_rate)].add(seed)
    _require(all(group == set(seeds) for group in grouped_seeds.values()), "Not every LR candidate uses the complete paired seed set")
    checks.append(
        {
            "name": "paired_lr_sweep",
            "status": "passed",
            "rows": len(sweep_rows),
            "seeds_per_candidate": len(seeds),
            "candidate_groups": len(grouped_seeds),
        }
    )

    summaries: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for row in summary_rows:
        summaries[(row["objective"], row["optimizer"])].append(
            {
                "learning_rate": float(row["learning_rate"]),
                "mean_tail_relative_gap": float(row["mean_tail_relative_gap"]),
                "sem_tail_relative_gap": float(row["sem_tail_relative_gap"]),
                "mean_log_gap_auc": float(row["mean_log_gap_auc"]),
                "divergence_rate": float(row["divergence_rate"]),
            }
        )

    selected_lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in selected_rows:
        key = (row["objective"], row["optimizer"])
        _require(key not in selected_lookup, f"Duplicate selected-LR row: {key}")
        selected_lookup[key] = row

    expected_selected_keys = {(objective_key, optimizer_key) for objective_key in objectives for optimizer_key in optimizers}
    _require(set(selected_lookup) == expected_selected_keys, "Selected-LR table does not cover every objective/optimizer pair")

    boundary_selections: list[str] = []
    for key, candidates in summaries.items():
        stable = [candidate for candidate in candidates if candidate["divergence_rate"] == 0.0]
        pool = stable if stable else candidates
        winner = min(
            pool,
            key=lambda candidate: (
                candidate["mean_tail_relative_gap"],
                candidate["mean_log_gap_auc"],
                candidate["sem_tail_relative_gap"],
                candidate["learning_rate"],
            ),
        )
        selected_lr = float(selected_lookup[key]["selected_learning_rate"])
        _require(_close(selected_lr, winner["learning_rate"]), f"Selected LR for {key} is not the declared metric winner")
        tested = sorted(candidate["learning_rate"] for candidate in candidates)
        if _close(selected_lr, tested[0]) or _close(selected_lr, tested[-1]):
            boundary_selections.append(f"{key[0]}/{key[1]}")
    if not manifest["quick_mode"]:
        _require(not boundary_selections, f"Full run contains boundary LR selections: {boundary_selections}")
    checks.append({"name": "independent_lr_selection", "status": "passed", "boundary_selections": boundary_selections})

    canonical_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in canonical_rows:
        canonical_groups[(row["objective"], row["optimizer"])].append(row)
    _require(set(canonical_groups) == expected_selected_keys, "Canonical-run table does not cover every objective/optimizer pair")

    for (objective_key, optimizer_key), rows in canonical_groups.items():
        ordered = sorted(rows, key=lambda row: int(row["step"]))
        expected_steps = int(objectives[objective_key]["effective_steps"])
        _require([int(row["step"]) for row in ordered] == list(range(expected_steps + 1)), f"Canonical steps are incomplete for {(objective_key, optimizer_key)}")
        numbers = np.asarray(
            [[float(row["x"]), float(row["y"]), float(row["objective_value"]), float(row["objective_gap"])] for row in ordered],
            dtype=np.float64,
        )
        _require(bool(np.isfinite(numbers).all()), f"Canonical run has non-finite values for {(objective_key, optimizer_key)}")
        _require(bool((numbers[:, 3] >= -1e-14).all()), f"Canonical run has a negative objective gap for {(objective_key, optimizer_key)}")
        start = np.asarray(objectives[objective_key]["start"], dtype=np.float64)
        _require(bool(np.allclose(numbers[0, :2], start, atol=1e-12, rtol=0.0)), f"Canonical start mismatch for {(objective_key, optimizer_key)}")
        selected_lr = float(selected_lookup[(objective_key, optimizer_key)]["selected_learning_rate"])
        _require(_close(float(ordered[0]["learning_rate"]), selected_lr), f"Canonical LR mismatch for {(objective_key, optimizer_key)}")
    checks.append({"name": "canonical_numerical_integrity", "status": "passed", "rows": len(canonical_rows)})

    verified_plot_files: list[str] = []
    for folder in PLOT_FOLDERS:
        for objective_key in objectives:
            png_path = output_dir / folder / f"{objective_key}.png"
            pdf_path = output_dir / folder / f"{objective_key}.pdf"
            _require(png_path.is_file() and png_path.stat().st_size > 20_000, f"Missing or undersized PNG: {png_path}")
            _require(pdf_path.is_file() and pdf_path.stat().st_size > 5_000, f"Missing or undersized PDF: {pdf_path}")
            with Image.open(png_path) as image:
                image.verify()
            with Image.open(png_path) as image:
                width, height = image.size
            _require(width >= 1600 and height >= 900, f"PNG resolution is too small: {png_path} ({width}x{height})")
            _require(pdf_path.read_bytes()[:5] == b"%PDF-", f"Invalid PDF header: {pdf_path}")
            verified_plot_files.extend((str(png_path.relative_to(output_dir)), str(pdf_path.relative_to(output_dir))))
    checks.append({"name": "plot_file_integrity", "status": "passed", "files": len(verified_plot_files)})

    live_objectives = {objective.key: objective for objective in objective_specs()}
    memory_objective = live_objectives.get("memory_trap")
    memory_evidence: dict[str, Any] | None = None
    if memory_objective is not None and "memory_trap" in objectives:
        hessian_point = torch.tensor([1.0, 0.0], dtype=torch.float64)
        hessian = torch.autograd.functional.hessian(memory_objective.value_fn, hessian_point)
        _require(float(hessian[0, 0]) < 0.0, "Custom memory-trap objective did not exhibit negative x-curvature at x=1")

        live_optimizers = {optimizer.key: optimizer for optimizer in optimizer_specs()}
        gd_two_steps = _run_optimizer(memory_objective, live_optimizers["gd"], 1.0, memory_objective.start, 2)
        gdm_two_steps = _run_optimizer(memory_objective, live_optimizers["gdm"], 1.0, memory_objective.start, 2)
        _require(bool(np.allclose(gd_two_steps.positions[1], gdm_two_steps.positions[1], atol=1e-12, rtol=0.0)), "GD and GDM should share the first step from zero memory")
        _require(float(gd_two_steps.values[1]) < 1e-6, "GD LR=1 did not reach the near-optimal region on step one")
        _require(float(gd_two_steps.values[2]) < float(gd_two_steps.values[1]), "GD did not remain near/converge toward the optimum on step two")
        _require(float(gdm_two_steps.values[2]) > 1.0, "GDM stale memory did not materially increase the objective on step two")

        gd_selected = selected_lookup[("memory_trap", "gd")]
        gdm_selected = selected_lookup[("memory_trap", "gdm")]
        gd_tail = float(gd_selected["mean_tail_relative_gap"])
        gdm_tail = float(gdm_selected["mean_tail_relative_gap"])
        _require(gd_tail <= 1.01e-12, "Tuned GD did not reach the numerical floor on the memory trap")
        _require(gdm_tail > 1e-2, "Tuned GDM unexpectedly erased the intended memory-trap gap")
        dense_candidates = [candidate for candidate in summaries[("memory_trap", "gdm")] if 0.2 <= candidate["learning_rate"] <= 0.5]
        _require(len(dense_candidates) >= 100, "Memory-trap GDM tuning was not dense enough in the sensitive LR interval")

        memory_evidence = {
            "hessian_xx_at_x_1": float(hessian[0, 0]),
            "shared_first_iterate": gd_two_steps.positions[1].tolist(),
            "gd_value_after_step_1_at_lr_1": float(gd_two_steps.values[1]),
            "gd_value_after_step_2_at_lr_1": float(gd_two_steps.values[2]),
            "gdm_value_after_step_2_at_lr_1": float(gdm_two_steps.values[2]),
            "tuned_gd_lr": float(gd_selected["selected_learning_rate"]),
            "tuned_gdm_lr": float(gdm_selected["selected_learning_rate"]),
            "tuned_gd_mean_tail_relative_gap": gd_tail,
            "tuned_gdm_mean_tail_relative_gap": gdm_tail,
            "tail_gap_ratio_gdm_to_gd_floor": gdm_tail / max(gd_tail, 1e-12),
            "dense_gdm_candidates_between_0_2_and_0_5": len(dense_candidates),
        }
        checks.append({"name": "memory_trap_construction", "status": "passed", **memory_evidence})

    _require(not manifest.get("validation_warnings"), f"Manifest contains unresolved warnings: {manifest.get('validation_warnings')}")
    checks.append({"name": "manifest_warnings", "status": "passed", "warnings": []})

    report = {
        "overall_assessment": "Ready to share",
        "validated_generation_time_utc": manifest["generated_at_utc"],
        "source_scope": manifest["source_scope"],
        "checks": checks,
        "memory_trap_evidence": memory_evidence,
        "selected_learning_rates": manifest["selected_learning_rates"],
        "verified_plot_files": verified_plot_files,
    }
    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Ready to share: {len(checks)} validation checks passed; report written to {report_path}")
    if memory_evidence is not None:
        print(
            "Memory trap: "
            f"GD tail={memory_evidence['tuned_gd_mean_tail_relative_gap']:.3g}, "
            f"GDM tail={memory_evidence['tuned_gdm_mean_tail_relative_gap']:.3g}, "
            f"dense tuned GDM LR={memory_evidence['tuned_gdm_lr']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
