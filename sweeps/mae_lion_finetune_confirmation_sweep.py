"""Select AGAM-Lion's probe winner and create paired Lion fine-tune runs."""

from __future__ import annotations

import argparse
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import wandb

ENTITY_NAME = "osuwaidi-khalifa-university"
PROJECT_NAME = "MAL_benchmark"
EXPECTED_SCREEN_RUNS = 24
EXPECTED_SEEDS = (42, 1337)
LION_SOURCE_RUN_IDS = ("pcqfobwq", "ajqbhr9a")
EXPECTED_CONFIRMATION_RUNS = 4


def current_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def final_probe_accuracy(run: Any) -> float:
    # ``final/probe_val_acc`` is the common historical field shared by the
    # original Lion screen and the newer AGAM-Lion screens.
    value = run.summary.get("final/probe_val_acc")
    if value is None:
        value = run.summary.get("linear_probe/final_val_top1_pct", run.summary.get("final_probe_val_acc"))
    if value is None:
        raise ValueError(f"Run {run.id} does not contain a final linear-probe accuracy.")
    return float(value)


def require_final_checkpoint(run: Any) -> Path:
    checkpoint = Path(str(run.summary.get("checkpoint", ""))).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Run {run.id} final checkpoint is unavailable: {checkpoint}")
    source_epochs = int(run.config.get("epochs", 300))
    if int(run.summary.get("epoch", source_epochs)) != source_epochs:
        raise ValueError(f"Run {run.id} did not reach its configured final epoch {source_epochs}.")
    return checkpoint


def select_agam_sources(
    runs: list[Any],
    *,
    expected_runs: int,
    expected_seeds: tuple[int, ...] = EXPECTED_SEEDS,
) -> tuple[list[Any], list[dict[str, Any]]]:
    # Recovery sweeps may coexist with interrupted source runs.  Only fully
    # completed runs are eligible, and the distinct completed-cell count must
    # still match the preregistered design exactly.
    finished_runs = [run for run in runs if run.state == "finished"]
    if len(finished_runs) != expected_runs:
        states = {run.id: run.state for run in runs if run.state != "finished"}
        raise ValueError(
            f"Expected {expected_runs} finished screen runs, found {len(finished_runs)}; "
            f"ineligible runs: {states}."
        )

    grouped: dict[tuple[float, float], list[Any]] = defaultdict(list)
    observed_cells: set[tuple[float, float, int]] = set()
    for run in finished_runs:
        if str(run.config.get("optimizer")) != "AGAM_Lion":
            raise ValueError(f"Unexpected optimizer in AGAM-Lion screen: {run.config.get('optimizer')!r}.")
        base_lr = float(run.config["base_lr"])
        weight_decay = float(run.config["weight_decay"])
        seed = int(run.config["seed"])
        cell = (base_lr, weight_decay, seed)
        if cell in observed_cells:
            raise ValueError(f"Duplicate AGAM-Lion screen cell across input sweeps: {cell}.")
        observed_cells.add(cell)
        grouped[(base_lr, weight_decay)].append(run)

    ranking: list[dict[str, Any]] = []
    for (base_lr, weight_decay), group_runs in grouped.items():
        seeds = tuple(sorted(int(run.config["seed"]) for run in group_runs))
        if seeds != expected_seeds:
            print(
                "EXCLUDED_INCOMPLETE_AGAM_CELL="
                f"base_lr={base_lr},weight_decay={weight_decay},"
                f"observed_seeds={','.join(map(str, seeds))}"
            )
            continue
        accuracies = [final_probe_accuracy(run) for run in group_runs]
        ranking.append(
            {
                "base_lr": base_lr,
                "weight_decay": weight_decay,
                "mean_final_probe": statistics.fmean(accuracies),
                "min_final_probe": min(accuracies),
                "std_final_probe": statistics.pstdev(accuracies),
                "runs": sorted(group_runs, key=lambda run: int(run.config["seed"])),
            }
        )

    if not ranking:
        raise ValueError("No AGAM-Lion LR/WD cell has both required seeds.")

    # Primary selection is the paired-seed mean. Worst-seed performance and
    # lower variability are deterministic robustness tie-breakers.
    ranking.sort(
        key=lambda row: (
            row["mean_final_probe"],
            row["min_final_probe"],
            -row["std_final_probe"],
            -row["weight_decay"],
            -row["base_lr"],
        ),
        reverse=True,
    )
    selected = ranking[0]
    for run in selected["runs"]:
        require_final_checkpoint(run)
    return list(selected["runs"]), ranking


def validate_lion_sources(
    api: wandb.Api,
    *,
    source_run_ids: tuple[str, ...] = LION_SOURCE_RUN_IDS,
    expected_seeds: tuple[int, ...] = EXPECTED_SEEDS,
) -> list[Any]:
    sources = [api.run(f"{ENTITY_NAME}/{PROJECT_NAME}/{run_id}") for run_id in source_run_ids]
    seeds = tuple(sorted(int(run.config["seed"]) for run in sources))
    if seeds != expected_seeds:
        raise ValueError(f"Lion sources have seeds {seeds}; expected {expected_seeds}.")
    for run in sources:
        if run.state != "finished":
            raise ValueError(f"Lion source {run.id} is not finished: {run.state}.")
        if str(run.config.get("optimizer")) != "Lion":
            raise ValueError(f"Lion source {run.id} has optimizer {run.config.get('optimizer')!r}.")
        if float(run.config["base_lr"]) != 1e-4 or float(run.config["weight_decay"]) != 0.5:
            raise ValueError(f"Lion source {run.id} is not the selected LR=1e-4, WD=0.5 configuration.")
        require_final_checkpoint(run)
    return sorted(sources, key=lambda run: int(run.config["seed"]))


def build_sweep(args: argparse.Namespace, source_run_ids: list[str]) -> dict[str, Any]:
    comparison_group = getattr(args, "comparison_group", "lion_vs_agam_lion_selected_mae_finetune_v1")
    study_stage = getattr(args, "study_stage", "selected_checkpoint_end_to_end_finetune")
    return {
        "program": args.program,
        "name": args.sweep_name,
        "method": "grid",
        "metric": {"name": "finetune/final_val_top1_pct", "goal": "maximize"},
        "parameters": {
            "source_run_id": {"values": source_run_ids},
            "comparison_group": {"values": (comparison_group,)},
            "study_stage": {"values": (study_stage,)},
            "selection_metric": {"values": ("finetune/final_val_top1_pct",)},
            "source_revision": {"values": (args.source_revision,)},
        },
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "--data_dir",
            args.data_dir,
            "--source_entity",
            ENTITY_NAME,
            "--source_project",
            args.project_name,
            "--epochs",
            "100",
            "--batch_size",
            "1024",
            "--max_micro_batch_size",
            "256",
            "--base_lr",
            "0.0005",
            "--min_lr",
            "0.000001",
            "--warmup_epochs",
            "5",
            "--weight_decay",
            "0.05",
            "--layer_decay",
            "0.65",
            "--drop_path",
            "0.1",
            "--mixup",
            "0.8",
            "--cutmix",
            "1.0",
            "--label_smoothing",
            "0.1",
            "--num_workers",
            "4",
            "--amp_dtype",
            "bfloat16",
            "--float32_precision",
            "tf32",
            "${args}",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", help="Fine-tune entry point (normally tasks/mae_finetune_eval.py)")
    parser.add_argument("--screen_path", "--screen-path", action="append", required=True)
    parser.add_argument("--expected_screen_runs", "--expected-screen-runs", type=int, default=EXPECTED_SCREEN_RUNS)
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", default=PROJECT_NAME)
    parser.add_argument("--data_dir", "--data-dir", required=True)
    parser.add_argument("--source_revision", "--source-revision", default=current_commit())
    parser.add_argument("--expected_seeds", "--expected-seeds", type=int, nargs="+", default=EXPECTED_SEEDS)
    parser.add_argument("--lion_source_run_id", "--lion-source-run-id", action="append")
    args = parser.parse_args()

    expected_seeds = tuple(sorted(args.expected_seeds))
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        parser.error("--expected_seeds must contain distinct seed values.")
    lion_source_run_ids = tuple(args.lion_source_run_id or LION_SOURCE_RUN_IDS)
    if len(lion_source_run_ids) != len(expected_seeds):
        parser.error("Provide exactly one Lion source run for every expected seed.")

    api = wandb.Api(timeout=180)
    screen_runs: list[Any] = []
    for screen_path in args.screen_path:
        screen_runs.extend(list(api.sweep(screen_path).runs))
    agam_sources, ranking = select_agam_sources(
        screen_runs,
        expected_runs=args.expected_screen_runs,
        expected_seeds=expected_seeds,
    )
    lion_sources = validate_lion_sources(
        api,
        source_run_ids=lion_source_run_ids,
        expected_seeds=expected_seeds,
    )
    sources = lion_sources + agam_sources
    source_ids = [run.id for run in sources]
    expected_confirmation_runs = 2 * len(expected_seeds)
    if len(source_ids) != expected_confirmation_runs or len(set(source_ids)) != expected_confirmation_runs:
        raise RuntimeError(f"Expected {expected_confirmation_runs} distinct fine-tune sources, got {source_ids}.")

    winner = ranking[0]
    print(
        "SELECTED_AGAM_CONFIG="
        f"base_lr={winner['base_lr']},weight_decay={winner['weight_decay']},"
        f"mean_final_probe={winner['mean_final_probe']:.6f},"
        f"min_final_probe={winner['min_final_probe']:.6f},"
        f"std_final_probe={winner['std_final_probe']:.6f}"
    )
    print(f"SOURCE_RUN_IDS={','.join(source_ids)}")
    print(f"SCREEN_PATHS={','.join(args.screen_path)}")
    for rank, row in enumerate(ranking, start=1):
        print(
            f"AGAM_RANK_{rank}=base_lr={row['base_lr']},weight_decay={row['weight_decay']},"
            f"mean={row['mean_final_probe']:.6f},min={row['min_final_probe']:.6f},"
            f"std={row['std_final_probe']:.6f}"
        )

    sweep_id = wandb.sweep(
        entity=ENTITY_NAME,
        project=args.project_name,
        sweep=build_sweep(args, source_ids),
    )
    print(f"SWEEP_PATH={ENTITY_NAME}/{args.project_name}/{sweep_id}")
    print(f"EXPECTED_RUNS={expected_confirmation_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
