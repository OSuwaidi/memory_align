"""Select one LR/WD cell per optimizer using only the screening dev loss."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import wandb

OPTIMIZERS = ("AdamW", "AM_AdamW", "AdaTAMW", "AGM_AdamW")
EXPECTED_SEEDS = (42, 1337)


def parse_case(value: str) -> tuple[str, float, float]:
    optimizer, learning_rate, weight_decay = value.split("::")
    return optimizer, float(learning_rate), float(weight_decay)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sweep = wandb.Api(timeout=180).sweep(args.sweep_path)
    observations: dict[tuple[str, float, float], list[tuple[int, float, str]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    for run in sweep.runs:
        config = dict(run.config)
        summary = dict(run.summary)
        metric = summary.get("final/val_loss", summary.get("final_val_loss"))
        if run.state != "finished" or not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
            rejected.append({"run_id": run.id, "state": run.state, "metric": metric})
            continue
        optimizer, learning_rate, weight_decay = parse_case(str(config["optimizer_case"]))
        observations[(optimizer, learning_rate, weight_decay)].append((int(config["seed"]), float(metric), run.id))

    rows: list[dict[str, Any]] = []
    for (optimizer, learning_rate, weight_decay), values in observations.items():
        values.sort()
        seeds = tuple(seed for seed, _metric, _run_id in values)
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(f"Cell {(optimizer, learning_rate, weight_decay)} has seeds {seeds}, expected {EXPECTED_SEEDS}.")
        losses = [metric for _seed, metric, _run_id in values]
        rows.append(
            {
                "optimizer": optimizer,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "mean_final_dev_loss": statistics.mean(losses),
                "std_final_dev_loss": statistics.stdev(losses),
                "seeds": list(seeds),
                "run_ids": [run_id for _seed, _metric, run_id in values],
                "losses": losses,
            }
        )
    if rejected:
        raise RuntimeError(f"The screening sweep contains unusable runs: {rejected}")

    selected: dict[str, dict[str, Any]] = {}
    for optimizer in OPTIMIZERS:
        candidates = [row for row in rows if row["optimizer"] == optimizer]
        if len(candidates) != 6:
            raise RuntimeError(f"Expected six LR/WD cells for {optimizer}, found {len(candidates)}.")
        # The first key is the predeclared selection metric. Remaining keys are
        # deterministic tie-breakers, not additional test-set optimization.
        winner = min(
            candidates,
            key=lambda row: (
                row["mean_final_dev_loss"],
                row["std_final_dev_loss"],
                row["learning_rate"],
                row["weight_decay"],
            ),
        )
        selected[optimizer] = winner

    receipt = {
        "source_sweep_path": args.sweep_path,
        "source_sweep_name": sweep.name,
        "selection_metric": "mean final dev cross-entropy across seeds 42 and 1337",
        "test_metrics_used_for_selection": False,
        "selected": selected,
        "all_cells": sorted(rows, key=lambda row: (row["optimizer"], row["learning_rate"], row["weight_decay"])),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(selected, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
