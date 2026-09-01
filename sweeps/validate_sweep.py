"""Validate that a W&B sweep has exactly the expected finished runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import wandb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_path")
    parser.add_argument("--expected_runs", "--expected-runs", type=int, required=True)
    args = parser.parse_args()

    sweep = wandb.Api(timeout=180).sweep(args.sweep_path)
    runs = list(sweep.runs)
    states = Counter(run.state for run in runs)
    receipt = {
        "sweep_path": args.sweep_path,
        "sweep_id": sweep.id,
        "sweep_name": sweep.name,
        "sweep_state": sweep.state,
        "expected_runs": args.expected_runs,
        "observed_runs": len(runs),
        "states": dict(states),
    }
    print(json.dumps(receipt, sort_keys=True))
    if len(runs) != args.expected_runs:
        return 2
    if states != Counter({"finished": args.expected_runs}):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
