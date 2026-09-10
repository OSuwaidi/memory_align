"""Continuously stop one optimizer's runs while allowing a sweep to proceed."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import wandb

ALLOWED_SWEEP_PREFIX = "osuwaidi-khalifa-university/MAL_benchmark/"


def write_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def run_signature(run: Any) -> tuple[Any, ...]:
    config = dict(run.config)
    return (
        config.get("optimizer"),
        config.get("MAL_config"),
        config.get("batch_size"),
        config.get("base_lr"),
        config.get("weight_decay"),
        config.get("seed"),
        config.get("use_scheduler"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_path")
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--expected-optimizer-runs", type=int, required=True)
    parser.add_argument("--expected-finished-at-start", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--record-path", type=Path, required=True)
    args = parser.parse_args()

    if not args.sweep_path.startswith(ALLOWED_SWEEP_PREFIX):
        parser.error(f"sweep path must start with {ALLOWED_SWEEP_PREFIX}")
    if args.expected_optimizer_runs <= 0 or args.expected_finished_at_start < 0:
        parser.error("expected run counts must be non-negative and the total must be positive")
    if args.expected_finished_at_start > args.expected_optimizer_runs:
        parser.error("the initial finished count cannot exceed the expected optimizer-run count")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")

    api = wandb.Api(timeout=180)
    stop_requested: set[str] = set()
    terminal_excluded: set[str] = set()
    initial_finished_ids: set[str] | None = None

    while True:
        api.flush()
        sweep = api.sweep(args.sweep_path)
        optimizer_runs = [run for run in sweep.runs if dict(run.config).get("optimizer") == args.optimizer]
        signatures = [run_signature(run) for run in optimizer_runs]
        if len(signatures) != len(set(signatures)):
            raise RuntimeError(f"Duplicate {args.optimizer} grid signatures detected in {args.sweep_path}.")
        if len(optimizer_runs) > args.expected_optimizer_runs:
            raise RuntimeError(
                f"Expected at most {args.expected_optimizer_runs} {args.optimizer} runs, found {len(optimizer_runs)}."
            )

        finished_ids = {run.id for run in optimizer_runs if run.state == "finished"}
        if initial_finished_ids is None:
            if len(finished_ids) != args.expected_finished_at_start:
                raise RuntimeError(
                    f"Expected {args.expected_finished_at_start} already-finished {args.optimizer} runs, "
                    f"found {len(finished_ids)}."
                )
            initial_finished_ids = finished_ids
        elif not initial_finished_ids.issubset(finished_ids):
            raise RuntimeError("An initially finished run changed state unexpectedly.")

        for run in optimizer_runs:
            if run.state == "running" and run.id not in stop_requested:
                print(f"Requesting stop for {run.id}: {run.name}", flush=True)
                run.stop()
                stop_requested.add(run.id)
            elif run.state not in {"finished", "running", "pending"}:
                # This covers runs stopped immediately before this monitor was
                # submitted as well as a future run that terminates between
                # API snapshots. It is excluded from the result set either way.
                terminal_excluded.add(run.id)

        excluded_ids = stop_requested | terminal_excluded

        payload = {
            "captured_at": datetime.now(UTC).isoformat(),
            "sweep_path": args.sweep_path,
            "sweep_state": sweep.state,
            "target_optimizer": args.optimizer,
            "expected_optimizer_runs": args.expected_optimizer_runs,
            "initial_finished_ids": sorted(initial_finished_ids),
            "stop_requested_ids": sorted(stop_requested),
            "terminal_excluded_ids": sorted(terminal_excluded),
            "excluded_nonfinished_ids": sorted(excluded_ids),
            "observed_optimizer_run_ids": sorted(run.id for run in optimizer_runs),
            "observed_states": {run.id: run.state for run in optimizer_runs},
        }
        write_record(args.record_path, payload)

        expected_stops = args.expected_optimizer_runs - args.expected_finished_at_start
        if len(optimizer_runs) == args.expected_optimizer_runs and len(excluded_ids) == expected_stops:
            print(
                f"All {expected_stops} non-finished {args.optimizer} grid runs were stopped or terminally excluded; "
                f"preserved {len(initial_finished_ids)} completed runs.",
                flush=True,
            )
            return 0
        if sweep.state in {"FINISHED", "CANCELED"}:
            raise RuntimeError(
                f"Sweep became {sweep.state} before all {args.optimizer} configurations were observed and stopped."
            )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
