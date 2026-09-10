"""Create the matched Modal experiment for MAL-AdamW ``in_place``.

Two phases avoid wasting runs while preserving a fully paired comparison:

``controls``
    Run the missing out-of-place update and metric controls at seed 42, which
    match the completed in-place Modal pilot exactly.

``replication``
    Run in-place versus out-of-place update alignment at seeds 1337 and 2026.
    Together with seed 42, this yields three paired seeds for the structural
    decision.
"""

from __future__ import annotations

import argparse
import subprocess
from typing import Any

import wandb

ENTITY_NAME = "osuwaidi-khalifa-university"
PROJECT_NAME = "MAL_benchmark"

MODEL = "vit_tiny_patch16_224"
IMAGE_SIZE = 64
PATCH_SIZE = 8
EPOCHS = 300
WARMUP_EPOCHS = 15
PROBE_EVERY = 50

REPRESENTATIVE_BATCH_SIZE = 1024
REPRESENTATIVE_BASE_LR = 1e-3
REPRESENTATIVE_WEIGHT_DECAY = 5e-2
COMPARISON_GROUP = "mal_adamw_in_place_matched_v1"

PHASES: dict[str, dict[str, Any]] = {
    "controls": {
        "MAL_configs": (
            "False,1.0,none,attenuate,update,complement",
            "False,1.0,none,attenuate,metric,complement",
        ),
        "seeds": (42,),
        "experiment_stage": "mal_adamw_in_place_seed42_controls",
        "expected_runs": 2,
    },
    "replication": {
        "MAL_configs": (
            "False,1.0,none,attenuate,update,complement",
            "True,1.0,none,attenuate,update,complement",
        ),
        "seeds": (1337, 2026),
        "experiment_stage": "mal_adamw_in_place_paired_replication",
        "expected_runs": 4,
    },
}


def current_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_sweep_configuration(args: argparse.Namespace) -> dict[str, Any]:
    phase = PHASES[args.phase]
    return {
        "program": args.program,
        "name": args.sweep_name,
        "method": "grid",
        "metric": {"name": "final_probe_val_acc", "goal": "maximize"},
        "parameters": {
            "optimizer": {"values": ("MAL_AdamW",)},
            "MAL_config": {"values": phase["MAL_configs"]},
            "batch_size": {"values": (REPRESENTATIVE_BATCH_SIZE,)},
            "base_lr": {"values": (REPRESENTATIVE_BASE_LR,)},
            "weight_decay": {"values": (REPRESENTATIVE_WEIGHT_DECAY,)},
            "seed": {"values": phase["seeds"]},
            "use_scheduler": {"values": (True,)},
            "compute_platform": {"values": ("modal",)},
            "comparison_group": {"values": (COMPARISON_GROUP,)},
            "experiment_stage": {"values": (phase["experiment_stage"],)},
            "source_revision": {"values": (args.source_revision,)},
        },
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "--data_dir",
            args.data_dir,
            "--arch",
            MODEL,
            "--image_size",
            str(IMAGE_SIZE),
            "--patch_size",
            str(PATCH_SIZE),
            "--epochs",
            str(args.epochs),
            "--warmup_epochs",
            str(args.warmup_epochs),
            "--probe_every",
            str(args.probe_every),
            "--num_workers",
            "4",
            "--amp_dtype",
            "bfloat16",
            "--float32_precision",
            "tf32",
            "--output_dir",
            args.output_dir,
            "--save_every",
            "0",
            "--beta2",
            "0.95",
            "${args}",
        ],
    }


def expected_run_count(configuration: dict[str, Any]) -> int:
    count = 1
    for parameter in configuration["parameters"].values():
        count *= len(parameter["values"])
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", help="MAE entry point (normally tasks/mae_pretrain.py)")
    parser.add_argument("--phase", choices=tuple(PHASES), required=True)
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", default=PROJECT_NAME)
    parser.add_argument("--data_dir", "--data-dir", default="/workspace/data/tiny-imagenet-200")
    parser.add_argument("--output_dir", "--output-dir", default="/tmp/memory-align/mae-in-place-matched")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--warmup_epochs", "--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--probe_every", "--probe-every", type=int, default=PROBE_EVERY)
    parser.add_argument("--source_revision", "--source-revision", default=current_commit())
    args = parser.parse_args()

    configuration = build_sweep_configuration(args)
    expected_runs = expected_run_count(configuration)
    phase_expected_runs = PHASES[args.phase]["expected_runs"]
    if expected_runs != phase_expected_runs:
        raise RuntimeError(
            f'Phase "{args.phase}" must contain exactly {phase_expected_runs} runs, found {expected_runs}.'
        )

    sweep_id = wandb.sweep(
        entity=ENTITY_NAME,
        project=args.project_name,
        sweep=configuration,
    )
    sweep_path = f"{ENTITY_NAME}/{args.project_name}/{sweep_id}"
    print(f"SWEEP_PATH={sweep_path}")
    print(f"EXPECTED_RUNS={expected_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
