"""Create a W&B sweep for full-parameter SmolLM2 fine-tuning.

The command-line selectors deliberately live in this sweep creator rather
than the training entry point: optimizer, effective batch size, LR multiplier,
and seed are W&B sweep parameters, while the fixed training recipe is passed
to ``tasks/llm_finetune.py`` through the sweep command.
"""

from __future__ import annotations

import argparse

import wandb

ENTITY_NAME = "osuwaidi-khalifa-university"
MODEL = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
DATASET = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"

SGDM_OPTIMIZERS = ("SGDM", "AM_MSGD", "CAUTIOUS_SGDM", "TAM_SGDM", "MAL_SGDM")
ADAMW_OPTIMIZERS = ("AdamW", "AM_AdamW", "AdaTAMW", "MAL_AdamW")
SEEDS = (42, 1337, 2026)
BATCH_SIZES = (32,)
LR_MULTIPLIERS = (0.3, 1.0, 3.0)
DEFAULT_MAL_CONFIG = "False,1.0,none,attenuate,update,complement"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f'expected a boolean value, got "{value}"')


def get_finished_run_ids(project_name: str, sweep_ids: list[str]) -> list[str]:
    api = wandb.Api()
    runs = api.runs(
        path=f"{ENTITY_NAME}/{project_name}",
        filters={"sweep": {"$in": sweep_ids}, "state": "finished"},
        per_page=100,
        lazy=True,
        include_sweeps=True,
    )
    return [run.id for run in runs]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", help="LLM training entry point (normally llm_finetune.py)")
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", required=True)
    optimizer_selection = parser.add_mutually_exclusive_group()
    optimizer_selection.add_argument("--family", choices=("sgdm", "adamw", "all"), default="adamw")
    optimizer_selection.add_argument(
        "--optimizers",
        nargs="+",
        choices=SGDM_OPTIMIZERS + ADAMW_OPTIMIZERS,
        help="Explicit optimizer subset; overrides the family selector.",
    )
    parser.add_argument("--prior_sweeps", "--prior-sweeps", nargs="+")
    parser.add_argument("--method", choices=("grid",), default="grid")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_sizes", "--batch-sizes", nargs="+", type=int, default=list(BATCH_SIZES))
    parser.add_argument("--lr_multipliers", "--lr-multipliers", nargs="+", type=float, default=list(LR_MULTIPLIERS))
    parser.add_argument("--use_scheduler", "--use-scheduler", type=parse_bool, default=True)
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=0.0)
    parser.add_argument("--cache_dir", "--cache-dir", default="./data/llm_cache")
    parser.add_argument("--amp_dtype", "--amp-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--float32_precision", "--float32-precision", choices=("tf32", "ieee"), default="tf32")
    parser.add_argument(
        "--mal_config",
        "--mal-config",
        default=DEFAULT_MAL_CONFIG,
        help="Single MAL configuration selected before the full LLM benchmark.",
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if not args.batch_sizes or any(batch_size <= 0 for batch_size in args.batch_sizes):
        parser.error("--batch_sizes must contain positive integers.")
    if not args.lr_multipliers or any(multiplier <= 0.0 for multiplier in args.lr_multipliers):
        parser.error("--lr_multipliers must contain positive values.")
    if args.weight_decay < 0.0:
        parser.error("--weight_decay must be non-negative.")

    optimizers = tuple(args.optimizers) if args.optimizers else {
        "sgdm": SGDM_OPTIMIZERS,
        "adamw": ADAMW_OPTIMIZERS,
        "all": SGDM_OPTIMIZERS + ADAMW_OPTIMIZERS,
    }[args.family]
    if any(optimizer in SGDM_OPTIMIZERS for optimizer in optimizers) and args.mal_config.split(",")[-1].strip().lower() != "fixed":
        parser.error("SGDM-family sweeps require a MAL_config with gradient_weight_mode=fixed.")

    sweep_configuration = {
        "program": args.program,
        "name": args.sweep_name,
        "method": args.method,
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {
            "optimizer": {"values": optimizers},
            "MAL_config": {"values": (args.mal_config,)},
            "batch_size": {"values": tuple(dict.fromkeys(args.batch_sizes))},
            "lr_multiplier": {"values": tuple(dict.fromkeys(args.lr_multipliers))},
            "seed": {"values": SEEDS},
        },
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "--model_name",
            MODEL,
            "--model_revision",
            MODEL_REVISION,
            "--dataset_name",
            DATASET,
            "--dataset_config",
            DATASET_CONFIG,
            "--dataset_revision",
            DATASET_REVISION,
            "--cache_dir",
            args.cache_dir,
            "--sequence_length",
            "512",
            "--epochs",
            str(args.epochs),
            "--warmup_ratio",
            "0.1",
            "--use_scheduler",
            str(args.use_scheduler),
            "--sgd_base_lr",
            "0.01",
            "--am_msgd_base_lr",
            "0.1",
            "--adamw_base_lr",
            "0.00005",
            "--reference_batch_size",
            "32",
            "--weight_decay",
            str(args.weight_decay),
            "--max_grad_norm",
            "1.0",
            "--momentum",
            "0.9",
            "--beta2",
            "0.999",
            "--amp_dtype",
            args.amp_dtype,
            "--float32_precision",
            args.float32_precision,
            "${args}",
        ],
    }

    prior_run_ids = None
    if args.prior_sweeps:
        prior_run_ids = get_finished_run_ids(args.project_name, args.prior_sweeps)
        print(f"Adding {len(prior_run_ids)} finished runs from prior sweep(s): {args.prior_sweeps}")

    sweep_id = wandb.sweep(
        entity=ENTITY_NAME,
        project=args.project_name,
        sweep=sweep_configuration,
        prior_runs=prior_run_ids,
    )
    expected_runs = (
        len(optimizers)
        * len(tuple(dict.fromkeys(args.batch_sizes)))
        * len(tuple(dict.fromkeys(args.lr_multipliers)))
        * len(SEEDS)
    )
    print(f"EXPECTED_RUNS={expected_runs}")
    print(f"Run with:\n$ uv run wandb agent --forward-signals {ENTITY_NAME}/{args.project_name}/{sweep_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
