"""Create the W&B sweep for full-parameter SmolLM2 fine-tuning."""

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
ADAMW_OPTIMIZERS = ("AdamW", "AM_AdamW", "CAUTIOUS_AdamW", "AdaTAMW", "MAL_AdamW")
SEEDS = (42, 1337, 2026)
BATCH_SIZES = (32,)
LR_MULTIPLIERS = (0.3, 1.0, 3.0)


def get_finished_run_ids(project_name: str, sweep_ids: list[str]) -> list[str]:
    api = wandb.Api()
    runs = api.runs(
        path=f"{ENTITY_NAME}/{project_name}",
        filters={"sweep": {"$in": sweep_ids}, "state": {"$in": ["finished", "running"]}},
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
    parser.add_argument("--family", choices=("sgdm", "adamw", "all"), default="all")
    parser.add_argument("--prior_sweeps", "--prior-sweeps", nargs="+")
    parser.add_argument("--method", choices=("grid", "random", "bayes"), default="grid")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--cache_dir", "--cache-dir", default="./data/llm_cache")
    parser.add_argument("--amp_dtype", "--amp-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--float32_precision", "--float32-precision", choices=("tf32", "ieee"), default="tf32")
    args = parser.parse_args()

    optimizers = {
        "sgdm": SGDM_OPTIMIZERS,
        "adamw": ADAMW_OPTIMIZERS,
        "all": SGDM_OPTIMIZERS + ADAMW_OPTIMIZERS,
    }[args.family]
    sweep_configuration = {
        "program": args.program,
        "name": args.sweep_name,
        "method": args.method,
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {
            "optimizer": {"values": optimizers},
            "batch_size": {"values": BATCH_SIZES},
            "lr_multiplier": {"values": LR_MULTIPLIERS},
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
            "True",
            "--sgd_base_lr",
            "0.01",
            "--am_msgd_base_lr",
            "0.1",
            "--adamw_base_lr",
            "0.00005",
            "--reference_batch_size",
            "32",
            "--weight_decay",
            "0.0",
            "--max_grad_norm",
            "1.0",
            "--momentum",
            "0.9",
            "--beta2",
            "0.999",
            "--mal_config",
            "False,1.0,True,attenuate,False",
            "--mal_align",
            "metric",
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
        print(f"Adding {len(prior_run_ids)} finished/running runs from prior sweep(s): {args.prior_sweeps}")

    sweep_id = wandb.sweep(
        entity=ENTITY_NAME,
        project=args.project_name,
        sweep=sweep_configuration,
        prior_runs=prior_run_ids,
    )
    print(f"Run with:\n$ uv run wandb agent --forward-signals {ENTITY_NAME}/{args.project_name}/{sweep_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
