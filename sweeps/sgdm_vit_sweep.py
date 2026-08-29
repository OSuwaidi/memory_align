"""Create the SGDM-family W&B sweep for Tiny-ImageNet MAE pre-training."""

from __future__ import annotations

import argparse

import wandb

# To initialize W&B sweep config:
# $ uv run sweeps/sgdm_vit_sweep.py <tasks/mae_pretrain.py> --sweep_name <___> --project_name <___> --> prints <entity/project/sweep/sweep_id>
# To assign/tag a run agent to a sweep:
# $ CUDA_VISIBLE_DEVICES=0 uv run wandb agent --forward-signals <entity/project/sweep_id>

ENTITY_NAME = "osuwaidi-khalifa-university"
MODEL = "vit_tiny_patch16_224"
IMAGE_SIZE = 64
PATCH_SIZE = 8
EPOCHS = 300
WARMUP_EPOCHS = 15
PROBE_EVERY = 50
SEEDS = (42, 1337, 2026)
BASE_LRS = (0.1, 0.3)
WEIGHT_DECAYS = (1e-4, 5e-4)
BATCH_SIZES = (4096, 256)
USE_SCHEDULER = (True,)
MAL_CONFIGS = ("False,1.0,True,attenuate,False",)


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
    parser.add_argument("program", help="MAE training entry point (normally mae_pretrain.py)")
    parser.add_argument("--data_dir", "--data-dir", default="./data/tiny-imagenet-200")
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", required=True)
    parser.add_argument("--prior_sweeps", "--prior-sweeps", nargs="+")
    parser.add_argument("--method", choices=("grid", "random", "bayes"), default="grid")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--warmup_epochs", "--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--probe_every", "--probe-every", type=int, default=PROBE_EVERY)
    parser.add_argument("--amp_dtype", "--amp-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--float32_precision", "--float32-precision", choices=("tf32", "ieee"), default="tf32")
    parser.add_argument("--output_dir", "--output-dir", default="./outputs/mae")
    args = parser.parse_args()

    sweep_configuration = {
        "program": args.program,
        "name": args.sweep_name,
        "method": args.method,
        "metric": {"name": "probe/val_acc", "goal": "maximize"},
        "parameters": {
            "optimizer": {"values": ("SGDM", "AM_MSGD", "CAUTIOUS_SGDM", "TAM_SGDM", "MAL_SGDM")},
            "mal_config": {"values": MAL_CONFIGS},
            "nesterov": {"values": (False,)},
            "batch_size": {"values": BATCH_SIZES},
            "base_lr": {"values": BASE_LRS},
            "weight_decay": {"values": WEIGHT_DECAYS},
            "seed": {"values": SEEDS},
            "use_scheduler": {"values": USE_SCHEDULER},
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
            "--amp_dtype",
            args.amp_dtype,
            "--float32_precision",
            args.float32_precision,
            "--output_dir",
            args.output_dir,
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
    print(f"Run with:\n$ uv run wandb agent --forward-signals {ENTITY_NAME}/{args.project_name}/{sweep_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
