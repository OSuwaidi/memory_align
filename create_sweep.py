import argparse

import wandb

# To initialize W&B sweep config:
# $ uv run create_sweep.py <main.py> --data <___> --sweep_name <___> --project_name <___> --> prints <entity/project/sweep/sweep_id>
# To assign/tag a run agent to a sweep:
# $ CUDA_VISIBLE_DEVICES=0 uv run wandb agent --forward-signals <entity/project/sweep_id>

ENTITY_NAME = "osuwaidi-khalifa-university"

SEEDS = (77, 433, 1024)
LRS = (0.1, 0.2, 0.4)
BATCH_SIZES = (128, 256)

# Hyperband waits for this many logged val_acc observations before it can
# terminate a run. Increase this if your method needs a longer warm-up.
HYPERBAND_MIN_ITER = 11
HYPERBAND_ETA = 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a dynamic W&B Sweep configuration."
    )
    parser.add_argument(
        "program",
        type=str,
        help="Python training script to run",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--sweep_name",
        type=str,
        required=True,
        help="Sweep name",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="Project name",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="bayes",
        choices=["grid", "random", "bayes"],
        help="Sweep search method",
    )

    args = parser.parse_args()

    sweep_configuration = {
        "program": args.program,
        "name": args.sweep_name,
        "method": args.method,
        "metric": {
            "name": "val_acc",
            "goal": "maximize",
        },
        "parameters": {
            "align": {
                "values": ("MAL_ada",),
            },
            "nesterov": {
                "values": (True, False),
            },
            "batch_size": {
                "values": BATCH_SIZES,
            },
            "lr": {
                "values": LRS,
            },
            "seed": {
                "values": SEEDS,
            },
            # Continuously sampled over [0, 1], rather than choosing
            # from a discrete list of values.
            "c": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 1.0,
            },
        },
        "early_terminate": {
            "type": "hyperband",
            # Do not prune until val_acc has been logged at least 10 times.
            "min_iter": HYPERBAND_MIN_ITER,
            # Subsequent brackets occur at approximately 10, 30, 90, ...
            "eta": HYPERBAND_ETA,
            # False is W&B's less aggressive Hyperband behavior.
            "strict": False,
        },
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "--data",
            args.data,
            "${args}",
        ],
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        entity=ENTITY_NAME,
        project=args.project_name,
    )

    print(
        "To run a W&B agent against the sweep:\n"
        f"$ uv run wandb agent --forward-signals "
        f"{ENTITY_NAME}/{args.project_name}/{sweep_id}"
    )
