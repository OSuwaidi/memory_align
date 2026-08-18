import argparse

import wandb

# To initialize W&B sweep config:
# $ uv run sweeps/cifar_resnet_sweep.py <main.py> --data <___> --sweep_name <___> --project_name <___> --> prints <entity/project/sweep/sweep_id>
# To assign/tag a run agent to a sweep:
# $ CUDA_VISIBLE_DEVICES=0 uv run wandb agent --forward-signals <entity/project/sweep_id>

ENTITY_NAME = "osuwaidi-khalifa-university"
SEEDS = (42, 1337,) #2026)
LRs = (
    0.025,
    0.05,
    0.1,
    0.2,
    # 0.4,
    # 0.8,
    # 1.0,
)
BATCH_SIZES = (64, 512, 2048, 4096,)[::-1]
WEIGHT_DECAY = (5e-4,)

def percentage(value: str) -> float:
    parsed_value = float(value)
    if not 0.0 <= parsed_value <= 100.0:
        raise argparse.ArgumentTypeError("must be between 0 and 100")
    return parsed_value


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=str, help="Dataset name", required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--arch",
        type=str,
        help="Architecture name",
        default="resnet50",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--val_acc_target",
        type=percentage,
        default=50.0,
        help="Validation-accuracy target percentage used for convergence-speed metrics (0-100).",
    )
    parser.add_argument(
        "--amp_dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="CUDA autocast dtype; defaults to bfloat16. float32 disables AMP.",
    )
    parser.add_argument(
        "--float32_precision",
        type=str,
        choices=("tf32", "ieee"),
        default="tf32",
        help="Internal precision for residual CUDA float32 matmuls/convolutions.",
    )


def get_finished_run_ids(sweep_ids: list[str]) -> list[str]:
    """
    Retrieves the IDs of all finished runs within a specified sweep.
    :param sweep_ids: The IDs of the prior sweep.
    :return: A list of strings containing the IDs of all finished runs.
    """
    api = wandb.Api()
    runs = api.runs(
        path=f"{ENTITY_NAME}/{args.project_name}",
        filters={
            "sweep": {"$in": sweep_ids},
            "state": "finished",
        },
        per_page=200,
        lazy=True,
        include_sweeps=True,
    )

    return [run.id for run in runs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dynamic W&B Sweep configuration.")
    parser.add_argument("program", type=str, help="Python training script to run")  # required by default since positional arg
    parser.add_argument("--sweep_name", type=str, help="Sweep name", required=True)
    parser.add_argument("--project_name", type=str, help="Project name", required=True)
    parser.add_argument(
        "--prior_sweeps",
        type=str,
        nargs="+",
        help="Previous sweep(s) ID",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="grid",
        choices=["grid", "random", "bayes"],
        help="Sweep search method",
    )
    add_training_args(parser)
    args = parser.parse_args()

    # 1. Define the sweep configuration
    sweep_configuration = {
        "program": args.program,
        "name": args.sweep_name,
        "method": args.method,  # 'grid' tries every combination. Use 'bayes' or 'random' for large searches.
        "metric": {
            "name": "test_acc",
            "goal": "maximize",
        },
        "parameters": {
            "align": {
                "values": (
                    "MAL",
                    # "none",
                    # "cautious",
                )
            },
            "inplace_pwr": {"values": ("True,0.5", "False,1.0", "True,1.0")},
            "scale": {"values": (True, False)},
            "per_unit": {"values": (True, False)},
            "nesterov": {"values": (False,)},
            "batch_size": {"values": BATCH_SIZES},
            "lr": {"values": LRs},
            "weight_decay": {"values": WEIGHT_DECAY},
            "seed": {"values": SEEDS},
        },
        # "command" key used to inject custom CLI args: the command agent uses to launch "program" (script)
        "command": [  # Order MATTERS: must form a valid run command
            "${env}",  # macros get expanded upon run
            "${interpreter}",
            "${program}",
            "--data",
            args.data,
            "--data_dir",
            args.data_dir,
            "--arch",
            args.arch,
            "--epochs",
            args.epochs,
            "--val_acc_target",
            args.val_acc_target,
            "--amp_dtype",
            args.amp_dtype,
            "--float32_precision",
            args.float32_precision,
            "${args}",  # MANDATORY at the end: expands all sweep parameters as CLI args
        ],
    }

    # Fetch successfully completed runs from previous sweep
    prior_run_ids = None
    if prior_sweeps := args.prior_sweeps:
        prior_run_ids = get_finished_run_ids(prior_sweeps)

        print(f"Adding {len(prior_run_ids)} finished runs from sweep ID(s): {prior_sweeps} as prior runs.")

    # 2. Initialize the sweep on W&B servers, seeded with completed runs
    sweep_id = wandb.sweep(
        entity=ENTITY_NAME,
        project=args.project_name,
        sweep=sweep_configuration,
        prior_runs=prior_run_ids,
    )
    print(f"To run a W&B agent against the sweep:\n$ uv run wandb agent --forward-signals {ENTITY_NAME}/{args.project_name}/{sweep_id}")

    # wandb.agent(
    #         sweep_id=sweep_id,
    #         function=lambda: main(),
    #         project=args.project_name,
    #         )
