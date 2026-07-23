import wandb
import argparse

# To initialize W&B sweep config: $ uv run create_sweep.py <main.py> --data <___> --sweep_name <___> --project_name <___> --> prints <entity/project/sweep/sweep_id>
# To assign/tag a run agent to a sweep: $ CUDA_VISIBLE_DEVICES=0 uv run wandb agent --forward-signals <entity/project/sweep_id>

ENTITY_NAME = "osuwaidi-khalifa-university"
SEEDS = (77, 433, 1024)
LRs = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0)
BATCH_SIZES = (64, 128, 256, 512, 1024, 2048)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dynamic W&B Sweep configuration.")
    parser.add_argument("program", type=str, help="Python training script to run")  # required by default since positional arg
    parser.add_argument("--data", type=str, help="Dataset name", required=True)
    parser.add_argument("--sweep_name", type=str, help="Sweep name", required=True)
    parser.add_argument("--project_name", type=str, help="Project name", required=True)
    parser.add_argument(
        "--method",
        type=str,
        default="grid",
        choices=["grid", "random", "bayes"],
        help="Sweep search method",
    )
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
                )
            },
            "nesterov": {"values": (False,)},
            "batch_size": {"values": BATCH_SIZES},
            "lr": {"values": LRs},
            "seed": {"values": SEEDS},
        },
        # "command" key used to inject custom CLI args: the command agent uses to launch "program" (script)
        "command": [  # Order MATTERS: must form a valid run command
            "${env}",  # macros get expanded upon run
            "${interpreter}",
            "${program}",
            "--data",
            f"{args.data}",
            "${args}",  # MANDATORY at the end: expands all sweep parameters as CLI args
        ],
    }

    # 2. Initialize the sweep on W&B servers
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=args.project_name,
    )
    print(
        f"To run a W&B agent against the sweep: $ uv run wandb agent --forward-signals {ENTITY_NAME}/{args.project_name}/{sweep_id}"
    )

    # wandb.agent(
    #         sweep_id=sweep_id,
    #         function=lambda: main(),
    #         project=args.project_name,
    #         )
