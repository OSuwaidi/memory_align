import wandb

# To initialize W&B sweep config: $ uv run create_sweep.py --> prints <entity/project/sweep_id>
# To assign/tag a run agent to a sweep: $ CUDA_VISIBLE_DEVICES=0 uv run wandb agent --forward-signals <entity/project/sweep_id>

ENTITY_NAME = "osuwaidi-khalifa-university"
PROJECT_NAME = "align_cifar100"
SEEDS = (77, 433, 1024)
LRs = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 0.1)
BATCH_SIZES = (64, 128, 256, 512, 1024, 2048)

if __name__ == "__main__":
    # 1. Define the sweep configuration
    sweep_configuration = {
        "program": "main_cifar100.py",
        "name": "bs_sweep",
        "method": "grid",  # 'grid' tries every combination. Use 'bayes' or 'random' for large searches.
        "metric": {
            "name": "test_acc",
            "goal": "maximize",
            },
        "parameters": {
            "align": {"values": (True, False)},
            "ema": {"values": (True, False)},
            "per": {"values": (True, False)},
            "batch_size": {"values": BATCH_SIZES},
            "lr": {"values": LRs},
            "seed": {"values": SEEDS},
            },
        # "command" key used to inject custom CLI args: the command agent uses to launch "program" (script)
        "command": [  # Order MATTERS: must form a valid run command
            "${env}",  # macros get expanded upon run
            "${interpreter}",
            "${program}",
            "--some_flag",
            "flag_value",
            "${args}",  # MANDATORY at the end: expands all sweep parameters as CLI args
            ],
        }

    # 2. Initialize the sweep on W&B servers
    sweep_id = wandb.sweep(
            sweep=sweep_configuration,
            project=PROJECT_NAME,
            )
    print(f"To run a W&B agent against the sweep: $ uv run wandb agent --forward-signals {ENTITY_NAME}/{PROJECT_NAME}/{sweep_id}")

    # wandb.agent(
    #         sweep_id=sweep_id,
    #         function=lambda: main(**args_dict),
    #         project=PROJECT_NAME,
    #         )
