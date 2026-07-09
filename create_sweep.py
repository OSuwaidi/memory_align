import wandb

# To run: $uv run create_sweeps.py --> prints <entity/project/sweep_id>

PROJECT_NAME = "momentum-cifar10"
SEEDS = (77, 433, 1024)
LRs = (0.05, 0.1, 0.5)

if __name__ == "__main__":
    # 1. Define the sweep configuration
    sweep_configuration = {
        "program": "main.py",
        "name": "mem_align",
        "method": "grid",  # 'grid' tries every combination. Use 'bayes' or 'random' for large searches.
        "metric": {
            "name": "test_acc",
            "goal": "maximize",
        },
        "parameters": {
            "ema": {"values": (True, False)},
            "couple": {"values": (True, False)},
            "per": {"values": (True, False)},
            "lr": {"values": LRs},
            "seed": {"values": SEEDS},
        },
    }

    # 2. Initialize the sweep on W&B servers
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=PROJECT_NAME,
    )
    print(f"Sweep ID: {sweep_id}")

    # wandb.agent(
    #         sweep_id=sweep_id,
    #         function=lambda: main(**args_dict),
    #         project=PROJECT_NAME,
    #         )
