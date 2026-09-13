"""Create the post-selection CIFAR heatmaps and scheduler ablation.

The optimizer and MAL structure are encoded in one ``optimizer_case`` sweep
parameter.  This avoids the silent Cartesian duplication that would result
from sweeping ``optimizer`` and ``MAL_config`` independently.
"""

from __future__ import annotations

import argparse
from typing import Any

import wandb

ENTITY_NAME = "osuwaidi-khalifa-university"
PROJECT_NAME = "MAL_benchmark"
SEEDS = (42, 1337, 2026)
WEIGHT_DECAY = 5e-4

T_ATT_U = "False,1.0,False,attenuate"
T_ATT_U_P05 = "False,0.5,False,attenuate"
I_ATT_U = "True,1.0,False,attenuate"
T_REP_U = "False,1.0,False,replace"
T_REP_N = "False,1.0,True,replace"

# Explicit seven-field forms for the QHM-style MAL-SGDM ablation:
# in_place,pwr,scale,gate_mode,align,gradient_weight_mode,unbias.
# ``align=moment`` names SGDM's only alignment geometry; it is metadata rather
# than an additional degree of freedom.
MAL_SGDM_COMPLEMENT_NONE = "False,1.0,False,attenuate,moment,complement,none"
MAL_SGDM_COMPLEMENT_BUFFER = "False,1.0,False,attenuate,moment,complement,buffer"
MAL_SGDM_COMPLEMENT_ESTIMATOR = "False,1.0,False,attenuate,moment,complement,estimator"

CIFAR10_BATCH_SIZES = (64, 128, 256, 512, 1024, 2048, 4096)
CIFAR10_LRS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
CIFAR10_QHM_LRS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
CIFAR100_BATCH_SIZES = (128, 256, 512, 1024, 2048, 4096)
CIFAR100_LRS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8)


def mal_case(label: str, config: str) -> str:
    return f"MAL_SGDM::{label}::{config}"


def validate_sgdm_mal_config(value: str) -> str:
    fields = value.split(",")
    if len(fields) not in (4, 7):
        raise argparse.ArgumentTypeError(
            "must be 'in_place,pwr,scale,gate_mode' or "
            "'in_place,pwr,scale,gate_mode,moment,gradient_weight_mode,unbias'"
        )
    if fields[0] not in {"True", "False"} or fields[2] not in {"True", "False"}:
        raise argparse.ArgumentTypeError("in_place and scale must be True or False")
    if fields[1] not in {"0.5", "1.0"}:
        raise argparse.ArgumentTypeError("pwr must be 0.5 or 1.0")
    if fields[3] not in {"attenuate", "replace"}:
        raise argparse.ArgumentTypeError("gate_mode must be attenuate or replace")
    if len(fields) == 7:
        align, gradient_weight_mode, unbias = fields[4:]
        if align != "moment":
            raise argparse.ArgumentTypeError("MAL-SGDM alignment must be moment")
        if gradient_weight_mode not in {"fixed", "complement"}:
            raise argparse.ArgumentTypeError("gradient_weight_mode must be fixed or complement")
        if unbias not in {"none", "buffer", "estimator"}:
            raise argparse.ArgumentTypeError("unbias must be none, buffer, or estimator")
        if gradient_weight_mode == "fixed" and unbias != "none":
            raise argparse.ArgumentTypeError("fixed gradient weighting requires unbias=none")
        if gradient_weight_mode == "complement" and fields[3] != "attenuate":
            raise argparse.ArgumentTypeError("complement gradient weighting requires gate_mode=attenuate")
    return value


def build_configuration(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    comparison_parameters: dict[str, Any] = {}
    if args.experiment == "cifar10-screen":
        data = "cifar10"
        arch = "resnet18"
        batch_sizes = CIFAR10_BATCH_SIZES
        learning_rates = CIFAR10_LRS
        target = 90.0
        optimizer_cases = (
            "AM_MSGD",
            "TAM_SGDM",
            mal_case("T-Att/U", T_ATT_U),
            mal_case("T-Rep/U", T_REP_U),
            mal_case("T-Rep/N", T_REP_N),
        )
        scheduler_values = (True,)
    elif args.experiment == "cifar10-mal-qhm":
        # Full matched heatmap against the 147 completed canonical MAL-SGDM
        # controls in sweep bps6rkim.  Reusing those exact cells avoids 126
        # redundant runs through LR=0.8; this commit's added step bookkeeping
        # does not alter the default recurrence (covered by the numerical check).
        data = "cifar10"
        arch = "resnet18"
        batch_sizes = CIFAR10_BATCH_SIZES
        learning_rates = CIFAR10_QHM_LRS
        target = 90.0
        optimizer_cases = (
            mal_case("MAL-complement-none", MAL_SGDM_COMPLEMENT_NONE),
            mal_case("MAL-complement-buffer", MAL_SGDM_COMPLEMENT_BUFFER),
            mal_case("MAL-complement-estimator", MAL_SGDM_COMPLEMENT_ESTIMATOR),
        )
        scheduler_values = (True,)
        comparison_parameters = {
            "comparison_sweep": {"value": "bps6rkim"},
            "comparison_optimizer_variant": {"value": "T-Att/U"},
            "comparison_MAL_config": {"value": "False,1.0,False,attenuate,False"},
        }
    elif args.experiment == "cifar100-benchmark":
        if not args.mal_sgdm_config:
            raise ValueError("--mal_sgdm_config is required for cifar100-benchmark")
        data = "cifar100"
        arch = "resnet50"
        batch_sizes = CIFAR100_BATCH_SIZES
        learning_rates = CIFAR100_LRS
        target = 70.0
        optimizer_cases = (
            "SGDM",
            "AM_MSGD",
            "CAUTIOUS_SGDM",
            "TAM_SGDM",
            mal_case("MAL-selected", args.mal_sgdm_config),
        )
        scheduler_values = (True,)
    elif args.experiment == "cifar100-scheduler-ablation":
        if not args.mal_sgdm_config:
            raise ValueError("--mal_sgdm_config is required for cifar100-scheduler-ablation")
        # One conventional cell from scheduled sweep c72berzj. Holding every
        # optimizer-independent hyperparameter fixed makes this a direct
        # scheduler-removal stress test instead of a second tuning sweep.
        data = "cifar100"
        arch = "resnet50"
        batch_sizes = (256,)
        learning_rates = (0.1,)
        target = 70.0
        optimizer_cases = (
            "SGDM",
            "AM_MSGD",
            "TAM_SGDM",
            mal_case("MAL-selected", args.mal_sgdm_config),
        )
        scheduler_values = (False,)
    elif args.experiment == "cifar10-scheduler-ablation":
        if not args.mal_sgdm_config:
            raise ValueError("--mal_sgdm_config is required for cifar10-scheduler-ablation")
        # The practical BS=256/LR=0.1 cell from scheduled sweep bps6rkim:
        # every method was stable, and its non-saturated AUC/accuracy gaps make
        # scheduler-removal effects measurable.  Weight decay remains 5e-4.
        data = "cifar10"
        arch = "resnet18"
        batch_sizes = (256,)
        learning_rates = (0.1,)
        target = 90.0
        optimizer_cases = (
            "SGDM",
            "AM_MSGD",
            "TAM_SGDM",
            "TAM_baseline",
            mal_case("MAL-selected", args.mal_sgdm_config),
        )
        scheduler_values = (False,)
    elif args.experiment == "cifar10-scheduled-controls":
        # bps6rkim already contains scheduled AM-MSGD, TAM-SGDM, and the
        # selected MAL-SGDM at this cell.  Only the two missing controls run.
        data = "cifar10"
        arch = "resnet18"
        batch_sizes = (256,)
        learning_rates = (0.1,)
        target = 90.0
        optimizer_cases = ("SGDM", "TAM_baseline")
        scheduler_values = (True,)
    elif args.experiment == "cifar100-tam-controls":
        # Add the fixed-gate TAM control on both sides of the exact scheduler
        # ablation cell.  The adaptive TAM/other methods come from c72berzj and
        # the paired scheduler-free sweep, so none of them is repeated here.
        data = "cifar100"
        arch = "resnet50"
        batch_sizes = (256,)
        learning_rates = (0.1,)
        target = 70.0
        optimizer_cases = ("TAM_baseline",)
        scheduler_values = (True, False)
    elif args.experiment == "cifar100-mal-structure-scheduler-free":
        # Exact recipe from scheduler-free sweep 52y7g41m, changing only one
        # MAL structural field at a time around its three matched default
        # cells.  Do not spend another three runs repeating that baseline.
        data = "cifar100"
        arch = "resnet50"
        batch_sizes = (256,)
        learning_rates = (0.1,)
        target = 70.0
        optimizer_cases = (
            mal_case("MAL-pwr0.5", T_ATT_U_P05),
            mal_case("MAL-in-place", I_ATT_U),
        )
        scheduler_values = (False,)
    else:
        raise ValueError(f"Unsupported experiment {args.experiment!r}")

    expected_runs = (
        len(optimizer_cases)
        * len(batch_sizes)
        * len(learning_rates)
        * len(SEEDS)
        * len(scheduler_values)
    )
    configuration: dict[str, Any] = {
        "program": args.program,
        "name": args.sweep_name,
        "method": "grid",
        # Even for an exhaustive grid, keeping the sweep metric validation-only
        # prevents the W&B UI from encouraging test-set model selection.
        "metric": {"name": "best_val_acc", "goal": "maximize"},
        "parameters": {
            "optimizer_case": {"values": optimizer_cases},
            "nesterov": {"values": (False,)},
            "batch_size": {"values": batch_sizes},
            "lr": {"values": learning_rates},
            "weight_decay": {"values": (WEIGHT_DECAY,)},
            "seed": {"values": SEEDS},
            "use_scheduler": {"values": scheduler_values},
            **comparison_parameters,
        },
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "--data",
            data,
            "--data_dir",
            args.data_dir,
            "--arch",
            arch,
            "--epochs",
            str(args.epochs),
            "--split_seed",
            str(args.split_seed),
            "--val_acc_target",
            str(target),
            "--amp_dtype",
            args.amp_dtype,
            "--float32_precision",
            args.float32_precision,
            "${args}",
        ],
    }
    return configuration, expected_runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program")
    parser.add_argument(
        "--experiment",
        choices=(
            "cifar10-screen",
            "cifar10-mal-qhm",
            "cifar100-benchmark",
            "cifar100-scheduler-ablation",
            "cifar10-scheduler-ablation",
            "cifar10-scheduled-controls",
            "cifar100-tam-controls",
            "cifar100-mal-structure-scheduler-free",
        ),
        required=True,
    )
    parser.add_argument("--sweep_name", "--sweep-name", required=True)
    parser.add_argument("--project_name", "--project-name", default=PROJECT_NAME)
    parser.add_argument("--data_dir", "--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--split_seed", "--split-seed", type=int, default=20260901)
    parser.add_argument("--amp_dtype", "--amp-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--float32_precision", "--float32-precision", choices=("tf32", "ieee"), default="tf32")
    parser.add_argument("--mal_sgdm_config", "--mal-sgdm-config", type=validate_sgdm_mal_config)
    args = parser.parse_args()

    configuration, expected_runs = build_configuration(args)
    sweep_id = wandb.sweep(entity=ENTITY_NAME, project=args.project_name, sweep=configuration)
    sweep_path = f"{ENTITY_NAME}/{args.project_name}/{sweep_id}"
    print(f"EXPECTED_RUNS={expected_runs}")
    print(f"Run with:\n$ uv run wandb agent --forward-signals {sweep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
