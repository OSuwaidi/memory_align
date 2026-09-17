import argparse
import os
import random
import signal
import sys
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.models import resnet18, resnet50
from torchvision.transforms import v2
from tqdm.auto import tqdm, trange
from wandb.sdk.internal.internal_api import Api as WandbInternalApi

# W&B executes this file by path, making ``tasks/`` (rather than the repository
# root) Python's import root. Add the repository root before importing siblings.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from optims.am_opt import AM_MSGD, AM_AdamW
from optims.cautious_opt import C_SGDM, C_AdamW
from optims.agam_opt import AGAM_SGD, AGAM_AdamW
from optims.tam_opt import TAM_SGDM, AdaTAMW, TAMBaselineSGDM
from sweeps.cifar_resnet_sweep import add_training_args
from tasks.wandb_metadata import task_metadata

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda")
WARMUP_EPOCHS = 5
BETA = 0.9
NUM_GPUS = torch.cuda.device_count()
ALLOCATED_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
NUM_WORKERS = min(max(ALLOCATED_CPUS // max(NUM_GPUS, 1), 1), 16)
EVAL_NUM_WORKERS = min(NUM_WORKERS, 6)
MAX_MICRO_BATCH_SIZE = 512
DEFAULT_MAL_SGDM_CONFIG = "False,1.0,False,attenuate"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f'Expected a boolean value, got "{value}".')


def configure_cuda_precision(
    amp_dtype_name: str,
    float32_precision: str,
) -> tuple[torch.dtype, bool]:
    """Resolve AMP settings and enable TF32 for residual FP32 CUDA operations."""
    amp_dtypes = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    amp_dtype = amp_dtypes[amp_dtype_name]
    amp_enabled = amp_dtype != torch.float32

    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support bfloat16 AMP; use --amp_dtype float32.")

    # These are the post-PyTorch-2.9 replacements for the deprecated
    # cuda.matmul.allow_tf32 and cudnn.allow_tf32 flags.
    torch.backends.cuda.matmul.fp32_precision = float32_precision
    torch.backends.cudnn.conv.fp32_precision = float32_precision  # pyright: ignore[reportAttributeAccessIssue]

    return amp_dtype, amp_enabled


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    elif torch.mps.is_available():
        torch.mps.manual_seed(seed)


# Must be defined on the global scope to be picklable and accessible to workers
def set_worker_seed(worker_id):
    worker_seed = torch.initial_seed() % 2**32  # PyTorch auto increments its seed (internally) to get a unique seed per worker: "torch.initial_seed()" reflects that
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def train_val_model(
    model,
    opt,
    epochs,
    val_acc_target,
    train_loader,
    val_loader,
    run,
    lr_scheduler=None,
    label_smoothing=0.0,
    amp_dtype=torch.bfloat16,
    amp_enabled=True,
    grad_accumulation_steps=1,
):
    # Make the first finite epoch selectable even in intentionally hostile
    # heatmap cells whose validation accuracy is exactly zero.
    best_val_acc = -1.0
    best_train_loss = 0.0
    AUC = 0.0
    rise = 0.0
    best_model: dict[str, Any] = {}
    best_val_epoch = 0
    epochs_to_target = epochs + 1
    target_reached = False
    diverged = False
    divergence_epoch = None

    print(f"Starting training on {next(model.parameters()).device} with {'AMP ' + str(amp_dtype) if amp_enabled else 'float32'}")
    optimizer_step = 0
    for epoch in trange(1, epochs + 1, desc="Training", unit="epoch", leave=True, position=0):
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        # Match drop_last=True at the effective-batch level: incomplete groups of
        # micro-batches must not produce a smaller optimizer update.
        micro_batches_per_epoch = (len(train_loader) // grad_accumulation_steps) * grad_accumulation_steps
        opt.zero_grad(set_to_none=True)
        for micro_batch_index, (x, y) in enumerate(train_loader, start=1):
            if micro_batch_index > micro_batches_per_epoch:
                break

            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(
                DEVICE.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                loss = F.cross_entropy(
                    model(x),
                    y,
                    label_smoothing=label_smoothing,
                )

            if not torch.isfinite(loss):
                diverged = True
                divergence_epoch = epoch
                print(f"Non-finite loss at epoch {epoch}, micro-batch {micro_batch_index}; marking the run diverged.")
                break

            (loss / grad_accumulation_steps).backward()

            if micro_batch_index % grad_accumulation_steps == 0:
                opt.step()
                optimizer_step += 1

                if lr_scheduler:
                    lr_scheduler.step()

                opt.zero_grad(set_to_none=True)

            n_batch = y.size(0)
            n_samples += n_batch
            epoch_loss += loss.item() * n_batch

        if diverged:
            break

        val_acc = eval_model(
            model,
            val_loader,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )

        if not target_reached and val_acc >= val_acc_target:
            epochs_to_target = epoch
            target_reached = True

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_loss = epoch_loss / n_samples
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_epoch = epoch

        AUC += val_acc
        rise += val_acc / epoch

        run.log(
            {
                "train_loss": epoch_loss / n_samples,
                "val_acc": val_acc,
                "train/loss": epoch_loss / n_samples,
                "val/acc": val_acc,
                "val/auc_so_far": AUC / epoch,
                "epoch": epoch,
                "AUC": AUC / epochs,
                "rise": rise,
                "lr": opt.param_groups[0]["lr"],
            },
        )

    run.summary["target_reached"] = int(target_reached)
    run.summary["epochs_2_target"] = epochs_to_target
    run.summary["best_val_acc"] = best_val_acc
    run.summary["best_train_loss"] = best_train_loss
    run.summary["best_val_epoch"] = best_val_epoch
    run.summary["val_auc"] = AUC / epochs
    run.summary["diverged"] = int(diverged)
    run.summary["selection_val_acc"] = 0.0 if diverged else best_val_acc
    run.summary["best/val_acc"] = best_val_acc
    run.summary["best/train_loss"] = best_train_loss
    run.summary["best/epoch"] = best_val_epoch
    run.summary["val/auc"] = AUC / epochs
    if divergence_epoch is not None:
        run.summary["divergence_epoch"] = divergence_epoch
    return best_model, diverged


def resolve_optimizer_case(config) -> tuple[str, str, str]:
    """Resolve a non-Cartesian optimizer case used by the heatmap sweeps.

    A combined case prevents irrelevant MAL configurations from multiplying
    every non-MAL optimizer in a W&B grid.  Legacy sweeps that provide separate
    ``optimizer`` and ``MAL_config`` keys remain supported.
    """
    raw_mal_config = str(config.get("MAL_config", DEFAULT_MAL_SGDM_CONFIG))
    optimizer_case = str(config.get("optimizer_case", "")).strip()
    if not optimizer_case:
        return str(config.optimizer), raw_mal_config, str(config.optimizer)

    fields = optimizer_case.split("::", maxsplit=2)
    if len(fields) == 1:
        return fields[0], raw_mal_config, fields[0]
    if len(fields) != 3 or fields[0] != "MAL_SGDM":
        raise ValueError("optimizer_case must be an optimizer name or 'MAL_SGDM::<variant-label>::<MAL_config>'.")
    return fields[0], fields[2], fields[1]


def parse_mal_config(value: str) -> dict[str, Any]:
    """Parse current MAL configs while retaining false legacy safeguard fields."""
    fields = [field.strip() for field in value.split(",")]
    if not 4 <= len(fields) <= 8:
        raise ValueError(
            "MAL_config must be "
            "'in_place,pwr,scale,gate_mode[,align[,gradient_weight_mode[,unbias]]]'."
        )

    in_place_text, pwr_text, scale_text, gate_mode, *tail = fields
    align = None
    gradient_weight_mode = "fixed"
    unbias = "none"
    if tail and tail[0].lower() in {"true", "false"}:
        legacy_safeguard = parse_bool(tail.pop(0))
        if legacy_safeguard:
            raise ValueError("MAL descent_safeguard was removed and can no longer be enabled.")
    if tail:
        align = tail.pop(0).lower()
    if tail:
        gradient_weight_mode = tail.pop(0).lower()
    if tail:
        unbias = tail.pop(0).lower()
    if tail:
        raise ValueError("MAL_config contains too many fields.")

    scale_key = scale_text.lower()
    if scale_key in {"true", "false"}:
        scale: bool | str = parse_bool(scale_text)
    elif scale_key in {"none", "step", "moment"}:
        scale = scale_key
    else:
        raise ValueError("MAL scale must be True, False, none, step, or moment.")

    parsed: dict[str, Any] = {
        "in_place": parse_bool(in_place_text),
        "pwr": float(pwr_text),
        "scale": scale,
        "gate_mode": gate_mode,
        "gradient_weight_mode": gradient_weight_mode,
        "unbias": unbias,
    }
    if parsed["pwr"] not in (0.5, 1.0):
        raise ValueError("MAL pwr must be 0.5 or 1.0.")
    if gate_mode not in ("attenuate", "replace"):
        raise ValueError("MAL gate_mode must be attenuate or replace.")
    if gradient_weight_mode not in ("fixed", "complement"):
        raise ValueError("MAL gradient_weight_mode must be fixed or complement.")
    if gradient_weight_mode == "complement" and gate_mode != "attenuate":
        raise ValueError('MAL gradient_weight_mode="complement" requires gate_mode="attenuate".')
    if unbias not in ("none", "buffer", "estimator"):
        raise ValueError("MAL unbias must be none, buffer, or estimator.")
    if gradient_weight_mode == "fixed" and unbias != "none":
        raise ValueError('MAL unbias requires gradient_weight_mode="complement".')
    if align is not None:
        if align not in ("update", "metric", "white", "moment"):
            raise ValueError("MAL align must be update, metric, white, or moment.")
        parsed["align"] = align
    return parsed


def split_weight_decay_params(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Match the repository optimizers' bias/norm weight-decay exemption."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim <= 1 else decay).append(parameter)
    groups: list[dict[str, Any]] = []
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    return groups


def build_lr_scheduler(
    optimizer: Optimizer,
    *,
    use_scheduler: bool,
    total_steps: int,
    warmup_steps: int,
) -> SequentialLR | None:
    """Build the step-wise warmup/cosine schedule without touching constant-LR runs."""
    if not use_scheduler:
        return None
    if warmup_steps <= 0 or total_steps <= warmup_steps:
        raise ValueError("Scheduled training requires 0 < warmup_steps < total_steps.")

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-5,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )


@torch.inference_mode()
def eval_model(
    model,
    eval_loader,
    *,
    amp_dtype=torch.bfloat16,
    amp_enabled=True,
) -> float:
    model.eval()
    correct = 0
    total = 0

    for x, y in tqdm(eval_loader, unit="batch", leave=False, position=1):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(
            DEVICE.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds.eq_(y)).sum().item()
        total += y.size(0)

    return 100.0 * correct / total


class TransformDataset(Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.T = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return self.T(x), y


def main():
    parser = argparse.ArgumentParser()
    add_training_args(parser)
    # "parse_known_args" only parses CLI args that are defined above; it doesn't capture/parse all args that are present in the actual command
    args, _unknown = parser.parse_known_args()  # W&B appends sweep params as CLI args; ignore them here as they're captured via "run.config"

    if not torch.cuda.is_available():
        raise RuntimeError("This training entry point requires a CUDA-capable PyTorch environment.")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    if args.data == "cifar10":
        DatasetCls = datasets.CIFAR10
        MEAN = (0.4914, 0.4822, 0.4465)
        STD = (0.2470, 0.2435, 0.2616)
        label_smoothing = 0.0
        args.arch = "resnet18"

    elif args.data == "cifar100":
        DatasetCls = datasets.CIFAR100
        MEAN = (0.5071, 0.4865, 0.4409)
        STD = (0.2673, 0.2564, 0.2762)
        label_smoothing = 0.1
    else:
        raise ValueError(f'The given dataset "{args.data}" is not valid')

    train_transform = v2.Compose(
        [
            v2.PILToTensor(),
            v2.RandomCrop(32, padding=4, padding_mode="reflect"),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandAugment(num_ops=2, magnitude=9),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(MEAN, STD),
            v2.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ]
    )

    eval_transform = v2.Compose(
        [
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(MEAN, STD),
        ]
    )

    raw_ds = DatasetCls(
        root=args.data_dir,
        train=True,
        download=True,
    )
    test_ds = DatasetCls(
        root=args.data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1000,
        shuffle=False,
        num_workers=EVAL_NUM_WORKERS,
        persistent_workers=False,
        pin_memory=True,
    )

    indices = list(range(len(raw_ds)))

    train_size = int(0.85 * len(raw_ds))  # 42,500 train; 7,500 validation (75/class for CIFAR-100).

    # Start W&B Sweeps (W&B Sweep injects its parameters as configs automatically):
    run = wandb.init(  # the "entity" is known from the `wandb` run command, and "project" is inherited from the sweep config
        job_type="train",
        tags=("optimizer-benchmark", "cifar"),
        config={
            **task_metadata(
                task="cifar_image_classification",
                task_type="supervised_image_classification",
                model_name=args.arch,
                model_source="torchvision",
                dataset_name=args.data,
                dataset_config="official_train_85_15_validation_official_test",
                dataset_source="torchvision",
                training_regime="supervised_from_scratch",
            ),
            "data": args.data,
            "model": args.arch,
            "epochs": args.epochs,
            "val_acc_target": args.val_acc_target,
            "beta": BETA,
            "label_smoothing": label_smoothing,
            "amp_dtype": args.amp_dtype,
            "float32_precision": args.float32_precision,
            "split_seed": args.split_seed,
        },
    )

    def handle_interruption(signum, frame):
        manual_stop = signum == signal.SIGINT
        if signum == signal.SIGTERM:
            try:
                manual_stop = WandbInternalApi().check_stop_requested(run.project, run.entity, run.id)
            except Exception as error:  # noqa: BLE001 - W&B internals expose heterogeneous errors.
                print(
                    f"Could not check W&B's stop flag ({error!r}); preserving the config by requeueing it.",
                    file=sys.stderr,
                )

        action = "skipping config" if manual_stop else "re-queueing config"
        print(f"Received signal {signum}; {action}...")

        if not manual_stop:
            run.mark_preempting()

        exit_code = 0 if manual_stop else 128 + signum
        run.finish(exit_code=exit_code)
        sys.exit(exit_code)

    signal.signal(signal.SIGTERM, handle_interruption)
    signal.signal(signal.SIGUSR1, handle_interruption)
    signal.signal(signal.SIGINT, handle_interruption)

    run.define_metric("*", step_metric="epoch")  # let epoch be the default x-axis for all metrics

    config = run.config

    optimizer, raw_mal_config, optimizer_variant = resolve_optimizer_case(config)
    nest = parse_bool(config.get("nesterov", False))
    bs = config.batch_size
    lr = config.lr
    weight_decay = config.weight_decay
    seed = config.seed
    use_scheduler = parse_bool(config.use_scheduler)
    if use_scheduler and args.epochs <= WARMUP_EPOCHS:
        parser.error(f"--epochs must be greater than {WARMUP_EPOCHS} warmup epochs when scheduling is enabled")

    mal_config = parse_mal_config(raw_mal_config)
    mal_align = str(mal_config.pop("align", "moment" if optimizer == "MAL_SGDM" else "metric"))
    optimizer_mal_config = dict(mal_config)
    if optimizer == "MAL_SGDM":
        if isinstance(optimizer_mal_config["scale"], str):
            if optimizer_mal_config["scale"] == "moment":
                raise ValueError('MAL-SGDM does not support scale="moment".')
            optimizer_mal_config["scale"] = optimizer_mal_config["scale"] == "step"
    elif optimizer == "MAL_AdamW":
        # ``unbias`` is an SGDM-QHM option. MAL-AdamW always performs its exact
        # first-moment coefficient normalization internally.
        sgdm_unbias = optimizer_mal_config.pop("unbias")
        if sgdm_unbias != "none":
            raise ValueError("MAL-AdamW does not accept the MAL-SGDM unbias modes.")
    run.config.update(
        {
            "optimizer": optimizer,
            "optimizer_variant": optimizer_variant,
            "MAL_config": raw_mal_config,
            "mal_align": mal_align,
            **{f"mal_{key}": value for key, value in mal_config.items()},
            # Preserve these historical top-level fields for existing W&B
            # analyses while the mal_* names remain unambiguous across families.
            **{key: mal_config[key] for key in ("in_place", "pwr", "scale", "gate_mode")},
        },
        allow_val_change=True,
    )
    if optimizer == "AM_MSGD":
        run.config.update({"am_beta_max": BETA, "am_model_lambda": 0.1}, allow_val_change=True)
        run.name = f"{optimizer}_bmax:{BETA}_lambda:0.1_bs:{bs}_{lr}_{seed}"
    elif optimizer == "MAL_SGDM":
        run.name = (
            f"{optimizer}_{optimizer_variant}_inp:{int(mal_config['in_place'])}"
            f"_pwr:{mal_config['pwr']}_scl:{str(mal_config['scale']).lower()}"
            f"_gate:{mal_config['gate_mode']}_gw:{mal_config['gradient_weight_mode']}"
            f"_ub:{mal_config['unbias']}_nest:{int(nest)}_bs:{bs}_{lr}_{seed}"
        )
    elif optimizer == "MAL_AdamW":
        run.name = (
            f"{optimizer}_inp:{int(mal_config['in_place'])}_pwr:{mal_config['pwr']}"
            f"_scl:{str(mal_config['scale']).lower()}_gate:{mal_config['gate_mode']}"
            f"_a:{mal_align}_gw:{mal_config['gradient_weight_mode']}_bs:{bs}_{lr}_{seed}"
        )
    elif optimizer == "TAM_SGDM":
        run.config.update(
            {"tam_adaptive_gate": True, "tam_gamma": 0.9, "tam_torque_eps": 1e-8},
            allow_val_change=True,
        )
        run.name = f"{optimizer}_adaptive_bs:{bs}_{lr}_{seed}"
    elif optimizer == "TAM_baseline":
        run.config.update(
            {"tam_adaptive_gate": False, "tam_gradient_scale": 0.5},
            allow_val_change=True,
        )
        run.name = f"{optimizer}_gscale:0.5_bs:{bs}_{lr}_{seed}"
    else:
        run.name = f"{optimizer}_nest:{str(nest)[0]}_bs:{bs}_{lr}_{seed}"
    if bs >= 2 * MAX_MICRO_BATCH_SIZE:
        if bs % MAX_MICRO_BATCH_SIZE != 0:
            raise ValueError(f"Batch size {bs} must be divisible by {MAX_MICRO_BATCH_SIZE} for gradient accumulation.")
        micro_batch_size = MAX_MICRO_BATCH_SIZE
        grad_accumulation_steps = bs // MAX_MICRO_BATCH_SIZE
    else:
        micro_batch_size = bs
        grad_accumulation_steps = 1

    amp_dtype, amp_enabled = configure_cuda_precision(
        args.amp_dtype,
        args.float32_precision,
    )

    set_seed(seed)

    if args.arch == "resnet18":
        model = resnet18
    elif args.arch == "resnet50":
        model = resnet50
    else:
        raise ValueError(f'Architecture "{args.arch}" is not defined.')

    model = model(norm_layer=lambda n_channels: nn.GroupNorm(num_groups=min(32, n_channels // 4), num_channels=n_channels))
    model.conv1 = nn.Conv2d(3, model.conv1.out_channels, 3, bias=model.conv1.bias is not None, padding=1)
    nn.init.kaiming_normal_(
        model.conv1.weight,
        mode="fan_out",
        nonlinearity="relu",
    )
    model.maxpool = nn.Identity()  # pyright: ignore[reportAttributeAccessIssue]
    model.fc = nn.Linear(model.fc.in_features, len(raw_ds.classes), bias=True)

    model.to(DEVICE)

    sgd_optimizer_names = {
        "SGDM",
        "AM_MSGD",
        "CAUTIOUS_SGDM",
        "TAM_SGDM",
        "TAM_baseline",
        "MAL_SGDM",
    }
    run.config.update(
        {
            "optimizer_family": "sgdm" if optimizer in sgd_optimizer_names else "adamw",
            "effective_batch_size": int(bs),
            "micro_batch_size": int(micro_batch_size),
            "gradient_accumulation_steps": int(grad_accumulation_steps),
            "base_learning_rate": float(lr),
            "learning_rate": float(lr),
            "train_examples": int(train_size),
            "validation_examples": int(len(raw_ds) - train_size),
            "test_examples": len(test_ds),
            "num_classes": len(raw_ds.classes),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        },
        allow_val_change=True,
    )

    train_indices, val_indices = train_test_split(
        indices,
        train_size=train_size,
        stratify=raw_ds.targets,
        random_state=args.split_seed,
    )

    train_ds = Subset(raw_ds, [int(index) for index in train_indices])
    val_ds = Subset(raw_ds, [int(index) for index in val_indices])
    train_ds, val_ds = (
        TransformDataset(train_ds, train_transform),
        TransformDataset(val_ds, eval_transform),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,  # torch pickles "worker_init_fn" + dataset + all its transforms and sends serialized copy to each worker
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
        drop_last=True,  # a final tiny batch is too noisy and can throw the model off, especially with BatchNorm
        worker_init_fn=set_worker_seed,
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1000,
        shuffle=False,
        num_workers=EVAL_NUM_WORKERS,
        persistent_workers=False,
        pin_memory=True,
    )

    if optimizer == "MAL_SGDM":
        optimizer = AGAM_SGD(
            model.parameters(),
            lr=lr,
            beta=BETA,
            weight_decay=weight_decay,
            nesterov=nest,
            **optimizer_mal_config,
        )

    elif optimizer == "MAL_AdamW":
        optimizer = AGAM_AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            align=mal_align,
            **optimizer_mal_config,
        )

    elif optimizer == "AM_AdamW":
        model_lambda = 0.1
        optimizer = AM_AdamW(
            model.parameters(),
            lr=lr,
            betas=(BETA - 0.1 * model_lambda, 0.999),
            model_lambda=model_lambda,
            weight_decay=weight_decay,
        )

    elif optimizer == "SGDM":
        optimizer = SGD(
            split_weight_decay_params(model, weight_decay),
            lr=lr,
            weight_decay=0.0,
            momentum=BETA,
            dampening=0.0,
            nesterov=nest,
        )

    elif optimizer == "AM_MSGD":
        if nest:
            raise ValueError("AM-MSGD does not define a Nesterov variant.")
        optimizer = AM_MSGD(
            model.parameters(),
            lr=lr,
            beta_max=BETA,
            model_lambda=0.1,
            weight_decay=weight_decay,
        )

    elif optimizer == "CAUTIOUS_SGDM":
        optimizer = C_SGDM(
            model.parameters(),
            lr=lr,
            beta=BETA,
            weight_decay=weight_decay,
            nesterov=nest,
        )

    elif optimizer == "CAUTIOUS_AdamW":
        optimizer = C_AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    elif optimizer == "TAM_SGDM":
        optimizer = TAM_SGDM(
            model.parameters(),
            lr=lr,
            beta=BETA,
            weight_decay=weight_decay,
        )

    elif optimizer == "TAM_baseline":
        optimizer = TAMBaselineSGDM(
            model.parameters(),
            lr=lr,
            beta=BETA,
            gradient_scale=0.5,
            weight_decay=weight_decay,
        )

    elif optimizer == "AdaTAMW":
        optimizer = AdaTAMW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(f'The given optimizerment method "{optimizer}" is not valid')

    steps_per_epoch = len(train_loader) // grad_accumulation_steps
    if steps_per_epoch <= 0:
        parser.error("The effective batch size exceeds the available training split.")
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS

    scheduler = build_lr_scheduler(
        optimizer,
        use_scheduler=use_scheduler,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    best_model, diverged = train_val_model(
        model,
        optimizer,
        args.epochs,
        args.val_acc_target,
        train_loader,
        val_loader,
        run,
        lr_scheduler=scheduler,
        label_smoothing=label_smoothing,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
        grad_accumulation_steps=grad_accumulation_steps,
    )

    if diverged:
        # Keep numerically unstable cells in the completed heatmap with an
        # explicit zero score, rather than crashing the W&B grid or evaluating
        # an arbitrary pre-divergence checkpoint.
        test_acc_at_final_epoch = 0.0
        test_acc_at_best_val = 0.0
    else:
        if not best_model:
            raise RuntimeError("Training completed without producing a validation checkpoint.")
        test_acc_at_final_epoch = eval_model(
            model,
            test_loader,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        model.load_state_dict(best_model)
        test_acc_at_best_val = eval_model(
            model,
            test_loader,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
    run.summary.update(
        {
            # Backward-compatible aliases denote the validation-selected
            # checkpoint; selection never uses the test set.
            "test_acc": test_acc_at_best_val,
            "test/acc": test_acc_at_best_val,
            "test_acc_at_final_epoch": test_acc_at_final_epoch,
            "test/acc_at_final_epoch": test_acc_at_final_epoch,
            "test_acc_at_best_val": test_acc_at_best_val,
            "test/acc_at_best_val": test_acc_at_best_val,
        }
    )

    run.finish(exit_code=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
