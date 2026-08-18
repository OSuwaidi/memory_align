import argparse
import random
import signal
import sys
from multiprocessing import cpu_count
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
import wandb
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.models import resnet18, resnet50
from torchvision.transforms import v2
from tqdm.auto import tqdm, trange

from cautious_opt import CAUTIOUS_SGD
from mal_opt import MAL_SGDM
from sweeps.cifar_resnet_sweep import add_training_args

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda")
WARMUP_EPOCHS = 5
BETA = 0.9
NUM_GPUS = torch.cuda.device_count()
NUM_WORKERS = min(cpu_count() // (2 * NUM_GPUS), 16)
MAX_MICRO_BATCH_SIZE = 512


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
    torch.backends.cudnn.conv.fp32_precision = float32_precision

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
    best_val_acc = 0.0
    best_train_loss = 0.0
    AUC = 0.0
    rise = 0.0
    best_model: dict[str, Any] = {}
    best_val_epoch = 0
    epochs_to_target = -1
    target_reached = False

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
                "train_loss": round(epoch_loss / n_samples, 2),
                "val_acc": val_acc,
                "epoch": epoch,
                "AUC": AUC / epochs,
                "rise": rise,
            },
        )

    run.summary["target_reached"] = target_reached
    run.summary["epochs_2_target"] = epochs_to_target
    run.summary["best_val_acc"] = round(best_val_acc, 2)
    run.summary["best_train_loss"] = round(best_train_loss, 2)
    run.summary["best_val_epoch"] = best_val_epoch
    return best_model


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

    acc = 100.0 * correct / total
    return round(acc, 2)


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
        num_workers=min(cpu_count() // (4 * NUM_GPUS), 6),
        persistent_workers=False,
        pin_memory=True,
    )

    indices = list(range(len(raw_ds)))

    train_size = int(0.85 * len(raw_ds))  # 42,500 (~64 val images per class held out)

    # Start W&B Sweeps (W&B Sweep injects its parameters as configs automatically):
    run = wandb.init(  # the "entity" is known from the `wandb` run command, and "project" is inherited from the sweep config
        job_type="train",
        tags=("MAL per vs tensor",),
        config={
            "data": args.data,
            "model": args.arch,
            "epochs": args.epochs,
            "val_acc_target": args.val_acc_target,
            "beta": BETA,
            "label_smoothing": label_smoothing,
            "amp_dtype": args.amp_dtype,
            "float32_precision": args.float32_precision,
        },
    )

    # Handle system-level interruptions (e.g., SLURM, AWS Spot, manual kills)
    def handle_preemption(signum, frame):
        print(f"Received signal {signum}; re-queueing sweep run...")

        run.mark_preempting()

        exit_code = 128 + signum
        run.finish(exit_code=exit_code)
        sys.exit(exit_code)

    signal.signal(signal.SIGTERM, handle_preemption)
    signal.signal(signal.SIGUSR1, handle_preemption)
    signal.signal(signal.SIGINT, handle_preemption)  # Ctrl+C

    run.define_metric("*", step_metric="epoch")  # let epoch be the default x-axis for all metrics

    config = run.config

    align = config.align
    nest = config.nesterov
    bs = config.batch_size
    lr = config.lr
    weight_decay = config.weight_decay
    seed = config.seed
    in_place, pwr = config.inplace_pwr.split(",")
    run.config.update({"in_place": in_place, "pwr": pwr}, allow_val_change=True)
    scale = config.scale
    per_unit = config.per_unit

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

    run.name = f"{align}_inp:{str(in_place)[0]}_pwr:{pwr}_scl:{str(scale)[0]}_per:{str(per_unit)[0]}_nest:{str(nest)[0]}_bs:{bs}_{lr}_{seed}"

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
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, len(raw_ds.classes), bias=True)

    model.to(DEVICE)

    train_indices, val_indices = train_test_split(indices, train_size=train_size, stratify=raw_ds.targets, random_state=seed)

    train_ds, val_ds = Subset(raw_ds, train_indices), Subset(raw_ds, val_indices)
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
        num_workers=min(cpu_count() // (4 * NUM_GPUS), 6),
        persistent_workers=False,
        pin_memory=True,
    )

    if align == "MAL":
        optimizer = MAL_SGDM(
            model.parameters(),
            lr=lr,
            beta=BETA,
            weight_decay=weight_decay,
            in_place=in_place == "True",
            scale=scale,
            pwr=float(pwr),
            nesterov=nest,
            per_unit=per_unit,
        )

    elif align == "none":
        optimizer = SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=BETA,
            dampening=0.0,
            nesterov=nest,
        )

    elif align == "cautious":
        optimizer = CAUTIOUS_SGD(
            model.parameters(),
            lr=lr,
            beta=BETA,
            weight_decay=weight_decay,
            nesterov=nest,
        )

    else:
        raise ValueError(f'The given alignment method "{align}" is not valid')

    steps_per_epoch = len(train_loader) // grad_accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)

    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps), eta_min=1e-5)

    # Combine schedulers sequentially at the iteration level
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    best_model = train_val_model(
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

    model.load_state_dict(best_model)
    test_acc = eval_model(
        model,
        test_loader,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )
    run.summary["test_acc"] = round(test_acc, 2)

    run.finish(exit_code=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
