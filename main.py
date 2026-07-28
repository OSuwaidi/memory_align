import argparse
import random
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

from cautious_sgd import CAUTIOUS_SGD
from mal_sgd import MAL_SGD

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda")
WARMUP_EPOCHS = 5
NUM_WORKERS = cpu_count() // 2


def configure_cuda_precision(
    amp_dtype_name: str,
    float32_precision: str,
) -> tuple[torch.dtype, bool]:
    """Resolve AMP settings and enable TF32 for residual FP32 CUDA operations."""
    amp_dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    amp_dtype = amp_dtypes[amp_dtype_name]
    amp_enabled = amp_dtype != torch.float32

    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support bfloat16 AMP; use --amp_dtype float16.")

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
    train_loader,
    val_loader,
    run,
    lr_scheduler=None,
    label_smoothing=0.0,
    amp_dtype=torch.float16,
    amp_enabled=True,
):
    best_val_acc = 0.0
    best_train_loss = 0.0
    best_model: dict[str, Any] = {}
    best_val_epoch = 0
    # run.define_metric("epoch")
    # run.define_metric("train_loss", step_metric="epoch")
    # run.define_metric("val_acc", step_metric="epoch")
    # run.define_metric("optimizer_step")
    # run.define_metric("mal_beta/*", step_metric="optimizer_step")

    scaler = torch.amp.GradScaler(
        DEVICE.type,
        enabled=amp_enabled and amp_dtype == torch.float16,
    )

    print(f"Starting training on {next(model.parameters()).device} with {'AMP ' + str(amp_dtype) if amp_enabled else 'float32'}")
    optimizer_step = 0
    for epoch in trange(1, epochs + 1, desc="Training", unit="epoch", leave=True, position=0):
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
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

            scaler.scale(loss).backward()

            # MAL and Cautious SGD inspect gradients inside optimizer.step().
            # They must see the true gradients, not GradScaler-scaled values.
            scaler.unscale_(opt)
            scale_before_step = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            optimizer_step_succeeded = scaler.get_scale() >= scale_before_step

            if optimizer_step_succeeded:
                optimizer_step += 1

                if lr_scheduler:
                    lr_scheduler.step()

            n_batch = y.size(0)
            n_samples += n_batch
            epoch_loss += loss.item() * n_batch

        val_acc = eval_model(
            model,
            val_loader,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_loss = epoch_loss / n_samples
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_epoch = epoch

        run.log(
            {
                "train_loss": round(epoch_loss / n_samples, 2),
                "val_acc": val_acc,
                "epoch": epoch,
            },
        )

    run.summary["best_val_acc"] = round(best_val_acc, 2)
    run.summary["best_train_loss"] = round(best_train_loss, 2)
    run.summary["best_val_epoch"] = best_val_epoch
    return best_model


@torch.inference_mode()
def eval_model(
    model,
    eval_loader,
    *,
    amp_dtype=torch.float16,
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
    parser.add_argument("--data", type=str)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--nesterov", default=False)
    parser.add_argument(
        "--amp_dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="CUDA autocast dtype; float32 disables AMP.",
    )
    parser.add_argument(
        "--float32_precision",
        choices=("tf32", "ieee"),
        default="tf32",
        help="Internal precision for residual CUDA float32 matmuls/convolutions.",
    )

    # "parse_known_args" only parses CLI args that are defined above; doesn't capture/prarse all args that are present in the command
    args, unknown = parser.parse_known_args()  # W&B appends sweep configs as CLI args; ignore them here as they're captured via "run.config"

    if args.data == "cifar10":
        MEAN = (0.4914, 0.4822, 0.4465)
        STD = (0.2470, 0.2435, 0.2616)
    elif args.data == "cifar100":
        MEAN = (0.5071, 0.4865, 0.4409)
        STD = (0.2673, 0.2564, 0.2762)

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

    if args.data == "cifar10":
        DatasetCls = datasets.CIFAR10
    elif args.data == "cifar100":
        DatasetCls = datasets.CIFAR100

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
        num_workers=1,
        persistent_workers=False,
        pin_memory=True,
    )

    indices = list(range(len(raw_ds)))

    train_size = int(0.85 * len(raw_ds))  # 42,500 (~64 val images per class held out)

    # Start W&B Sweeps (W&B Sweeps injects the configs automatically):
    run = wandb.init(  # the "entity" is known from the run command, and "project" is inherited from the sweep config
        job_type="train",
        tags=("BS x LR",),
        config={
            "model": args.arch,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "beta": args.beta,
            "amp_dtype": args.amp_dtype,
            "float32_precision": args.float32_precision,
            "label_smoothing": args.label_smoothing,
        },
    )  # individual runs are forced into the parent sweep's project name

    config = run.config

    align = config.align
    nest = config.nesterov
    bs = config.batch_size
    lr = config.lr
    seed = config.seed

    amp_dtype, amp_enabled = configure_cuda_precision(
        str(args.amp_dtype),
        str(args.float32_precision),
    )

    run.name = f"{align}_nest:{str(nest)[0]}_bs:{bs}_{lr}_{seed}"

    set_seed(seed)

    if args.arch == "resnet50":
        model = resnet50(norm_layer=lambda n_channels: nn.GroupNorm(num_groups=min(32, n_channels // 4), num_channels=n_channels))
        model.conv1 = nn.Conv2d(3, model.conv1.out_channels, 3, bias=model.conv1.bias is not None)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, len(raw_ds.classes), bias=True)
    else:
        model = timm.create_model(args.arch, pretrained=False, num_classes=len(raw_ds.classes), drop_rate=0.0)

    model.to(DEVICE)

    train_indices, val_indices = train_test_split(indices, train_size=train_size, stratify=raw_ds.targets, random_state=seed)

    train_ds, val_ds = Subset(raw_ds, train_indices), Subset(raw_ds, val_indices)
    train_ds, val_ds = (
        TransformDataset(train_ds, train_transform),
        TransformDataset(val_ds, eval_transform),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
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
        num_workers=1,
        persistent_workers=False,
        pin_memory=True,
    )

    if "MAL" in align:
        if "ada" in align:
            adaptive = True
        else:
            adaptive = False

        optimizer = MAL_SGD(
            model.parameters(),
            lr=lr,
            weight_decay=args.weight_decay,
            beta=args.beta,
            adaptive=adaptive,
            nesterov=nest,
        )

    elif align == "none":
        optimizer = SGD(
            model.parameters(),
            lr=lr,
            weight_decay=args.weight_decay,
            momentum=args.beta,
            dampening=0.0,
            nesterov=nest,
        )

    elif align == "cautious":
        optimizer = CAUTIOUS_SGD(
            model.parameters(),
            lr=lr,
            weight_decay=args.weight_decay,
            beta=args.beta,
            nesterov=nest,
        )

    else:
        raise ValueError(f"The given alignment method {align} is not valid.")

    steps_per_epoch = len(train_loader)
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
        train_loader,
        val_loader,
        run,
        lr_scheduler=scheduler,
        label_smoothing=args.label_smoothing,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
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
