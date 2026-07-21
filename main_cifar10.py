from typing import Any
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.models import resnet18
import numpy as np
import random
from multiprocessing import cpu_count
from tqdm.auto import trange, tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.transforms import v2
from sgd import create_sgd_optimizer
import wandb
import argparse
import timm

# -------------------------
# Config
# -------------------------
DEVICE = "cuda"
WARMUP_EPOCHS = 5
NUM_WORKERS = cpu_count() // 4


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
    worker_seed = (
            torch.initial_seed() % 2 ** 32
    )  # PyTorch auto increments its seed (internally) to get a unique seed per worker: "torch.initial_seed()" reflects that
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def train_val(model, opt, epochs, train_loader, val_loader, run, lr_scheduler=None):
    best_val_acc = 0.0
    best_train_loss = 0.0
    best_model: dict[str, Any] = {}
    best_val_epoch = 0

    print(f"Starting training on GPU: {next(model.parameters()).get_device()}")
    for epoch in trange(1, epochs + 1, desc="Training", unit="epoch", leave=True, position=0):
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            n_batch = y.size(0)
            n_samples += n_batch
            epoch_loss += loss.item() * n_batch
            if lr_scheduler:
                lr_scheduler.step()

        val_acc = eval_model(model, val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_loss = epoch_loss / n_samples
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_epoch = epoch

        run.log(
                dict(train_loss=round(epoch_loss / n_samples, 2), val_acc=val_acc),
                step=epoch,
                )

    run.summary["final_val_acc"] = round(val_acc, 2)
    run.summary["best_val_acc"] = round(best_val_acc, 2)
    run.summary["best_train_loss"] = round(best_train_loss, 2)
    run.summary["best_val_epoch"] = best_val_epoch
    return best_model


@torch.inference_mode()
def eval_model(model, eval_loader) -> float:
    model.eval()
    correct = 0
    total = 0

    for x, y in tqdm(eval_loader, unit="batch", leave=False, position=1):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
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
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    # "parse_known_args" only parses CLI args that are defined above; doesn't capture/prarse all args that are present in the command
    args, unknown = parser.parse_known_args()  # W&B appends sweep configs as CLI args; ignore them here as they're captured via "run.config"

    train_transform = v2.Compose(
            [
                v2.PILToTensor(),
                v2.RandomCrop(32, padding=4, padding_mode="reflect"),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandAugment(num_ops=2, magnitude=9),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                v2.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                ]
            )

    eval_transform = v2.Compose(
            [
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                ]
            )

    raw_ds = datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=True,
            )
    indices = list(range(len(raw_ds)))

    train_size = int(0.85 * len(raw_ds))  # 42,500

    test_ds = datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            download=True,
            transform=eval_transform,
            )
    test_loader = DataLoader(
            test_ds,
            batch_size=500,
            shuffle=False,
            num_workers=2,
            persistent_workers=False,
            pin_memory=True,
            )

    # Start W&B Sweeps (W&B Sweeps injects the configs automatically):
    run = wandb.init(  # the "entity" is known from the run command, and "project" is inherited from the sweep config
            job_type="train",
            tags=("batch_sizes",),
            config=dict(
                    model=args.arch,
                    epochs=args.epochs,
                    weight_decay=args.weight_decay,
                    couple=True,
                    tau=0.0,
                    ),
            )  # individual runs are forced into the parent sweep's project name

    config = run.config

    align = config.align
    per = config.per
    bs = config.batch_size
    lr = config.lr
    seed = config.seed
    optimizer_impl = "mem_align_sgd" if align else "torch_sgdm"
    config.update(dict(optimizer=optimizer_impl), allow_val_change=True)

    f = lambda truth: str(truth)[0]

    run.name = f"align:{f(align)}_per:{f(per)}_bs:{bs}_{lr}_{seed}"

    set_seed(seed)

    if args.arch == "resnet18":
        model = resnet18(
                norm_layer=lambda n_channels: nn.GroupNorm(
                        num_groups=min(32, n_channels // 4), num_channels=n_channels
                        )
                )
        model.conv1 = nn.Conv2d(3, 64, 3, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, len(raw_ds.classes), bias=True)
    else:
        model = timm.create_model(
                args.arch, pretrained=False, num_classes=len(raw_ds.classes), drop_rate=0.0
                )

    model.to(DEVICE)

    train_indices, val_indices = train_test_split(
            indices, train_size=train_size, stratify=raw_ds.targets, random_state=seed
            )

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
            drop_last=True,  # a final small batchsize (4 here) is too noisy and can throw the model off, especially with BatchNorm
            worker_init_fn=set_worker_seed,
            generator=torch.Generator().manual_seed(seed),
            )

    val_loader = DataLoader(
            val_ds,
            batch_size=500,
            shuffle=False,
            num_workers=2,
            persistent_workers=False,
            pin_memory=True,
            )

    optimizer = create_sgd_optimizer(
            model.parameters(),
            lr=lr,
            weight_decay=args.weight_decay,
            mem_align=align,
            per=per,
            tau=0.0,
            )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS

    warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
            )

    cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=(total_steps - warmup_steps), eta_min=1e-6
            )

    # Combine schedulers sequentially at the iteration level
    scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
            )

    best_model = train_val(
            model, optimizer, args.epochs, train_loader, val_loader, run, lr_scheduler=scheduler
            )

    model.load_state_dict(best_model)
    test_acc = eval_model(model, test_loader)
    run.summary["test_acc"] = round(test_acc, 2)

    run.finish(exit_code=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
