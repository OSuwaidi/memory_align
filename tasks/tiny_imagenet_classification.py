"""Supervised Tiny-ImageNet confirmation task for MAL structural selection.

The 100,000 official training images are split once into 90,000 training and
10,000 validation images with a fixed stratified split.  Tiny-ImageNet's
official labelled validation set is held out as the final test set.  This lets
the sweep select checkpoints by validation accuracy while retaining an honest
generalization metric.

The task supports two deliberately different, realistic confirmation settings:

* ResNet-50 from scratch with MAL-SGDM at 64 px.
* ImageNet-pretrained timm ViT-Tiny fine-tuning with MAL-AdamW at 224 px.

Optimizer hyperparameters come only from ``wandb.run.config``.  Training-recipe
arguments are ordinary CLI options supplied by the sweep command.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
import wandb
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode, v2
from tqdm.auto import tqdm, trange

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tasks.mae_pretrain import (
    MAL_ALIGN_CHOICES,
    TinyImageNetAnnotatedVal,
    TransformView,
    build_optimizer,
    parse_bool,
    parse_mal_config,
    resolve_tiny_imagenet_root,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_ARCHITECTURES = {"resnet50", "vit_tiny_patch16_224"}
SUPPORTED_OPTIMIZERS = {"MAL_SGDM", "MAL_AdamW"}
REQUIRED_SWEEP_KEYS = frozenset(("optimizer", "MAL_config", "batch_size", "weight_decay", "seed", "use_scheduler"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_worker_seed(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def configure_precision(
    device: torch.device,
    amp_dtype_name: str,
    float32_precision: str,
) -> tuple[torch.dtype, bool]:
    amp_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[amp_dtype_name]
    amp_enabled = amp_dtype != torch.float32
    if device.type != "cuda":
        raise RuntimeError("This confirmation task requires CUDA.")
    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support bfloat16 AMP.")
    torch.backends.cuda.matmul.fp32_precision = float32_precision
    torch.backends.cudnn.conv.fp32_precision = float32_precision  # pyright: ignore[reportAttributeAccessIssue]
    return amp_dtype, amp_enabled


def build_transforms(image_size: int) -> tuple[Any, Any]:
    resize_size = int(round(image_size * 256 / 224))
    train_transform = v2.Compose(
        (
            v2.RandomResizedCrop(
                (image_size, image_size),
                scale=(0.5, 1.0),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(),
            v2.RandAugment(num_ops=2, magnitude=9, interpolation=InterpolationMode.BICUBIC),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            v2.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        )
    )
    eval_transform = v2.Compose(
        (
            v2.Resize(resize_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop((image_size, image_size)),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        )
    )
    return train_transform, eval_transform


def build_datasets(
    data_dir: str | Path,
    *,
    image_size: int,
    split_seed: int,
) -> tuple[Any, Any, Any, int, Path]:
    root = resolve_tiny_imagenet_root(data_dir)
    raw_train = ImageFolder(root / "train")
    if len(raw_train) != 100_000 or len(raw_train.classes) != 200:
        raise ValueError("Expected the complete 100,000-image, 200-class Tiny-ImageNet training set.")

    indices = np.arange(len(raw_train))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=0.1,
        random_state=split_seed,
        shuffle=True,
        stratify=raw_train.targets,
    )
    train_targets = np.asarray(raw_train.targets)[train_indices]
    validation_targets = np.asarray(raw_train.targets)[validation_indices]
    if not np.all(np.bincount(train_targets, minlength=200) == 450):
        raise RuntimeError("The fixed training split is not balanced at 450 images per class.")
    if not np.all(np.bincount(validation_targets, minlength=200) == 50):
        raise RuntimeError("The fixed validation split is not balanced at 50 images per class.")

    official_validation = TinyImageNetAnnotatedVal(root / "val", raw_train.class_to_idx)
    train_transform, eval_transform = build_transforms(image_size)
    train_dataset = TransformView(Subset(raw_train, train_indices.tolist()), train_transform)
    validation_dataset = TransformView(Subset(raw_train, validation_indices.tolist()), eval_transform)
    test_dataset = TransformView(official_validation, eval_transform)
    return train_dataset, validation_dataset, test_dataset, len(raw_train.classes), root


def build_model(arch: str, *, num_classes: int, pretrained: bool) -> nn.Module:
    if arch not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f'Unsupported architecture "{arch}".')
    if arch == "resnet50" and pretrained:
        raise ValueError("The ResNet-50 confirmation recipe is intentionally trained from scratch.")
    if arch == "vit_tiny_patch16_224" and not pretrained:
        raise ValueError("The ViT-Tiny confirmation recipe intentionally fine-tunes an ImageNet-pretrained backbone.")
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)


def cosine_warmup_lr(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return peak_lr * step / warmup_steps
    progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer: Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    examples = 0
    for images, targets in tqdm(loader, desc="Evaluation", unit="batch", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
        loss_sum += loss.item() * targets.size(0)
        correct += logits.argmax(dim=1).eq(targets).sum().item()
        examples += targets.size(0)
    return loss_sum / examples, 100.0 * correct / examples


def validate_config(args: argparse.Namespace, config: Any, parser: argparse.ArgumentParser) -> None:
    missing = REQUIRED_SWEEP_KEYS.difference(config.keys())
    if missing:
        parser.error(f"Missing W&B sweep parameter(s): {', '.join(sorted(missing))}.")
    if str(config.optimizer) not in SUPPORTED_OPTIMIZERS:
        parser.error(f"optimizer must be one of {sorted(SUPPORTED_OPTIMIZERS)}.")
    if str(config.optimizer) == "MAL_SGDM" and "lr" not in config:
        parser.error("MAL_SGDM requires the sweep parameter lr.")
    if str(config.optimizer) == "MAL_AdamW" and "base_lr" not in config:
        parser.error("MAL_AdamW requires the sweep parameter base_lr.")
    if args.epochs <= 0 or args.warmup_epochs < 0 or args.epochs <= args.warmup_epochs:
        parser.error("epochs must be positive and greater than warmup_epochs.")
    if int(config.batch_size) <= 0 or args.max_micro_batch_size <= 0:
        parser.error("batch sizes must be positive.")
    micro_batch_size = min(int(config.batch_size), args.max_micro_batch_size)
    if int(config.batch_size) % micro_batch_size:
        parser.error("The effective batch size must be divisible by the selected micro-batch size.")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_dir", "--data-dir", default="./data/tiny-imagenet-200")
    parser.add_argument("--arch", choices=sorted(SUPPORTED_ARCHITECTURES), required=True)
    parser.add_argument("--pretrained", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--image_size", "--image-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--warmup_epochs", "--warmup-epochs", type=int, default=5)
    parser.add_argument("--min_lr", "--min-lr", type=float, default=0.0)
    parser.add_argument("--val_acc_target", "--val-acc-target", type=float, required=True)
    parser.add_argument("--split_seed", "--split-seed", type=int, default=20260901)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--label_smoothing", "--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max_micro_batch_size", "--max-micro-batch-size", type=int, default=256)
    parser.add_argument("--eval_batch_size", "--eval-batch-size", type=int, default=512)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp_dtype", "--amp-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--float32_precision", "--float32-precision", choices=("tf32", "ieee"), default="tf32")
    parser.add_argument("--wandb_project", "--wandb-project", default=None)
    parser.add_argument("--wandb_entity", "--wandb-entity", default=None)
    parser.add_argument("--wandb_mode", "--wandb-mode", choices=("online", "offline", "disabled"), default=None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args, _unknown = parser.parse_known_args()

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        job_type="tiny-imagenet-confirmation",
        config=vars(args),
        tags=("mal-confirmatory", "tiny-imagenet", "supervised", args.arch),
    )
    config = run.config
    validate_config(args, config, parser)

    optimizer_name = str(config.optimizer)
    batch_size = int(config.batch_size)
    weight_decay = float(config.weight_decay)
    seed = int(config.seed)
    use_scheduler = parse_bool(config.use_scheduler)
    nesterov = parse_bool(config.get("nesterov", False))
    device = torch.device(args.device)
    amp_dtype, amp_enabled = configure_precision(device, args.amp_dtype, args.float32_precision)
    set_seed(seed)

    train_dataset, validation_dataset, test_dataset, num_classes, data_root = build_datasets(
        args.data_dir,
        image_size=args.image_size,
        split_seed=args.split_seed,
    )
    model = build_model(args.arch, num_classes=num_classes, pretrained=args.pretrained).to(device)

    if optimizer_name == "MAL_SGDM":
        nominal_lr = float(config.lr)
        actual_lr = nominal_lr
    else:
        nominal_lr = float(config.base_lr)
        actual_lr = nominal_lr * batch_size / 256.0

    raw_mal_config = str(config.MAL_config)
    mal_config = parse_mal_config(raw_mal_config)
    mal_align = str(mal_config.pop("align", "metric")).lower()
    if mal_align not in MAL_ALIGN_CHOICES:
        parser.error(f"MAL align must be one of {sorted(MAL_ALIGN_CHOICES)}.")
    optimizer = build_optimizer(
        optimizer_name,
        model,
        lr=actual_lr,
        weight_decay=weight_decay,
        momentum=args.momentum,
        beta2=args.beta2,
        nesterov=nesterov,
        mal_config=mal_config,
        mal_align=mal_align,
    )

    micro_batch_size = min(batch_size, args.max_micro_batch_size)
    accumulation_steps = batch_size // micro_batch_size
    num_workers = min(cpu_count(), 16) if args.num_workers < 0 else args.num_workers
    eval_workers = min(num_workers, 6)
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        worker_init_fn=set_worker_seed,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=eval_workers,
        persistent_workers=False,
        pin_memory=True,
        worker_init_fn=set_worker_seed,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=eval_workers,
        persistent_workers=False,
        pin_memory=True,
        worker_init_fn=set_worker_seed,
        generator=torch.Generator().manual_seed(seed + 2),
    )
    steps_per_epoch = len(train_loader) // accumulation_steps
    if steps_per_epoch <= 0:
        parser.error("The effective batch size exceeds the available training set.")

    run.config.update(
        {
            "MAL_config": raw_mal_config,
            "actual_lr": actual_lr,
            "nominal_lr": nominal_lr,
            "micro_batch_size": micro_batch_size,
            "accumulation_steps": accumulation_steps,
            "steps_per_epoch": steps_per_epoch,
            "num_classes": num_classes,
            "train_examples": len(train_dataset),
            "validation_examples": len(validation_dataset),
            "test_examples": len(test_dataset),
            "resolved_data_dir": str(data_root),
            "mal_align": mal_align,
            **{f"mal_{key}": value for key, value in mal_config.items()},
        },
        allow_val_change=True,
    )
    run.name = (
        f"{optimizer_name}_{args.arch}_inp{int(mal_config['in_place'])}_p{mal_config['pwr']:g}"
        f"_scl{str(mal_config['scale']).lower()}_g{mal_config['gate_mode']}_a{mal_align}"
        f"_sched{int(use_scheduler)}_lr{nominal_lr:g}_s{seed}"
    )
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")
    run.define_metric("lr", step_metric="epoch")

    print(
        f"Training {args.arch} on Tiny-ImageNet: {len(train_dataset):,} train / "
        f"{len(validation_dataset):,} validation / {len(test_dataset):,} test. "
        f"Optimizer={optimizer_name}, config={raw_mal_config}, scheduler={use_scheduler}, "
        f"effective batch={batch_size} ({micro_batch_size} x {accumulation_steps}), "
        f"nominal LR={nominal_lr:g}, actual LR={actual_lr:g}."
    )

    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    best_val_acc = -math.inf
    best_val_loss = math.inf
    best_val_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    validation_accuracies: list[float] = []
    epoch_to_target = args.epochs + 1
    optimizer_step = 0
    exit_code = 0
    try:
        for epoch in trange(1, args.epochs + 1, desc="Supervised confirmation", unit="epoch"):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            examples = 0
            usable_micro_batches = (len(train_loader) // accumulation_steps) * accumulation_steps
            current_lr = actual_lr
            for micro_batch_index, (images, targets) in enumerate(train_loader, start=1):
                if micro_batch_index > usable_micro_batches:
                    break
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                    loss = F.cross_entropy(model(images), targets, label_smoothing=args.label_smoothing)
                (loss / accumulation_steps).backward()

                if micro_batch_index % accumulation_steps == 0:
                    optimizer_step += 1
                    if use_scheduler:
                        current_lr = cosine_warmup_lr(
                            optimizer_step,
                            total_steps=total_steps,
                            warmup_steps=warmup_steps,
                            peak_lr=actual_lr,
                            min_lr=args.min_lr,
                        )
                        set_optimizer_lr(optimizer, current_lr)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                examples += targets.size(0)
                loss_sum += loss.item() * targets.size(0)

            train_loss = loss_sum / examples
            val_loss, val_acc = evaluate(
                model,
                validation_loader,
                device=device,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            validation_accuracies.append(val_acc)
            if epoch_to_target == args.epochs + 1 and val_acc >= args.val_acc_target:
                epoch_to_target = epoch
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_val_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

            run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/auc_so_far": float(np.mean(validation_accuracies)),
                    "lr": current_lr,
                }
            )
            run.summary.update(
                {
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "best_val_epoch": best_val_epoch,
                    "epoch_to_target": epoch_to_target,
                    "target_reached": int(epoch_to_target <= args.epochs),
                }
            )

        if not best_state:
            raise RuntimeError("Training completed without a validation checkpoint.")
        model.load_state_dict(best_state)
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            device=device,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        run.summary.update(
            {
                "final_val_acc": validation_accuracies[-1],
                "val_auc": float(np.mean(validation_accuracies)),
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
    except BaseException:
        exit_code = 1
        raise
    finally:
        run.finish(exit_code=exit_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
