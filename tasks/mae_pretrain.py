"""MAE pre-training for a patch-8 timm ViT-Tiny on Tiny-ImageNet.

The MAE encoder/decoder and recipe follow the official facebookresearch/mae
implementation, adapted to Tiny-ImageNet's 64 px images and this repository's
optimizer family.  The encoder itself is assembled from timm's
``vit_tiny_patch16_224`` implementation with ``patch_size=8``.

Populate the default data path first with
``uv run download_datasets.py --task tiny-imagenet``.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from collections.abc import Iterable
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, cast

import numpy as np
import timm
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from timm.data import Mixup
from timm.layers import DropPath
from timm.layers.patch_embed import PatchEmbed
from timm.loss import SoftTargetCrossEntropy
from timm.models.vision_transformer import Block, VisionTransformer
from timm.optim.lars import Lars
from torch import nn
from torch.optim import SGD, AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import InterpolationMode, v2
from tqdm.auto import tqdm, trange

# W&B executes this file by path, making ``tasks/`` (rather than the repository
# root) Python's import root. Add the repository root before importing siblings.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from optims.agam_opt import AGAM_SGD, AGAM_AdamW
from optims.am_opt import AM_MSGD, AM_AdamW
from optims.cautious_opt import C_SGDM, C_AdamW
from optims.lion_opt import AGAM_Lion, Lion
from optims.tam_opt import TAM_SGDM, AdaTAMW
from tasks.wandb_metadata import task_metadata

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_ARCH = "vit_tiny_patch16_224"
SGD_OPTIMIZERS = {"SGDM", "AM_MSGD", "CAUTIOUS_SGDM", "TAM_SGDM", "MAL_SGDM"}
ADAMW_OPTIMIZERS = {"AdamW", "AM_AdamW", "CAUTIOUS_AdamW", "AdaTAMW", "MAL_AdamW"}
LION_OPTIMIZERS = {"Lion", "AGAM_Lion"}
ALL_OPTIMIZERS = SGD_OPTIMIZERS | ADAMW_OPTIMIZERS | LION_OPTIMIZERS
MAL_ALIGN = "white"  # Backward-compatible fallback for five-field sweep configs.
MAL_ALIGN_CHOICES = frozenset(("update", "metric", "white", "moment"))
REQUIRED_SWEEP_KEYS = frozenset(("optimizer", "batch_size", "base_lr", "weight_decay", "seed", "use_scheduler"))
MAL_CONFIG_KEYS = ("MAL_config", "mal_config")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f'expected a boolean value, got "{value}"')


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


def configure_precision(device: torch.device, amp_dtype_name: str, float32_precision: str) -> tuple[torch.dtype, bool]:
    amp_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[amp_dtype_name]
    amp_enabled = amp_dtype != torch.float32

    if device.type != "cuda" and amp_enabled:
        raise RuntimeError("bfloat16 AMP is supported by this entry point only on CUDA; use --amp_dtype float32.")
    if device.type == "cuda":
        if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not support bfloat16 AMP; use --amp_dtype float32.")
        torch.backends.cuda.matmul.fp32_precision = float32_precision
        torch.backends.cudnn.conv.fp32_precision = float32_precision  # pyright: ignore[reportAttributeAccessIssue]

    return amp_dtype, amp_enabled


def _get_1d_sincos_pos_embed(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("A sine/cosine position-embedding dimension must be even.")
    frequencies = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    frequencies = 1.0 / (10_000**frequencies)
    angles = np.einsum("m,d->md", positions.reshape(-1), frequencies)
    return np.concatenate((np.sin(angles), np.cos(angles)), axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, include_cls_token: bool = True) -> np.ndarray:
    if embed_dim % 4 != 0:
        raise ValueError("A 2D sine/cosine position-embedding dimension must be divisible by four.")
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, -1)
    embedding = np.concatenate(
        (
            _get_1d_sincos_pos_embed(embed_dim // 2, grid[0]),
            _get_1d_sincos_pos_embed(embed_dim // 2, grid[1]),
        ),
        axis=1,
    )
    if include_cls_token:
        embedding = np.concatenate((np.zeros((1, embed_dim)), embedding), axis=0)
    return embedding


class MaskedAutoencoderViT(nn.Module):
    """MAE with a timm ViT encoder and a lightweight transformer decoder."""

    def __init__(
        self,
        *,
        arch: str = SUPPORTED_ARCH,
        image_size: int = 64,
        patch_size: int = 8,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 4,
        decoder_num_heads: int = 4,
        norm_pix_loss: bool = True,
    ) -> None:
        super().__init__()
        if arch != SUPPORTED_ARCH:
            raise ValueError(f'Only "{SUPPORTED_ARCH}" is supported, got "{arch}".')
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        if decoder_embed_dim % decoder_num_heads != 0:
            raise ValueError("decoder_embed_dim must be divisible by decoder_num_heads.")

        encoder = cast(
            VisionTransformer,
            timm.create_model(
                arch,
                pretrained=False,
                img_size=image_size,
                patch_size=patch_size,
                num_classes=0,
                global_pool="",
            ),
        )
        if encoder.num_prefix_tokens != 1 or encoder.reg_token is not None:
            raise RuntimeError(f'Architecture "{arch}" does not expose the single-class-token ViT layout expected by MAE.')

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = 3
        self.encoder_embed_dim = cast(int, encoder.embed_dim)
        self.patch_embed = cast(PatchEmbed, encoder.patch_embed)
        self.cls_token = cast(nn.Parameter, encoder.cls_token)
        self.pos_embed = cast(nn.Parameter, encoder.pos_embed)
        self.pos_embed.requires_grad_(False)
        self.blocks = cast(nn.Sequential, encoder.blocks)
        self.norm = cast(nn.LayerNorm, encoder.norm)

        num_patches = self.patch_embed.num_patches
        self.decoder_embed = nn.Linear(self.encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        decoder_norm = partial(nn.LayerNorm, eps=1e-6)
        self.decoder_blocks = nn.Sequential(
            *[
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=decoder_norm,  # pyright: ignore[reportArgumentType]
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = decoder_norm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * self.in_channels, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    @property
    def num_patches(self) -> int:
        return self.patch_embed.num_patches

    def initialize_weights(self) -> None:
        grid_size = int(math.sqrt(self.num_patches))
        if grid_size * grid_size != self.num_patches:
            raise ValueError("This MAE implementation requires a square patch grid.")

        encoder_pos = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        decoder_pos = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(encoder_pos).float().unsqueeze(0))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos).float().unsqueeze(0))

        patch_weight = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(patch_weight.view(patch_weight.shape[0], -1))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        n, channels, height, width = images.shape
        if channels != self.in_channels or height != width or height % p != 0:
            raise ValueError(f"Expected square RGB images divisible by patch size {p}, got {tuple(images.shape)}.")
        grid = height // p
        patches = images.reshape(n, channels, grid, p, grid, p)
        patches = torch.einsum("nchpwq->nhwpqc", patches)
        return patches.reshape(n, grid * grid, p * p * channels)

    def random_masking(
        self,
        tokens: torch.Tensor,
        mask_ratio: float,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, length, dim = tokens.shape
        len_keep = int(length * (1.0 - mask_ratio))
        noise = torch.rand(n, length, device=tokens.device, generator=generator)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        visible = torch.gather(tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, dim))

        mask = torch.ones((n, length), device=tokens.device, dtype=tokens.dtype)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return visible, mask, ids_restore

    def forward_encoder(
        self,
        images: torch.Tensor,
        mask_ratio: float,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.patch_embed(images)
        tokens = tokens + self.pos_embed[:, 1:, :]
        tokens, mask, ids_restore = self.random_masking(tokens, mask_ratio, generator=generator)

        cls_token = (self.cls_token + self.pos_embed[:, :1, :]).expand(images.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = self.blocks(tokens)
        return self.norm(tokens), mask, ids_restore

    def forward_decoder(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        tokens = self.decoder_embed(latent)
        num_mask_tokens = ids_restore.shape[1] + 1 - tokens.shape[1]
        mask_tokens = self.mask_token.expand(tokens.shape[0], num_mask_tokens, -1)
        patch_tokens = torch.cat((tokens[:, 1:, :], mask_tokens), dim=1)
        patch_tokens = torch.gather(
            patch_tokens,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
        )
        tokens = torch.cat((tokens[:, :1, :], patch_tokens), dim=1)
        tokens = self.decoder_blocks(tokens + self.decoder_pos_embed)
        tokens = self.decoder_norm(tokens)
        return self.decoder_pred(tokens)[:, 1:, :]

    def forward_loss(self, images: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify(images)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            variance = target.var(dim=-1, keepdim=True)
            target = (target - mean) / torch.sqrt(variance + 1e-6)
        loss_per_patch = (prediction - target).square().mean(dim=-1)
        return (loss_per_patch * mask).sum() / mask.sum()

    def encode_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return the unmasked, frozen-probe representation (the MAE class token)."""
        tokens = self.patch_embed(images) + self.pos_embed[:, 1:, :]
        cls_token = (self.cls_token + self.pos_embed[:, :1, :]).expand(images.shape[0], -1, -1)
        tokens = self.blocks(torch.cat((cls_token, tokens), dim=1))
        return self.norm(tokens)[:, 0]

    def forward(
        self,
        images: torch.Tensor,
        mask_ratio: float = 0.75,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(images, mask_ratio, generator=generator)
        prediction = self.forward_decoder(latent, ids_restore)
        return self.forward_loss(images, prediction, mask), prediction, mask


class MAEEncoderClassifier(nn.Module):
    """MAE encoder adapted to the official global-pooling fine-tune protocol."""

    def __init__(
        self,
        mae: MaskedAutoencoderViT,
        *,
        num_classes: int,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        self.patch_embed = mae.patch_embed
        self.cls_token = mae.cls_token
        self.pos_embed = mae.pos_embed
        self.blocks = mae.blocks
        self.fc_norm = nn.LayerNorm(mae.encoder_embed_dim, eps=1e-6)
        self.head = nn.Linear(mae.encoder_embed_dim, num_classes)

        # The MAE positional embedding is fixed during SSL pre-training but the
        # complete ViT backbone is trainable during end-to-end fine-tuning.
        self.pos_embed.requires_grad_(True)
        drop_path_rates = torch.linspace(0.0, drop_path_rate, len(self.blocks)).tolist()
        for block, rate in zip(self.blocks, drop_path_rates, strict=True):
            block.drop_path1 = DropPath(rate) if rate > 0.0 else nn.Identity()
            block.drop_path2 = DropPath(rate) if rate > 0.0 else nn.Identity()

        nn.init.ones_(self.fc_norm.weight)
        nn.init.zeros_(self.fc_norm.bias)
        nn.init.trunc_normal_(self.head.weight, std=2e-5)
        nn.init.zeros_(self.head.bias)
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(images) + self.pos_embed[:, 1:, :]
        cls_token = (self.cls_token + self.pos_embed[:, :1, :]).expand(images.shape[0], -1, -1)
        tokens = self.blocks(torch.cat((cls_token, tokens), dim=1))
        pooled = tokens[:, 1:, :].mean(dim=1)
        return self.head(self.fc_norm(pooled))


class TransformView(Dataset):
    """Apply a transform while sharing the underlying image/label dataset."""

    def __init__(self, dataset: Any, transform: Any) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, target = self.dataset[index]
        return self.transform(image), target


class TinyImageNetAnnotatedVal(Dataset):
    """Read Tiny-ImageNet's original flat ``val/images`` layout."""

    def __init__(self, val_dir: Path, class_to_idx: dict[str, int]) -> None:
        annotation_file = val_dir / "val_annotations.txt"
        image_dir = val_dir / "images"
        if not annotation_file.is_file() or not image_dir.is_dir():
            raise FileNotFoundError(f"Expected {annotation_file} and {image_dir}.")

        labels: dict[str, int] = {}
        with annotation_file.open(encoding="utf-8") as annotations:
            for line in annotations:
                filename, wnid, *_ = line.rstrip("\n").split("\t")
                if wnid not in class_to_idx:
                    raise ValueError(f'Validation annotation contains unknown class "{wnid}".')
                labels[filename] = class_to_idx[wnid]

        self.samples = [(image_dir / filename, labels[filename]) for filename in sorted(labels)]
        missing = [str(path) for path, _ in self.samples if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Validation annotation references a missing image: {missing[0]}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        path, target = self.samples[index]
        return default_loader(str(path)), target


def resolve_tiny_imagenet_root(data_dir: str | Path) -> Path:
    root = Path(data_dir).expanduser().resolve()
    candidates = (root, root / "tiny-imagenet-200")
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate
    raise FileNotFoundError(f'Could not find Tiny-ImageNet under "{root}". Expected train/ and val/ either there or under tiny-imagenet-200/.')


def build_transforms(image_size: int) -> tuple[Any, Any]:
    train_transform = v2.Compose(
        (
            v2.RandomResizedCrop(
                (image_size, image_size),
                scale=(0.2, 1.0),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        )
    )
    eval_transform = v2.Compose(
        (
            v2.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        )
    )
    return train_transform, eval_transform


def build_datasets(data_dir: str | Path, image_size: int) -> tuple[Any, Any, Any, int, Path]:
    root = resolve_tiny_imagenet_root(data_dir)
    train_transform, eval_transform = build_transforms(image_size)
    raw_train = ImageFolder(root / "train")
    if len(raw_train.classes) < 2:
        raise ValueError(f"Expected at least two Tiny-ImageNet classes under {root / 'train'}.")

    val_dir = root / "val"
    if (val_dir / "val_annotations.txt").is_file():
        raw_val: Any = TinyImageNetAnnotatedVal(val_dir, raw_train.class_to_idx)
    else:
        reorganized_val = ImageFolder(val_dir)
        if reorganized_val.class_to_idx != raw_train.class_to_idx:
            raise ValueError("The reorganized validation classes do not match the training class-to-index mapping.")
        raw_val = reorganized_val

    pretrain_train = TransformView(raw_train, train_transform)
    probe_train = TransformView(raw_train, eval_transform)
    validation = TransformView(raw_val, eval_transform)
    return pretrain_train, probe_train, validation, len(raw_train.classes), root


def build_finetune_transforms(image_size: int) -> tuple[Any, Any]:
    """Strong train augmentation and deterministic evaluation for MAE fine-tuning."""
    train_transform = v2.Compose(
        (
            v2.RandomResizedCrop(
                (image_size, image_size),
                scale=(0.08, 1.0),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(),
            v2.RandAugment(num_ops=2, magnitude=9, interpolation=InterpolationMode.BICUBIC),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        )
    )
    eval_transform = v2.Compose(
        (
            v2.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        )
    )
    return train_transform, eval_transform


def layerwise_finetune_param_groups(
    model: MAEEncoderClassifier,
    *,
    weight_decay: float,
    layer_decay: float,
) -> list[dict[str, Any]]:
    """Create MAE-style layer-wise AdamW groups, from embeddings to head."""
    final_layer_id = len(model.blocks) + 1
    grouped: dict[tuple[int, bool], dict[str, Any]] = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in {"cls_token", "pos_embed"} or name.startswith("patch_embed."):
            layer_id = 0
        elif name.startswith("blocks."):
            block_index = int(name.split(".", 2)[1])
            layer_id = block_index + 1
        else:
            layer_id = final_layer_id

        no_decay = parameter.ndim <= 1 or name in {"cls_token", "pos_embed"}
        key = (layer_id, no_decay)
        if key not in grouped:
            grouped[key] = {
                "params": [],
                "weight_decay": 0.0 if no_decay else weight_decay,
                "lr_scale": layer_decay ** (final_layer_id - layer_id),
            }
        grouped[key]["params"].append(parameter)

    groups = list(grouped.values())
    grouped_parameters = sum(len(group["params"]) for group in groups)
    trainable_parameters = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    if grouped_parameters != trainable_parameters:
        raise RuntimeError("Fine-tune parameter grouping omitted or duplicated parameters.")
    return groups


@torch.inference_mode()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> tuple[float, float, float]:
    model.eval()
    loss_sum = 0.0
    top1_correct = 0
    top5_correct = 0
    examples = 0
    for images, targets in tqdm(loader, desc="Fine-tune validation", unit="batch", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
        topk = min(5, logits.shape[1])
        predictions = logits.topk(topk, dim=1).indices
        matches = predictions.eq(targets.unsqueeze(1))
        batch_size = targets.shape[0]
        loss_sum += loss.item() * batch_size
        top1_correct += matches[:, :1].sum().item()
        top5_correct += matches.sum().item()
        examples += batch_size
    return (
        loss_sum / examples,
        100.0 * top1_correct / examples,
        100.0 * top5_correct / examples,
    )


def run_end_to_end_finetune(
    mae: MaskedAutoencoderViT,
    raw_train_dataset: Any,
    raw_val_dataset: Any,
    *,
    num_classes: int,
    image_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    epochs: int,
    batch_size: int,
    max_micro_batch_size: int,
    base_lr: float,
    minimum_lr: float,
    warmup_epochs: int,
    weight_decay: float,
    layer_decay: float,
    drop_path_rate: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    label_smoothing: float,
    num_workers: int,
    seed: int,
    run: Any,
) -> dict[str, float | int]:
    """Fine-tune the complete final MAE encoder with one fixed evaluator recipe."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    classifier = MAEEncoderClassifier(
        mae,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    ).to(device)
    train_transform, eval_transform = build_finetune_transforms(image_size)
    train_dataset = TransformView(raw_train_dataset, train_transform)
    val_dataset = TransformView(raw_val_dataset, eval_transform)

    micro_batch_size = min(batch_size, max_micro_batch_size)
    if batch_size % micro_batch_size:
        raise ValueError("Fine-tune batch size must be divisible by its micro-batch size.")
    accumulation_steps = batch_size // micro_batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
        worker_init_fn=set_worker_seed,
        generator=torch.Generator().manual_seed(seed),
    )
    eval_workers = min(num_workers, 6)
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(1024, batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=eval_workers,
        persistent_workers=False,
        pin_memory=device.type == "cuda",
        worker_init_fn=set_worker_seed,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    steps_per_epoch = len(train_loader) // accumulation_steps
    if steps_per_epoch <= 0:
        raise ValueError("Fine-tune effective batch size exceeds the training set.")

    peak_lr = base_lr * batch_size / 256.0
    parameter_groups = layerwise_finetune_param_groups(
        classifier,
        weight_decay=weight_decay,
        layer_decay=layer_decay,
    )
    optimizer = AdamW(parameter_groups, lr=peak_lr, betas=(0.9, 0.999))
    mixup = Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=label_smoothing,
        num_classes=num_classes,
    )
    criterion = SoftTargetCrossEntropy()
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    optimizer_step = 0
    best_top1 = -math.inf
    best_top5 = 0.0
    best_loss = math.inf
    best_epoch = 0
    final_loss = math.inf
    final_top1 = 0.0
    final_top5 = 0.0

    # Isolate all stochastic downstream choices from the completed SSL run.
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        for finetune_epoch in trange(1, epochs + 1, desc="End-to-end fine-tuning", unit="epoch"):
            classifier.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss_sum = 0.0
            train_examples = 0
            current_lr = 0.0
            usable_micro_batches = steps_per_epoch * accumulation_steps

            for micro_batch_index, (images, targets) in enumerate(train_loader, start=1):
                if micro_batch_index > usable_micro_batches:
                    break
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                images, soft_targets = mixup(images, targets)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    loss = criterion(classifier(images), soft_targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite fine-tune loss at epoch {finetune_epoch}, batch {micro_batch_index}: {loss.item()}"
                    )
                (loss / accumulation_steps).backward()

                if micro_batch_index % accumulation_steps == 0:
                    current_lr = cosine_warmup_lr(
                        optimizer_step,
                        total_steps=total_steps,
                        warmup_steps=warmup_steps,
                        peak_lr=peak_lr,
                        min_lr=minimum_lr,
                    )
                    set_optimizer_lr(optimizer, current_lr)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

                micro_batch_examples = targets.shape[0]
                train_loss_sum += loss.detach().item() * micro_batch_examples
                train_examples += micro_batch_examples

            train_loss = train_loss_sum / train_examples
            final_loss, final_top1, final_top5 = evaluate_classifier(
                classifier,
                val_loader,
                device=device,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            if final_top1 > best_top1 or (final_top1 == best_top1 and final_loss < best_loss):
                best_top1 = final_top1
                best_top5 = final_top5
                best_loss = final_loss
                best_epoch = finetune_epoch

            run.log(
                {
                    "finetune/epoch": finetune_epoch,
                    "finetune/train_loss": train_loss,
                    "finetune/val_loss": final_loss,
                    "finetune/val_top1_pct": final_top1,
                    "finetune/val_top5_pct": final_top5,
                    "finetune/lr": current_lr,
                }
            )

    return {
        "finetune/final_val_loss": final_loss,
        "finetune/final_val_top1_pct": final_top1,
        "finetune/final_val_top5_pct": final_top5,
        "finetune/best_val_loss_at_best_top1": best_loss,
        "finetune/best_val_top1_pct": best_top1,
        "finetune/best_val_top5_pct_at_best_top1": best_top5,
        "finetune/best_epoch": best_epoch,
    }


def split_weight_decay_params(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
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


def parse_mal_config(value: str) -> dict[str, Any]:
    """Parse the compact MAL sweep representation.

    The current format is ``in_place,pwr,scale,gate_mode[,align[,gradient_weight_mode]]``.
    Historical strings containing a fifth, false ``descent_safeguard`` field
    remain readable so completed sweeps can be reproduced, but the removed
    option cannot be enabled.
    """
    fields = [field.strip() for field in value.split(",")]
    if not 4 <= len(fields) <= 7:
        raise ValueError("MAL_config must be 'in_place,pwr,scale,gate_mode[,align[,gradient_weight_mode]]'.")

    in_place_text, pwr_text, scale_text, gate_mode, *tail = fields
    align = None
    gradient_weight_mode = "fixed"

    # Backward compatibility for old ...,[descent_safeguard[,align]] strings.
    if tail and tail[0].lower() in {"true", "false"} and parse_bool(tail.pop(0)):
        raise ValueError("MAL descent_safeguard was removed and can no longer be enabled.")
    if tail:
        align = tail.pop(0).lower()
    if tail:
        gradient_weight_mode = tail.pop(0).lower()
    if tail:
        raise ValueError("MAL_config contains too many fields.")

    scale_key = scale_text.strip().lower()
    if scale_key in {"true", "false"}:
        scale: bool | str = parse_bool(scale_text)
    elif scale_key in {"none", "step", "moment"}:
        scale = scale_key
    else:
        raise ValueError("MAL scale must be True, False, none, step, or moment.")

    config = {
        "in_place": parse_bool(in_place_text),
        "pwr": float(pwr_text),
        "scale": scale,
        "gate_mode": gate_mode,
        "gradient_weight_mode": gradient_weight_mode,
    }
    if config["pwr"] not in (0.5, 1.0):
        raise ValueError("MAL pwr must be 0.5 or 1.0.")
    if gate_mode not in ("replace", "attenuate"):
        raise ValueError("MAL gate_mode must be replace or attenuate.")
    if gradient_weight_mode not in ("fixed", "complement"):
        raise ValueError("MAL gradient_weight_mode must be fixed or complement.")
    if gradient_weight_mode == "complement" and gate_mode != "attenuate":
        raise ValueError('MAL gradient_weight_mode="complement" requires gate_mode="attenuate".')
    if align is not None:
        align = align.strip().lower()
        if align not in MAL_ALIGN_CHOICES:
            raise ValueError(f"MAL align must be one of {sorted(MAL_ALIGN_CHOICES)}.")
        config["align"] = align
    return config


def build_optimizer(
    name: str,
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    beta2: float,
    nesterov: bool,
    mal_config: dict[str, Any],
    mal_align: str,
) -> Optimizer:
    if name not in ALL_OPTIMIZERS:
        raise ValueError(f'Unknown optimizer "{name}". Choose one of {sorted(ALL_OPTIMIZERS)}.')
    if name not in {"SGDM", "CAUTIOUS_SGDM", "MAL_SGDM"} and nesterov:
        raise ValueError(f'Optimizer "{name}" does not implement the requested Nesterov variant.')

    parameters: Iterable[nn.Parameter] = model.parameters()
    if name == "SGDM":
        return SGD(
            split_weight_decay_params(model, weight_decay),
            lr=lr,
            momentum=momentum,
            dampening=0.0,
            nesterov=nesterov,
        )
    if name == "AM_MSGD":
        return AM_MSGD(parameters, lr=lr, beta_max=momentum, model_lambda=0.1, weight_decay=weight_decay)
    if name == "CAUTIOUS_SGDM":
        return C_SGDM(parameters, lr=lr, beta=momentum, weight_decay=weight_decay, nesterov=nesterov)
    if name == "TAM_SGDM":
        return TAM_SGDM(parameters, lr=lr, beta=momentum, weight_decay=weight_decay)
    if name == "MAL_SGDM":
        sgdm_mal_config = dict(mal_config)
        gradient_weight_mode = sgdm_mal_config.pop("gradient_weight_mode", "fixed")
        if gradient_weight_mode != "fixed":
            raise ValueError("MAL-SGDM does not implement gradient_weight_mode; use fixed.")
        if isinstance(sgdm_mal_config["scale"], str):
            if sgdm_mal_config["scale"] == "moment":
                raise ValueError('MAL-SGDM does not support scale="moment".')
            sgdm_mal_config["scale"] = sgdm_mal_config["scale"] == "step"
        return AGAM_SGD(
            parameters,
            lr=lr,
            beta=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            **sgdm_mal_config,
        )
    if name == "AdamW":
        return AdamW(split_weight_decay_params(model, weight_decay), lr=lr, betas=(momentum, beta2))
    if name == "AM_AdamW":
        model_lambda = 0.1
        beta1_max = momentum - 0.1 * model_lambda
        return AM_AdamW(
            parameters,
            lr=lr,
            betas=(beta1_max, beta2),
            weight_decay=weight_decay,
            model_lambda=model_lambda,
        )
    if name == "CAUTIOUS_AdamW":
        return C_AdamW(parameters, lr=lr, betas=(momentum, beta2), weight_decay=weight_decay)
    if name == "AdaTAMW":
        return AdaTAMW(parameters, lr=lr, betas=(momentum, beta2), weight_decay=weight_decay)
    if name == "MAL_AdamW":
        return AGAM_AdamW(
            parameters,
            lr=lr,
            betas=(momentum, beta2),
            weight_decay=weight_decay,
            align=mal_align,
            **mal_config,
        )
    if name == "Lion":
        return Lion(
            split_weight_decay_params(model, weight_decay),
            lr=lr,
            betas=(momentum, beta2),
            weight_decay=weight_decay,
        )
    if name == "AGAM_Lion":
        return AGAM_Lion(
            split_weight_decay_params(model, weight_decay),
            lr=lr,
            betas=(momentum, beta2),
            weight_decay=weight_decay,
            pwr=1.0,
        )
    raise AssertionError(f'Optimizer dispatch is incomplete for "{name}".')


def cosine_warmup_lr(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * step / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer: Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)


def train_one_epoch(
    model: MaskedAutoencoderViT,
    optimizer: Optimizer,
    loader: DataLoader,
    *,
    device: torch.device,
    epoch: int,
    epochs: int,
    steps_per_epoch: int,
    accumulation_steps: int,
    mask_ratio: float,
    peak_lr: float,
    min_lr: float,
    warmup_epochs: int,
    use_scheduler: bool,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_samples = 0
    usable_micro_batches = steps_per_epoch * accumulation_steps
    last_lr = 0.0
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    update_in_epoch = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}", unit="batch", leave=False, total=usable_micro_batches)
    for micro_batch_index, (images, _labels) in enumerate(progress, start=1):
        if micro_batch_index > usable_micro_batches:
            break
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            loss, _prediction, _mask = model(images, mask_ratio)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, batch {micro_batch_index}: {loss.item()}")
        (loss / accumulation_steps).backward()

        if micro_batch_index % accumulation_steps == 0:
            global_step = (epoch - 1) * steps_per_epoch + update_in_epoch
            if use_scheduler:
                last_lr = cosine_warmup_lr(
                    global_step,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    peak_lr=peak_lr,
                    min_lr=min_lr,
                )
            else:
                last_lr = peak_lr
            set_optimizer_lr(optimizer, last_lr)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_in_epoch += 1

        batch_size = images.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{total_loss / total_samples:.4f}", lr=f"{last_lr:.2e}")

    return total_loss / total_samples, last_lr


@torch.inference_mode()
def evaluate_reconstruction_loss(
    model: MaskedAutoencoderViT,
    loader: DataLoader,
    *,
    device: torch.device,
    mask_ratio: float,
    mask_seed: int,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> float:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(mask_seed)
    total_loss = 0.0
    total_samples = 0
    for images, _labels in tqdm(loader, desc="Validation", unit="batch", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            loss, _prediction, _mask = model(images, mask_ratio, generator=generator)
        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / total_samples


@torch.inference_mode()
def extract_features(
    model: MaskedAutoencoderViT,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    description: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    for images, targets in tqdm(loader, desc=description, unit="batch", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            representation = model.encode_features(images)
        features.append(representation.float().cpu())
        labels.append(targets.long().cpu())
    return torch.cat(features), torch.cat(labels)


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    num_classes: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    base_lr: float,
    warmup_epochs: int,
    seed: int,
) -> tuple[float, float]:
    linear = nn.Linear(train_features.shape[1], num_classes)
    head = nn.Sequential(nn.BatchNorm1d(train_features.shape[1], affine=False, eps=1e-6), linear).to(device)
    nn.init.trunc_normal_(linear.weight, std=0.01)
    nn.init.zeros_(linear.bias)

    # This matches the official MAE linear-probe head/optimizer: LARS is applied
    # to the matrix, while the one-dimensional bias is exempt from adaptation.
    optimizer = Lars(
        (
            {"params": (linear.weight,), "always_adapt": True},
            {"params": (linear.bias,), "always_adapt": False},
        ),
        lr=base_lr * batch_size / 256.0,
        momentum=0.9,  # pyright: ignore[reportArgumentType]
        weight_decay=0.0,  # pyright: ignore[reportArgumentType]
        trust_coeff=0.001,
    )
    steps_per_epoch = math.ceil(len(train_features) / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * min(warmup_epochs, max(epochs - 1, 0))
    peak_lr = base_lr * batch_size / 256.0
    generator = torch.Generator().manual_seed(seed)
    total_loss = 0.0

    for probe_epoch in trange(epochs, desc="Linear probe", unit="epoch", leave=False):
        head.train()
        permutation = torch.randperm(len(train_features), generator=generator)
        epoch_loss = 0.0
        seen = 0
        for batch_index, start in enumerate(range(0, len(permutation), batch_size)):
            indices = permutation[start : start + batch_size]
            features = train_features[indices].to(device, non_blocking=True)
            labels = train_labels[indices].to(device, non_blocking=True)
            step = probe_epoch * steps_per_epoch + batch_index
            lr = cosine_warmup_lr(
                step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                peak_lr=peak_lr,
                min_lr=0.0,
            )
            set_optimizer_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(features), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item() * len(indices)
            seen += len(indices)
        total_loss = epoch_loss / seen

    head.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for start in range(0, len(val_features), batch_size):
            features = val_features[start : start + batch_size].to(device, non_blocking=True)
            labels = val_labels[start : start + batch_size].to(device, non_blocking=True)
            predictions = head(features).argmax(dim=1)
            correct += predictions.eq(labels).sum().item()
            total += len(labels)
    return total_loss, 100.0 * correct / total


def periodic_linear_probe(
    model: MaskedAutoencoderViT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    num_classes: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    base_lr: float,
    warmup_epochs: int,
    seed: int,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> tuple[float, float]:
    was_training = model.training
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # Probe initialization/training must not perturb the pre-training RNG stream.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        train_features, train_labels = extract_features(
            model,
            train_loader,
            device=device,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
            description="Probe features (train)",
        )
        val_features, val_labels = extract_features(
            model,
            val_loader,
            device=device,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
            description="Probe features (val)",
        )
        result = train_linear_probe(
            train_features,
            train_labels,
            val_features,
            val_labels,
            num_classes=num_classes,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            seed=seed,
        )
    model.train(was_training)
    return result


def save_checkpoint(
    path: Path,
    *,
    model: MaskedAutoencoderViT,
    optimizer: Optimizer,
    epoch: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_dir", "--data-dir", default="./data/tiny-imagenet-200")
    parser.add_argument("--arch", default=SUPPORTED_ARCH)
    parser.add_argument("--image_size", "--image-size", type=int, default=64)
    parser.add_argument("--patch_size", "--patch-size", type=int, default=8)
    parser.add_argument("--decoder_embed_dim", "--decoder-embed-dim", type=int, default=128)
    parser.add_argument("--decoder_depth", "--decoder-depth", type=int, default=4)
    parser.add_argument("--decoder_num_heads", "--decoder-num-heads", type=int, default=4)
    parser.add_argument("--mask_ratio", "--mask-ratio", type=float, default=0.75)
    parser.add_argument("--norm_pix_loss", "--norm-pix-loss", type=parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--no_norm_pix_loss", "--no-norm-pix-loss", action="store_false", dest="norm_pix_loss")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup_epochs", "--warmup-epochs", type=int, default=15)
    parser.add_argument("--max_micro_batch_size", "--max-micro-batch-size", type=int, default=256)
    parser.add_argument("--min_lr", "--min-lr", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)

    parser.add_argument("--probe_every", "--probe-every", type=int, default=50)
    parser.add_argument("--probe_epochs", "--probe-epochs", type=int, default=90)
    parser.add_argument("--probe_batch_size", "--probe-batch-size", type=int, default=4096)
    parser.add_argument("--probe_base_lr", "--probe-base-lr", type=float, default=0.1)
    parser.add_argument("--probe_warmup_epochs", "--probe-warmup-epochs", type=int, default=10)

    parser.add_argument("--run_finetune", "--run-finetune", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--finetune_epochs", "--finetune-epochs", type=int, default=100)
    parser.add_argument("--finetune_batch_size", "--finetune-batch-size", type=int, default=1024)
    parser.add_argument("--finetune_max_micro_batch_size", "--finetune-max-micro-batch-size", type=int, default=256)
    parser.add_argument("--finetune_base_lr", "--finetune-base-lr", type=float, default=5e-4)
    parser.add_argument("--finetune_min_lr", "--finetune-min-lr", type=float, default=1e-6)
    parser.add_argument("--finetune_warmup_epochs", "--finetune-warmup-epochs", type=int, default=5)
    parser.add_argument("--finetune_weight_decay", "--finetune-weight-decay", type=float, default=0.05)
    parser.add_argument("--finetune_layer_decay", "--finetune-layer-decay", type=float, default=0.65)
    parser.add_argument("--finetune_drop_path", "--finetune-drop-path", type=float, default=0.1)
    parser.add_argument("--finetune_mixup", "--finetune-mixup", type=float, default=0.8)
    parser.add_argument("--finetune_cutmix", "--finetune-cutmix", type=float, default=1.0)
    parser.add_argument("--finetune_label_smoothing", "--finetune-label-smoothing", type=float, default=0.1)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", "--num-workers", type=int, default=-1)
    parser.add_argument("--eval_batch_size", "--eval-batch-size", type=int, default=1024)
    parser.add_argument("--amp_dtype", "--amp-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--float32_precision", "--float32-precision", choices=("tf32", "ieee"), default="tf32")
    parser.add_argument("--output_dir", "--output-dir", default="./outputs/mae")
    parser.add_argument("--save_every", "--save-every", type=int, default=50)
    parser.add_argument("--wandb_project", "--wandb-project", default=None)
    parser.add_argument("--wandb_entity", "--wandb-entity", default=None)
    parser.add_argument("--wandb_mode", "--wandb-mode", choices=("online", "offline", "disabled"), default=None)


def validate_config(args: argparse.Namespace, config: Any, parser: argparse.ArgumentParser) -> None:
    missing_sweep_keys = REQUIRED_SWEEP_KEYS.difference(config.keys())
    optimizer_name = str(config.get("optimizer", ""))
    if optimizer_name in {"MAL_SGDM", "MAL_AdamW"} and not any(key in config for key in MAL_CONFIG_KEYS):
        missing_sweep_keys = set(missing_sweep_keys) | {"MAL_config"}
    if missing_sweep_keys:
        parser.error(f"Missing W&B sweep parameter(s): {', '.join(sorted(missing_sweep_keys))}.")
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.warmup_epochs < 0:
        parser.error("--warmup_epochs must be non-negative.")
    if parse_bool(config.use_scheduler) and args.epochs <= args.warmup_epochs:
        parser.error("--epochs must be greater than --warmup_epochs when scheduling is enabled.")
    if not 0.0 < args.mask_ratio < 1.0:
        parser.error("--mask_ratio must be strictly between 0 and 1.")
    for name in ("max_micro_batch_size", "eval_batch_size", "probe_batch_size", "probe_epochs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive.")
    if config.batch_size <= 0:
        parser.error("Sweep parameter batch_size must be positive.")
    if config.batch_size % min(config.batch_size, args.max_micro_batch_size) != 0:
        parser.error("--batch_size must be divisible by the selected micro-batch size.")
    if args.probe_every < 0:
        parser.error("--probe_every must be non-negative (zero disables the periodic probe).")
    if args.probe_warmup_epochs < 0:
        parser.error("--probe_warmup_epochs must be non-negative.")
    if args.run_finetune:
        if args.finetune_epochs <= 0 or args.finetune_batch_size <= 0 or args.finetune_max_micro_batch_size <= 0:
            parser.error("Fine-tune epochs and batch sizes must be positive.")
        finetune_micro_batch_size = min(args.finetune_batch_size, args.finetune_max_micro_batch_size)
        if args.finetune_batch_size % finetune_micro_batch_size:
            parser.error("--finetune_batch_size must be divisible by its selected micro-batch size.")
        if not 0 <= args.finetune_warmup_epochs < args.finetune_epochs:
            parser.error("--finetune_warmup_epochs must be non-negative and smaller than --finetune_epochs.")
        if min(args.finetune_base_lr, args.finetune_min_lr, args.finetune_weight_decay) < 0.0:
            parser.error("Fine-tune learning rates and weight decay must be non-negative.")
        if not 0.0 < args.finetune_layer_decay <= 1.0:
            parser.error("--finetune_layer_decay must be in (0, 1].")
        if not 0.0 <= args.finetune_drop_path < 1.0:
            parser.error("--finetune_drop_path must be in [0, 1).")
        if min(args.finetune_mixup, args.finetune_cutmix, args.finetune_label_smoothing) < 0.0:
            parser.error("Fine-tune augmentation and smoothing coefficients must be non-negative.")
    if args.save_every < 0:
        parser.error("--save_every must be non-negative (zero saves only the final checkpoint).")
    if config.base_lr < 0.0 or args.min_lr < 0.0 or args.probe_base_lr < 0.0 or config.weight_decay < 0.0:
        parser.error("Learning rates and weight decay must be non-negative.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args, _unknown = parser.parse_known_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("A CUDA device was requested, but CUDA is unavailable; pass --device cpu for a smoke test.")

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        job_type="mae-pretrain",
        config={
            **vars(args),
            **task_metadata(
                task="tiny_imagenet_mae_pretraining",
                task_type="self_supervised_image_pretraining",
                model_name=f"{args.arch}_mae_patch{args.patch_size}",
                model_source="timm_mae",
                dataset_name="tiny-imagenet-200",
                dataset_config="official_train_pretraining_official_validation_evaluation",
                dataset_source="official_tiny_imagenet",
                training_regime="self_supervised_from_scratch",
            ),
        },
        tags=("mae", "tiny-imagenet", "vit-tiny", "patch8"),
    )
    config = run.config
    validate_config(args, config, parser)
    optimizer_name = str(config.optimizer)
    batch_size = int(config.batch_size)
    base_lr = float(config.base_lr)
    weight_decay = float(config.weight_decay)
    seed = int(config.seed)
    nesterov = parse_bool(config.get("nesterov", False))
    use_scheduler = parse_bool(config.use_scheduler)

    device = torch.device(args.device)
    amp_dtype, amp_enabled = configure_precision(device, args.amp_dtype, args.float32_precision)
    set_seed(seed)

    pretrain_dataset, probe_train_dataset, val_dataset, num_classes, data_root = build_datasets(args.data_dir, args.image_size)
    raw_train_dataset = probe_train_dataset.dataset
    raw_val_dataset = val_dataset.dataset
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
    num_workers = min(allocated_cpus, 16) if args.num_workers < 0 else args.num_workers
    micro_batch_size = min(batch_size, args.max_micro_batch_size)
    accumulation_steps = batch_size // micro_batch_size
    train_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": set_worker_seed,
    }
    eval_num_workers = min(num_workers, 6)
    eval_loader_kwargs = {
        "num_workers": eval_num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": set_worker_seed,
    }
    train_loader = DataLoader(
        pretrain_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        persistent_workers=eval_num_workers > 0,
        generator=torch.Generator().manual_seed(seed + 1),
        **eval_loader_kwargs,
    )
    probe_train_loader = DataLoader(
        probe_train_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(seed + 2),
        **eval_loader_kwargs,
    )
    probe_val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(seed + 3),
        **eval_loader_kwargs,
    )
    steps_per_epoch = len(train_loader) // accumulation_steps
    if steps_per_epoch == 0:
        parser.error("The effective batch size is larger than the available training data.")

    model = MaskedAutoencoderViT(
        arch=args.arch,
        image_size=args.image_size,
        patch_size=args.patch_size,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        norm_pix_loss=args.norm_pix_loss,
    ).to(device)
    actual_lr = base_lr * batch_size / 256.0
    raw_mal_config: str | None = None
    mal_config: dict[str, Any] = {}
    mal_align = MAL_ALIGN
    if optimizer_name in {"MAL_SGDM", "MAL_AdamW"}:
        raw_mal_config = str(next(config[key] for key in MAL_CONFIG_KEYS if key in config))
        mal_config = parse_mal_config(raw_mal_config)
        mal_align = str(mal_config.pop("align", config.get("mal_align", MAL_ALIGN))).lower()
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

    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    encoder_parameter_count = (
        sum(parameter.numel() for module in (model.patch_embed, model.blocks, model.norm) for parameter in module.parameters() if parameter.requires_grad)
        + model.cls_token.numel()
    )

    optimizer_metadata: dict[str, Any] = {}
    if raw_mal_config is not None:
        # Uppercase is the canonical W&B grouping field. Reading the old
        # lowercase key above keeps prior sweep definitions reproducible.
        optimizer_metadata = {
            "MAL_config": raw_mal_config,
            "optimizer_config": raw_mal_config,
            "mal_align": mal_align,
            **{f"mal_{key}": value for key, value in mal_config.items()},
        }
    run.config.update(
        {
            "actual_lr": actual_lr,
            "micro_batch_size": micro_batch_size,
            "accumulation_steps": accumulation_steps,
            "steps_per_epoch": steps_per_epoch,
            "num_classes": num_classes,
            "train_examples": len(pretrain_dataset),
            "validation_examples": len(val_dataset),
            "test_examples": 0,
            "trainable_parameters": parameter_count,
            "encoder_trainable_parameters": encoder_parameter_count,
            "optimizer_family": (
                "sgdm" if optimizer_name in SGD_OPTIMIZERS else "adamw" if optimizer_name in ADAMW_OPTIMIZERS else "lion"
            ),
            "effective_batch_size": batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "base_learning_rate": base_lr,
            "learning_rate": actual_lr,
            "resolved_data_dir": str(data_root),
            "downstream_evaluation": "linear_probe_and_end_to_end_finetune" if args.run_finetune else "linear_probe",
            "downstream_accuracy_unit": "percent",
            **(
                {
                    "finetune_optimizer": "AdamW",
                    "finetune_actual_lr": args.finetune_base_lr * args.finetune_batch_size / 256.0,
                    "finetune_global_pool": True,
                    "finetune_backbone_trainable": True,
                    "finetune_initialization": "final_epoch_mae_encoder",
                }
                if args.run_finetune
                else {}
            ),
            **optimizer_metadata,
            **(
                {
                    "AGAM_config": "False,1.0,none,attenuate,lion_sign_update,complement",
                    "optimizer_config": "False,1.0,none,attenuate,lion_sign_update,complement",
                    "agam_gate_target": "lion_immediate_beta1",
                    "agam_alignment_direction": "signed_update",
                    "agam_persistent_state": "base_lion_beta2_ema",
                }
                if optimizer_name == "AGAM_Lion"
                else {}
            ),
            **({"am_beta_max": args.momentum, "am_model_lambda": 0.1} if optimizer_name == "AM_MSGD" else {}),
            **({"am_beta1_max": args.momentum - 0.1 * 0.1, "am_model_lambda": 0.1} if optimizer_name == "AM_AdamW" else {}),
        },
        allow_val_change=True,
    )
    mal_suffix = ""
    if optimizer_name.startswith("MAL_"):
        mal_suffix = (
            f"_inp{int(mal_config['in_place'])}_p{mal_config['pwr']:g}"
            f"_scl{str(mal_config['scale']).lower()}_g{mal_config['gate_mode']}"
            f"_a{mal_align}_gw{mal_config['gradient_weight_mode']}"
        )
    run.name = f"{optimizer_name}{mal_suffix}_bs{batch_size}_blr{base_lr:g}_wd{weight_decay:g}_s{seed}"
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")
    run.define_metric("probe/*", step_metric="epoch")
    run.define_metric("linear_probe/*", step_metric="epoch")
    run.define_metric("finetune/epoch")
    run.define_metric("finetune/*", step_metric="finetune/epoch")
    run.define_metric("diagnostic/*", step_metric="epoch")
    run.define_metric("lr", step_metric="epoch")

    print(
        f"Training {args.arch} MAE on {data_root} ({len(pretrain_dataset):,} train / {len(val_dataset):,} val, "
        f"{num_classes} classes).\n"
        f"Parameters: {parameter_count / 1e6:.2f}M total, {encoder_parameter_count / 1e6:.2f}M encoder. "
        f"Effective batch: {batch_size} = {micro_batch_size} x {accumulation_steps}; "
        f"base LR: {base_lr:.3g}, actual LR: {actual_lr:.3g}."
    )

    checkpoint_path = Path(args.output_dir).expanduser().resolve() / run.id / "checkpoint-last.pt" if args.output_dir else None
    best_val_loss = math.inf
    best_probe_accuracy = 0.0
    best_probe_epoch = 0
    exit_code = 0
    try:
        for epoch in trange(1, args.epochs + 1, desc="MAE pre-training", unit="epoch"):
            train_loss, current_lr = train_one_epoch(
                model,
                optimizer,
                train_loader,
                device=device,
                epoch=epoch,
                epochs=args.epochs,
                steps_per_epoch=steps_per_epoch,
                accumulation_steps=accumulation_steps,
                mask_ratio=args.mask_ratio,
                peak_lr=actual_lr,
                min_lr=args.min_lr,
                warmup_epochs=args.warmup_epochs,
                use_scheduler=use_scheduler,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            val_loss = evaluate_reconstruction_loss(
                model,
                val_loader,
                device=device,
                mask_ratio=args.mask_ratio,
                mask_seed=seed + 10_000,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            best_val_loss = min(best_val_loss, val_loss)
            metrics: dict[str, float | int] = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr": current_lr,
            }

            should_probe = args.probe_every and (epoch % args.probe_every == 0 or epoch == args.epochs)
            probe_accuracy: float | None = None
            if should_probe:
                probe_train_loss, probe_accuracy = periodic_linear_probe(
                    model,
                    probe_train_loader,
                    probe_val_loader,
                    num_classes=num_classes,
                    device=device,
                    epochs=args.probe_epochs,
                    batch_size=args.probe_batch_size,
                    base_lr=args.probe_base_lr,
                    warmup_epochs=args.probe_warmup_epochs,
                    # Hold the probe initialization and minibatch order fixed
                    # across checkpoints so changes reflect the encoder only.
                    seed=seed + 20_000,
                    amp_dtype=amp_dtype,
                    amp_enabled=amp_enabled,
                )
                if probe_accuracy > best_probe_accuracy:
                    best_probe_accuracy = probe_accuracy
                    best_probe_epoch = epoch
                metrics.update(
                    {
                        "probe/train_loss": probe_train_loss,
                        "probe/val_acc": probe_accuracy,
                        "linear_probe/train_loss": probe_train_loss,
                        "linear_probe/val_top1_pct": probe_accuracy,
                    }
                )
                if epoch == args.epochs:
                    run.summary["final_probe_val_acc"] = probe_accuracy
                    run.summary["linear_probe/final_val_top1_pct"] = probe_accuracy

            if isinstance(optimizer, AM_MSGD) and optimizer.last_beta is not None:
                metrics["diagnostic/am_beta"] = float(optimizer.last_beta)
                assert optimizer.last_effective_momentum is not None
                metrics["diagnostic/am_effective_momentum"] = float(optimizer.last_effective_momentum)
            elif isinstance(optimizer, AM_AdamW) and optimizer.last_beta is not None:
                metrics["diagnostic/am_beta_mean"] = float(optimizer.last_beta)
                assert optimizer.last_beta_min is not None and optimizer.last_beta_max is not None
                metrics["diagnostic/am_beta_min"] = float(optimizer.last_beta_min)
                metrics["diagnostic/am_beta_max"] = float(optimizer.last_beta_max)
            run.log(metrics)
            run.summary["best_val_loss"] = best_val_loss
            run.summary["best_probe_val_acc"] = best_probe_accuracy
            run.summary["best/val_loss"] = best_val_loss
            run.summary["best/probe_val_acc"] = best_probe_accuracy
            run.summary["linear_probe/best_val_top1_pct"] = best_probe_accuracy
            run.summary["linear_probe/best_pretrain_epoch"] = best_probe_epoch
            if epoch == args.epochs and probe_accuracy is not None:
                run.summary["final/probe_val_acc"] = probe_accuracy

            should_save = epoch == args.epochs or (args.save_every and epoch % args.save_every == 0)
            if checkpoint_path is not None and should_save:
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=dict(run.config),
                )
                run.summary["checkpoint"] = str(checkpoint_path)

        if args.run_finetune:
            finetune_summary = run_end_to_end_finetune(
                model,
                raw_train_dataset,
                raw_val_dataset,
                num_classes=num_classes,
                image_size=args.image_size,
                device=device,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
                epochs=args.finetune_epochs,
                batch_size=args.finetune_batch_size,
                max_micro_batch_size=args.finetune_max_micro_batch_size,
                base_lr=args.finetune_base_lr,
                minimum_lr=args.finetune_min_lr,
                warmup_epochs=args.finetune_warmup_epochs,
                weight_decay=args.finetune_weight_decay,
                layer_decay=args.finetune_layer_decay,
                drop_path_rate=args.finetune_drop_path,
                mixup_alpha=args.finetune_mixup,
                cutmix_alpha=args.finetune_cutmix,
                label_smoothing=args.finetune_label_smoothing,
                num_workers=num_workers,
                seed=seed + 30_000,
                run=run,
            )
            run.summary.update(finetune_summary)
    except BaseException:
        exit_code = 1
        raise
    finally:
        run.finish(exit_code=exit_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
