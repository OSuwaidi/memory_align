"""End-to-end Tiny-ImageNet evaluation of a completed MAE checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import torch
import wandb

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tasks.mae_pretrain import (
    SUPPORTED_ARCH,
    MaskedAutoencoderViT,
    build_datasets,
    configure_precision,
    parse_bool,
    run_end_to_end_finetune,
)
from tasks.wandb_metadata import task_metadata


SUPPORTED_SOURCE_OPTIMIZERS = {
    "AdamW",
    "AM_AdamW",
    "AdaTAMW",
    "MAL_AdamW",
    "AdaMAL",
    "Lion",
    "AGAM_Lion",
}


def require_number(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run_id", "--source-run-id", required=True)
    parser.add_argument("--source_entity", "--source-entity", default="osuwaidi-khalifa-university")
    parser.add_argument("--source_project", "--source-project", default="MAL_benchmark")
    parser.add_argument("--data_dir", "--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=1024)
    parser.add_argument("--max_micro_batch_size", "--max-micro-batch-size", type=int, default=256)
    parser.add_argument("--base_lr", "--base-lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", "--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", "--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=0.05)
    parser.add_argument("--layer_decay", "--layer-decay", type=float, default=0.65)
    parser.add_argument("--drop_path", "--drop-path", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--label_smoothing", "--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp_dtype", "--amp-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--float32_precision", "--float32-precision", choices=("tf32", "ieee"), default="tf32")
    parser.add_argument("--wandb_project", "--wandb-project", default=None)
    parser.add_argument("--wandb_entity", "--wandb-entity", default=None)
    parser.add_argument("--wandb_mode", "--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    args, _unknown = parser.parse_known_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("A CUDA device was requested, but CUDA is unavailable.")
    if args.epochs <= 0 or args.batch_size <= 0 or args.max_micro_batch_size <= 0:
        parser.error("Fine-tune epochs and batch sizes must be positive.")
    micro_batch_size = min(args.batch_size, args.max_micro_batch_size)
    if args.batch_size % micro_batch_size:
        parser.error("--batch_size must be divisible by its selected micro-batch size.")
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error("--warmup_epochs must be non-negative and smaller than --epochs.")
    if args.base_lr <= 0.0 or min(args.min_lr, args.weight_decay) < 0.0:
        parser.error("Learning rates must be positive/non-negative and weight decay must be non-negative.")
    if not 0.0 < args.layer_decay <= 1.0:
        parser.error("--layer_decay must be in (0, 1].")
    if not 0.0 <= args.drop_path < 1.0:
        parser.error("--drop_path must be in [0, 1).")
    if min(args.mixup, args.cutmix) < 0.0 or not 0.0 <= args.label_smoothing < 1.0:
        parser.error("Mixup/CutMix must be non-negative and label smoothing must be in [0, 1).")

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        job_type="mae-end-to-end-finetune",
        config={
            **vars(args),
            **task_metadata(
                task="tiny_imagenet_mae_end_to_end_finetuning",
                task_type="supervised_image_finetuning",
                model_name="vit_tiny_patch8_mae_encoder",
                model_source="timm_mae_checkpoint",
                dataset_name="tiny-imagenet-200",
                dataset_config="official_train_finetuning_official_validation_evaluation",
                dataset_source="official_tiny_imagenet",
                training_regime="full_backbone_finetuning_from_mae_checkpoint",
            ),
        },
        tags=("mae", "tiny-imagenet", "vit-tiny", "patch8", "end-to-end-finetune"),
    )

    source_run_id = str(run.config.source_run_id)
    source = wandb.Api(timeout=180).run(f"{args.source_entity}/{args.source_project}/{source_run_id}")
    if source.state != "finished":
        raise RuntimeError(f"Source run {source_run_id} is not finished: {source.state}.")
    checkpoint_path = Path(str(source.summary.get("checkpoint", ""))).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Source checkpoint is unavailable on this host: {checkpoint_path}")

    source_config = dict(source.config)
    source_optimizer = str(source_config.get("optimizer", ""))
    if source_optimizer not in SUPPORTED_SOURCE_OPTIMIZERS:
        raise ValueError(
            f"Unsupported MAE source optimizer {source_optimizer!r}; "
            f"expected one of {sorted(SUPPORTED_SOURCE_OPTIMIZERS)}."
        )
    source_seed = int(source_config["seed"])
    source_epochs = int(source_config.get("epochs", 300))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_epoch = int(checkpoint["epoch"])
    if checkpoint_epoch != source_epochs:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is epoch {checkpoint_epoch}, expected final epoch {source_epochs}."
        )

    arch = str(source_config.get("arch", SUPPORTED_ARCH))
    image_size = int(source_config.get("image_size", 64))
    patch_size = int(source_config.get("patch_size", 8))
    model = MaskedAutoencoderViT(
        arch=arch,
        image_size=image_size,
        patch_size=patch_size,
        decoder_embed_dim=int(source_config.get("decoder_embed_dim", 128)),
        decoder_depth=int(source_config.get("decoder_depth", 4)),
        decoder_num_heads=int(source_config.get("decoder_num_heads", 4)),
        norm_pix_loss=parse_bool(source_config.get("norm_pix_loss", True)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)

    _pretrain, probe_train, validation, num_classes, data_root = build_datasets(args.data_dir, image_size)
    device = torch.device(args.device)
    amp_dtype, amp_enabled = configure_precision(device, args.amp_dtype, args.float32_precision)
    model.to(device)

    # Use the historical field shared by every MAE screening run. Newer runs
    # also expose the equivalent, more descriptive linear_probe alias.
    source_final_probe = source.summary.get("final/probe_val_acc")
    if source_final_probe is None:
        source_final_probe = source.summary.get(
            "linear_probe/final_val_top1_pct", source.summary.get("final_probe_val_acc")
        )
    source_sweep_id = source.sweep.id if source.sweep is not None else None
    run.config.update(
        {
            "pretraining_optimizer": source_optimizer,
            "optimizer": source_optimizer,
            "source_run_id": source_run_id,
            "source_sweep_id": source_sweep_id,
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_epoch": checkpoint_epoch,
            "source_seed": source_seed,
            "seed": source_seed,
            "source_batch_size": int(source_config["batch_size"]),
            "source_base_lr": require_number(source_config["base_lr"], "source_base_lr"),
            "source_weight_decay": require_number(source_config["weight_decay"], "source_weight_decay"),
            "source_optimizer_config": str(
                source_config.get("MAL_config", source_config.get("optimizer_config", "base"))
            ),
            "source_first_moment_correction": str(
                source_config.get(
                    "agam_first_moment_correction",
                    source_config.get("mal_first_moment_correction", "adaptive"),
                )
                if source_optimizer == "MAL_AdamW"
                else "not_applicable"
            ),
            "source_final_linear_probe_val_top1_pct": require_number(source_final_probe, "source final probe"),
            "resolved_data_dir": str(data_root),
            "finetune_optimizer": "AdamW",
            "finetune_actual_lr": args.base_lr * args.batch_size / 256.0,
            "finetune_global_pool": True,
            "finetune_backbone_trainable": True,
            "finetune_initialization": "final_epoch_mae_encoder",
        },
        allow_val_change=True,
    )
    run.name = f"{source_optimizer}_mae-ft_src{source_run_id}_s{source_seed}"
    run.define_metric("finetune/epoch")
    run.define_metric("finetune/*", step_metric="finetune/epoch")

    try:
        result = run_end_to_end_finetune(
            model,
            probe_train.dataset,
            validation.dataset,
            num_classes=num_classes,
            image_size=image_size,
            device=device,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_micro_batch_size=args.max_micro_batch_size,
            base_lr=args.base_lr,
            minimum_lr=args.min_lr,
            warmup_epochs=args.warmup_epochs,
            weight_decay=args.weight_decay,
            layer_decay=args.layer_decay,
            drop_path_rate=args.drop_path,
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            label_smoothing=args.label_smoothing,
            num_workers=args.num_workers,
            seed=source_seed + 30_000,
            run=run,
        )
        run.summary.update(result)
        run.summary["source/final_linear_probe_val_top1_pct"] = float(source_final_probe)
        return 0
    finally:
        run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
