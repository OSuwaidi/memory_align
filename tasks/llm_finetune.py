"""Full-parameter causal-LM fine-tuning from a pretrained checkpoint.

The default experiment fine-tunes SmolLM2-135M on fixed-length WikiText-2
blocks.  Unlike adapter tuning, every pretrained parameter is optimized, so
the experiment directly tests how each optimizer manages momentum near a
well-trained starting point.  Validation loss is measured once before the
first update and after every epoch; test loss is measured once at the end.
Missing assets are downloaded automatically.  To prefetch them before a GPU
job, run ``uv run download_datasets.py --task llm``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from array import array
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from datasets import load_dataset
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD, AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from wandb.sdk.internal.internal_api import Api as WandbInternalApi

from optims.am_opt import AM_MSGD, AM_AdamW
from optims.cautious_opt import CAUTIOUS_ADAMW, CAUTIOUS_SGD
from optims.mal_opt import MAL_SGDM, MAL_AdamW
from optims.tam_opt import TAM_SGDM, AdaTAMW

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
DEFAULT_DATASET = "Salesforce/wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"
DEFAULT_DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"

SGD_OPTIMIZERS = {"SGDM", "AM_MSGD", "CAUTIOUS_SGDM", "TAM_SGDM", "MAL_SGDM"}
ADAMW_OPTIMIZERS = {"AdamW", "AM_AdamW", "CAUTIOUS_AdamW", "AdaTAMW", "MAL_AdamW"}
ALL_OPTIMIZERS = SGD_OPTIMIZERS | ADAMW_OPTIMIZERS
REQUIRED_SWEEP_KEYS = frozenset(("optimizer", "batch_size", "lr_multiplier", "seed"))


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
        raise RuntimeError("This entry point enables bfloat16 AMP only on CUDA; use --amp_dtype float32 on CPU.")
    if device.type == "cuda":
        if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not support bfloat16 AMP; use --amp_dtype float32.")
        torch.backends.cuda.matmul.fp32_precision = float32_precision
        torch.backends.cudnn.conv.fp32_precision = float32_precision  # pyright: ignore[reportAttributeAccessIssue]
    return amp_dtype, amp_enabled


class TokenBlockDataset(Dataset[torch.Tensor]):
    """Non-overlapping, fixed-length causal-LM blocks backed by one tensor."""

    def __init__(self, token_ids: torch.Tensor, sequence_length: int) -> None:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be one-dimensional.")
        num_blocks = token_ids.numel() // sequence_length
        if num_blocks == 0:
            raise ValueError(f"The split has fewer than {sequence_length} tokens.")
        self.blocks = token_ids[: num_blocks * sequence_length].view(num_blocks, sequence_length)

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        # Cache int32 to halve disk/RAM use; embedding indices must be int64.
        return self.blocks[index].long()


def _token_cache_key(
    *,
    model_name: str,
    model_revision: str,
    dataset_name: str,
    dataset_config: str,
    dataset_revision: str,
    sequence_length: int,
) -> str:
    payload = json.dumps(
        {
            "model_name": model_name,
            "model_revision": model_revision,
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "dataset_revision": dataset_revision,
            "sequence_length": sequence_length,
            "packing_version": 1,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _tokenize_split(texts: Iterable[str], tokenizer: PreTrainedTokenizerBase, batch_size: int = 1_000) -> torch.Tensor:
    eos_token_id = tokenizer.eos_token_id
    if not isinstance(eos_token_id, int):
        raise TypeError("The tokenizer must define one integer eos_token_id for deterministic document packing.")

    packed = array("I")
    batch: list[str] = []

    def consume(current_batch: Sequence[str]) -> None:
        encoded = tokenizer(list(current_batch), add_special_tokens=False, return_attention_mask=False)["input_ids"]
        for token_ids in encoded:
            if token_ids:
                packed.extend(token_ids)
                packed.append(eos_token_id)

    for text in texts:
        if text and not text.isspace():
            batch.append(text)
        if len(batch) == batch_size:
            consume(batch)
            batch.clear()
    if batch:
        consume(batch)

    return torch.tensor(packed, dtype=torch.int32)


def load_token_blocks(
    tokenizer: PreTrainedTokenizerBase,
    *,
    model_name: str,
    model_revision: str,
    dataset_name: str,
    dataset_config: str,
    dataset_revision: str,
    sequence_length: int,
    cache_dir: Path,
) -> dict[str, TokenBlockDataset]:
    cache_key = _token_cache_key(
        model_name=model_name,
        model_revision=model_revision,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_revision=dataset_revision,
        sequence_length=sequence_length,
    )
    cache_path = cache_dir.expanduser().resolve() / f"packed-{cache_key}.pt"
    if cache_path.is_file():
        packed_splits = torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        raw_dataset = load_dataset(
            dataset_name,
            dataset_config,
            revision=dataset_revision,
            cache_dir=str(cache_dir.expanduser().resolve() / "huggingface"),
        )
        packed_splits = {}
        for split in ("train", "validation", "test"):
            if split not in raw_dataset:
                raise ValueError(f'Dataset "{dataset_name}/{dataset_config}" has no {split!r} split.')
            packed_splits[split] = _tokenize_split(raw_dataset[split]["text"], tokenizer)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        torch.save(packed_splits, temporary_path)
        os.replace(temporary_path, cache_path)

    if not isinstance(packed_splits, dict) or any(split not in packed_splits for split in ("train", "validation", "test")):
        raise ValueError(f"Malformed packed-token cache: {cache_path}")
    return {split: TokenBlockDataset(packed_splits[split], sequence_length) for split in ("train", "validation", "test")}


def split_weight_decay_params(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim <= 1 else decay).append(parameter)
    groups: list[dict[str, Any]] = []
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    return groups


def parse_mal_config(value: str) -> dict[str, Any]:
    fields = value.split(",")
    if len(fields) == 4:
        in_place_text, pwr_text, scale_text, gate_mode = fields
        descent_safeguard_text = "False"
    elif len(fields) == 5:
        in_place_text, pwr_text, scale_text, gate_mode, descent_safeguard_text = fields
    else:
        raise ValueError("mal_config must be 'in_place,pwr,scale,gate_mode[,descent_safeguard]'.")
    config = {
        "in_place": parse_bool(in_place_text),
        "pwr": float(pwr_text),
        "scale": parse_bool(scale_text),
        "gate_mode": gate_mode,
        "descent_safeguard": parse_bool(descent_safeguard_text),
    }
    if config["pwr"] not in (0.5, 1.0):
        raise ValueError("MAL pwr must be 0.5 or 1.0.")
    if gate_mode not in ("replace", "attenuate", "cap"):
        raise ValueError("MAL gate_mode must be replace, attenuate, or cap.")
    return config


def build_optimizer(
    name: str,
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    beta2: float,
    mal_config: dict[str, Any],
    mal_align: str,
) -> Optimizer:
    if name not in ALL_OPTIMIZERS:
        raise ValueError(f'Unknown optimizer "{name}". Choose one of {sorted(ALL_OPTIMIZERS)}.')
    parameters: Iterable[nn.Parameter] = model.parameters()
    if name == "SGDM":
        return SGD(split_weight_decay_params(model, weight_decay), lr=lr, momentum=momentum, dampening=0.0, nesterov=False, foreach=False)
    if name == "AM_MSGD":
        return AM_MSGD(parameters, lr=lr, beta_max=momentum, model_lambda=0.1, weight_decay=weight_decay)
    if name == "CAUTIOUS_SGDM":
        return CAUTIOUS_SGD(parameters, lr=lr, beta=momentum, weight_decay=weight_decay, nesterov=False)
    if name == "TAM_SGDM":
        return TAM_SGDM(parameters, lr=lr, beta=momentum, weight_decay=weight_decay)
    if name == "MAL_SGDM":
        return MAL_SGDM(parameters, lr=lr, beta=momentum, weight_decay=weight_decay, nesterov=False, **mal_config)
    if name == "AdamW":
        return AdamW(split_weight_decay_params(model, weight_decay), lr=lr, betas=(momentum, beta2), foreach=False, fused=False)
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
        return CAUTIOUS_ADAMW(parameters, lr=lr, betas=(momentum, beta2), weight_decay=weight_decay)
    if name == "AdaTAMW":
        return AdaTAMW(parameters, lr=lr, betas=(momentum, beta2), weight_decay=weight_decay)
    return MAL_AdamW(parameters, lr=lr, betas=(momentum, beta2), weight_decay=weight_decay, align=mal_align, **mal_config)


def scheduled_lr(step: int, *, total_steps: int, warmup_steps: int, peak_lr: float, use_scheduler: bool) -> float:
    if not use_scheduler:
        return peak_lr
    if warmup_steps and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        return 0.0
    progress = min(max((step - warmup_steps) / (decay_steps - 1), 0.0), 1.0)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer: Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[torch.Tensor],
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    description: str,
) -> float:
    model.eval()
    loss_sum = 0.0
    predicted_tokens = 0
    for input_ids in tqdm(loader, desc=description, unit="batch", leave=False):
        input_ids = input_ids.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        num_tokens = input_ids.shape[0] * (input_ids.shape[1] - 1)
        loss_sum += float(outputs.loss) * num_tokens
        predicted_tokens += num_tokens
    if predicted_tokens == 0:
        raise RuntimeError("Evaluation loader produced no predicted tokens.")
    return loss_sum / predicted_tokens


def train_one_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    loader: DataLoader[torch.Tensor],
    *,
    device: torch.device,
    epoch: int,
    steps_per_epoch: int,
    accumulation_steps: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    use_scheduler: bool,
    max_grad_norm: float,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> tuple[float, float, float, int, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    usable_micro_batches = steps_per_epoch * accumulation_steps
    loss_sum = 0.0
    predicted_tokens = 0
    grad_norm_sum = torch.zeros((), device=device)
    last_lr = 0.0
    start_time = time.perf_counter()
    update_in_epoch = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}", unit="batch", leave=False, total=usable_micro_batches)
    for micro_batch_index, input_ids in enumerate(progress, start=1):
        if micro_batch_index > usable_micro_batches:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            loss = outputs.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, micro-batch {micro_batch_index}: {float(loss)}")
        (loss / accumulation_steps).backward()

        num_tokens = input_ids.shape[0] * (input_ids.shape[1] - 1)
        loss_sum += float(loss.detach()) * num_tokens
        predicted_tokens += num_tokens

        if micro_batch_index % accumulation_steps == 0:
            global_step = (epoch - 1) * steps_per_epoch + update_in_epoch
            last_lr = scheduled_lr(
                global_step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                peak_lr=peak_lr,
                use_scheduler=use_scheduler,
            )
            set_optimizer_lr(optimizer, last_lr)
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, error_if_nonfinite=True)
            grad_norm_sum.add_(grad_norm.detach().float())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_in_epoch += 1

        progress.set_postfix(loss=f"{loss_sum / predicted_tokens:.4f}", lr=f"{last_lr:.2e}")

    elapsed = time.perf_counter() - start_time
    mean_grad_norm = float(grad_norm_sum / update_in_epoch)
    return loss_sum / predicted_tokens, last_lr, mean_grad_norm, predicted_tokens, predicted_tokens / elapsed


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_name", "--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--model_revision", "--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--dataset_name", "--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset_config", "--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset_revision", "--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--cache_dir", "--cache-dir", default="./data/llm_cache")
    parser.add_argument("--sequence_length", "--sequence-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup_ratio", "--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--use_scheduler", "--use-scheduler", type=parse_bool, default=True)
    parser.add_argument("--max_micro_batch_size", "--max-micro-batch-size", type=int, default=8)
    parser.add_argument("--eval_batch_size", "--eval-batch-size", type=int, default=16)
    parser.add_argument("--reference_batch_size", "--reference-batch-size", type=int, default=32)
    parser.add_argument("--sgd_base_lr", "--sgd-base-lr", type=float, default=1e-2)
    parser.add_argument("--am_msgd_base_lr", "--am-msgd-base-lr", type=float, default=1e-1)
    parser.add_argument("--adamw_base_lr", "--adamw-base-lr", type=float, default=5e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", "--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mal_config", "--mal-config", default="False,1.0,True,attenuate,False")
    parser.add_argument("--mal_align", "--mal-align", choices=("update", "metric", "white", "moment"), default="metric")
    parser.add_argument("--gradient_checkpointing", "--gradient-checkpointing", type=parse_bool, default=False)
    parser.add_argument("--attn_implementation", "--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", "--num-workers", type=int, default=2)
    parser.add_argument("--amp_dtype", "--amp-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--float32_precision", "--float32-precision", choices=("tf32", "ieee"), default="tf32")
    parser.add_argument("--wandb_project", "--wandb-project", default=None)
    parser.add_argument("--wandb_entity", "--wandb-entity", default=None)
    parser.add_argument("--wandb_mode", "--wandb-mode", choices=("online", "offline", "disabled"), default=None)


def validate_config(args: argparse.Namespace, config: Any, parser: argparse.ArgumentParser) -> None:
    missing = REQUIRED_SWEEP_KEYS.difference(config.keys())
    if missing:
        parser.error(f"Missing W&B sweep parameter(s): {', '.join(sorted(missing))}.")
    for name in ("sequence_length", "epochs", "max_micro_batch_size", "eval_batch_size", "reference_batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup_ratio must lie in [0, 1).")
    if args.max_grad_norm <= 0.0:
        parser.error("--max_grad_norm must be positive.")
    if args.sgd_base_lr < 0.0 or args.am_msgd_base_lr < 0.0 or args.adamw_base_lr < 0.0 or args.weight_decay < 0.0:
        parser.error("Learning rates and weight decay must be non-negative.")
    if int(config.batch_size) <= 0 or float(config.lr_multiplier) <= 0.0:
        parser.error("Sweep batch_size and lr_multiplier must be positive.")
    micro_batch_size = min(int(config.batch_size), args.max_micro_batch_size)
    if int(config.batch_size) % micro_batch_size != 0:
        parser.error("Sweep batch_size must be divisible by the selected micro-batch size.")


def install_signal_handlers(run: Any, process_state: dict[str, int]) -> None:
    def handle_interruption(signum: int, _frame: Any) -> None:
        manual_stop = signum == signal.SIGINT
        if signum == signal.SIGTERM:
            try:
                manual_stop = WandbInternalApi().check_stop_requested(run.project, run.entity, run.id)
            except Exception as error:  # noqa: BLE001 - signal cleanup must survive W&B/network failures
                print(f"Could not query W&B stop state during signal handling: {error}", file=sys.stderr)
        if not manual_stop:
            run.mark_preempting()
        exit_code = 0 if manual_stop else 128 + signum
        process_state["exit_code"] = exit_code
        sys.exit(exit_code)

    signal.signal(signal.SIGTERM, handle_interruption)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, handle_interruption)
    signal.signal(signal.SIGINT, handle_interruption)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args, _unknown = parser.parse_known_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("A CUDA device was requested, but CUDA is unavailable; pass --device cpu --amp_dtype float32 for a smoke test.")

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        job_type="llm-full-finetune",
        config=vars(args),
        tags=("llm", "full-finetune", "causal-lm", "near-saturation", "wikitext-2"),
    )
    process_state = {"exit_code": 0}
    install_signal_handlers(run, process_state)
    config = run.config
    validate_config(args, config, parser)

    optimizer_name = str(config.optimizer)
    if optimizer_name not in ALL_OPTIMIZERS:
        parser.error(f'Unknown optimizer "{optimizer_name}". Choose one of {sorted(ALL_OPTIMIZERS)}.')
    batch_size = int(config.batch_size)
    lr_multiplier = float(config.lr_multiplier)
    seed = int(config.seed)
    optimizer_base_lr = args.am_msgd_base_lr if optimizer_name == "AM_MSGD" else args.sgd_base_lr if optimizer_name in SGD_OPTIMIZERS else args.adamw_base_lr
    peak_lr = optimizer_base_lr * lr_multiplier * batch_size / args.reference_batch_size
    micro_batch_size = min(batch_size, args.max_micro_batch_size)
    accumulation_steps = batch_size // micro_batch_size

    device = torch.device(args.device)
    amp_dtype, amp_enabled = configure_precision(device, args.amp_dtype, args.float32_precision)
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.model_revision, cache_dir=args.cache_dir)
    if tokenizer.eos_token_id is None:
        parser.error("The selected tokenizer has no EOS token, which this packing recipe requires.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    datasets = load_token_blocks(
        tokenizer,
        model_name=args.model_name,
        model_revision=args.model_revision,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_revision=args.dataset_revision,
        sequence_length=args.sequence_length,
        cache_dir=Path(args.cache_dir),
    )

    loader_common = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": set_worker_seed,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        datasets["train"],
        batch_size=micro_batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_common,
    )
    eval_loader_common = {**loader_common, "persistent_workers": False}
    val_loader = DataLoader(datasets["validation"], batch_size=args.eval_batch_size, shuffle=False, drop_last=False, **eval_loader_common)
    test_loader = DataLoader(datasets["test"], batch_size=args.eval_batch_size, shuffle=False, drop_last=False, **eval_loader_common)
    steps_per_epoch = len(train_loader) // accumulation_steps
    if steps_per_epoch == 0:
        parser.error("The effective batch size exceeds the number of packed training blocks.")
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = round(total_steps * args.warmup_ratio) if args.use_scheduler else 0

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        cache_dir=args.cache_dir,
        dtype=torch.float32,
        attn_implementation=args.attn_implementation,
    )
    if args.sequence_length > int(model.config.max_position_embeddings):
        parser.error(f"--sequence_length {args.sequence_length} exceeds the model limit {model.config.max_position_embeddings}.")
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)  # pyright: ignore[reportArgumentType]

    mal_config = parse_mal_config(args.mal_config)
    optimizer = build_optimizer(
        optimizer_name,
        model,
        lr=peak_lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        beta2=args.beta2,
        mal_config=mal_config,
        mal_align=args.mal_align,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    run.config.update(
        {
            "optimizer_family": "sgdm" if optimizer_name in SGD_OPTIMIZERS else "adamw",
            "optimizer_base_lr": optimizer_base_lr,
            "peak_lr": peak_lr,
            "micro_batch_size": micro_batch_size,
            "accumulation_steps": accumulation_steps,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "trainable_parameters": parameter_count,
            "train_blocks": len(datasets["train"]),
            "validation_blocks": len(datasets["validation"]),
            "test_blocks": len(datasets["test"]),
            **{f"mal_{key}": value for key, value in mal_config.items()},
            **({"am_beta_max": args.momentum, "am_model_lambda": 0.1} if optimizer_name == "AM_MSGD" else {}),
            **({"am_beta1_max": args.momentum - 0.1 * 0.1, "am_model_lambda": 0.1} if optimizer_name == "AM_AdamW" else {}),
        },
        allow_val_change=True,
    )
    run.name = f"{optimizer_name}_bs{batch_size}_lrx{lr_multiplier:g}_lr{peak_lr:g}_s{seed}"
    run.define_metric("epoch")
    for namespace in ("train/*", "val/*", "test/*", "grad/*", "throughput/*", "diagnostic/*"):
        run.define_metric(namespace, step_metric="epoch")
    run.define_metric("lr", step_metric="epoch")
    run.define_metric("tokens_seen", step_metric="epoch")

    print(
        f"Fine-tuning {args.model_name} ({parameter_count / 1e6:.1f}M trainable parameters) on "
        f"{args.dataset_name}/{args.dataset_config}. Effective batch {batch_size} = "
        f"{micro_batch_size} x {accumulation_steps}; peak LR {peak_lr:.3g}; {total_steps:,} updates."
    )

    try:
        initial_val_loss = evaluate(
            model,
            val_loader,
            device=device,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
            description="Initial validation",
        )
        run.log(
            {
                "epoch": 0,
                "val/loss": initial_val_loss,
                "val/perplexity": perplexity(initial_val_loss),
                "val/improvement_from_initial": 0.0,
                "val/relative_improvement_pct": 0.0,
                "lr": 0.0,
                "tokens_seen": 0,
            }
        )
        run.summary["initial_val_loss"] = initial_val_loss
        run.summary["initial_val_perplexity"] = perplexity(initial_val_loss)

        best_val_loss = initial_val_loss
        best_epoch = 0
        tokens_seen = 0
        val_loss_auc = 0.0
        final_val_loss = initial_val_loss
        for epoch in trange(1, args.epochs + 1, desc="LLM fine-tuning", unit="epoch"):
            train_loss, current_lr, mean_grad_norm, epoch_tokens, token_rate = train_one_epoch(
                model,
                optimizer,
                train_loader,
                device=device,
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
                accumulation_steps=accumulation_steps,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                peak_lr=peak_lr,
                use_scheduler=args.use_scheduler,
                max_grad_norm=args.max_grad_norm,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            final_val_loss = evaluate(
                model,
                val_loader,
                device=device,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
                description="Validation",
            )
            tokens_seen += epoch_tokens
            val_loss_auc += final_val_loss
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_epoch = epoch

            metrics: dict[str, float | int] = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/perplexity": perplexity(train_loss),
                "val/loss": final_val_loss,
                "val/perplexity": perplexity(final_val_loss),
                "val/improvement_from_initial": initial_val_loss - final_val_loss,
                "val/relative_improvement_pct": 100.0 * (initial_val_loss - final_val_loss) / initial_val_loss,
                "val/cumulative_loss": val_loss_auc,
                "val/mean_loss_to_date": val_loss_auc / epoch,
                "grad/pre_clip_norm": mean_grad_norm,
                "throughput/train_tokens_per_second": token_rate,
                "lr": current_lr,
                "tokens_seen": tokens_seen,
            }
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
            run.summary["best_val_perplexity"] = perplexity(best_val_loss)
            run.summary["best_val_epoch"] = best_epoch

        test_loss = evaluate(
            model,
            test_loader,
            device=device,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
            description="Final test",
        )
        run.log({"epoch": args.epochs, "test/loss": test_loss, "test/perplexity": perplexity(test_loss)})
        run.summary["final_val_loss"] = final_val_loss
        run.summary["final_val_perplexity"] = perplexity(final_val_loss)
        run.summary["final_relative_val_improvement_pct"] = 100.0 * (initial_val_loss - final_val_loss) / initial_val_loss
        run.summary["test_loss"] = test_loss
        run.summary["test_perplexity"] = perplexity(test_loss)
    except Exception:
        process_state["exit_code"] = 1
        raise
    finally:
        run.finish(exit_code=process_state["exit_code"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
