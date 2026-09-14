"""From-scratch SmolLM2 pre-training for optimizer benchmarking.

The task consumes deterministic packed token streams produced by
``prepare_fineweb_edu.py``. It initializes the pinned SmolLM2 architecture from
configuration only, trains for a fixed number of tokens, and logs held-out loss
against both tokens and measured training time. Hyperparameters are carried in
one ``optimizer_case`` value so W&B never forms unintended Cartesian products.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import socket
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import wandb
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from optims.am_opt import AM_AdamW
from optims.mal_opt import MAL_AdamW
from optims.tam_opt import AdaTAMW
from tasks.wandb_metadata import task_metadata

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
OPTIMIZERS = ("AdamW", "AM_AdamW", "AdaTAMW", "AGM_AdamW")
AGM_CONFIG = "False,1.0,step,attenuate,update,complement"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


@dataclass(frozen=True)
class OptimizerCase:
    optimizer: str
    learning_rate: float
    weight_decay: float

    @classmethod
    def parse(cls, value: str) -> OptimizerCase:
        fields = value.split("::")
        if len(fields) != 3:
            raise ValueError("optimizer_case must be '<optimizer>::<learning_rate>::<weight_decay>'.")
        optimizer, learning_rate_text, weight_decay_text = fields
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer {optimizer!r}; choose one of {OPTIMIZERS}.")
        learning_rate = float(learning_rate_text)
        weight_decay = float(weight_decay_text)
        if learning_rate <= 0.0 or weight_decay < 0.0:
            raise ValueError("Learning rate must be positive and weight decay non-negative.")
        return cls(optimizer, learning_rate, weight_decay)


class PackedTokenDataset(Dataset[torch.Tensor]):
    def __init__(self, path: Path, sequence_length: int) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        size = path.stat().st_size
        itemsize = np.dtype("<i4").itemsize
        if size % itemsize:
            raise ValueError(f"Token file has a partial int32 value: {path}")
        self.tokens = np.memmap(path, mode="r", dtype="<i4")
        self.sequence_length = sequence_length
        self.num_sequences = len(self.tokens) // sequence_length
        if self.num_sequences == 0:
            raise ValueError(f"Token file is shorter than one sequence: {path}")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> torch.Tensor:
        start = index * self.sequence_length
        values = np.array(self.tokens[start : start + self.sequence_length], dtype=np.int64, copy=True)
        return torch.from_numpy(values)


class ChunkShuffleSampler(Sampler[int]):
    """Shuffle large contiguous chunks while preserving local I/O locality."""

    def __init__(self, size: int, *, seed: int, chunk_size: int, start_index: int = 0) -> None:
        if size <= 0 or chunk_size <= 0:
            raise ValueError("sampler size and chunk_size must be positive")
        if not 0 <= start_index <= size:
            raise ValueError("start_index lies outside the sampler")
        self.size = size
        self.seed = seed
        self.chunk_size = chunk_size
        self.start_index = start_index

    def __len__(self) -> int:
        return self.size - self.start_index

    def __iter__(self) -> Iterator[int]:
        chunks = np.arange(math.ceil(self.size / self.chunk_size))
        np.random.default_rng(self.seed).shuffle(chunks)
        skipped = 0
        for chunk in chunks:
            start = int(chunk) * self.chunk_size
            end = min(start + self.chunk_size, self.size)
            length = end - start
            if skipped + length <= self.start_index:
                skipped += length
                continue
            local_start = max(start, start + self.start_index - skipped)
            yield from range(local_start, end)
            skipped += length


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_worker_seed(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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


def build_optimizer(
    case: OptimizerCase,
    model: nn.Module,
    *,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> Optimizer:
    parameters: Iterable[nn.Parameter] = model.parameters()
    if case.optimizer == "AdamW":
        return AdamW(
            split_weight_decay_params(model, case.weight_decay),
            lr=case.learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            fused=True,
        )
    if case.optimizer == "AM_AdamW":
        model_lambda = 0.1
        return AM_AdamW(
            parameters,
            lr=case.learning_rate,
            betas=(beta1 - 0.1 * model_lambda, beta2),
            eps=epsilon,
            weight_decay=case.weight_decay,
            model_lambda=model_lambda,
        )
    if case.optimizer == "AdaTAMW":
        return AdaTAMW(
            parameters,
            lr=case.learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=case.weight_decay,
        )
    if case.optimizer == "AGM_AdamW":
        return MAL_AdamW(
            parameters,
            lr=case.learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=case.weight_decay,
            pwr=1.0,
            align="update",
            in_place=False,
            scale="step",
            gate_mode="attenuate",
            gradient_weight_mode="complement",
        )
    raise AssertionError(f"Optimizer dispatch is incomplete for {case.optimizer!r}.")


def scheduled_learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    minimum_lr_ratio: float,
) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
    multiplier = minimum_lr_ratio + 0.5 * (1.0 - minimum_lr_ratio) * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return peak_lr * multiplier


def set_optimizer_lr(optimizer: Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[torch.Tensor],
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    description: str,
) -> float:
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    prediction_tokens = 0
    for input_ids in tqdm(loader, desc=description, unit="batch", leave=False):
        input_ids = input_ids.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True):
            loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        batch_prediction_tokens = input_ids.shape[0] * (input_ids.shape[1] - 1)
        loss_sum.add_(loss.detach().to(dtype=torch.float64), alpha=batch_prediction_tokens)
        prediction_tokens += batch_prediction_tokens
    model.train()
    return float(loss_sum / prediction_tokens)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def snapshot_model(model: nn.Module, path: Path) -> None:
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    atomic_torch_save(state, path)


def load_data_metadata(data_dir: Path) -> dict[str, Any]:
    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "tokenizer_name": MODEL_NAME,
        "tokenizer_revision": MODEL_REVISION,
        "dtype": "int32-little-endian",
    }
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatches:
        raise ValueError(f"FineWeb-Edu metadata does not match the pinned recipe: {mismatches}")
    for split in ("train", "dev", "test"):
        path = data_dir / f"{split}.bin"
        expected_bytes = int(metadata[f"{split}_tokens"]) * np.dtype("<i4").itemsize
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ValueError(f"Invalid or missing {split} token stream: {path}")
    return metadata


def config_fingerprint(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", "--data-dir", type=Path, required=True)
    parser.add_argument("--optimizer_case", "--optimizer-case", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence_length", "--sequence-length", type=int, default=2048)
    parser.add_argument("--micro_batch_size", "--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", "--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max_steps", "--max-steps", type=int, required=True)
    parser.add_argument("--warmup_ratio", "--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--minimum_lr_ratio", "--minimum-lr-ratio", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", "--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log_every", "--log-every", type=int, default=10)
    parser.add_argument("--eval_every", "--eval-every", type=int, default=250)
    parser.add_argument("--eval_batch_size", "--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval_max_sequences", "--eval-max-sequences", type=int, default=0)
    parser.add_argument("--shuffle_chunk_size", "--shuffle-chunk-size", type=int, default=4096)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=-1)
    parser.add_argument("--checkpoint_every", "--checkpoint-every", type=int, default=500)
    parser.add_argument("--output_dir", "--output-dir", type=Path, default=Path("./outputs/llm-pretrain"))
    parser.add_argument("--evaluate_test", "--evaluate-test", type=parse_bool, default=False)
    parser.add_argument("--keep_checkpoints", "--keep-checkpoints", type=parse_bool, default=False)
    parser.add_argument("--amp_dtype", "--amp-dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--float32_precision", "--float32-precision", choices=("tf32", "ieee"), default="tf32")
    parser.add_argument("--wandb_entity", "--wandb-entity", default="osuwaidi-khalifa-university")
    parser.add_argument("--wandb_project", "--wandb-project", default="MAL_benchmark")
    parser.add_argument("--wandb_mode", "--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    parser.add_argument("--study_stage", "--study-stage", default="direct")
    parser.add_argument("--selection_rule", "--selection-rule")
    parser.add_argument("--hyperparameter_selection_receipt", "--hyperparameter-selection-receipt")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("This benchmark requires CUDA.")
    for name in ("sequence_length", "micro_batch_size", "gradient_accumulation_steps", "max_steps", "log_every", "eval_every", "eval_batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0 or not 0.0 <= args.minimum_lr_ratio <= 1.0:
        parser.error("warmup_ratio and minimum_lr_ratio must lie in [0, 1), [0, 1] respectively")
    if not 0.0 <= args.beta1 < 1.0 or not 0.0 <= args.beta2 < 1.0:
        parser.error("beta coefficients must lie in [0, 1)")
    if args.eval_max_sequences < 0:
        parser.error("--eval_max_sequences must be non-negative")

    data_dir = args.data_dir.expanduser().resolve()
    metadata = load_data_metadata(data_dir)
    case = OptimizerCase.parse(args.optimizer_case)
    tokens_per_update = args.sequence_length * args.micro_batch_size * args.gradient_accumulation_steps
    token_budget = tokens_per_update * args.max_steps
    if token_budget > int(metadata["train_tokens"]):
        parser.error(f"Requested {token_budget:,} tokens but the materialized training stream has {metadata['train_tokens']:,}.")

    fixed_config = {
        **task_metadata(
            task="fineweb_edu_causal_lm_pretraining",
            task_type="causal_language_model_pretraining",
            model_name="SmolLM2-360M-random-init",
            model_source="HuggingFaceTB architecture config",
            dataset_name=DATASET_NAME,
            dataset_config=DATASET_CONFIG,
            dataset_source="Hugging Face FineWeb-Edu",
            training_regime="from_scratch_fixed_token_budget",
        ),
        **vars(args),
        "data_dir": str(data_dir),
        "output_dir": str(args.output_dir),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_initialization": "random_from_pinned_config",
        "dataset_revision": DATASET_REVISION,
        "method_name": "AGM-AdamW" if case.optimizer == "AGM_AdamW" else case.optimizer,
        "optimizer": case.optimizer,
        "learning_rate": case.learning_rate,
        "weight_decay": case.weight_decay,
        "AGM_config": AGM_CONFIG if case.optimizer == "AGM_AdamW" else None,
        "effective_batch_tokens": tokens_per_update,
        "training_token_budget": token_budget,
        "dev_tokens": metadata["dev_tokens"],
        "test_tokens": metadata["test_tokens"],
        "data_split_strategy": metadata["split_strategy"],
        "data_split_sha256": {split: metadata["splits"][split]["sha256"] for split in ("train", "dev", "test")},
        "weight_decay_rule": "decoupled; excluded for bias and one-dimensional parameters",
    }
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        mode=args.wandb_mode,
        job_type="llm-pretrain",
        tags=("llm-pretraining", "fineweb-edu", "smollm2-360m", "optimizer-benchmark"),
        config=fixed_config,
    )
    config = dict(run.config)
    case = OptimizerCase.parse(str(config["optimizer_case"]))
    seed = int(config["seed"])
    run.name = f"{case.optimizer}_lr{case.learning_rate:g}_wd{case.weight_decay:g}_tok{token_budget}_s{seed}"

    device = torch.device("cuda")
    torch.backends.cuda.matmul.fp32_precision = args.float32_precision
    torch.backends.cudnn.conv.fp32_precision = args.float32_precision  # pyright: ignore[reportAttributeAccessIssue]
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The allocated GPU does not support bfloat16.")
    amp_dtype = torch.bfloat16
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, use_fast=True)
    model_config = AutoConfig.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model_config.use_cache = False
    model_config.dtype = "float32"
    model = AutoModelForCausalLM.from_config(model_config, attn_implementation="sdpa", dtype=torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    tokenizer_size = len(tokenizer)
    metadata_vocab_size = int(metadata["vocab_size"])
    if tokenizer_size != model_config.vocab_size or tokenizer_size != metadata_vocab_size:
        raise RuntimeError(f"Tokenizer/model/data vocabulary mismatch: {tokenizer_size} != {model_config.vocab_size} != {metadata_vocab_size}")

    optimizer = build_optimizer(case, model, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)
    warmup_steps = round(args.max_steps * args.warmup_ratio)
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
    num_workers = min(allocated_cpus, 8) if args.num_workers < 0 else args.num_workers

    train_dataset = PackedTokenDataset(data_dir / "train.bin", args.sequence_length)
    dev_dataset = PackedTokenDataset(data_dir / "dev.bin", args.sequence_length)
    test_dataset = PackedTokenDataset(data_dir / "test.bin", args.sequence_length)
    if args.eval_max_sequences:
        dev_dataset = Subset(dev_dataset, range(min(args.eval_max_sequences, len(dev_dataset))))
        test_dataset = Subset(test_dataset, range(min(args.eval_max_sequences, len(test_dataset))))
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=min(num_workers, 4), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=min(num_workers, 4), pin_memory=True)

    resume_identity = {
        "optimizer_case": str(config["optimizer_case"]),
        "seed": seed,
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
    }
    checkpoint_dir = args.output_dir.expanduser().resolve() / config_fingerprint(resume_identity)
    last_checkpoint = checkpoint_dir / "checkpoint-last.pt"
    best_checkpoint = checkpoint_dir / "checkpoint-best-model.pt"
    start_step = 0
    training_elapsed = 0.0
    optimizer_elapsed = 0.0
    cumulative_loss_sum = 0.0
    cumulative_loss_count = 0
    best_val_loss = math.inf
    best_val_step = 0
    if last_checkpoint.is_file():
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("identity") != resume_identity:
            raise RuntimeError(f"Checkpoint identity mismatch at {last_checkpoint}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["next_step"])
        training_elapsed = float(checkpoint["training_elapsed"])
        optimizer_elapsed = float(checkpoint["optimizer_elapsed"])
        cumulative_loss_sum = float(checkpoint["cumulative_loss_sum"])
        cumulative_loss_count = int(checkpoint["cumulative_loss_count"])
        best_val_loss = float(checkpoint["best_val_loss"])
        best_val_step = int(checkpoint["best_val_step"])
        del checkpoint
        print(f"Resuming {case.optimizer} at optimizer step {start_step:,}/{args.max_steps:,}.")

    sequences_consumed = start_step * args.micro_batch_size * args.gradient_accumulation_steps
    sampler = ChunkShuffleSampler(
        len(train_dataset),
        seed=seed + 10_000,
        chunk_size=args.shuffle_chunk_size,
        start_index=sequences_consumed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        worker_init_fn=set_worker_seed,
    )
    train_iterator = iter(train_loader)

    run.config.update(
        {
            "trainable_parameters": parameter_count,
            "warmup_steps": warmup_steps,
            "optimizer_family": "adamw",
            "optimizer_implementation": type(optimizer).__name__,
            "resumed_from_step": start_step,
            "num_workers": num_workers,
            "attention_implementation": "sdpa",
            "gradient_checkpointing": True,
            "parameter_dtype": "float32",
            "compute_dtype": "bfloat16",
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_count": 1,
            "host_name": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "transformers_version": transformers.__version__,
            "optimizer_backend": "torch_fused" if case.optimizer == "AdamW" else "repository_reference",
        },
        allow_val_change=True,
    )
    run.define_metric("tokens_seen")
    run.define_metric("optimizer_step")
    run.define_metric("train_elapsed_seconds")
    run.define_metric("train/*", step_metric="tokens_seen")
    run.define_metric("val/*", step_metric="tokens_seen")
    run.define_metric("wall_clock/*", step_metric="train_elapsed_seconds")
    run.define_metric("throughput/*", step_metric="tokens_seen")

    def save_resume_checkpoint(next_step: int) -> None:
        atomic_torch_save(
            {
                "identity": resume_identity,
                "next_step": next_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "training_elapsed": training_elapsed,
                "optimizer_elapsed": optimizer_elapsed,
                "cumulative_loss_sum": cumulative_loss_sum,
                "cumulative_loss_count": cumulative_loss_count,
                "best_val_loss": best_val_loss,
                "best_val_step": best_val_step,
            },
            last_checkpoint,
        )

    run_started = time.perf_counter()
    initial_val_loss = evaluate(model, dev_loader, device=device, amp_dtype=amp_dtype, description="Initial validation")
    if initial_val_loss < best_val_loss:
        best_val_loss = initial_val_loss
        best_val_step = start_step
        if args.evaluate_test:
            snapshot_model(model, best_checkpoint)
    run.log(
        {
            "optimizer_step": start_step,
            "tokens_seen": start_step * tokens_per_update,
            "train_elapsed_seconds": training_elapsed,
            "val/loss": initial_val_loss,
            "val/perplexity": perplexity(initial_val_loss),
            "wall_clock/val_loss": initial_val_loss,
            "wall_clock/val_perplexity": perplexity(initial_val_loss),
        }
    )

    window_loss_sum = 0.0
    window_loss_count = 0
    final_val_loss = initial_val_loss
    model.train()
    progress = tqdm(range(start_step, args.max_steps), initial=start_step, total=args.max_steps, desc=case.optimizer, unit="step")
    for step in progress:
        learning_rate = scheduled_learning_rate(
            step,
            total_steps=args.max_steps,
            warmup_steps=warmup_steps,
            peak_lr=case.learning_rate,
            minimum_lr_ratio=args.minimum_lr_ratio,
        )
        set_optimizer_lr(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        update_started = time.perf_counter()
        update_loss_tensor = torch.zeros((), device=device, dtype=torch.float32)
        for _micro_step in range(args.gradient_accumulation_steps):
            input_ids = next(train_iterator).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
            (loss / args.gradient_accumulation_steps).backward()
            update_loss_tensor.add_(loss.detach().float())
        gradient_norm = float(clip_grad_norm_(model.parameters(), args.max_grad_norm))
        torch.cuda.synchronize()
        optimizer_started = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        optimizer_elapsed += time.perf_counter() - optimizer_started
        training_elapsed += time.perf_counter() - update_started

        # One host read per optimizer update keeps timing honest. Reading each
        # micro-batch loss separately would introduce an avoidable CUDA sync.
        update_loss = float(update_loss_tensor / args.gradient_accumulation_steps)
        cumulative_loss_sum += update_loss
        cumulative_loss_count += 1
        window_loss_sum += update_loss
        window_loss_count += 1
        completed_step = step + 1
        tokens_seen = completed_step * tokens_per_update
        progress.set_postfix(loss=f"{update_loss:.4f}", lr=f"{learning_rate:.2e}")

        if completed_step % args.log_every == 0 or completed_step == args.max_steps:
            mean_window_loss = window_loss_sum / window_loss_count
            run.log(
                {
                    "optimizer_step": completed_step,
                    "tokens_seen": tokens_seen,
                    "train_elapsed_seconds": training_elapsed,
                    "train/loss": mean_window_loss,
                    "train/cumulative_loss": cumulative_loss_sum / cumulative_loss_count,
                    "train/learning_rate": learning_rate,
                    "train/gradient_norm": gradient_norm,
                    "wall_clock/train_loss": mean_window_loss,
                    "throughput/tokens_per_second": tokens_seen / max(training_elapsed, 1e-9),
                    "throughput/optimizer_seconds_per_step": optimizer_elapsed / completed_step,
                    "throughput/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            )
            window_loss_sum = 0.0
            window_loss_count = 0

        should_evaluate = completed_step % args.eval_every == 0 or completed_step == args.max_steps
        if should_evaluate:
            final_val_loss = evaluate(model, dev_loader, device=device, amp_dtype=amp_dtype, description=f"Validation step {completed_step}")
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_val_step = completed_step
                if args.evaluate_test:
                    snapshot_model(model, best_checkpoint)
            run.log(
                {
                    "optimizer_step": completed_step,
                    "tokens_seen": tokens_seen,
                    "train_elapsed_seconds": training_elapsed,
                    "val/loss": final_val_loss,
                    "val/perplexity": perplexity(final_val_loss),
                    "wall_clock/val_loss": final_val_loss,
                    "wall_clock/val_perplexity": perplexity(final_val_loss),
                }
            )

        if args.checkpoint_every and completed_step % args.checkpoint_every == 0 and completed_step < args.max_steps:
            save_resume_checkpoint(completed_step)

    final_test_loss: float | None = None
    best_val_test_loss: float | None = None
    if args.evaluate_test:
        final_test_loss = evaluate(model, test_loader, device=device, amp_dtype=amp_dtype, description="Test at final token budget")
        if best_val_step == args.max_steps:
            best_val_test_loss = final_test_loss
        else:
            if not best_checkpoint.is_file():
                raise RuntimeError("The validation-selected model checkpoint is missing.")
            best_state = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(best_state)
            model.to(device)
            best_val_test_loss = evaluate(model, test_loader, device=device, amp_dtype=amp_dtype, description="Test at best validation")

    total_wall_clock = time.perf_counter() - run_started
    summary: dict[str, Any] = {
        "initial_val_loss": initial_val_loss,
        "final_val_loss": final_val_loss,
        "final/val_loss": final_val_loss,
        "final/val_perplexity": perplexity(final_val_loss),
        "best_val_loss": best_val_loss,
        "best/val_loss": best_val_loss,
        "best/val_perplexity": perplexity(best_val_loss),
        "best_val_step": best_val_step,
        "best_val_tokens": best_val_step * tokens_per_update,
        "final_tokens_seen": token_budget,
        "training_elapsed_seconds": training_elapsed,
        "total_wall_clock_seconds": total_wall_clock,
        "mean_tokens_per_second": token_budget / max(training_elapsed, 1e-9),
        "mean_optimizer_seconds_per_step": optimizer_elapsed / args.max_steps,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    if final_test_loss is not None and best_val_test_loss is not None:
        summary.update(
            {
                "test/loss_at_final_tokens": final_test_loss,
                "test/perplexity_at_final_tokens": perplexity(final_test_loss),
                "test/loss_at_best_val": best_val_test_loss,
                "test/perplexity_at_best_val": perplexity(best_val_test_loss),
            }
        )
    run.summary.update(summary)
    run.log(
        {
            "optimizer_step": args.max_steps,
            "tokens_seen": token_budget,
            "train_elapsed_seconds": training_elapsed,
            "wall_clock/total_seconds": total_wall_clock,
        }
    )
    run.finish()

    if not args.keep_checkpoints:
        last_checkpoint.unlink(missing_ok=True)
        best_checkpoint.unlink(missing_ok=True)
        try:
            checkpoint_dir.rmdir()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
