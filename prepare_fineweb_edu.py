"""Materialize deterministic, document-disjoint FineWeb-Edu token splits.

The output is a set of raw little-endian int32 token streams plus a metadata
receipt. Documents are assigned to train/dev/test by a stable hash of their
FineWeb identifier, so no document can cross evaluation boundaries. Each
document is terminated by the pinned SmolLM2 tokenizer's EOS token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-360M"
TOKENIZER_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
DEFAULT_TRAIN_TOKENS = 1_499_463_680  # 5,720 updates x 262,144 tokens/update.
DEFAULT_EVAL_TOKENS = 8_388_608  # 4,096 sequences at context length 2,048.
SPLIT_HASH_MODULUS = 1_000
DEV_BUCKETS = frozenset(range(10))
TEST_BUCKETS = frozenset(range(10, 20))


@dataclass
class SplitWriter:
    name: str
    target_tokens: int
    final_path: Path

    def __post_init__(self) -> None:
        self.temporary_path = self.final_path.with_suffix(self.final_path.suffix + ".partial")
        self.handle = self.temporary_path.open("wb")
        self.tokens = 0
        self.documents = 0
        self.digest = hashlib.sha256()

    @property
    def complete(self) -> bool:
        return self.tokens >= self.target_tokens

    def write(self, token_ids: list[int]) -> None:
        if self.complete or not token_ids:
            return
        remaining = self.target_tokens - self.tokens
        values = np.asarray(token_ids[:remaining], dtype="<i4")
        payload = values.tobytes(order="C")
        self.handle.write(payload)
        self.digest.update(payload)
        self.tokens += len(values)
        self.documents += 1

    def finish(self) -> dict[str, Any]:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        if self.tokens != self.target_tokens:
            raise RuntimeError(f"{self.name} contains {self.tokens:,} tokens; expected {self.target_tokens:,}.")
        self.temporary_path.replace(self.final_path)
        return {
            "path": self.final_path.name,
            "tokens": self.tokens,
            "documents_contributing_tokens": self.documents,
            "bytes": self.final_path.stat().st_size,
            "sha256": self.digest.hexdigest(),
        }

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()
        self.temporary_path.unlink(missing_ok=True)


def stable_split(document_id: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(document_id.encode("utf-8")).digest()[:8], "big") % SPLIT_HASH_MODULUS
    if bucket in DEV_BUCKETS:
        return "dev"
    if bucket in TEST_BUCKETS:
        return "test"
    return "train"


def expected_metadata(train_tokens: int, eval_tokens: int) -> dict[str, Any]:
    return {
        "format_version": 1,
        "dtype": "int32-little-endian",
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_revision": TOKENIZER_REVISION,
        "train_tokens": train_tokens,
        "dev_tokens": eval_tokens,
        "test_tokens": eval_tokens,
        "split_strategy": "sha256(document_id) modulo 1000: dev=0..9, test=10..19, train=20..999",
        "document_separator": "eos_token_id",
    }


def existing_dataset_is_valid(output_dir: Path, expected: dict[str, Any]) -> bool:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except OSError, json.JSONDecodeError:
        return False
    if any(metadata.get(key) != value for key, value in expected.items()):
        return False
    for split, tokens in (("train", expected["train_tokens"]), ("dev", expected["dev_tokens"]), ("test", expected["test_tokens"])):
        path = output_dir / f"{split}.bin"
        if not path.is_file() or path.stat().st_size != int(tokens) * np.dtype("<i4").itemsize:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    parser.add_argument("--train_tokens", "--train-tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--eval_tokens", "--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS)
    parser.add_argument("--tokenizer_batch_size", "--tokenizer-batch-size", type=int, default=128)
    parser.add_argument("--cache_dir", "--cache-dir", type=Path)
    parser.add_argument(
        "--hard_exit_after_success",
        "--hard-exit-after-success",
        action="store_true",
        help="Exit without waiting for third-party streaming cleanup threads after every output has been fsynced.",
    )
    args = parser.parse_args()
    if args.train_tokens <= 0 or args.eval_tokens <= 0:
        parser.error("token counts must be positive")
    if args.tokenizer_batch_size <= 0:
        parser.error("--tokenizer_batch_size must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = expected_metadata(args.train_tokens, args.eval_tokens)
    if existing_dataset_is_valid(output_dir, expected):
        print(f"Reusing verified-size FineWeb-Edu token streams in {output_dir}")
        return 0

    for partial in output_dir.glob("*.partial"):
        partial.unlink()

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        revision=TOKENIZER_REVISION,
        cache_dir=args.cache_dir,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("The pinned SmolLM2 tokenizer has no EOS token.")

    writers = {
        "train": SplitWriter("train", args.train_tokens, output_dir / "train.bin"),
        "dev": SplitWriter("dev", args.eval_tokens, output_dir / "dev.bin"),
        "test": SplitWriter("test", args.eval_tokens, output_dir / "test.bin"),
    }
    stream = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        revision=DATASET_REVISION,
        streaming=True,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    progress = tqdm(total=sum(writer.target_tokens for writer in writers.values()), unit="token", unit_scale=True, desc="Tokenizing FineWeb-Edu")
    source_documents = 0
    try:
        for batch in stream.iter(batch_size=args.tokenizer_batch_size):
            texts = batch["text"]
            identifiers = batch.get("id")
            if identifiers is None:
                identifiers = batch.get("url")
            if identifiers is None:
                identifiers = [f"source-row-{source_documents + index}" for index in range(len(texts))]
            encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
            for identifier, token_ids in zip(identifiers, encoded, strict=True):
                source_documents += 1
                split = stable_split(str(identifier))
                writer = writers[split]
                if writer.complete:
                    continue
                before = writer.tokens
                writer.write([*token_ids, tokenizer.eos_token_id])
                progress.update(writer.tokens - before)
            if all(writer.complete for writer in writers.values()):
                break
        if not all(writer.complete for writer in writers.values()):
            missing = {name: writer.target_tokens - writer.tokens for name, writer in writers.items() if not writer.complete}
            raise RuntimeError(f"FineWeb-Edu stream ended before targets were met: {missing}")

        split_metadata = {name: writer.finish() for name, writer in writers.items()}
        metadata = {
            **expected,
            "eos_token_id": tokenizer.eos_token_id,
            "vocab_size": len(tokenizer),
            "source_documents_scanned": source_documents,
            "splits": split_metadata,
        }
        temporary_metadata = output_dir / "metadata.json.partial"
        temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        temporary_metadata.replace(output_dir / "metadata.json")
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise
    finally:
        progress.close()

    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    if args.hard_exit_after_success:
        # On the AUS Python 3.14 stack, datasets/fsspec can leave a helper
        # thread blocked after a streaming iterator reaches its target. All
        # payloads and metadata are already fsynced and atomically renamed.
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
