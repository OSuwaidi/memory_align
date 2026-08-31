"""Idempotently download the datasets and model assets used by this project.

Examples::

    uv run download_datasets.py --task cifar100
    uv run download_datasets.py --task tiny-imagenet
    uv run download_datasets.py --task llm
    uv run download_datasets.py --task all

The LLM training entry point can populate its cache on demand; the ``llm`` task
exists so downloads and token packing can be completed before a GPU job starts.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

TINY_IMAGENET_URL = "https://huggingface.co/datasets/cnak47/cv-datasets/resolve/main/tiny-imagenet-200.zip"
TINY_IMAGENET_MD5 = "90528d7ca1a48142e341f4ef8d21d0de"
TINY_IMAGENET_DIRECTORY = "tiny-imagenet-200"
CIFAR100_URL = "https://huggingface.co/datasets/nakroy/cifar100-python/resolve/main/cifar-100-python.tar.gz"
CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
CIFAR100_ARCHIVE = "cifar-100-python.tar.gz"
CIFAR100_DIRECTORY = "cifar-100-python"


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "memory-align-dataset-downloader/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary_path.open("wb") as output:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    print(f"\rDownloading {destination.name}: {100.0 * downloaded / total_size:5.1f}%", end="", flush=True)
            if total_size:
                print()
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    extraction_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if not member_path.is_relative_to(extraction_root):
                raise ValueError(f'Unsafe path in archive: "{member.filename}"')
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise ValueError(f'Archive contains an unsupported symbolic link: "{member.filename}"')
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f'ZIP integrity check failed at "{bad_member}".')
        archive.extractall(destination)


def validate_tiny_imagenet(root: Path) -> None:
    required = (root / "train", root / "val" / "images", root / "val" / "val_annotations.txt", root / "wnids.txt")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Tiny-ImageNet is incomplete; missing {missing[0]}")

    class_ids = [line.strip() for line in (root / "wnids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(class_ids) != 200:
        raise ValueError(f"Expected 200 Tiny-ImageNet class IDs, found {len(class_ids)}.")
    absent_class_dirs = [class_id for class_id in class_ids if not (root / "train" / class_id / "images").is_dir()]
    if absent_class_dirs:
        raise FileNotFoundError(f'Missing training directory for class "{absent_class_dirs[0]}".')

    train_image_count = sum(1 for _path in (root / "train").glob("*/images/*.JPEG"))
    if train_image_count != 100_000:
        raise ValueError(f"Expected 100,000 Tiny-ImageNet training images, found {train_image_count}.")

    annotation_count = sum(1 for line in (root / "val" / "val_annotations.txt").read_text(encoding="utf-8").splitlines() if line)
    if annotation_count != 10_000:
        raise ValueError(f"Expected 10,000 Tiny-ImageNet validation annotations, found {annotation_count}.")
    validation_image_count = sum(1 for _path in (root / "val" / "images").glob("*.JPEG"))
    if validation_image_count != 10_000:
        raise ValueError(f"Expected 10,000 Tiny-ImageNet validation images, found {validation_image_count}.")


def download_tiny_imagenet(parent_dir: Path, *, keep_archive: bool) -> Path:
    parent_dir = parent_dir.expanduser().resolve()
    dataset_root = parent_dir / TINY_IMAGENET_DIRECTORY
    if dataset_root.exists():
        validate_tiny_imagenet(dataset_root)
        print(f"Tiny-ImageNet is already complete at {dataset_root}")
        return dataset_root

    archive_path = parent_dir / f"{TINY_IMAGENET_DIRECTORY}.zip"
    if archive_path.exists():
        checksum = file_md5(archive_path)
        if checksum != TINY_IMAGENET_MD5:
            raise ValueError(f"Existing archive has MD5 {checksum}, expected {TINY_IMAGENET_MD5}: {archive_path}. Remove or rename it before retrying.")
        print(f"Using verified existing archive {archive_path}")
    else:
        print(f"Downloading Tiny-ImageNet from {TINY_IMAGENET_URL}")
        download_file(TINY_IMAGENET_URL, archive_path)
        checksum = file_md5(archive_path)
        if checksum != TINY_IMAGENET_MD5:
            archive_path.unlink(missing_ok=True)
            raise ValueError(f"Downloaded Tiny-ImageNet MD5 {checksum}; expected {TINY_IMAGENET_MD5}.")

    parent_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tiny-imagenet-extract-", dir=parent_dir) as temporary_dir:
        temporary_root = Path(temporary_dir)
        _safe_extract_zip(archive_path, temporary_root)
        extracted_root = temporary_root / TINY_IMAGENET_DIRECTORY
        validate_tiny_imagenet(extracted_root)
        if dataset_root.exists():
            validate_tiny_imagenet(dataset_root)
        else:
            extracted_root.replace(dataset_root)

    if not keep_archive:
        archive_path.unlink(missing_ok=True)
    print(f"Tiny-ImageNet is ready at {dataset_root}")
    return dataset_root


def download_cifar100(parent_dir: Path, *, keep_archive: bool) -> Path:
    """Populate torchvision's CIFAR-100 layout using a faster verified mirror."""
    from torchvision.datasets import CIFAR100

    parent_dir = parent_dir.expanduser().resolve()
    dataset_root = parent_dir / CIFAR100_DIRECTORY

    # torchvision validates every extracted batch file, so this is stronger
    # than checking for the directory alone and makes repeated calls cheap.
    try:
        CIFAR100(root=str(parent_dir), train=True, download=False)
        CIFAR100(root=str(parent_dir), train=False, download=False)
    except RuntimeError:
        archive_path = parent_dir / CIFAR100_ARCHIVE
        if archive_path.exists():
            checksum = file_md5(archive_path)
            if checksum != CIFAR100_MD5:
                raise ValueError(
                    f"Existing archive has MD5 {checksum}, expected {CIFAR100_MD5}: "
                    f"{archive_path}. Remove or rename it before retrying."
                )
            print(f"Using verified existing archive {archive_path}")
        else:
            print(f"Downloading CIFAR-100 from {CIFAR100_URL}")
            download_file(CIFAR100_URL, archive_path)
            checksum = file_md5(archive_path)
            if checksum != CIFAR100_MD5:
                archive_path.unlink(missing_ok=True)
                raise ValueError(f"Downloaded CIFAR-100 MD5 {checksum}; expected {CIFAR100_MD5}.")

        # With the verified archive already at torchvision's expected path,
        # download=True performs only its built-in integrity check/extraction.
        CIFAR100(root=str(parent_dir), train=True, download=True)
        CIFAR100(root=str(parent_dir), train=False, download=True)
        if not keep_archive:
            archive_path.unlink(missing_ok=True)

    print(f"CIFAR-100 is ready at {dataset_root}")
    return dataset_root


def prefetch_llm_assets(
    cache_dir: Path,
    *,
    model_name: str,
    model_revision: str,
    dataset_name: str,
    dataset_config: str,
    dataset_revision: str,
    sequence_length: int,
) -> Path:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from tasks.llm_finetune import load_token_blocks

    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading the pinned model/tokenizer snapshot for {model_name}")
    snapshot_download(
        repo_id=model_name,
        revision=model_revision,
        cache_dir=cache_dir,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "merges.txt", "vocab.json"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision, cache_dir=cache_dir)
    splits = load_token_blocks(
        tokenizer,
        model_name=model_name,
        model_revision=model_revision,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_revision=dataset_revision,
        sequence_length=sequence_length,
        cache_dir=cache_dir,
    )
    split_sizes = ", ".join(f"{name}={len(split):,}" for name, split in splits.items())
    print(f"LLM assets and packed {sequence_length}-token blocks are ready in {cache_dir} ({split_sizes}).")
    return cache_dir


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=("cifar100", "tiny-imagenet", "llm", "all"), default="all")
    parser.add_argument("--cifar100_dir", "--cifar100-dir", default="./data")
    parser.add_argument("--tiny_imagenet_dir", "--tiny-imagenet-dir", default="./data")
    parser.add_argument("--keep_archive", "--keep-archive", action="store_true")
    parser.add_argument("--llm_cache_dir", "--llm-cache-dir", default="./data/llm_cache")
    parser.add_argument("--model_name", "--model-name")
    parser.add_argument("--model_revision", "--model-revision")
    parser.add_argument("--dataset_name", "--dataset-name")
    parser.add_argument("--dataset_config", "--dataset-config")
    parser.add_argument("--dataset_revision", "--dataset-revision")
    parser.add_argument("--sequence_length", "--sequence-length", type=int, default=512)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    if args.sequence_length <= 1:
        parser.error("--sequence_length must be greater than one.")

    if args.task in ("cifar100", "all"):
        download_cifar100(Path(args.cifar100_dir), keep_archive=args.keep_archive)
    if args.task in ("tiny-imagenet", "all"):
        download_tiny_imagenet(Path(args.tiny_imagenet_dir), keep_archive=args.keep_archive)
    if args.task in ("llm", "all"):
        from tasks.llm_finetune import (
            DEFAULT_DATASET,
            DEFAULT_DATASET_CONFIG,
            DEFAULT_DATASET_REVISION,
            DEFAULT_MODEL,
            DEFAULT_MODEL_REVISION,
        )

        prefetch_llm_assets(
            Path(args.llm_cache_dir),
            model_name=args.model_name or DEFAULT_MODEL,
            model_revision=args.model_revision or DEFAULT_MODEL_REVISION,
            dataset_name=args.dataset_name or DEFAULT_DATASET,
            dataset_config=args.dataset_config or DEFAULT_DATASET_CONFIG,
            dataset_revision=args.dataset_revision or DEFAULT_DATASET_REVISION,
            sequence_length=args.sequence_length,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nDownload interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
