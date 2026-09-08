"""Canonical W&B task metadata shared by every training entry point."""

from __future__ import annotations

from typing import Any


def task_metadata(
    *,
    task: str,
    task_type: str,
    model_name: str,
    model_source: str,
    dataset_name: str,
    dataset_config: str,
    dataset_source: str,
    training_regime: str,
) -> dict[str, Any]:
    """Return the stable, filterable metadata contract for W&B runs."""
    required = {
        "task": task,
        "task_type": task_type,
        "model_name": model_name,
        "model_source": model_source,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_source": dataset_source,
        "training_regime": training_regime,
    }
    empty = [key for key, value in required.items() if not value.strip()]
    if empty:
        raise ValueError(f"W&B task metadata cannot be empty: {', '.join(empty)}")
    return {"logging_schema_version": 1, **required}
