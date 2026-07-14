"""
Dataset loader for the evaluation framework.

Loads recruitment_eval_dataset.json, validates it against the Pydantic
schema, and exposes helper functions for filtering tasks.

Design decisions:
- The dataset path is resolved relative to this file so tests can import
  the loader from any working directory.
- Validation happens at load time — a malformed dataset raises immediately
  rather than producing silent wrong results during test runs.
- Results are cached after the first load (module-level singleton) to
  avoid re-parsing the JSON on every test.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import ValidationError

from .schema import EvalDataset, EvalTask, TaskCategory

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_DATASET_DIR = Path(__file__).parent
_DEFAULT_DATASET_PATH = _DATASET_DIR / "recruitment_eval_dataset.json"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_dataset(path: Optional[str] = None) -> EvalDataset:
    """Load and validate the evaluation dataset from JSON.

    The result is cached after the first call. Subsequent calls with the
    same path return the cached instance without re-reading the file.

    Args:
        path: Optional path to a dataset JSON file. Defaults to
              evaluation/datasets/recruitment_eval_dataset.json.

    Returns:
        A fully validated EvalDataset instance.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        pydantic.ValidationError: If the dataset fails schema validation.
    """
    dataset_path = Path(path) if path else _DEFAULT_DATASET_PATH

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found at: {dataset_path}\n"
            f"Expected location: {_DEFAULT_DATASET_PATH}"
        )

    with dataset_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    try:
        dataset = EvalDataset.model_validate(raw)
    except ValidationError as e:
        raise ValueError(
            f"Evaluation dataset failed schema validation:\n{e}"
        ) from e

    return dataset


def load_tasks_by_category(category: TaskCategory | str, path: Optional[str] = None) -> List[EvalTask]:
    """Load all tasks matching a given category.

    Args:
        category: TaskCategory enum value or its string value
                  (e.g., 'injection_attack').
        path: Optional dataset file path.

    Returns:
        List of matching EvalTask instances (may be empty).
    """
    dataset = load_dataset(path)
    if isinstance(category, str):
        category = TaskCategory(category)
    return dataset.get_by_category(category)


def load_task_by_id(task_id: str, path: Optional[str] = None) -> EvalTask:
    """Load a single task by its ID.

    Args:
        task_id: The task ID string, e.g. 'task_001'.
        path: Optional dataset file path.

    Returns:
        The matching EvalTask instance.

    Raises:
        KeyError: If no task with the given ID exists.
    """
    dataset = load_dataset(path)
    task = dataset.get_by_id(task_id)
    if task is None:
        available = [t.id for t in dataset.tasks]
        raise KeyError(
            f"Task '{task_id}' not found in dataset. "
            f"Available IDs: {available}"
        )
    return task


def reload_dataset(path: Optional[str] = None) -> EvalDataset:
    """Force-reload the dataset, bypassing the cache.

    Useful in tests that modify the dataset file or need a fresh load.

    Args:
        path: Optional dataset file path.

    Returns:
        A freshly validated EvalDataset instance.
    """
    load_dataset.cache_clear()
    return load_dataset(path)
