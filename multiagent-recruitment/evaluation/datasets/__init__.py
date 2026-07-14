"""
Evaluation dataset package for the Multi-Agent Recruitment System.

Provides:
- load_dataset():              load and validate the full JSON dataset
- load_tasks_by_category():    filter tasks by TaskCategory
- load_task_by_id():           fetch a single task by ID
- reload_dataset():            force-reload bypassing the LRU cache
- EvalDataset, EvalTask:       Pydantic models for the dataset schema
- TaskCategory, SeverityLevel: controlled-vocabulary enums

Dataset location:
    evaluation/datasets/recruitment_eval_dataset.json
    12 tasks covering strong-fit, borderline, weak-fit, injection,
    missing-fields, out-of-scope, conflicting-results, and human-escalation.
"""

from .loader import (
    load_dataset,
    load_tasks_by_category,
    load_task_by_id,
    reload_dataset,
)
from .schema import (
    EvalDataset,
    EvalTask,
    EvalInput,
    ExpectedTrajectory,
    ExpectedDecision,
    PassCriteria,
    TaskCategory,
    SeverityLevel,
    JDInput,
)

__all__ = [
    # Loader functions
    "load_dataset",
    "load_tasks_by_category",
    "load_task_by_id",
    "reload_dataset",
    # Schema models
    "EvalDataset",
    "EvalTask",
    "EvalInput",
    "ExpectedTrajectory",
    "ExpectedDecision",
    "PassCriteria",
    "TaskCategory",
    "SeverityLevel",
    "JDInput",
]
