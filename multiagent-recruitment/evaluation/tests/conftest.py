"""
Shared pytest fixtures for the evaluation framework.

These fixtures are available to all test modules under evaluation/tests/
without explicit import — pytest discovers conftest.py automatically.

Design decisions:
- All fixtures use the project root on sys.path so imports like
  `from graph.workflow import build_recruitment_graph` work regardless
  of the working directory pytest is invoked from.
- Demo mode is forced ON in all evaluation tests so no real API calls
  are made. This keeps tests deterministic and fast.
- State-builder fixtures match each task category so test modules can
  request a ready-made RecruitmentState without boilerplate.
- The `eval_dataset` fixture is session-scoped — the JSON is parsed once
  and shared across the entire test session.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Path setup — add project root to sys.path
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parents[2]  # evaluation/tests/ -> project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Force demo mode for all evaluation tests — no API calls
os.environ["RECRUITMENT_DEMO_MODE"] = "true"

# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------

from evaluation.datasets.loader import load_dataset, load_tasks_by_category, load_task_by_id
from evaluation.datasets.schema import (
    EvalDataset,
    EvalTask,
    TaskCategory,
)


@pytest.fixture(scope="session")
def eval_dataset() -> EvalDataset:
    """Load and validate the full evaluation dataset once per session."""
    return load_dataset()


@pytest.fixture(scope="session")
def strong_fit_tasks(eval_dataset: EvalDataset) -> List[EvalTask]:
    """All strong-fit evaluation tasks."""
    return eval_dataset.get_by_category(TaskCategory.STRONG_FIT)


@pytest.fixture(scope="session")
def borderline_tasks(eval_dataset: EvalDataset) -> List[EvalTask]:
    """All borderline evaluation tasks."""
    return eval_dataset.get_by_category(TaskCategory.BORDERLINE)


@pytest.fixture(scope="session")
def weak_fit_tasks(eval_dataset: EvalDataset) -> List[EvalTask]:
    """All weak-fit evaluation tasks."""
    return eval_dataset.get_by_category(TaskCategory.WEAK_FIT)


@pytest.fixture(scope="session")
def injection_tasks(eval_dataset: EvalDataset) -> List[EvalTask]:
    """All injection-attack evaluation tasks."""
    return eval_dataset.get_by_category(TaskCategory.INJECTION_ATTACK)


@pytest.fixture(scope="session")
def missing_field_tasks(eval_dataset: EvalDataset) -> List[EvalTask]:
    """All missing-field evaluation tasks."""
    return eval_dataset.get_by_category(TaskCategory.MISSING_FIELDS)


@pytest.fixture(scope="session")
def out_of_scope_tasks(eval_dataset: EvalDataset) -> List[EvalTask]:
    """All out-of-scope evaluation tasks."""
    return eval_dataset.get_by_category(TaskCategory.OUT_OF_SCOPE)


@pytest.fixture(scope="session")
def conflicting_tasks(eval_dataset: EvalDataset) -> List[EvalTask]:
    """All conflicting-results evaluation tasks."""
    return eval_dataset.get_by_category(TaskCategory.CONFLICTING_RESULTS)


@pytest.fixture(scope="session")
def escalation_tasks(eval_dataset: EvalDataset) -> List[EvalTask]:
    """All human-escalation evaluation tasks."""
    return eval_dataset.get_by_category(TaskCategory.HUMAN_ESCALATION)


@pytest.fixture(scope="session")
def critical_tasks(eval_dataset: EvalDataset) -> List[EvalTask]:
    """All tasks with severity=critical."""
    return eval_dataset.get_critical_tasks()


# ---------------------------------------------------------------------------
# Workflow fixtures
# ---------------------------------------------------------------------------

from models.state import JDInput, RecruitmentState
from graph.workflow import build_recruitment_graph


@pytest.fixture(scope="session")
def recruitment_graph():
    """Build the LangGraph workflow once per session (expensive)."""
    return build_recruitment_graph()


def _build_state_from_task(task: EvalTask) -> RecruitmentState:
    """Convert an EvalTask input into a RecruitmentState dict.

    Args:
        task: The evaluation task whose input to convert.

    Returns:
        A RecruitmentState ready for graph.stream().
    """
    jd_data = task.input.jd.model_dump()
    jd = JDInput(**jd_data)

    return RecruitmentState(
        jd=jd,
        candidates=list(task.input.candidates),
        parsed_profiles=[],
        scorecards=[],
        verified_scores=[],
        revision_count=task.input.override_revision_count or 0,
        shortlist=[],
        step_count=0,
        errors=[],
        needs_human_escalation=False,
        human_approved=task.input.human_approved,
        schedules=[],
    )


@pytest.fixture
def build_state():
    """Factory fixture: returns a function that builds RecruitmentState from EvalTask."""
    return _build_state_from_task


@pytest.fixture
def run_workflow(recruitment_graph):
    """Factory fixture: runs the graph on a task and returns the collected events.

    Usage in tests:
        events = run_workflow(task)
        nodes_run = [name for event in events for name in event.keys()]
    """
    def _run(task: EvalTask) -> list:
        state = _build_state_from_task(task)
        return list(recruitment_graph.stream(state))

    return _run


@pytest.fixture
def run_workflow_full(recruitment_graph):
    """Factory fixture: runs the graph with human_approved=True.

    Use this to test the full pipeline including the scheduler.
    """
    def _run(task: EvalTask) -> list:
        state = _build_state_from_task(task)
        state["human_approved"] = True
        return list(recruitment_graph.stream(state))

    return _run


# ---------------------------------------------------------------------------
# Helper utilities available as fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extract_nodes():
    """Fixture: returns a helper function to extract node names from stream events."""
    def _extract(events: list) -> List[str]:
        """Return ordered list of node names from graph.stream() output."""
        return [node_name for event in events for node_name in event.keys()]

    return _extract


@pytest.fixture
def extract_final_state():
    """Fixture: returns a helper that merges all stream events into a final state dict."""
    def _extract(events: list) -> dict:
        """Merge all stream event outputs into a single state snapshot."""
        merged: dict = {}
        for event in events:
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    for key, value in node_output.items():
                        # Lists are accumulated (mirrors operator.add reducer)
                        if isinstance(value, list) and isinstance(merged.get(key), list):
                            merged[key] = merged[key] + value
                        else:
                            merged[key] = value
        return merged

    return _extract
