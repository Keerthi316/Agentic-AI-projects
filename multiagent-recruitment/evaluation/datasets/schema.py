"""
Pydantic schema for the evaluation dataset.

Every field in recruitment_eval_dataset.json is validated against these
models. Importing this module and calling EvalDataset.model_validate(data)
raises a clear ValidationError if the dataset is malformed — catching
drift between the dataset file and the code that consumes it.

Design decisions:
- Enums enforce controlled vocabularies for category and severity so
  test code can branch on TaskCategory.BORDERLINE rather than magic strings.
- Optional fields use None defaults rather than empty strings so callers
  can distinguish "not applicable" from "not provided".
- JDInput is redefined locally (not imported from models.state) to avoid
  coupling the evaluation module to the production code's import graph.
- PassCriteria holds both machine-checkable rules (score_range, injection_flag)
  and human-readable invariant descriptions (trace_invariants, critical_failures)
  so automated checkers and reviewers share the same source of truth.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Controlled Vocabularies
# ---------------------------------------------------------------------------


class TaskCategory(str, Enum):
    """Task category — matches the 'category' field in the JSON dataset."""

    STRONG_FIT = "strong_fit"
    BORDERLINE = "borderline"
    WEAK_FIT = "weak_fit"
    INJECTION_ATTACK = "injection_attack"
    MISSING_FIELDS = "missing_fields"
    OUT_OF_SCOPE = "out_of_scope"
    CONFLICTING_RESULTS = "conflicting_results"
    HUMAN_ESCALATION = "human_escalation"


class SeverityLevel(str, Enum):
    """Task severity — used to weight findings in the scorecard report."""

    NA = "n/a"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class JDInput(BaseModel):
    """
    Job description input for a single evaluation task.

    Mirrors models.state.JDInput but kept local so the evaluation package
    has no import dependency on the production code at schema-load time.
    Tests that actually run the graph build the real JDInput from this data.
    """

    title: str
    description: str
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    min_experience_years: int = 0
    education_requirement: str = ""


class EvalInput(BaseModel):
    """
    Input payload for a single evaluation task.

    'candidates' is a list of raw resume strings — the same format accepted
    by the workflow's RecruitmentState['candidates'] field.
    """

    jd: JDInput
    candidates: List[str] = Field(..., min_length=1)

    # Optional overrides for pre-seeding workflow state in edge-case tests.
    # human_approved=True allows full-pipeline tests that include the scheduler.
    # override_revision_count seeds revision_count to test max-retry behaviour.
    human_approved: bool = False
    override_revision_count: Optional[int] = None
    notes: Optional[str] = None


class ExpectedTrajectory(BaseModel):
    """
    Describes the expected LangGraph execution path for a task.

    'nodes_executed' is the minimum set of nodes that must appear in the
    trace. Tests use invariant-based checks (not exact sequence matching)
    so retry loops that repeat nodes do not cause false failures.
    """

    nodes_executed: List[str] = Field(
        ...,
        description="LangGraph node names expected to appear in the trace.",
    )
    verifier_triggered: bool = Field(
        ...,
        description="True when a borderline score must route through the verifier.",
    )
    human_approval_required: bool = Field(
        ...,
        description="True when the workflow must pause at human_approval_gate.",
    )
    scheduler_runs: bool = Field(
        ...,
        description="True when the scheduler node is expected to execute.",
    )
    retry_possible: bool = Field(
        default=False,
        description="True when the task may trigger the retry loop.",
    )
    notes: Optional[str] = None


class ExpectedDecision(BaseModel):
    """
    Expected output state after the workflow completes.

    All fields are Optional — only the fields relevant to a given task
    category are specified. Unspecified fields are not checked by the
    automated evaluation runner.
    """

    candidate_shortlisted: Optional[bool] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    status: Optional[str] = None
    injection_detected: Optional[bool] = None
    min_injection_confidence: Optional[float] = None
    fairness_corrected: Optional[bool] = None
    needs_human_escalation: Optional[bool] = None
    workflow_crashed: Optional[bool] = None
    hallucinated_profile: Optional[bool] = None
    error_logged: Optional[bool] = None
    step_budget_respected: Optional[bool] = None
    no_infinite_loop: Optional[bool] = None


class PassCriteria(BaseModel):
    """
    Rules that must hold for a task to be marked PASS.

    Two kinds of rules co-exist here:
    - Machine-checkable rules (score_range, injection_flag, verifier_must_run, …)
      are evaluated programmatically by the evaluation runner.
    - Human-readable descriptions (trace_invariants, critical_failures) are
      checked by TraceValidator in Phase 2 and serve as documentation for
      reviewers.
    """

    # ── Trace-level invariants (evaluated by TraceValidator in Phase 2) ─
    trace_invariants: List[str] = Field(
        default_factory=list,
        description="Human-readable invariant descriptions for TraceValidator.",
    )

    # ── Score range check ────────────────────────────────────────────────
    score_range: Optional[List[float]] = Field(
        default=None,
        description="[min, max] inclusive. Final candidate score must fall here.",
    )

    # ── Injection checks ─────────────────────────────────────────────────
    injection_flag: Optional[bool] = Field(
        default=None,
        description="Expected value of CandidateProfile.is_injection_detected.",
    )
    min_injection_confidence: Optional[float] = Field(
        default=None,
        description="Minimum injection_confidence when injection_flag is True.",
    )

    # ── Shortlist status check ───────────────────────────────────────────
    status_must_be: Optional[str] = None
    status_must_be_one_of: Optional[List[str]] = None

    # ── Structural checks ────────────────────────────────────────────────
    verifier_must_run: bool = False
    fairness_must_correct: bool = False
    workflow_must_not_crash: bool = False
    error_must_be_logged: bool = False
    no_hallucinated_skills: bool = False
    no_infinite_loop: bool = False
    max_steps: Optional[int] = None

    # ── Critical failure descriptions ────────────────────────────────────
    # Any one of these conditions occurring constitutes a CRITICAL failure.
    critical_failures: List[str] = Field(
        default_factory=list,
        description="Conditions that constitute a critical failure if observed.",
    )

    @field_validator("score_range")
    @classmethod
    def validate_score_range(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None:
            if len(v) != 2:
                raise ValueError("score_range must have exactly 2 elements: [min, max]")
            if v[0] > v[1]:
                raise ValueError(
                    f"score_range min ({v[0]}) must be <= max ({v[1]})"
                )
        return v


# ---------------------------------------------------------------------------
# Top-level task model
# ---------------------------------------------------------------------------


class EvalTask(BaseModel):
    """
    A single evaluation task in the dataset.

    Each task is self-contained: it carries its own input, expected
    execution path, expected tool-call sequence, expected decision, and
    pass criteria. The evaluation runner needs nothing else to execute
    and assess a task.
    """

    id: str = Field(..., description="Unique identifier, e.g. 'task_001'.")
    name: str = Field(..., description="Short human-readable name.")
    category: TaskCategory
    severity: SeverityLevel
    description: str = Field(..., description="What this task is testing and why.")

    input: EvalInput
    expected_trajectory: ExpectedTrajectory
    expected_tool_call_sequence: List[str] = Field(
        ...,
        description="Ordered list of tool/node names expected to be called.",
    )
    expected_decision: ExpectedDecision
    pass_criteria: PassCriteria
    tags: List[str] = Field(default_factory=list)

    @field_validator("expected_tool_call_sequence")
    @classmethod
    def sequence_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError(
                "expected_tool_call_sequence must contain at least one entry"
            )
        return v


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class EvalDataset(BaseModel):
    """
    The complete evaluation dataset loaded from the JSON file.

    Validates at construction time that:
    - All task IDs are unique.
    - At least one task is present.
    - Every task passes its own sub-model validation.
    """

    version: str
    description: str
    tasks: List[EvalTask] = Field(..., min_length=1)

    @field_validator("tasks")
    @classmethod
    def unique_task_ids(cls, v: List[EvalTask]) -> List[EvalTask]:
        ids = [t.id for t in v]
        if len(ids) != len(set(ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"Duplicate task IDs found in dataset: {duplicates}")
        return v

    # ── Filtering helpers ────────────────────────────────────────────────

    def get_by_category(self, category: TaskCategory) -> List[EvalTask]:
        """Return all tasks whose category matches the given value."""
        return [t for t in self.tasks if t.category == category]

    def get_by_id(self, task_id: str) -> Optional[EvalTask]:
        """Return the task with the given ID, or None if not found."""
        return next((t for t in self.tasks if t.id == task_id), None)

    def get_by_tag(self, tag: str) -> List[EvalTask]:
        """Return all tasks that carry the given tag string."""
        return [t for t in self.tasks if tag in t.tags]

    def get_critical_tasks(self) -> List[EvalTask]:
        """Return all tasks whose severity is CRITICAL."""
        return [t for t in self.tasks if t.severity == SeverityLevel.CRITICAL]

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of dataset statistics."""
        by_cat: Dict[str, int] = {}
        for t in self.tasks:
            by_cat[t.category.value] = by_cat.get(t.category.value, 0) + 1
        return {
            "version": self.version,
            "total_tasks": len(self.tasks),
            "by_category": by_cat,
            "critical_count": len(self.get_critical_tasks()),
        }
