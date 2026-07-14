"""
Tool-Call Accuracy Metrics — Layer 2 of the evaluation framework.

Compares the actual node execution sequence captured by TraceCapture
against the expected_tool_call_sequence defined in the EvalTask.

Three sub-metrics are measured per task:
  1. Sequence accuracy  — actual nodes match expected in order
  2. Coverage           — all expected nodes were called (none skipped)
  3. Precision          — no unexpected extra nodes were inserted
  4. Argument validity  — state objects passed to each node are valid Pydantic models

Design decisions:
- "Sequence" is checked with subsequence matching (not strict positional equality)
  because the retry loop may repeat nodes; we verify that the expected subsequence
  appears in the actual sequence rather than requiring an exact match.
- Argument validity is evaluated on the outputs each node emits into state
  (which are themselves the next node's inputs), using the Pydantic models
  already defined in models/state.py.
- Results are aggregated into a ToolCallReport with per-task and overall accuracy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from evaluation.traces.capture import TraceCapture
from evaluation.datasets.schema import EvalTask
from models.state import CandidateProfile, Scorecard, VerifiedScore, ShortlistEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ToolCallResult:
    """Result of tool-call evaluation for a single EvalTask."""

    task_id: str
    task_name: str

    # Sub-metric results
    sequence_accurate: bool
    """True if all expected tools appear in the correct relative order."""

    coverage_complete: bool
    """True if every expected tool was called (no skips)."""

    no_extra_tools: bool
    """True if the actual sequence contains no unexpected extra tool calls."""

    arguments_valid: bool
    """True if all state objects emitted by nodes pass Pydantic validation."""

    # Detail
    expected_sequence: List[str]
    actual_sequence: List[str]
    missing_tools: List[str]
    extra_tools: List[str]
    argument_errors: List[str]

    @property
    def passed(self) -> bool:
        """All sub-metrics must pass for the overall result to pass."""
        return (
            self.sequence_accurate
            and self.coverage_complete
            and self.no_extra_tools
            and self.arguments_valid
        )

    @property
    def accuracy_score(self) -> float:
        """Fractional score 0.0–1.0 based on sub-metrics passed."""
        sub = [
            self.sequence_accurate,
            self.coverage_complete,
            self.no_extra_tools,
            self.arguments_valid,
        ]
        return sum(sub) / len(sub)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.task_id} — accuracy={self.accuracy_score:.0%} "
            f"(seq={self.sequence_accurate}, cov={self.coverage_complete}, "
            f"prec={self.no_extra_tools}, args={self.arguments_valid})"
        )


@dataclass
class ToolCallReport:
    """Aggregated tool-call evaluation report across all tasks."""

    results: List[ToolCallResult] = field(default_factory=list)

    @property
    def overall_accuracy(self) -> float:
        """Mean accuracy score across all tasks."""
        if not self.results:
            return 0.0
        return sum(r.accuracy_score for r in self.results) / len(self.results)

    @property
    def pass_rate(self) -> float:
        """Fraction of tasks where all sub-metrics passed."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def passed(self) -> bool:
        """True if overall accuracy >= 0.85."""
        return self.overall_accuracy >= 0.85

    def failures(self) -> List[ToolCallResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        return (
            f"Tool-Call Report: {passed}/{total} tasks passed "
            f"(accuracy={self.overall_accuracy:.1%}, pass_rate={self.pass_rate:.1%})"
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


# Map from LangGraph node name → the tool name used in the eval dataset
_NODE_TO_TOOL_NAME: Dict[str, str] = {
    "resume_analyst": "parse_resume",
    "scorer": "score_candidates",
    "verifier": "verify_scores",
    "decider": "generate_shortlist",
    "human_approval_gate": "human_approval_gate",
    "scheduler": "schedule_interviews",
}

# Reverse map: tool name → node name
_TOOL_TO_NODE_NAME: Dict[str, str] = {v: k for k, v in _NODE_TO_TOOL_NAME.items()}


def _node_sequence_to_tool_names(node_sequence: List[str]) -> List[str]:
    """Convert a list of LangGraph node names to tool names used in the dataset.

    Unknown node names are passed through unchanged.
    """
    return [_NODE_TO_TOOL_NAME.get(n, n) for n in node_sequence]


def _is_subsequence(expected: List[str], actual: List[str]) -> bool:
    """Return True if `expected` is a subsequence of `actual`.

    A subsequence means all elements of `expected` appear in `actual`
    in the same relative order, but not necessarily contiguously.
    This allows for retry loops that repeat nodes.
    """
    it = iter(actual)
    return all(item in it for item in expected)


def _validate_node_arguments(trace: TraceCapture) -> List[str]:
    """Validate that each node's emitted state objects pass Pydantic validation.

    We inspect the output dicts of each node event and try to re-validate
    any Pydantic objects against their model class. Errors are collected
    as strings for reporting.

    Args:
        trace: The captured execution trace.

    Returns:
        List of error strings (empty if all valid).
    """
    errors: List[str] = []

    for event in trace.events:
        output = event.output
        node = event.node_name

        # Validate parsed_profiles (emitted by resume_analyst)
        if node == "resume_analyst":
            for p in output.get("parsed_profiles", []):
                if isinstance(p, CandidateProfile):
                    try:
                        CandidateProfile.model_validate(p.model_dump())
                    except ValidationError as e:
                        errors.append(f"[{node}] CandidateProfile validation error: {e}")
                else:
                    errors.append(f"[{node}] Expected CandidateProfile, got {type(p).__name__}")

        # Validate scorecards (emitted by scorer)
        elif node == "scorer":
            for sc in output.get("scorecards", []):
                if isinstance(sc, Scorecard):
                    try:
                        Scorecard.model_validate(sc.model_dump())
                    except ValidationError as e:
                        errors.append(f"[{node}] Scorecard validation error: {e}")
                else:
                    errors.append(f"[{node}] Expected Scorecard, got {type(sc).__name__}")

        # Validate verified_scores (emitted by verifier)
        elif node == "verifier":
            for vs in output.get("verified_scores", []):
                if isinstance(vs, VerifiedScore):
                    try:
                        VerifiedScore.model_validate(vs.model_dump())
                    except ValidationError as e:
                        errors.append(f"[{node}] VerifiedScore validation error: {e}")
                else:
                    errors.append(f"[{node}] Expected VerifiedScore, got {type(vs).__name__}")

        # Validate shortlist entries (emitted by decider)
        elif node == "decider":
            for se in output.get("shortlist", []):
                if isinstance(se, ShortlistEntry):
                    try:
                        ShortlistEntry.model_validate(se.model_dump())
                    except ValidationError as e:
                        errors.append(f"[{node}] ShortlistEntry validation error: {e}")
                else:
                    errors.append(f"[{node}] Expected ShortlistEntry, got {type(se).__name__}")

        # Validate step_count is an integer
        if "step_count" in output and not isinstance(output["step_count"], int):
            errors.append(f"[{node}] step_count must be int, got {type(output['step_count']).__name__}")

    return errors


# ---------------------------------------------------------------------------
# Main metrics class
# ---------------------------------------------------------------------------


class ToolCallMetrics:
    """Evaluates tool-call accuracy for the recruitment workflow.

    Usage:
        metrics = ToolCallMetrics()
        result = metrics.evaluate(task, trace)
        report = metrics.evaluate_all(tasks_and_traces)
    """

    def evaluate(self, task: EvalTask, trace: TraceCapture) -> ToolCallResult:
        """Evaluate tool-call accuracy for a single task.

        Args:
            task:  The EvalTask from the dataset (defines expected sequence).
            trace: The captured execution trace for the task.

        Returns:
            ToolCallResult with per-sub-metric results and detail.
        """
        expected: List[str] = list(task.expected_tool_call_sequence)
        actual_tools: List[str] = _node_sequence_to_tool_names(trace.node_sequence)

        # ── 1. Coverage: every expected tool was called ──────────────
        missing_tools = [t for t in expected if t not in actual_tools]
        coverage_complete = len(missing_tools) == 0

        # ── 2. Sequence: expected tools appear in correct relative order
        # We use subsequence matching to tolerate retry repetitions
        sequence_accurate = _is_subsequence(expected, actual_tools) if coverage_complete else False

        # ── 3. Precision: no extra unexpected tools ───────────────────
        # Tools that appear in actual but not in expected
        # Allow duplicates (retries) of expected tools, flag truly new ones
        expected_set = set(expected)
        extra_tools = [t for t in actual_tools if t not in expected_set]
        no_extra_tools = len(extra_tools) == 0

        # ── 4. Argument validity ──────────────────────────────────────
        argument_errors = _validate_node_arguments(trace)
        arguments_valid = len(argument_errors) == 0

        result = ToolCallResult(
            task_id=task.id,
            task_name=task.name,
            sequence_accurate=sequence_accurate,
            coverage_complete=coverage_complete,
            no_extra_tools=no_extra_tools,
            arguments_valid=arguments_valid,
            expected_sequence=expected,
            actual_sequence=actual_tools,
            missing_tools=missing_tools,
            extra_tools=extra_tools,
            argument_errors=argument_errors,
        )

        if result.passed:
            logger.info("Tool-call evaluation PASS: %s", task.id)
        else:
            logger.warning(
                "Tool-call evaluation FAIL: %s — missing=%s, extra=%s, arg_errors=%d",
                task.id, missing_tools, extra_tools, len(argument_errors),
            )

        return result

    def evaluate_all(
        self,
        tasks_and_traces: List[tuple[EvalTask, TraceCapture]],
    ) -> ToolCallReport:
        """Evaluate tool-call accuracy across multiple tasks.

        Args:
            tasks_and_traces: List of (EvalTask, TraceCapture) pairs.

        Returns:
            ToolCallReport with per-task results and aggregate metrics.
        """
        report = ToolCallReport()
        for task, trace in tasks_and_traces:
            result = self.evaluate(task, trace)
            report.results.append(result)

        logger.info(report.summary())
        return report
