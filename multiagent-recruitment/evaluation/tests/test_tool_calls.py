"""
pytest suite for Tool-Call Evaluation (Layer 2).

Tests are organised into four classes:

1. TestToolCallMetricsUnit     — unit tests for ToolCallMetrics logic
                                 using hand-built TraceCapture objects
2. TestToolCallSequences       — integration tests: actual vs expected sequences
                                 per dataset category
3. TestArgumentValidity        — integration tests: Pydantic argument validation
                                 for each node's emitted state objects
4. TestToolCallAccuracyReport  — asserts aggregate tool-call accuracy >= 85%

Every integration test uses DEMO MODE (forced in conftest.py) — no API calls.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import pytest

_PROJECT_ROOT = Path(__file__).parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("RECRUITMENT_DEMO_MODE", "true")

from evaluation.traces.capture import TraceCapture, NodeEvent, capture_trace
from evaluation.metrics.tool_call import (
    ToolCallMetrics,
    ToolCallResult,
    ToolCallReport,
    _is_subsequence,
    _node_sequence_to_tool_names,
    _validate_node_arguments,
)
from evaluation.datasets.schema import EvalTask, TaskCategory
from models.state import (
    CandidateProfile,
    JDInput,
    RecruitmentState,
    Scorecard,
    ShortlistEntry,
    VerifiedScore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_to_state(task: EvalTask) -> RecruitmentState:
    jd = JDInput(**task.input.jd.model_dump())
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
    )


def _make_minimal_capture(node_sequence: List[str]) -> TraceCapture:
    """Build a minimal TraceCapture with valid Pydantic objects for testing."""
    tc = TraceCapture()
    tc.node_sequence = list(node_sequence)
    for i, name in enumerate(node_sequence):
        output = {"step_count": i + 1}

        if name == "resume_analyst":
            output["parsed_profiles"] = [
                CandidateProfile(candidate_id="c0", name="Test User")
            ]
        elif name == "scorer":
            output["scorecards"] = [
                Scorecard(candidate_id="c0", total_score=85.0)
            ]
        elif name == "verifier":
            output["verified_scores"] = [
                VerifiedScore(
                    candidate_id="c0",
                    original_score=85.0,
                    blind_score=84.0,
                )
            ]
        elif name == "decider":
            output["shortlist"] = [
                ShortlistEntry(
                    candidate_id="c0",
                    name="Test User",
                    final_score=85.0,
                    rank=1,
                    status="shortlisted",
                )
            ]

        event = NodeEvent(
            node_name=name,
            output=output,
            step_index=i,
            step_count_after=i + 1,
        )
        tc.events.append(event)
        for k, v in output.items():
            if isinstance(v, list) and isinstance(tc.final_state.get(k), list):
                tc.final_state[k] = tc.final_state[k] + v
            else:
                tc.final_state[k] = v

    return tc


# ---------------------------------------------------------------------------
# 1. Unit tests for ToolCallMetrics logic
# ---------------------------------------------------------------------------


class TestToolCallMetricsUnit:
    """Unit tests for the _is_subsequence and _node_sequence_to_tool_names helpers."""

    def test_is_subsequence_exact_match(self):
        assert _is_subsequence(["a", "b", "c"], ["a", "b", "c"]) is True

    def test_is_subsequence_with_extras(self):
        # Expected is a subset — should pass
        assert _is_subsequence(["a", "c"], ["a", "b", "c", "d"]) is True

    def test_is_subsequence_wrong_order(self):
        assert _is_subsequence(["b", "a"], ["a", "b"]) is False

    def test_is_subsequence_missing_element(self):
        assert _is_subsequence(["a", "d"], ["a", "b", "c"]) is False

    def test_is_subsequence_empty_expected(self):
        assert _is_subsequence([], ["a", "b"]) is True

    def test_is_subsequence_with_repeats(self):
        # Retry loop: verifier appears twice in actual
        assert _is_subsequence(
            ["parse_resume", "score_candidates", "verify_scores"],
            ["parse_resume", "score_candidates", "verify_scores", "scorer", "verify_scores"],
        ) is True

    def test_node_to_tool_names_mapping(self):
        nodes = ["resume_analyst", "scorer", "verifier", "decider", "human_approval_gate", "scheduler"]
        tools = _node_sequence_to_tool_names(nodes)
        assert tools == [
            "parse_resume",
            "score_candidates",
            "verify_scores",
            "generate_shortlist",
            "human_approval_gate",
            "schedule_interviews",
        ]

    def test_node_to_tool_names_unknown_passthrough(self):
        tools = _node_sequence_to_tool_names(["unknown_node"])
        assert tools == ["unknown_node"]

    def test_argument_validation_valid_objects(self):
        tc = _make_minimal_capture(["resume_analyst", "scorer", "verifier", "decider"])
        errors = _validate_node_arguments(tc)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_argument_validation_invalid_scorecard(self):
        tc = TraceCapture()
        tc.node_sequence = ["scorer"]
        # Inject a raw dict instead of Scorecard
        event = NodeEvent(
            node_name="scorer",
            output={"scorecards": [{"bad": "data"}], "step_count": 1},
            step_index=0,
            step_count_after=1,
        )
        tc.events.append(event)
        errors = _validate_node_arguments(tc)
        # A raw dict in the scorecards list should trigger a type error
        assert len(errors) > 0

    def test_tool_call_result_passed_all_correct(self):
        tc = _make_minimal_capture(
            ["resume_analyst", "scorer", "decider", "human_approval_gate"]
        )
        # Build a minimal fake task
        from evaluation.datasets.schema import (
            EvalInput, JDInput as EvalJDInput, ExpectedTrajectory,
            ExpectedDecision, PassCriteria, SeverityLevel
        )
        task = EvalTask(
            id="unit_test",
            name="Unit test task",
            category=TaskCategory.STRONG_FIT,
            severity=SeverityLevel.NA,
            description="Unit test",
            input=EvalInput(
                jd=EvalJDInput(title="Engineer", description="Test JD"),
                candidates=["Alice\nSKILLS\nPython"],
            ),
            expected_trajectory=ExpectedTrajectory(
                nodes_executed=["resume_analyst", "scorer", "decider", "human_approval_gate"],
                verifier_triggered=False,
                human_approval_required=True,
                scheduler_runs=False,
            ),
            expected_tool_call_sequence=["parse_resume", "score_candidates", "generate_shortlist", "human_approval_gate"],
            expected_decision=ExpectedDecision(),
            pass_criteria=PassCriteria(),
        )

        metrics = ToolCallMetrics()
        result = metrics.evaluate(task, tc)

        assert result.sequence_accurate is True
        assert result.coverage_complete is True
        assert result.no_extra_tools is True
        assert result.arguments_valid is True
        assert result.passed is True
        assert result.accuracy_score == 1.0

    def test_tool_call_result_missing_tool(self):
        tc = _make_minimal_capture(["resume_analyst", "scorer"])
        from evaluation.datasets.schema import (
            EvalInput, JDInput as EvalJDInput, ExpectedTrajectory,
            ExpectedDecision, PassCriteria, SeverityLevel
        )
        task = EvalTask(
            id="unit_test_missing",
            name="Missing tool test",
            category=TaskCategory.STRONG_FIT,
            severity=SeverityLevel.NA,
            description="Test missing tool",
            input=EvalInput(
                jd=EvalJDInput(title="Engineer", description="Test JD"),
                candidates=["Alice\nSKILLS\nPython"],
            ),
            expected_trajectory=ExpectedTrajectory(
                nodes_executed=["resume_analyst", "scorer", "decider"],
                verifier_triggered=False,
                human_approval_required=True,
                scheduler_runs=False,
            ),
            expected_tool_call_sequence=["parse_resume", "score_candidates", "generate_shortlist"],
            expected_decision=ExpectedDecision(),
            pass_criteria=PassCriteria(),
        )

        metrics = ToolCallMetrics()
        result = metrics.evaluate(task, tc)
        assert result.coverage_complete is False
        assert "generate_shortlist" in result.missing_tools
        assert result.passed is False


# ---------------------------------------------------------------------------
# 2. Integration tests for tool-call sequences per category
# ---------------------------------------------------------------------------


class TestToolCallSequences:
    """Integration tests: run real graph and check actual vs expected sequences."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.metrics = ToolCallMetrics()

    def test_strong_fit_sequence(self, strong_fit_tasks, recruitment_graph):
        """Strong-fit: parse_resume → score_candidates → generate_shortlist → human_approval_gate."""
        for task in strong_fit_tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            result = self.metrics.evaluate(task, trace)

            assert result.coverage_complete, (
                f"[{task.id}] Missing tools: {result.missing_tools}\n"
                f"Actual: {result.actual_sequence}\n"
                f"Expected: {result.expected_sequence}"
            )
            assert result.sequence_accurate, (
                f"[{task.id}] Tool sequence out of order.\n"
                f"Actual: {result.actual_sequence}\n"
                f"Expected: {result.expected_sequence}"
            )

    def test_borderline_includes_verify_scores(self, borderline_tasks, recruitment_graph):
        """Borderline: sequence must include verify_scores between score_candidates and generate_shortlist."""
        for task in borderline_tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            result = self.metrics.evaluate(task, trace)

            assert result.coverage_complete, (
                f"[{task.id}] Missing verify_scores: {result.missing_tools}\n"
                f"Actual: {result.actual_sequence}"
            )
            # verify_scores must appear in actual sequence
            assert "verify_scores" in result.actual_sequence, (
                f"[{task.id}] verify_scores not in actual tool sequence: {result.actual_sequence}"
            )

    def test_weak_fit_no_verify_scores(self, weak_fit_tasks, recruitment_graph):
        """Weak-fit: verify_scores must NOT appear (score < 50, not borderline)."""
        for task in weak_fit_tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            result = self.metrics.evaluate(task, trace)

            assert "verify_scores" not in result.actual_sequence, (
                f"[{task.id}] verify_scores should not run for weak-fit candidate. "
                f"Actual: {result.actual_sequence}"
            )

    def test_injection_sequence_parse_then_score(self, injection_tasks, recruitment_graph):
        """Injection tasks: parse_resume must always precede score_candidates."""
        for task in injection_tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            actual = _node_sequence_to_tool_names(trace.node_sequence)

            if "parse_resume" in actual and "score_candidates" in actual:
                parse_idx = actual.index("parse_resume")
                score_idx = actual.index("score_candidates")
                assert parse_idx < score_idx, (
                    f"[{task.id}] parse_resume must precede score_candidates. "
                    f"Actual: {actual}"
                )

    def test_escalation_task_sequence(self, escalation_tasks, recruitment_graph):
        """Escalation tasks: decider must appear in the sequence after exhausting retries."""
        for task in escalation_tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            actual = _node_sequence_to_tool_names(trace.node_sequence)

            assert "generate_shortlist" in actual, (
                f"[{task.id}] Decider must run after exhausting retries. "
                f"Actual sequence: {actual}"
            )

    def test_full_pipeline_with_approval_includes_scheduler(self, strong_fit_tasks, recruitment_graph):
        """With human_approved=True, schedule_interviews must appear in sequence."""
        approved_tasks = [t for t in strong_fit_tasks if t.input.human_approved]
        if not approved_tasks:
            pytest.skip("No strong-fit task with human_approved=True")

        for task in approved_tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            result = self.metrics.evaluate(task, trace)

            assert "schedule_interviews" in result.actual_sequence, (
                f"[{task.id}] schedule_interviews missing from actual sequence: "
                f"{result.actual_sequence}"
            )
            assert result.coverage_complete, (
                f"[{task.id}] Missing tools: {result.missing_tools}"
            )


# ---------------------------------------------------------------------------
# 3. Argument validity tests
# ---------------------------------------------------------------------------


class TestArgumentValidity:
    """Integration tests: verify Pydantic objects emitted by each node are valid."""

    def test_parsed_profiles_are_valid_pydantic(self, eval_dataset, recruitment_graph):
        """Every CandidateProfile emitted by resume_analyst must pass Pydantic validation."""
        from pydantic import ValidationError as PydanticValidationError

        for task in eval_dataset.tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)

            for profile in trace.parsed_profiles():
                try:
                    CandidateProfile.model_validate(profile.model_dump())
                except PydanticValidationError as e:
                    pytest.fail(
                        f"[{task.id}] CandidateProfile validation failed: {e}\n"
                        f"Profile: {profile}"
                    )

    def test_scorecards_are_valid_pydantic(self, eval_dataset, recruitment_graph):
        """Every Scorecard emitted by scorer must be a valid Pydantic model."""
        from pydantic import ValidationError as PydanticValidationError

        for task in eval_dataset.tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)

            for sc in trace.scorecards():
                try:
                    Scorecard.model_validate(sc.model_dump())
                except PydanticValidationError as e:
                    pytest.fail(
                        f"[{task.id}] Scorecard validation failed: {e}\n"
                        f"Scorecard: {sc}"
                    )
                # Score must be in range 0-100
                assert 0.0 <= sc.total_score <= 100.0, (
                    f"[{task.id}] total_score out of range: {sc.total_score}"
                )

    def test_verified_scores_are_valid_pydantic(self, borderline_tasks, conflicting_tasks, recruitment_graph):
        """Every VerifiedScore emitted by verifier must be a valid Pydantic model."""
        from pydantic import ValidationError as PydanticValidationError

        all_tasks = borderline_tasks + conflicting_tasks
        for task in all_tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)

            for vs in trace.verified_scores():
                try:
                    VerifiedScore.model_validate(vs.model_dump())
                except PydanticValidationError as e:
                    pytest.fail(
                        f"[{task.id}] VerifiedScore validation failed: {e}\n"
                        f"VerifiedScore: {vs}"
                    )
                # Blind score must be in range
                assert 0.0 <= vs.blind_score <= 100.0, (
                    f"[{task.id}] blind_score out of range: {vs.blind_score}"
                )

    def test_shortlist_entries_are_valid_pydantic(self, eval_dataset, recruitment_graph):
        """Every ShortlistEntry emitted by decider must be a valid Pydantic model."""
        from pydantic import ValidationError as PydanticValidationError

        for task in eval_dataset.tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)

            for entry in trace.shortlist():
                try:
                    ShortlistEntry.model_validate(entry.model_dump())
                except PydanticValidationError as e:
                    pytest.fail(
                        f"[{task.id}] ShortlistEntry validation failed: {e}\n"
                        f"Entry: {entry}"
                    )
                # Status must be in valid set
                assert entry.status in ("shortlisted", "hold", "rejected"), (
                    f"[{task.id}] Invalid shortlist status: {entry.status}"
                )

    def test_no_argument_validation_errors_any_task(self, eval_dataset, recruitment_graph):
        """_validate_node_arguments must find no errors across all tasks."""
        all_errors = []
        for task in eval_dataset.tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            errors = _validate_node_arguments(trace)
            for e in errors:
                all_errors.append(f"[{task.id}] {e}")

        assert not all_errors, (
            f"Argument validation errors found:\n"
            + "\n".join(all_errors)
        )


# ---------------------------------------------------------------------------
# 4. Aggregate accuracy report
# ---------------------------------------------------------------------------


class TestToolCallAccuracyReport:
    """Assert aggregate tool-call accuracy >= 85% across all dataset tasks."""

    ACCURACY_THRESHOLD = 0.85

    def test_aggregate_tool_call_accuracy(self, eval_dataset, recruitment_graph):
        """Overall tool-call accuracy across all 12 tasks must be >= 85%."""
        metrics = ToolCallMetrics()
        tasks_and_traces = []

        for task in eval_dataset.tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            tasks_and_traces.append((task, trace))

        report = metrics.evaluate_all(tasks_and_traces)

        # Print per-task summary
        print(f"\n{'='*60}")
        print("  TOOL-CALL ACCURACY REPORT")
        print(f"{'='*60}")
        for result in report.results:
            print(f"  {result.summary()}")
        print(f"{'='*60}")
        print(f"  Overall Accuracy: {report.overall_accuracy:.1%}")
        print(f"  Pass Rate:        {report.pass_rate:.1%}")
        print(f"{'='*60}\n")

        assert report.overall_accuracy >= self.ACCURACY_THRESHOLD, (
            f"Tool-call accuracy {report.overall_accuracy:.1%} below threshold "
            f"{self.ACCURACY_THRESHOLD:.0%}. "
            f"Failing tasks: {[r.task_id for r in report.failures()]}"
        )

    def test_no_task_has_argument_validation_failures(self, eval_dataset, recruitment_graph):
        """Every task must pass argument validation — no invalid Pydantic objects."""
        metrics = ToolCallMetrics()
        arg_failures = []

        for task in eval_dataset.tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            result = metrics.evaluate(task, trace)

            if not result.arguments_valid:
                arg_failures.append(
                    f"[{task.id}] {result.argument_errors}"
                )

        assert not arg_failures, (
            f"Argument validation failures:\n" + "\n".join(arg_failures)
        )
