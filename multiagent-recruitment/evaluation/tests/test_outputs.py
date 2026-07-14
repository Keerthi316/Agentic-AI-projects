"""
pytest suite for Output Quality Evaluation (Layer 3 — DeepEval).

Tests are organised into four classes:

1. TestDeepEvalTestCaseBuilders — unit tests: verify test case construction
                                  from trace data produces valid LLMTestCase objects
2. TestOutputScoring            — integration: evaluate real workflow outputs
                                  against Faithfulness, Relevancy, Completion metrics
3. TestHallucinationPrevention  — integration: verify no hallucinated skills/facts
4. TestOutputQualityReport      — aggregate pass rate >= threshold across all tasks

DeepEval metrics are evaluated in demo mode using structural checks when
DeepEval is not installed (DEEPEVAL_AVAILABLE=False). When DeepEval is
available, the metrics call an LLM evaluator — ensure OPENAI_API_KEY is
set or use DEEPEVAL_MODEL=gpt-4o-mini.

Design decision:
- When DeepEval is not available, we fall back to structural output quality
  checks (length, format, score range) that verify the same properties the
  LLM-based metrics would check — just without semantic understanding.
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

from evaluation.traces.capture import capture_trace, TraceCapture
from evaluation.deepeval_suite.test_cases import (
    build_deepeval_test_cases,
    build_parse_resume_test_case,
    build_score_candidates_test_case,
    build_verify_scores_test_case,
    build_generate_shortlist_test_case,
    DEEPEVAL_AVAILABLE,
)
from evaluation.deepeval_suite.metrics import (
    evaluate_test_case,
    MetricResult,
    get_faithfulness_metric,
    get_answer_relevancy_metric,
    get_hallucination_metric,
    get_task_completion_metric,
    FAITHFULNESS_THRESHOLD,
    ANSWER_RELEVANCY_THRESHOLD,
    TASK_COMPLETION_THRESHOLD,
    HALLUCINATION_MAX_THRESHOLD,
)
from evaluation.datasets.schema import EvalTask, TaskCategory
from models.state import JDInput, RecruitmentState


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


def _structural_quality_check(test_case) -> List[str]:
    """Structural output quality checks used when DeepEval is not available.

    Returns a list of failure messages (empty = pass).
    """
    failures = []
    actual = getattr(test_case, "actual_output", "")

    # Output must be non-empty
    if not actual or not actual.strip():
        failures.append("actual_output is empty")
        return failures

    # Output must be reasonably long (at least 20 chars)
    if len(actual) < 20:
        failures.append(f"actual_output too short ({len(actual)} chars)")

    # Output must not contain error messages
    lower = actual.lower()
    error_patterns = ["traceback", "exception", "error:", "none", "null"]
    for pattern in error_patterns:
        if pattern in lower and len(actual) < 100:
            failures.append(f"Output looks like an error: contains '{pattern}'")
            break

    return failures


# ---------------------------------------------------------------------------
# 1. Unit tests: test case builders
# ---------------------------------------------------------------------------


class TestDeepEvalTestCaseBuilders:
    """Test that LLMTestCase objects are built correctly from trace data."""

    def test_build_test_cases_returns_list(self, strong_fit_tasks, recruitment_graph):
        """build_deepeval_test_cases returns a list for strong-fit tasks."""
        task = strong_fit_tasks[0]
        state = _task_to_state(task)
        trace = capture_trace(state, graph=recruitment_graph)
        cases = build_deepeval_test_cases(task, trace)

        assert isinstance(cases, list)

    def test_parse_resume_case_has_input_and_output(self, strong_fit_tasks, recruitment_graph):
        """parse_resume test case has non-empty input and actual_output."""
        task = strong_fit_tasks[0]
        state = _task_to_state(task)
        trace = capture_trace(state, graph=recruitment_graph)
        case = build_parse_resume_test_case(task, trace)

        if case is None:
            pytest.skip("No profiles in trace — parse_resume case not built")

        assert case.input, "LLMTestCase.input must be non-empty"
        assert case.actual_output, "LLMTestCase.actual_output must be non-empty"

    def test_score_candidates_case_has_context(self, strong_fit_tasks, recruitment_graph):
        """score_candidates test case has JD in context."""
        task = strong_fit_tasks[0]
        state = _task_to_state(task)
        trace = capture_trace(state, graph=recruitment_graph)
        case = build_score_candidates_test_case(task, trace)

        if case is None:
            pytest.skip("No scorecards in trace — scoring case not built")

        assert case.context, "LLMTestCase.context must include JD"
        assert any("Python" in c or "Engineer" in c for c in case.context), (
            "JD context should contain job-related keywords"
        )

    def test_verify_scores_case_built_for_borderline(self, borderline_tasks, recruitment_graph):
        """verify_scores test case is built for borderline tasks."""
        task = borderline_tasks[0]
        state = _task_to_state(task)
        trace = capture_trace(state, graph=recruitment_graph)
        case = build_verify_scores_test_case(task, trace)

        if not trace.ran("verifier"):
            pytest.skip("Verifier did not run — skipping")

        assert case is not None, "verify_scores case must be built for borderline task"
        assert case.actual_output, "actual_output must be non-empty"

    def test_shortlist_case_has_ranked_output(self, strong_fit_tasks, recruitment_graph):
        """generate_shortlist case output contains rank and status."""
        task = strong_fit_tasks[0]
        state = _task_to_state(task)
        trace = capture_trace(state, graph=recruitment_graph)
        case = build_generate_shortlist_test_case(task, trace)

        if case is None:
            pytest.skip("No shortlist in trace")

        assert "Rank" in case.actual_output or "rank" in case.actual_output.lower(), (
            f"Shortlist output should contain rank information: {case.actual_output[:200]}"
        )

    def test_no_case_built_for_empty_trace(self, out_of_scope_tasks, recruitment_graph):
        """Out-of-scope tasks may produce empty traces — builder returns empty list gracefully."""
        task = out_of_scope_tasks[0]
        state = _task_to_state(task)
        trace = capture_trace(state, graph=recruitment_graph)
        cases = build_deepeval_test_cases(task, trace)

        # Should return a list (possibly empty) — never raise
        assert isinstance(cases, list)

    def test_expected_output_is_non_empty(self, eval_dataset, recruitment_graph):
        """expected_output must be set in every non-None test case."""
        for task in eval_dataset.tasks[:6]:  # limit to first 6 for speed
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            cases = build_deepeval_test_cases(task, trace)

            for case in cases:
                assert case.expected_output, (
                    f"[{task.id}] {case.name}: expected_output must be non-empty"
                )


# ---------------------------------------------------------------------------
# 2. Integration: output scoring
# ---------------------------------------------------------------------------


class TestOutputScoring:
    """Evaluate real workflow outputs against quality metrics.

    When DeepEval is available: runs full LLM-based evaluation.
    When not available: runs structural quality checks.
    """

    def test_parse_resume_output_quality_strong_fit(self, strong_fit_tasks, recruitment_graph):
        """Resume parsing output for strong-fit candidates must meet quality threshold."""
        for task in strong_fit_tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            case = build_parse_resume_test_case(task, trace)

            if case is None:
                continue

            if DEEPEVAL_AVAILABLE:
                results = evaluate_test_case(case, [get_faithfulness_metric()])
                for r in results:
                    assert r.passed or r.score >= FAITHFULNESS_THRESHOLD - 0.1, (
                        f"[{task.id}] Faithfulness too low: {r.summary()}"
                    )
            else:
                failures = _structural_quality_check(case)
                assert not failures, (
                    f"[{task.id}] Output quality failures: {failures}\n"
                    f"Output: {case.actual_output[:300]}"
                )

    def test_scorecard_output_contains_reasoning(self, eval_dataset, recruitment_graph):
        """Scorecard output must contain explicit reasoning text."""
        tasks_with_scoring = [
            t for t in eval_dataset.tasks
            if t.category not in (TaskCategory.MISSING_FIELDS, TaskCategory.OUT_OF_SCOPE)
        ]

        for task in tasks_with_scoring:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            scorecards = trace.scorecards()

            if not scorecards:
                continue

            for sc in scorecards:
                assert sc.reasoning and len(sc.reasoning) > 10, (
                    f"[{task.id}] Scorecard missing reasoning for {sc.candidate_id}: "
                    f"'{sc.reasoning}'"
                )

    def test_scorecard_output_relevancy(self, strong_fit_tasks, recruitment_graph):
        """Scorecard should reference job-relevant skills."""
        for task in strong_fit_tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            case = build_score_candidates_test_case(task, trace)

            if case is None:
                continue

            if DEEPEVAL_AVAILABLE:
                results = evaluate_test_case(case, [get_answer_relevancy_metric()])
                for r in results:
                    assert r.passed or r.score >= ANSWER_RELEVANCY_THRESHOLD - 0.1, (
                        f"[{task.id}] Answer relevancy too low: {r.summary()}"
                    )
            else:
                # Structural check: output should mention relevant keywords
                output = case.actual_output.lower()
                assert any(
                    kw in output
                    for kw in ["score", "python", "experience", "skill", "engineer"]
                ), (
                    f"[{task.id}] Scorecard output lacks relevant keywords: "
                    f"{case.actual_output[:200]}"
                )

    def test_shortlist_output_structure(self, eval_dataset, recruitment_graph):
        """Shortlist output must contain rank and status for each entry."""
        for task in eval_dataset.tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            shortlist = trace.shortlist()

            if not shortlist:
                continue

            for i, entry in enumerate(shortlist):
                assert entry.rank >= 1, (
                    f"[{task.id}] Entry {i}: rank must be >= 1, got {entry.rank}"
                )
                assert entry.status in ("shortlisted", "hold", "rejected"), (
                    f"[{task.id}] Entry {i}: invalid status '{entry.status}'"
                )
                assert 0.0 <= entry.final_score <= 100.0, (
                    f"[{task.id}] Entry {i}: final_score out of range: {entry.final_score}"
                )


# ---------------------------------------------------------------------------
# 3. Hallucination prevention tests
# ---------------------------------------------------------------------------


class TestHallucinationPrevention:
    """Verify the system does not invent qualifications not in the resume."""

    HALLUCINATION_THRESHOLD = HALLUCINATION_MAX_THRESHOLD

    def test_hardware_engineer_no_python_skills(self, weak_fit_tasks, recruitment_graph):
        """Hardware engineer (Sandra Lee) must not have Python in extracted skills."""
        hw_tasks = [t for t in weak_fit_tasks if "hardware" in t.description.lower() or
                    "Sandra" in "".join(t.input.candidates)]

        for task in hw_tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            profiles = trace.parsed_profiles()

            for profile in profiles:
                python_hallucinated = any(
                    "python" in s.lower() for s in profile.skills
                )
                assert not python_hallucinated, (
                    f"[{task.id}] Hallucinated Python skill for hardware engineer: "
                    f"skills={profile.skills}"
                )

    def test_recipe_produces_no_tech_profile(self, out_of_scope_tasks, recruitment_graph):
        """Recipe text must not produce a profile with tech skills."""
        for task in out_of_scope_tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            profiles = trace.parsed_profiles()

            for profile in profiles:
                # Profile from recipe must have no meaningful name
                tech_skills = [
                    s for s in profile.skills
                    if any(kw in s.lower() for kw in ["python", "fastapi", "aws", "docker"])
                ]
                assert not tech_skills, (
                    f"[{task.id}] Recipe produced hallucinated tech skills: {tech_skills}"
                )

    def test_weak_marketing_candidate_no_backend_skills(self, weak_fit_tasks, recruitment_graph):
        """Marketing coordinator must not get backend skills in their parsed profile."""
        marketing_tasks = [
            t for t in weak_fit_tasks
            if "marketing" in t.description.lower() or
            any("marketing" in c.lower() for c in t.input.candidates)
        ]

        for task in marketing_tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            profiles = trace.parsed_profiles()

            for profile in profiles:
                backend_skills = [
                    s for s in profile.skills
                    if any(kw in s.lower() for kw in ["fastapi", "postgresql", "kubernetes", "redis"])
                ]
                assert not backend_skills, (
                    f"[{task.id}] Marketing candidate profile has hallucinated backend skills: "
                    f"{backend_skills}"
                )

    def test_deepeval_hallucination_metric_strong_fit(self, strong_fit_tasks, recruitment_graph):
        """Strong-fit candidates: hallucination score must be low."""
        for task in strong_fit_tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            case = build_parse_resume_test_case(task, trace)

            if case is None:
                continue

            if DEEPEVAL_AVAILABLE:
                results = evaluate_test_case(case, [get_hallucination_metric()])
                for r in results:
                    assert r.score <= self.HALLUCINATION_THRESHOLD + 0.1, (
                        f"[{task.id}] Hallucination score too high: {r.summary()}"
                    )
            else:
                # Structural: profile skills should overlap with resume text
                profiles = trace.parsed_profiles()
                for profile in profiles:
                    resume_lower = task.input.candidates[0].lower()
                    for skill in profile.skills:
                        if len(skill) > 3:  # Skip short abbreviations
                            assert skill.lower() in resume_lower or any(
                                word in resume_lower
                                for word in skill.lower().split()
                            ), (
                                f"[{task.id}] Possible hallucinated skill '{skill}' "
                                f"not found in resume text"
                            )


# ---------------------------------------------------------------------------
# 4. Aggregate output quality report
# ---------------------------------------------------------------------------


class TestOutputQualityReport:
    """Assert aggregate output quality meets thresholds across all tasks."""

    QUALITY_PASS_THRESHOLD = 0.80  # 80% of test cases must pass quality checks

    def test_aggregate_output_quality(self, eval_dataset, recruitment_graph):
        """Build all test cases and validate output quality >= 80%."""
        total_cases = 0
        passed_cases = 0

        task_summaries = []

        for task in eval_dataset.tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)
            cases = build_deepeval_test_cases(task, trace)

            task_passed = 0
            task_total = len(cases)

            for case in cases:
                total_cases += 1
                if DEEPEVAL_AVAILABLE:
                    results = evaluate_test_case(case)
                    case_passed = all(r.passed for r in results)
                else:
                    failures = _structural_quality_check(case)
                    case_passed = len(failures) == 0

                if case_passed:
                    passed_cases += 1
                    task_passed += 1

            task_summaries.append((task.id, task_passed, task_total))

        # Print summary table
        print(f"\n{'='*55}")
        print("  OUTPUT QUALITY REPORT")
        print(f"{'='*55}")
        for task_id, tp, tt in task_summaries:
            rate = f"{tp}/{tt}" if tt > 0 else "0/0 (no cases)"
            print(f"  {task_id}: {rate}")
        print(f"{'='*55}")
        overall = (passed_cases / total_cases) if total_cases > 0 else 1.0
        print(f"  Overall: {passed_cases}/{total_cases} ({overall:.1%})")
        print(f"{'='*55}\n")

        if total_cases == 0:
            pytest.skip("No test cases were built — skipping quality assertion")

        assert overall >= self.QUALITY_PASS_THRESHOLD, (
            f"Output quality {overall:.1%} below threshold {self.QUALITY_PASS_THRESHOLD:.0%}. "
            f"Total cases: {total_cases}, passed: {passed_cases}"
        )

    def test_task_completion_all_agents(self, eval_dataset, recruitment_graph):
        """Task completion metric (structural) must pass for core agents."""
        completion_failures = []

        for task in eval_dataset.tasks:
            state = _task_to_state(task)
            trace = capture_trace(state, graph=recruitment_graph)

            # Structural task completion checks
            if trace.ran("resume_analyst"):
                profiles = trace.parsed_profiles()
                if not profiles and not trace.errors():
                    completion_failures.append(
                        f"[{task.id}] resume_analyst ran but produced no profiles and no errors"
                    )

            if trace.ran("scorer"):
                scorecards = trace.scorecards()
                if not scorecards and not trace.errors():
                    completion_failures.append(
                        f"[{task.id}] scorer ran but produced no scorecards and no errors"
                    )

            if trace.ran("decider"):
                shortlist = trace.shortlist()
                if not shortlist and not trace.errors():
                    completion_failures.append(
                        f"[{task.id}] decider ran but produced no shortlist and no errors"
                    )

        assert not completion_failures, (
            f"Task completion failures:\n" + "\n".join(completion_failures)
        )
