"""
DeepEval test case builders for the Multi-Agent Recruitment System.

Maps captured LangGraph execution traces into DeepEval LLMTestCase objects
so they can be evaluated by DeepEval metrics (Faithfulness, AnswerRelevancy,
TaskCompletion, Hallucination).

Each agent in the workflow maps to a different conceptual "test case":

  • parse_resume    → LLMTestCase: input=resume_text, actual_output=profile JSON,
                       context=jd_text (for relevancy)
  • score_candidates → LLMTestCase: input=profile+JD, actual_output=scorecard reasoning
  • verify_scores    → LLMTestCase: input=blind_profile+JD, actual_output=verification notes
  • generate_shortlist → LLMTestCase: input=scorecards, actual_output=shortlist
  • schedule_interviews → LLMTestCase: input=shortlist+JD, actual_output=schedule

Design decisions:
- DeepEval is imported with a graceful fallback so the rest of the framework
  still works if DeepEval is not installed.
- LLMTestCase.expected_output is set to the dataset's expected_decision description
  where available.
- retrieval_context is populated with the JD text so faithfulness and relevancy
  metrics can compare the output against the source context.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from evaluation.traces.capture import TraceCapture
from evaluation.datasets.schema import EvalTask
from models.state import CandidateProfile, Scorecard, VerifiedScore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional DeepEval import
# ---------------------------------------------------------------------------

try:
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    DEEPEVAL_AVAILABLE = False
    logger.warning(
        "DeepEval not installed. Test cases will be built as plain dicts. "
        "Install with: pip install deepeval"
    )

    # Provide a lightweight stub so the rest of the module works
    class LLMTestCase:  # type: ignore[no-redef]
        """Stub LLMTestCase used when deepeval is not installed."""

        def __init__(
            self,
            input: str = "",
            actual_output: str = "",
            expected_output: str = "",
            context: Optional[List[str]] = None,
            retrieval_context: Optional[List[str]] = None,
            name: str = "",
        ):
            self.input = input
            self.actual_output = actual_output
            self.expected_output = expected_output
            self.context = context or []
            self.retrieval_context = retrieval_context or []
            self.name = name


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _jd_to_context(task: EvalTask) -> List[str]:
    """Format the job description into a context list for DeepEval."""
    jd = task.input.jd
    return [
        f"Job Title: {jd.title}",
        f"Description: {jd.description}",
        f"Required Skills: {', '.join(jd.required_skills)}",
        f"Preferred Skills: {', '.join(jd.preferred_skills)}",
        f"Min Experience: {jd.min_experience_years} years",
        f"Education: {jd.education_requirement}",
    ]


def _profile_to_str(profile: CandidateProfile) -> str:
    """Serialise a CandidateProfile to a readable string."""
    return json.dumps(profile.model_dump(), indent=2, default=str)


def _scorecard_to_str(sc: Scorecard) -> str:
    """Serialise a Scorecard to a readable string."""
    return (
        f"Candidate: {sc.candidate_id}\n"
        f"Total Score: {sc.total_score}\n"
        f"Skill Score: {sc.skill_score}\n"
        f"Experience Score: {sc.experience_score}\n"
        f"Education Score: {sc.education_score}\n"
        f"Reasoning: {sc.reasoning}\n"
        f"Borderline: {sc.is_borderline}"
    )


def _verified_to_str(vs: VerifiedScore) -> str:
    return (
        f"Candidate: {vs.candidate_id}\n"
        f"Original Score: {vs.original_score}\n"
        f"Blind Score: {vs.blind_score}\n"
        f"Difference: {vs.score_difference}\n"
        f"Fair: {vs.is_fair}\n"
        f"Notes: {vs.fairness_notes}"
    )


def _expected_decision_summary(task: EvalTask) -> str:
    """Format the expected decision into a human-readable expected output."""
    d = task.expected_decision
    parts = []
    if d.candidate_shortlisted is not None:
        parts.append(f"shortlisted={d.candidate_shortlisted}")
    if d.status:
        parts.append(f"status={d.status}")
    if d.injection_detected is not None:
        parts.append(f"injection_detected={d.injection_detected}")
    if d.min_score is not None:
        parts.append(f"min_score={d.min_score}")
    if d.max_score is not None:
        parts.append(f"max_score={d.max_score}")
    return "; ".join(parts) if parts else task.description


# ---------------------------------------------------------------------------
# Per-agent test case builders
# ---------------------------------------------------------------------------


def build_parse_resume_test_case(
    task: EvalTask,
    trace: TraceCapture,
) -> Optional[LLMTestCase]:
    """Build a DeepEval test case for the Resume Analyst (parse_resume) step.

    Input: raw resume text
    Output: extracted CandidateProfile as JSON
    Context: job description (for relevancy check)
    """
    profiles = trace.parsed_profiles()
    candidates = task.input.candidates

    if not profiles or not candidates:
        return None

    profile = profiles[0]
    resume_text = candidates[0][:2000]  # limit for context window

    actual_output = _profile_to_str(profile)
    expected = (
        f"Structured profile with name, skills, experience. "
        f"injection_detected={task.expected_decision.injection_detected}"
    )

    return LLMTestCase(
        name=f"{task.id}_parse_resume",
        input=f"Extract a structured candidate profile from this resume:\n{resume_text}",
        actual_output=actual_output,
        expected_output=expected,
        context=_jd_to_context(task),
        retrieval_context=[resume_text],
    )


def build_score_candidates_test_case(
    task: EvalTask,
    trace: TraceCapture,
) -> Optional[LLMTestCase]:
    """Build a DeepEval test case for the Scorer (score_candidates) step.

    Input: candidate profile + JD
    Output: scorecard reasoning
    Context: JD text
    """
    scorecards = trace.scorecards()
    profiles = trace.parsed_profiles()

    if not scorecards:
        return None

    sc = scorecards[0]
    profile_text = _profile_to_str(profiles[0]) if profiles else "Profile unavailable"
    jd_context = _jd_to_context(task)

    input_text = (
        f"Score this candidate against the job description.\n"
        f"Profile:\n{profile_text}\n\n"
        f"Job:\n{chr(10).join(jd_context)}"
    )

    expected_min = task.expected_decision.min_score
    expected_max = task.expected_decision.max_score
    if expected_min is not None and expected_max is not None:
        expected = f"Score between {expected_min} and {expected_max}"
    elif expected_min is not None:
        expected = f"Score >= {expected_min}"
    elif expected_max is not None:
        expected = f"Score <= {expected_max}"
    else:
        expected = "Valid score between 0 and 100 with reasoning"

    return LLMTestCase(
        name=f"{task.id}_score_candidates",
        input=input_text,
        actual_output=_scorecard_to_str(sc),
        expected_output=expected,
        context=jd_context,
        retrieval_context=[profile_text],
    )


def build_verify_scores_test_case(
    task: EvalTask,
    trace: TraceCapture,
) -> Optional[LLMTestCase]:
    """Build a DeepEval test case for the Verifier (verify_scores) step."""
    verified = trace.verified_scores()
    scorecards = trace.scorecards()

    if not verified or not scorecards:
        return None

    vs = verified[0]
    sc = scorecards[0]

    input_text = (
        f"Blind re-score this candidate against the job description.\n"
        f"Original Score: {sc.total_score}\n"
        f"Reasoning: {sc.reasoning}"
    )

    expected = (
        f"Blind score within {'±10 of ' + str(sc.total_score) if vs.is_fair else 'acceptable range'}. "
        f"fairness_notes explaining the comparison."
    )

    return LLMTestCase(
        name=f"{task.id}_verify_scores",
        input=input_text,
        actual_output=_verified_to_str(vs),
        expected_output=expected,
        context=_jd_to_context(task),
        retrieval_context=[],
    )


def build_generate_shortlist_test_case(
    task: EvalTask,
    trace: TraceCapture,
) -> Optional[LLMTestCase]:
    """Build a DeepEval test case for the Decider (generate_shortlist) step."""
    shortlist = trace.shortlist()
    scorecards = trace.scorecards()

    if not shortlist:
        return None

    scorecards_text = "\n".join(_scorecard_to_str(sc) for sc in scorecards)
    shortlist_text = "\n".join(
        f"Rank {e.rank}: {e.name} — {e.final_score} — {e.status}"
        for e in shortlist
    )

    expected = _expected_decision_summary(task)

    return LLMTestCase(
        name=f"{task.id}_generate_shortlist",
        input=f"Generate a ranked shortlist from these scorecards:\n{scorecards_text}",
        actual_output=shortlist_text,
        expected_output=expected,
        context=_jd_to_context(task),
        retrieval_context=[scorecards_text],
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_deepeval_test_cases(
    task: EvalTask,
    trace: TraceCapture,
) -> List[LLMTestCase]:
    """Build all relevant DeepEval test cases for a single task.

    Creates one test case per agent step that produced output in the trace.
    Test cases that cannot be built (missing trace data) are silently skipped.

    Args:
        task:  The EvalTask providing expected outputs and JD context.
        trace: The captured execution trace for the task.

    Returns:
        List of LLMTestCase objects (may be empty if trace has no relevant output).
    """
    cases: List[LLMTestCase] = []

    builders = [
        build_parse_resume_test_case,
        build_score_candidates_test_case,
        build_verify_scores_test_case,
        build_generate_shortlist_test_case,
    ]

    for builder in builders:
        try:
            case = builder(task, trace)
            if case is not None:
                cases.append(case)
        except Exception as exc:
            logger.warning(
                "Failed to build DeepEval test case for %s with %s: %s",
                task.id, builder.__name__, exc,
            )

    logger.debug("Built %d DeepEval test cases for task %s", len(cases), task.id)
    return cases


def build_all_test_cases(
    tasks_and_traces: List[tuple[EvalTask, TraceCapture]],
) -> Dict[str, List[LLMTestCase]]:
    """Build DeepEval test cases for all tasks.

    Args:
        tasks_and_traces: List of (EvalTask, TraceCapture) pairs.

    Returns:
        Dict mapping task_id → list of LLMTestCase objects.
    """
    all_cases: Dict[str, List[LLMTestCase]] = {}
    for task, trace in tasks_and_traces:
        cases = build_deepeval_test_cases(task, trace)
        all_cases[task.id] = cases
    return all_cases
