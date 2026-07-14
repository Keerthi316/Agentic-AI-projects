"""
DeepEval metric configurations for the Multi-Agent Recruitment System.

Defines and configures four DeepEval metrics:
  1. Faithfulness       — output is grounded in the provided resume/context
  2. Answer Relevancy   — output addresses the job description requirements
  3. Task Completion    — agent completed its assigned task correctly
  4. Hallucination      — agent did not invent qualifications or skills

Each metric is configured with a threshold appropriate for the recruitment
domain. Thresholds are more strict for safety-relevant metrics.

Design decisions:
- All metrics use OpenRouter as the LLM backend via DeepEval's built-in
  ``OpenRouterModel``.  No requests are ever made to api.openai.com.
- The model is configurable via OPENROUTER_EVAL_MODEL (or DEEPEVAL_MODEL as
  a fallback), defaulting to ``openai/gpt-4o-mini`` on OpenRouter.
- All metrics have a graceful no-op stub so the codebase compiles even
  when deepeval is not installed (CI without LLM access).
- Thresholds are chosen conservatively:
  - Faithfulness ≥ 0.70  (model may summarise, not just copy)
  - Relevancy    ≥ 0.70  (some tangential info is OK)
  - Completion   ≥ 0.80  (task must be substantially done)
  - Hallucination ≤ 0.20 (very low tolerance for invented facts)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

FAITHFULNESS_THRESHOLD = 0.70
ANSWER_RELEVANCY_THRESHOLD = 0.70
TASK_COMPLETION_THRESHOLD = 0.80
HALLUCINATION_MAX_THRESHOLD = 0.20  # This is a "lower is better" metric

# ---------------------------------------------------------------------------
# Optional DeepEval import with stubs
# ---------------------------------------------------------------------------

try:
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        HallucinationMetric,
    )
    try:
        # GEval is available in newer deepeval versions
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCaseParams
        GEVAL_AVAILABLE = True
    except ImportError:
        GEVAL_AVAILABLE = False

    DEEPEVAL_AVAILABLE = True
    logger.info("DeepEval metrics loaded successfully")

except ImportError:
    DEEPEVAL_AVAILABLE = False
    GEVAL_AVAILABLE = False
    logger.warning(
        "DeepEval not installed — using stub metrics. "
        "Install with: pip install deepeval"
    )

    # ── Stubs ──────────────────────────────────────────────────────────

    class _StubMetric:
        """No-op metric stub used when deepeval is not installed."""

        def __init__(self, threshold: float = 0.5, name: str = "stub", **kwargs):
            self.threshold = threshold
            self.name = name
            self.score = 0.0
            self.success = True
            self.reason = "deepeval not installed — metric not evaluated"

        def measure(self, test_case: Any) -> float:
            return self.score

        def is_successful(self) -> bool:
            return self.success

    FaithfulnessMetric = _StubMetric  # type: ignore[assignment, misc]
    AnswerRelevancyMetric = _StubMetric  # type: ignore[assignment, misc]
    HallucinationMetric = _StubMetric  # type: ignore[assignment, misc]

    class GEval(_StubMetric):  # type: ignore[no-redef]
        pass

    class LLMTestCaseParams:  # type: ignore[no-redef]
        ACTUAL_OUTPUT = "actual_output"
        EXPECTED_OUTPUT = "expected_output"
        INPUT = "input"
        CONTEXT = "context"
        RETRIEVAL_CONTEXT = "retrieval_context"


# ---------------------------------------------------------------------------
# OpenRouter model — lazy singleton
# ---------------------------------------------------------------------------

_openrouter_model: Any = None  # cached instance


def _get_eval_model() -> Any:
    """Return a cached OpenRouterModel instance for all DeepEval metrics.

    The model is created once and reused across all metric factory calls to
    avoid re-reading the environment and constructing multiple clients.

    Returns
    -------
    deepeval.models.OpenRouterModel, or a plain model-name string if deepeval
    is not installed (stub mode).

    Raises
    ------
    EnvironmentError
        If ``OPENROUTER_API_KEY`` is not set when deepeval is available.
    """
    global _openrouter_model
    if _openrouter_model is not None:
        return _openrouter_model

    if not DEEPEVAL_AVAILABLE:
        # Stub mode — model object is not used, return a placeholder string
        _openrouter_model = "openrouter-stub"
        return _openrouter_model

    from evaluation.deepeval_suite.openrouter_llm import get_openrouter_model
    _openrouter_model = get_openrouter_model()
    return _openrouter_model


# ---------------------------------------------------------------------------
# Metric factory functions
# ---------------------------------------------------------------------------


def get_faithfulness_metric() -> FaithfulnessMetric:
    """Return a configured FaithfulnessMetric backed by OpenRouter.

    Checks that the LLM output is grounded in the retrieved context
    (the resume text and JD). Penalises claims not supported by the source.

    Returns:
        FaithfulnessMetric with threshold=0.70.
    """
    if DEEPEVAL_AVAILABLE:
        return FaithfulnessMetric(
            threshold=FAITHFULNESS_THRESHOLD,
            model=_get_eval_model(),
            include_reason=True,
        )
    return FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, name="faithfulness")


def get_answer_relevancy_metric() -> AnswerRelevancyMetric:
    """Return a configured AnswerRelevancyMetric backed by OpenRouter.

    Checks that the agent's output is relevant to the input (JD + resume).
    Penalises generic, off-topic, or verbatim-template responses.

    Returns:
        AnswerRelevancyMetric with threshold=0.70.
    """
    if DEEPEVAL_AVAILABLE:
        return AnswerRelevancyMetric(
            threshold=ANSWER_RELEVANCY_THRESHOLD,
            model=_get_eval_model(),
            include_reason=True,
        )
    return AnswerRelevancyMetric(threshold=ANSWER_RELEVANCY_THRESHOLD, name="answer_relevancy")


def get_hallucination_metric() -> HallucinationMetric:
    """Return a configured HallucinationMetric backed by OpenRouter.

    Checks that the agent does not invent skills, experience, or qualifications
    not present in the original resume.

    Returns:
        HallucinationMetric with threshold=0.20 (lower is better).
    """
    if DEEPEVAL_AVAILABLE:
        return HallucinationMetric(
            threshold=HALLUCINATION_MAX_THRESHOLD,
            model=_get_eval_model(),
            include_reason=True,
        )
    return HallucinationMetric(threshold=HALLUCINATION_MAX_THRESHOLD, name="hallucination")


def get_task_completion_metric() -> Any:
    """Return a configured TaskCompletion metric backed by OpenRouter.

    Uses GEval (G-Eval framework) to assess whether the agent completed
    its assigned task. Checks:
    - Resume Analyst: structured profile was extracted
    - Scorer: score is within 0-100 with reasoning
    - Verifier: blind score produced with fairness notes
    - Decider: ranked shortlist with statuses

    Falls back to a stub if GEval is unavailable.

    Returns:
        GEval metric or stub.
    """
    if DEEPEVAL_AVAILABLE and GEVAL_AVAILABLE:
        return GEval(
            name="task_completion",
            criteria=(
                "Determine whether the agent successfully completed its assigned task. "
                "A complete task means: "
                "(1) Resume parsing produces a structured profile with name and skills. "
                "(2) Scoring produces a numeric score 0-100 with explicit reasoning. "
                "(3) Verification produces a blind score and fairness notes. "
                "(4) Shortlist generation produces a ranked list with status for each candidate. "
                "Score 1.0 if the task is completely and correctly done, "
                "0.5 if partially done, 0.0 if not done or critically wrong."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=TASK_COMPLETION_THRESHOLD,
            model=_get_eval_model(),
        )
    return GEval(threshold=TASK_COMPLETION_THRESHOLD, name="task_completion")


def get_deepeval_metrics() -> List[Any]:
    """Return the full list of configured DeepEval metrics.

    All metrics share the same OpenRouterModel instance (via ``_get_eval_model``),
    so the API key and base_url are resolved only once.

    Returns:
        List of metric objects ready for use in deepeval.evaluate().
    """
    return [
        get_faithfulness_metric(),
        get_answer_relevancy_metric(),
        get_hallucination_metric(),
        get_task_completion_metric(),
    ]


# ---------------------------------------------------------------------------
# Metric result types (for use when deepeval is not available)
# ---------------------------------------------------------------------------


class MetricResult:
    """Lightweight container for a metric evaluation result.

    Used by test_outputs.py when deepeval is not installed, and also
    to normalise results from deepeval metrics for report generation.
    """

    def __init__(
        self,
        metric_name: str,
        score: float,
        threshold: float,
        passed: bool,
        reason: str = "",
    ):
        self.metric_name = metric_name
        self.score = score
        self.threshold = threshold
        self.passed = passed
        self.reason = reason

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.metric_name}: score={self.score:.2f}, "
            f"threshold={self.threshold:.2f} — {self.reason[:80]}"
        )


def evaluate_test_case(test_case: Any, metrics: Optional[List[Any]] = None) -> List[MetricResult]:
    """Evaluate a single test case against DeepEval metrics.

    Args:
        test_case: A DeepEval LLMTestCase or stub.
        metrics:   Optional list of metrics. Defaults to get_deepeval_metrics().

    Returns:
        List of MetricResult objects.
    """
    if metrics is None:
        metrics = get_deepeval_metrics()

    results: List[MetricResult] = []

    for metric in metrics:
        try:
            if DEEPEVAL_AVAILABLE:
                metric.measure(test_case)
                score = float(metric.score)
                passed = metric.is_successful()
                reason = getattr(metric, "reason", "")
            else:
                # Stub metrics always pass with score 0.0 placeholder
                score = 0.0
                passed = True
                reason = "deepeval not installed — skipped"

            results.append(MetricResult(
                metric_name=getattr(metric, "name", type(metric).__name__),
                score=score,
                threshold=getattr(metric, "threshold", 0.5),
                passed=passed,
                reason=str(reason)[:200] if reason else "",
            ))
        except Exception as exc:
            logger.warning("Metric %s failed: %s", type(metric).__name__, exc)
            results.append(MetricResult(
                metric_name=type(metric).__name__,
                score=0.0,
                threshold=getattr(metric, "threshold", 0.5),
                passed=False,
                reason=f"Evaluation error: {exc}",
            ))

    return results
