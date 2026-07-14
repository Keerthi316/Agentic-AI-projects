"""
DeepEval integration package for the Multi-Agent Recruitment System.

Provides:
- LLMTestCase builders for each agent step (parse_resume, score_candidates,
  verify_scores, generate_shortlist)
- Configured DeepEval metrics (Faithfulness, AnswerRelevancy, TaskCompletion,
  Hallucination) with appropriate thresholds for the recruitment domain
- OpenRouter LLM factory (``get_openrouter_model``) so all metrics route
  through OpenRouter instead of OpenAI

All exports degrade gracefully if deepeval is not installed.
"""

from .openrouter_llm import get_openrouter_model
from .test_cases import (
    build_deepeval_test_cases,
    build_all_test_cases,
    build_parse_resume_test_case,
    build_score_candidates_test_case,
    build_verify_scores_test_case,
    build_generate_shortlist_test_case,
    DEEPEVAL_AVAILABLE,
)
from .metrics import (
    get_deepeval_metrics,
    get_faithfulness_metric,
    get_answer_relevancy_metric,
    get_hallucination_metric,
    get_task_completion_metric,
    evaluate_test_case,
    MetricResult,
    FAITHFULNESS_THRESHOLD,
    ANSWER_RELEVANCY_THRESHOLD,
    TASK_COMPLETION_THRESHOLD,
    HALLUCINATION_MAX_THRESHOLD,
)

__all__ = [
    # OpenRouter LLM factory
    "get_openrouter_model",
    # Test case builders
    "build_deepeval_test_cases",
    "build_all_test_cases",
    "build_parse_resume_test_case",
    "build_score_candidates_test_case",
    "build_verify_scores_test_case",
    "build_generate_shortlist_test_case",
    "DEEPEVAL_AVAILABLE",
    # Metrics
    "get_deepeval_metrics",
    "get_faithfulness_metric",
    "get_answer_relevancy_metric",
    "get_hallucination_metric",
    "get_task_completion_metric",
    "evaluate_test_case",
    "MetricResult",
    "FAITHFULNESS_THRESHOLD",
    "ANSWER_RELEVANCY_THRESHOLD",
    "TASK_COMPLETION_THRESHOLD",
    "HALLUCINATION_MAX_THRESHOLD",
]
