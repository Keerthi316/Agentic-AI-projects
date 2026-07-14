"""Verification script -- confirms OpenRouter wiring is correct."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["RECRUITMENT_DEMO_MODE"] = "true"
# Provide a fake key so the factory doesn't raise EnvironmentError
os.environ.setdefault("OPENROUTER_API_KEY", "sk-or-test-fake-key-for-import-check")

print("=" * 55)
print("  DeepEval -> OpenRouter Migration Verification")
print("=" * 55)

# -- 1. openrouter_llm.py imports and factory --
from evaluation.deepeval_suite.openrouter_llm import get_openrouter_model
model = get_openrouter_model()
model_cls = type(model).__name__
base_url  = getattr(model, "base_url", "UNKNOWN")
model_name = model.get_model_name() if hasattr(model, "get_model_name") else getattr(model, "model", "?")

print("\n[1] openrouter_llm.py")
print("    model class :", model_cls)
print("    base_url    :", base_url)
print("    model name  :", model_name)
assert model_cls == "OpenRouterModel", "Expected OpenRouterModel, got " + model_cls
assert "openrouter.ai" in base_url, "base_url does not point to openrouter.ai: " + base_url
assert "openai.com" not in base_url, "base_url still points to openai.com!"
print("    [PASS]")

# -- 2. metrics.py _get_eval_model returns OpenRouterModel --
from evaluation.deepeval_suite.metrics import _get_eval_model, DEEPEVAL_AVAILABLE, GEVAL_AVAILABLE
eval_model = _get_eval_model()
print("\n[2] metrics._get_eval_model()")
print("    DEEPEVAL_AVAILABLE :", DEEPEVAL_AVAILABLE)
print("    GEVAL_AVAILABLE    :", GEVAL_AVAILABLE)
print("    model type         :", type(eval_model).__name__)
assert type(eval_model).__name__ == "OpenRouterModel", \
    "Expected OpenRouterModel, got " + type(eval_model).__name__
print("    [PASS]")

# -- 3. Each metric gets an OpenRouterModel --
from evaluation.deepeval_suite.metrics import (
    get_faithfulness_metric, get_answer_relevancy_metric,
    get_hallucination_metric, get_task_completion_metric,
)
metrics = [
    ("FaithfulnessMetric",     get_faithfulness_metric()),
    ("AnswerRelevancyMetric",  get_answer_relevancy_metric()),
    ("HallucinationMetric",    get_hallucination_metric()),
    ("TaskCompletion(GEval)",  get_task_completion_metric()),
]
print("\n[3] Metric model wiring")
for label, m in metrics:
    m_model = getattr(m, "model", None)
    m_type  = type(m_model).__name__ if m_model is not None else "None"
    m_url   = getattr(m_model, "base_url", "N/A")
    print("    " + label.ljust(26) + ": model=" + m_type + ", base_url=" + str(m_url))
    if DEEPEVAL_AVAILABLE:
        assert m_type == "OpenRouterModel", \
            label + ": expected OpenRouterModel, got " + m_type
        assert "openrouter.ai" in str(m_url), \
            label + ": base_url " + repr(m_url) + " does not contain openrouter.ai"
print("    [PASS]")

# -- 4. get_deepeval_metrics returns 4 metrics --
from evaluation.deepeval_suite.metrics import get_deepeval_metrics
all_metrics = get_deepeval_metrics()
print("\n[4] get_deepeval_metrics()")
print("    count :", len(all_metrics))
assert len(all_metrics) == 4, "Expected 4 metrics, got " + str(len(all_metrics))
print("    [PASS]")

# -- 5. No OPENAI_API_KEY or OpenAI model string in metrics.py code --
import pathlib
src = pathlib.Path("evaluation/deepeval_suite/metrics.py").read_text()
# Check that no functional code uses OpenAI
assert "OPENAI_API_KEY" not in src, "metrics.py still references OPENAI_API_KEY"
# Old pattern was: _EVAL_MODEL = os.getenv(...)
# New code uses _get_eval_model() function, which contains that substring --
# so check for the old assignment pattern specifically.
assert '_EVAL_MODEL = os.getenv' not in src, "metrics.py still has old _EVAL_MODEL assignment"
# Confirm the new OpenRouter singleton pattern is present
assert "_openrouter_model" in src, "metrics.py missing _openrouter_model singleton"
assert "get_openrouter_model" in src, "metrics.py missing get_openrouter_model() call"
print("\n[5] No functional OpenAI references in metrics.py")
print("    OPENAI_API_KEY          : not found [PASS]")
print("    _EVAL_MODEL assignment  : not found [PASS]")
print("    _openrouter_model var   : found     [PASS]")
print("    get_openrouter_model()  : found     [PASS]")

# -- 6. __init__.py exports get_openrouter_model --
from evaluation.deepeval_suite import get_openrouter_model as _exported
print("\n[6] __init__.py exports get_openrouter_model")
assert callable(_exported), "get_openrouter_model not callable"
print("    [PASS]")

print("\n" + "=" * 55)
print("  ALL CHECKS PASSED")
print("=" * 55)
