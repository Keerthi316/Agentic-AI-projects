"""
OpenRouter LLM factory for DeepEval metrics.

DeepEval 4.1.0+ ships a first-class ``OpenRouterModel`` that wraps the
OpenAI-compatible ``/chat/completions`` endpoint at
https://openrouter.ai/api/v1.

This module provides a single ``get_openrouter_model()`` factory that:
- Reads ``OPENROUTER_API_KEY`` from the environment (required when DeepEval
  metrics actually run; not needed in demo/stub mode).
- Reads ``OPENROUTER_EVAL_MODEL`` (or falls back to ``DEEPEVAL_MODEL``) to
  let callers choose the model without touching source code.
- Raises a clear ``EnvironmentError`` early if the key is missing and the
  caller is not in stub mode, rather than surfacing a cryptic 401 later.

Usage
-----
    from evaluation.deepeval_suite.openrouter_llm import get_openrouter_model

    model = get_openrouter_model()
    metric = FaithfulnessMetric(model=model, threshold=0.7)

Environment variables
---------------------
OPENROUTER_API_KEY      Required. Your OpenRouter secret key.
OPENROUTER_EVAL_MODEL   Optional. Model string to use for evaluation
                        (e.g. ``openai/gpt-4o-mini``).  Falls back to
                        DEEPEVAL_MODEL, then to the DeepEval default
                        (currently ``openai/gpt-4o-mini`` on OpenRouter).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model — cheap, capable, OpenRouter-compatible
# ---------------------------------------------------------------------------
_DEFAULT_EVAL_MODEL = "openai/gpt-4o-mini"

# ---------------------------------------------------------------------------
# Optional DeepEval import
# ---------------------------------------------------------------------------
try:
    from deepeval.models import OpenRouterModel as _OpenRouterModel
    _DEEPEVAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DEEPEVAL_AVAILABLE = False
    _OpenRouterModel = None  # type: ignore[assignment]


def get_openrouter_model(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> "_OpenRouterModel":  # type: ignore[type-arg]
    """Return a configured ``OpenRouterModel`` for DeepEval metrics.

    Parameters
    ----------
    model:
        Model string to use (e.g. ``"openai/gpt-4o-mini"``).
        Defaults to ``OPENROUTER_EVAL_MODEL`` env var, then ``DEEPEVAL_MODEL``,
        then ``openai/gpt-4o-mini``.
    api_key:
        OpenRouter API key.  Defaults to ``OPENROUTER_API_KEY`` env var.

    Returns
    -------
    deepeval.models.OpenRouterModel

    Raises
    ------
    ImportError
        If ``deepeval`` is not installed.
    EnvironmentError
        If ``OPENROUTER_API_KEY`` is not set and ``api_key`` is not provided.
    """
    if not _DEEPEVAL_AVAILABLE:
        raise ImportError(
            "deepeval is not installed. "
            "Install it with: pip install deepeval"
        )

    resolved_model = (
        model
        or os.getenv("OPENROUTER_EVAL_MODEL")
        or os.getenv("DEEPEVAL_MODEL")
        or _DEFAULT_EVAL_MODEL
    )

    resolved_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not resolved_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file or export it before running evaluations."
        )

    logger.info(
        "Creating OpenRouterModel: model=%s, base_url=https://openrouter.ai/api/v1",
        resolved_model,
    )

    return _OpenRouterModel(
        model=resolved_model,
        api_key=resolved_key,
        base_url="https://openrouter.ai/api/v1",
    )
