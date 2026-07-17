"""
OpenRouter LLM wrapper using langchain-openai.
All agents import get_llm() from here.
"""
from __future__ import annotations

from functools import lru_cache
from langchain_openai import ChatOpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL


@lru_cache(maxsize=4)
def get_llm(model: str = OPENROUTER_MODEL, temperature: float = 0.1) -> ChatOpenAI:
    """Return a cached ChatOpenAI instance pointing at OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=1024,          # keep well within free-tier credit limits
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://recruitmentai.local",
            "X-Title": "AI Recruitment Multi-Agent System",
        },
    )
