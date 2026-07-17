"""Miscellaneous helpers shared across the app."""
from __future__ import annotations

import json
import re
from typing import Any


def parse_json_from_llm(text: str) -> dict:
    """
    Robustly extract the first JSON object from an LLM response.
    Handles markdown code fences, trailing text, etc.
    """
    # strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    # find first { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # last resort: try the whole string
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def truncate(text: str, max_chars: int = 200) -> str:
    return text[:max_chars] + "…" if len(text) > max_chars else text


def score_color(score: float) -> str:
    """Return a hex color based on score 0-100."""
    if score >= 80:
        return "#22c55e"   # green
    if score >= 60:
        return "#f59e0b"   # amber
    return "#ef4444"       # red


def recommendation_icon(rec: str) -> str:
    mapping = {
        "Interview": "✅",
        "Hold": "⏸️",
        "Reject": "❌",
        "Need Human Review": "🔍",
    }
    return mapping.get(rec, "❓")


def recommendation_color(rec: str) -> str:
    mapping = {
        "Interview": "#22c55e",
        "Hold": "#f59e0b",
        "Reject": "#ef4444",
        "Need Human Review": "#3b82f6",
    }
    return mapping.get(rec, "#6b7280")


def format_list(items: list, max_items: int = 6) -> str:
    if not items:
        return "—"
    shown = items[:max_items]
    suffix = f" (+{len(items) - max_items} more)" if len(items) > max_items else ""
    return ", ".join(shown) + suffix
