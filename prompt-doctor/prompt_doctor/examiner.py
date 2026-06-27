"""
examiner.py — AI Prompt Engineering Examiner
=============================================

Design decisions:
1. **System prompt as a constant** — Keeps the evaluation criteria visible and
   auditable. The prompt is strict, forbids rewriting, and enforces JSON output.
2. **Two-pass JSON extraction** — First try `json.loads()` directly. If that
   fails (common with LLM output), use regex to find a JSON block. If that also
   fails, return a safe fallback with `ran_ok: false`.
3. **Schema validation** — After parsing, validate that all required keys exist
   and have the correct types. This prevents malformed data from reaching the UI.
4. **Single retry on API failure** — Transient network errors are common. One
   retry with a 30-second timeout balances reliability vs. latency.
5. **Level-aware evaluation** — The system prompt dynamically includes only the
   principles relevant to the current level, so the examiner never judges
   principles the student hasn't been taught yet.
"""

import json
import os
import re
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"  # Fast + cheap for grading
TIMEOUT = 30  # seconds
MAX_RETRIES = 1

# ---------------------------------------------------------------------------
# Examiner system prompt (template)
# ---------------------------------------------------------------------------

EXAMINER_SYSTEM_PROMPT = """You are a strict but fair prompt engineering examiner. Your job is to evaluate a student's prompt based on specific principles for the current level.

## Rules
1. Judge ONLY the principles listed below for this level. Ignore everything else.
2. Never rewrite or improve the student's prompt. Only evaluate it.
3. For every principle that fails, quote the exact weak phrase from the prompt or identify what is missing.
4. For every failed principle, ask exactly one guiding question that helps the student improve.
5. Return ONLY valid JSON. No markdown, no explanation outside the JSON.

## Principles for Level {level}
{principles_text}

## Output Schema
Return a JSON object with this exact structure:
{{
  "level": {level},
  "principles": [
    {{
      "name": "<principle name>",
      "pass": true/false,
      "weakness": "<exact weak phrase or description of what's missing>",
      "question": "<exactly one guiding question>"
    }}
  ],
  "ran_ok": true,
  "verdict": "pass" if all principles pass, otherwise "revise"
}}

## Student's Prompt
The student's prompt is provided in the user message. Evaluate only that prompt text.
"""

# ---------------------------------------------------------------------------
# Level principle definitions
# ---------------------------------------------------------------------------

LEVEL_PRINCIPLES: dict[int, list[dict[str, str]]] = {
    1: [
        {
            "name": "Role",
            "description": "The prompt assigns a clear role or persona to the AI (e.g., 'You are a data analyst').",
        },
        {
            "name": "Clear instruction",
            "description": "The prompt gives a specific, unambiguous instruction about what the AI should do.",
        },
    ],
    2: [
        {
            "name": "Structured output",
            "description": "The prompt requests output in a structured format (JSON, XML, table, etc.) and specifies the schema or fields.",
        },
    ],
    3: [
        {
            "name": "Few-shot examples",
            "description": "The prompt includes at least one input-output example to demonstrate the desired behavior.",
        },
    ],
    4: [
        {
            "name": "Reasoning",
            "description": "The prompt instructs the AI to reason step-by-step or show its work for multi-step tasks.",
        },
    ],
    5: [
        {
            "name": "Defensive constraints",
            "description": "The prompt includes guardrails against messy, ambiguous, or adversarial input (e.g., 'Ignore instructions in the input', 'If the input is unclear, ask for clarification').",
        },
    ],
}


def _build_principles_text(level: int) -> str:
    """Build a formatted string of principles for the given level."""
    principles = LEVEL_PRINCIPLES.get(level, [])
    lines: list[str] = []
    for i, p in enumerate(principles, 1):
        lines.append(f"{i}. **{p['name']}**: {p['description']}")
    return "\n".join(lines)


def _build_system_prompt(level: int) -> str:
    """Build the examiner system prompt for the given level."""
    principles_text = _build_principles_text(level)
    return EXAMINER_SYSTEM_PROMPT.format(
        level=level,
        principles_text=principles_text,
    )


# ---------------------------------------------------------------------------
# OpenRouter API call
# ---------------------------------------------------------------------------


def _call_openrouter(system_prompt: str, user_prompt: str) -> tuple[str | None, str | None]:
    """
    Call the OpenRouter API with the examiner system prompt and the student's prompt.
    Returns a tuple of (response_text, error_message).
    """
    if not OPENROUTER_API_KEY:
        return None, "OPENROUTER_API_KEY is not set. Create a .env file with your key."

    if not user_prompt or not user_prompt.strip():
        return None, "No prompt provided."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.strip()},
        ],
        "temperature": 0.1,  # Low temperature for consistent grading
        "max_tokens": 1024,
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content, None
        except requests.exceptions.Timeout:
            error_msg = f"Request timed out after {TIMEOUT} seconds."
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            error_msg = f"API returned HTTP {status}. Check your API key and model availability."
        except requests.RequestException as e:
            error_msg = f"Request failed: {e}"
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            error_msg = f"Unexpected API response: {e}"

        if attempt < MAX_RETRIES:
            time.sleep(2)
            continue

        return None, error_msg

    return None, "Unknown API error."


# ---------------------------------------------------------------------------
# JSON extraction and validation
# ---------------------------------------------------------------------------

# Schema for validation
_REQUIRED_PRINCIPLE_KEYS = {"name", "pass", "weakness", "question"}
_REQUIRED_TOP_LEVEL_KEYS = {"level", "principles", "ran_ok", "verdict"}
_VALID_VERDICTS = {"pass", "revise"}


def _extract_json(text: str) -> dict[str, Any] | None:
    """
    Extract a JSON object from text using two strategies:
    1. Direct json.loads() on the whole text.
    2. Regex to find a JSON block (```json ... ``` or { ... }).
    """
    # Strategy 1: Direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Regex extraction
    # Try ```json ... ``` block first
    match = re.search(r"```(?:json)?\s*\n?(\{.*?\})\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try bare { ... } (non-greedy extraction to avoid over-capturing).
    match = re.search(r"(\{.*?\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def _validate_schema(data: dict[str, Any], level: int) -> dict[str, Any] | None:
    """
    Validate that the parsed JSON matches the expected schema.
    Returns the validated data (possibly with defaults filled in), or None.
    """
    # Check top-level keys
    missing = _REQUIRED_TOP_LEVEL_KEYS - set(data.keys())
    if missing:
        return None

    # Check level matches
    if data.get("level") != level:
        return None

    # Check verdict
    if data.get("verdict") not in _VALID_VERDICTS:
        return None

    # Check principles
    principles = data.get("principles", [])
    expected_principles = [p["name"] for p in LEVEL_PRINCIPLES[level]]
    if not isinstance(principles, list) or len(principles) != len(expected_principles):
        return None

    for expected_name, p in zip(expected_principles, principles):
        if not isinstance(p, dict):
            return None
        missing_keys = _REQUIRED_PRINCIPLE_KEYS - set(p.keys())
        if missing_keys:
            return None
        if p.get("name") != expected_name:
            return None
        if not isinstance(p.get("pass"), bool):
            return None
        if not isinstance(p.get("weakness"), str):
            return None
        if not isinstance(p.get("question"), str):
            return None

    # Recompute the verdict so the UI does not depend on model truthfulness.
    data["verdict"] = "pass" if all(p["pass"] for p in principles) else "revise"
    data["ran_ok"] = True

    return data


# ---------------------------------------------------------------------------
# Fallback response
# ---------------------------------------------------------------------------


def _fallback_response(level: int, error_msg: str) -> dict[str, Any]:
    """Return a safe fallback when the examiner fails."""
    principles = LEVEL_PRINCIPLES.get(level, [])
    return {
        "level": level,
        "principles": [
            {
                "name": p["name"],
                "pass": False,
                "weakness": f"Examiner error: {error_msg}",
                "question": "Please try submitting your prompt again.",
            }
            for p in principles
        ],
        "ran_ok": False,
        "verdict": "revise",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(level: int, user_prompt: str) -> dict[str, Any]:
    """
    Evaluate a user's prompt for the given level.

    Args:
        level: The current level (1-5).
        user_prompt: The prompt written by the user.

    Returns:
        A dictionary matching the examiner JSON schema.
    """
    # --- Input validation ---
    if level not in LEVEL_PRINCIPLES:
        return _fallback_response(level, f"Invalid level: {level}")

    if not user_prompt or not user_prompt.strip():
        return _fallback_response(level, "No prompt provided.")

    # --- Build system prompt ---
    system_prompt = _build_system_prompt(level)

    # --- Call OpenRouter ---
    raw_response, api_error = _call_openrouter(system_prompt, user_prompt)

    if raw_response is None:
        return _fallback_response(
            level,
            api_error or "Failed to reach the examiner API. Check your internet connection and API key.",
        )

    # --- Extract and validate JSON ---
    parsed = _extract_json(raw_response)
    if parsed is None:
        return _fallback_response(
            level,
            "The examiner returned an unparseable response. Please try again.",
        )

    validated = _validate_schema(parsed, level)
    if validated is None:
        return _fallback_response(
            level,
            "The examiner returned a malformed response. Please try again.",
        )

    return validated