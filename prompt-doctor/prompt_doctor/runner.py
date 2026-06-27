"""
runner.py — Prompt Execution Engine
====================================

Design decisions:
1. **Trims the user prompt** — Users often include extra whitespace. Strip it.
2. **Appends sample input** — The user's prompt and the sample input are sent
   together so the AI can operate on the data.
3. **Low temperature but not zero** — Temperature 0.3 gives deterministic but
   slightly varied outputs, which mimics real-world LLM usage.
4. **Separate from examiner** — The runner and examiner use independent API calls
   so a failure in one doesn't affect the other.
"""

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"
TIMEOUT = 45  # seconds (generation may take longer than grading)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_prompt(user_prompt: str, sample_input: str) -> dict[str, Any]:
    """
    Execute a user's prompt against the sample input via OpenRouter.

    Args:
        user_prompt: The prompt written by the user.
        sample_input: The test data to feed into the prompt.

    Returns:
        A dict with:
            - "ran_ok": bool — True if the API call succeeded.
            - "output": str — The AI's raw response text (if ran_ok).
            - "error": str — Error message (if not ran_ok).
    """
    if not OPENROUTER_API_KEY:
        return {
            "ran_ok": False,
            "output": "",
            "error": "OPENROUTER_API_KEY is not set. Create a .env file with your key.",
        }

    if not user_prompt or not user_prompt.strip():
        return {
            "ran_ok": False,
            "output": "",
            "error": "No prompt provided.",
        }

    # Build the full message: user prompt + sample input
    full_prompt = f"{user_prompt.strip()}\n\n{sample_input.strip()}"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": full_prompt,
            }
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        return {
            "ran_ok": True,
            "output": content,
            "error": "",
        }
    except requests.exceptions.Timeout:
        return {
            "ran_ok": False,
            "output": "",
            "error": "Request timed out after 45 seconds. Try a shorter prompt.",
        }
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        return {
            "ran_ok": False,
            "output": "",
            "error": f"API returned HTTP {status}. Check your API key and model availability.",
        }
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        return {
            "ran_ok": False,
            "output": "",
            "error": f"Failed to run prompt: {e}",
        }