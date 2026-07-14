"""
LLM invocation utilities.

Design decisions:
1. Wrap LangChain's model invocation behind a simple interface so agents
   don't depend directly on LangChain. This makes testing and swapping
   models trivial.

2. Use low temperature (0.1) for structured extraction tasks to ensure
   consistent, deterministic JSON output from the LLM.

3. Provide both raw invoke and structured output methods.
   The structured output method parses JSON and validates against Pydantic models.

4. Auto-detect demo mode: If no OPENAI_API_KEY is set, fall back to
   mock data so the workflow can be demonstrated without API credentials.
"""

import json
import logging
from typing import Optional, Type, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from models.config import Settings
from tools.demo import demo_invoke_llm, demo_invoke_llm_structured, is_demo_mode

logger = logging.getLogger(__name__)
settings = Settings()

T = TypeVar("T", bound=BaseModel)


def get_llm() -> ChatOpenAI:
    """Get the configured LLM instance.

    Uses a low temperature for consistent structured output.
    Override model/temperature via Settings or environment variables.
    """
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


def invoke_llm(prompt: str) -> str:
    """Invoke the LLM with a prompt and return raw text response.

    Auto-detects demo mode: if no API key is configured, uses mock data.

    Args:
        prompt: The full prompt to send to the LLM.

    Returns:
        Raw text response from the LLM.

    Raises:
        RuntimeError: If the LLM call fails (only in production mode).
    """
    if is_demo_mode():
        logger.info("DEMO MODE: Using mock LLM response")
        return demo_invoke_llm(prompt)

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        content = response.content.strip()
        logger.debug(f"LLM response (first 200 chars): {content[:200]}...")
        return content
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        raise RuntimeError(f"LLM invocation failed: {e}") from e


def invoke_llm_structured(prompt: str, model_class: Type[T]) -> Optional[T]:
    """Invoke the LLM and parse the response into a Pydantic model.

    Auto-detects demo mode: if no API key is configured, uses mock data.

    Args:
        prompt: The full prompt to send to the LLM.
        model_class: A Pydantic BaseModel subclass to validate against.

    Returns:
        An instance of model_class if parsing succeeds, None otherwise.
    """
    if is_demo_mode():
        logger.info(f"DEMO MODE: Using mock structured response for {model_class.__name__}")
        return demo_invoke_llm_structured(prompt, model_class)

    content = invoke_llm(prompt)

    # Strip markdown code fences if present (LLMs sometimes add them)
    cleaned = _clean_json_response(content)

    try:
        data = json.loads(cleaned)
        validated = model_class.model_validate(data)
        logger.info(f"Successfully parsed response into {model_class.__name__}")
        return validated
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.debug(f"Raw response:\n{content}")
        return None
    except ValidationError as e:
        logger.error(f"Pydantic validation failed for {model_class.__name__}: {e}")
        logger.debug(f"Parsed data:\n{data if 'data' in locals() else 'N/A'}")
        return None


def _clean_json_response(content: str) -> str:
    """Remove markdown code fences and leading/trailing whitespace from LLM output.

    Args:
        content: Raw LLM response string.

    Returns:
        Cleaned JSON string.
    """
    content = content.strip()

    # Remove ```json ... ``` fences
    if content.startswith("```json"):
        content = content[len("```json"):]
        if content.endswith("```"):
            content = content[:-3]
    elif content.startswith("```"):
        content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

    return content.strip()
