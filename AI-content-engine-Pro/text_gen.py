"""Text generation — OpenRouter via OpenAI-compatible SDK.

All public functions accept an optional `feedback_context` parameter.
When provided, it is appended to the user prompt so the LLM can fix
issues flagged by the AI critic loop in critic.py.
"""

import json
from openai import OpenAI
import config

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE)
    return _client


def _chat(messages: list[dict], temperature: float = 0.8) -> str:
    resp = _get_client().chat.completions.create(
        model=config.TEXT_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def _append_feedback(user_content: str, feedback_context: str | None) -> str:
    """Append critic feedback to a user message when regenerating."""
    if not feedback_context:
        return user_content
    return user_content + f"\n\n[IMPROVEMENT REQUIRED]\n{feedback_context}"


def generate_tagline(
    product: str,
    audience: str,
    tone: str,
    feedback_context: str | None = None,
) -> str:
    """Generate a campaign tagline using few-shot prompting.

    Args:
        product:          Product name.
        audience:         Target audience description.
        tone:             Brand tone string.
        feedback_context: Optional critic feedback for regeneration.

    Returns:
        Tagline string (no quotes, no explanation).
    """
    user_msg = f"Product: {product} | Audience: {audience} | Tone: {tone}"
    messages = [
        {"role": "system", "content": (
            "You are an award-winning copywriter. "
            "Create short, punchy campaign taglines. "
            "Return ONLY the tagline — no quotes, no explanation."
        )},
        {"role": "user", "content": "Product: Nike Air Max | Audience: Young athletes | Tone: Modern"},
        {"role": "assistant", "content": "Move Faster. Live Louder."},
        {"role": "user", "content": "Product: Organic Green Tea | Audience: Health-conscious adults | Tone: Eco"},
        {"role": "assistant", "content": "Sip the Earth. Feel Alive."},
        {"role": "user", "content": _append_feedback(user_msg, feedback_context)},
    ]
    return _chat(messages, temperature=0.9)


def generate_blog_intro(
    product: str,
    audience: str,
    tone: str,
    tagline: str,
    feedback_context: str | None = None,
) -> str:
    """Generate a 200-word blog introduction via role prompting + prompt chaining.

    Args:
        product:          Product name.
        audience:         Target audience description.
        tone:             Brand tone string.
        tagline:          Campaign tagline to chain from.
        feedback_context: Optional critic feedback for regeneration.

    Returns:
        Blog introduction as flowing prose (~200 words).
    """
    user_msg = (
        f"Campaign tagline: \"{tagline}\"\n"
        f"Product: {product}\nTarget audience: {audience}\nBrand tone: {tone}\n\n"
        "Write a compelling 200-word blog introduction for this campaign."
    )
    messages = [
        {"role": "system", "content": (
            "You are a senior Content Strategist specialising in brand storytelling. "
            "Write EXACTLY 200 words. No headers, no bullet points — flowing prose only."
        )},
        {"role": "user", "content": _append_feedback(user_msg, feedback_context)},
    ]
    return _chat(messages, temperature=0.75)


def generate_social_posts(
    product: str,
    audience: str,
    tone: str,
    tagline: str,
    feedback_context: str | None = None,
) -> dict:
    """Generate social posts as structured JSON {twitter, instagram, linkedin}.

    Args:
        product:          Product name.
        audience:         Target audience description.
        tone:             Brand tone string.
        tagline:          Campaign tagline to chain from.
        feedback_context: Optional critic feedback for regeneration.

    Returns:
        Dict with keys: twitter, instagram, linkedin.

    Raises:
        ValueError: If the LLM response is not valid JSON.
    """
    user_msg = (
        f"Campaign tagline: \"{tagline}\"\n"
        f"Product: {product}\nTarget audience: {audience}\nBrand tone: {tone}\n\n"
        "Generate social posts for Twitter, Instagram, and LinkedIn."
    )
    messages = [
        {"role": "system", "content": (
            "You are a social media strategist. "
            "Return ONLY valid JSON — no markdown fences.\n"
            'Schema: {"twitter": "", "instagram": "", "linkedin": ""}\n'
            "Limits: twitter ≤280, instagram ≤2200, linkedin ≤3000. Add hashtags."
        )},
        {"role": "user", "content": _append_feedback(user_msg, feedback_context)},
    ]
    raw = _chat(messages, temperature=0.8)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
        return {k: data.get(k, "") for k in ("twitter", "instagram", "linkedin")}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Social posts response was not valid JSON:\n{raw}") from exc


def generate_image_prompt(
    product: str,
    audience: str,
    tone: str,
    tone_style: str,
) -> str:
    """Generate a hero image prompt using Subject+Style+Composition+Constraints formula.

    Args:
        product:    Product name.
        audience:   Target audience description.
        tone:       Brand tone string.
        tone_style: Visual style description from TONE_STYLE_MAP.

    Returns:
        Single-paragraph image prompt string.
    """
    messages = [
        {"role": "system", "content": (
            "You are a visual art director. "
            "Create image prompts using: Subject + Style + Composition + Constraints. "
            "Return ONLY the prompt — one paragraph, no labels."
        )},
        {"role": "user", "content": (
            f"Product: {product}\nTarget audience: {audience}\n"
            f"Brand tone: {tone}\nVisual style: {tone_style}\n\n"
            "Generate a hero image prompt for this product campaign."
        )},
    ]
    return _chat(messages, temperature=0.85)
