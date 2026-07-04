"""Text generation — OpenRouter via OpenAI-compatible SDK."""

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


def generate_tagline(product: str, audience: str, tone: str) -> str:
    """Generate a campaign tagline using few-shot prompting."""
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
        {"role": "user", "content": f"Product: {product} | Audience: {audience} | Tone: {tone}"},
    ]
    return _chat(messages, temperature=0.9)


def generate_blog_intro(product: str, audience: str, tone: str, tagline: str) -> str:
    """Generate a 200-word blog introduction via role prompting + prompt chaining."""
    messages = [
        {"role": "system", "content": (
            "You are a senior Content Strategist specialising in brand storytelling. "
            "Write EXACTLY 200 words. No headers, no bullet points — flowing prose only."
        )},
        {"role": "user", "content": (
            f"Campaign tagline: \"{tagline}\"\n"
            f"Product: {product}\nTarget audience: {audience}\nBrand tone: {tone}\n\n"
            "Write a compelling 200-word blog introduction for this campaign."
        )},
    ]
    return _chat(messages, temperature=0.75)


def generate_social_posts(product: str, audience: str, tone: str, tagline: str) -> dict:
    """Generate social posts as structured JSON {twitter, instagram, linkedin}."""
    messages = [
        {"role": "system", "content": (
            "You are a social media strategist. "
            "Return ONLY valid JSON — no markdown fences.\n"
            'Schema: {"twitter": "", "instagram": "", "linkedin": ""}\n'
            "Limits: twitter ≤280, instagram ≤2200, linkedin ≤3000. Add hashtags."
        )},
        {"role": "user", "content": (
            f"Campaign tagline: \"{tagline}\"\n"
            f"Product: {product}\nTarget audience: {audience}\nBrand tone: {tone}\n\n"
            "Generate social posts for Twitter, Instagram, and LinkedIn."
        )},
    ]
    raw = _chat(messages, temperature=0.8)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
        return {k: data.get(k, "") for k in ("twitter", "instagram", "linkedin")}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Social posts response was not valid JSON:\n{raw}") from exc


def generate_image_prompt(product: str, audience: str, tone: str, tone_style: str) -> str:
    """Generate a hero image prompt using Subject+Style+Composition+Constraints formula."""
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
