"""Multi-Channel Adaptation — rewrites text assets for a target distribution channel.

Supported channels (shown in UI dropdown):
  - Original            — no rewrite, return as-is
  - B2B LinkedIn        — professional, data-driven, C-suite language
  - Gen-Z TikTok        — casual, emoji-heavy, trend-aware, ultra-short
  - Parents Facebook    — warm, trustworthy, family-centric, conversational

Only the text assets are rewritten (tagline, blog intro, social posts).
The hero image and promo video are channel-agnostic and remain unchanged.
"""

from __future__ import annotations

import json

from text_gen import _chat

# ── Channel registry ──────────────────────────────────────────────────────────

# Maps UI label → detailed rewrite instruction for the LLM system prompt.
CHANNEL_INSTRUCTIONS: dict[str, str] = {
    "Original": "",  # No-op sentinel

    "B2B LinkedIn": (
        "Rewrite for a B2B LinkedIn audience: senior decision-makers, C-suite, and investors. "
        "Use professional, data-driven language. Emphasise ROI, efficiency, scalability, and "
        "industry impact. Avoid slang. Tone: authoritative yet approachable. "
        "LinkedIn post: professional narrative with 2–3 relevant hashtags, ≤1 300 chars. "
        "Tagline: crisp value proposition ≤10 words. "
        "Blog: 200 words, business storytelling style."
    ),

    "Gen-Z TikTok": (
        "Rewrite for Gen-Z TikTok audience (ages 16–24). "
        "Use casual, punchy language. Add relevant emojis (3–5 per post). "
        "References to trends, challenges, or viral culture are encouraged. "
        "Keep sentences very short — 5–8 words max. Use all-lowercase where it feels natural. "
        "TikTok caption: ≤150 chars with 3–5 trending hashtags. "
        "Tagline: ultra-catchy, 4–6 words. "
        "Blog: rewrite as a punchy 'why you need this' narrative, 150–180 words."
    ),

    "Parents Facebook": (
        "Rewrite for parents on Facebook (ages 28–45). "
        "Use warm, friendly, trust-building language. Emphasise safety, value for money, "
        "family benefits, and ease of use. Avoid jargon. Tone: like a trusted friend giving advice. "
        "Facebook post: conversational, 2–3 short paragraphs, 1–2 gentle hashtags, ≤800 chars. "
        "Tagline: warm benefit statement ≤12 words. "
        "Blog: 200 words, reassuring and informative tone."
    ),
}

# The displayable list of channels for the Streamlit dropdown
CHANNEL_OPTIONS: list[str] = list(CHANNEL_INSTRUCTIONS.keys())


# ── LLM rewrite prompt ────────────────────────────────────────────────────────

_ADAPT_SYSTEM_BASE = """\
You are a multi-channel content strategist. Your task is to rewrite marketing assets
for a specific distribution channel, following the channel instructions exactly.

You will receive the original assets and must return ONLY valid JSON — no markdown fences.
Schema:
{
  "tagline": "<rewritten tagline>",
  "blog": "<rewritten blog introduction>",
  "twitter": "<rewritten Twitter post>",
  "instagram": "<rewritten Instagram post>",
  "linkedin": "<rewritten LinkedIn post>"
}

Preserve the product name exactly. Do not invent new product features.
"""


# ── Public API ────────────────────────────────────────────────────────────────

def adapt_for_channel(
    channel: str,
    product: str,
    audience: str,
    tone: str,
    tagline: str,
    blog: str,
    posts: dict,
) -> dict:
    """Rewrite text assets for the specified channel.

    Args:
        channel:  One of CHANNEL_OPTIONS. If "Original", returns assets unchanged.
        product:  Product name.
        audience: Target audience (original brief, for context).
        tone:     Brand tone (original brief, for context).
        tagline:  Current campaign tagline.
        blog:     Current blog introduction.
        posts:    Dict with keys: twitter, instagram, linkedin.

    Returns:
        Dict with keys: tagline, blog, twitter, instagram, linkedin.
        All values are strings.

    Raises:
        ValueError:  If channel is unrecognised or LLM returns invalid JSON.
        RuntimeError: On API failure.
    """
    if channel not in CHANNEL_INSTRUCTIONS:
        raise ValueError(
            f"Unknown channel '{channel}'. Valid options: {list(CHANNEL_INSTRUCTIONS)}"
        )

    # "Original" is a no-op — return the existing assets as-is
    if channel == "Original":
        return {
            "tagline": tagline,
            "blog": blog,
            "twitter": posts.get("twitter", ""),
            "instagram": posts.get("instagram", ""),
            "linkedin": posts.get("linkedin", ""),
        }

    channel_instruction = CHANNEL_INSTRUCTIONS[channel]

    system_prompt = _ADAPT_SYSTEM_BASE + f"\n\nChannel instructions:\n{channel_instruction}"

    user_msg = (
        f"Product: {product}\n"
        f"Original target audience: {audience}\n"
        f"Original brand tone: {tone}\n\n"
        f"ORIGINAL TAGLINE:\n{tagline}\n\n"
        f"ORIGINAL BLOG INTRODUCTION:\n{blog}\n\n"
        f"ORIGINAL SOCIAL POSTS:\n"
        f"Twitter: {posts.get('twitter', '')}\n"
        f"Instagram: {posts.get('instagram', '')}\n"
        f"LinkedIn: {posts.get('linkedin', '')}\n\n"
        f"Rewrite all assets for the '{channel}' channel."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    raw = _chat(messages, temperature=0.75)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Channel adaptation response was not valid JSON:\n{raw}"
        ) from exc

    return {
        "tagline":   data.get("tagline", tagline),
        "blog":      data.get("blog", blog),
        "twitter":   data.get("twitter", posts.get("twitter", "")),
        "instagram": data.get("instagram", posts.get("instagram", "")),
        "linkedin":  data.get("linkedin", posts.get("linkedin", "")),
    }
