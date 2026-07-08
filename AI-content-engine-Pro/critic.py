"""AI Self-Critique Loop — evaluates and conditionally regenerates text assets.

Pipeline for each asset:
  1. Run critic prompt against the generated text.
  2. Parse structured verdict: pass | warn | fail + notes.
  3. If 'fail', regenerate that asset once using critic feedback (max 2 attempts).
  4. Return the final asset, verdict, and any warnings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Literal

from text_gen import _chat  # reuse the existing low-level helper

# ── Types ─────────────────────────────────────────────────────────────────────

Verdict = Literal["pass", "warn", "fail"]


@dataclass
class CriticResult:
    """Holds the outcome of a critic evaluation for one asset."""

    asset_name: str
    final_text: str
    verdict: Verdict
    notes: str
    attempts: int  # 1 = first gen passed, 2 = regenerated once
    warnings: list[str] = field(default_factory=list)


# ── Internal helpers ──────────────────────────────────────────────────────────

_CRITIC_SYSTEM = """\
You are a senior brand strategist and content quality reviewer.
Evaluate the given marketing asset strictly against the brief.

Check ALL of the following:
1. Tone match  — Does the copy match the requested brand tone?
2. Audience fit — Is the language appropriate for the target audience?
3. Length      — Is it within expected limits (tagline ≤15 words, blog ~200 words, each social post ≤platform limit)?
4. Factual integrity — No invented claims or contradictions about the product?
5. Clarity     — Is the message clear and compelling?

Return ONLY valid JSON — no markdown fences.
Schema:
{
  "verdict": "pass" | "warn" | "fail",
  "notes": "<concise explanation, max 120 chars>",
  "warnings": ["<optional specific warnings>"]
}

- "pass"  → asset is good, no action needed
- "warn"  → minor issues, asset can be used but note the warnings
- "fail"  → significant issues, asset should be regenerated
"""


def _run_critic(
    asset_name: str,
    asset_text: str,
    product: str,
    audience: str,
    tone: str,
    extra_context: str = "",
) -> tuple[Verdict, str, list[str]]:
    """Call the critic LLM and parse its JSON verdict.

    Returns:
        (verdict, notes, warnings)
    """
    user_msg = (
        f"Asset type: {asset_name}\n"
        f"Product: {product}\n"
        f"Target audience: {audience}\n"
        f"Brand tone: {tone}\n"
    )
    if extra_context:
        user_msg += f"Additional context: {extra_context}\n"
    user_msg += f"\nAsset to evaluate:\n{asset_text}"

    messages = [
        {"role": "system", "content": _CRITIC_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    raw = _chat(messages, temperature=0.2)
    # Strip any accidental markdown fences
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        data = json.loads(raw)
        verdict: Verdict = data.get("verdict", "pass")
        if verdict not in ("pass", "warn", "fail"):
            verdict = "pass"
        notes: str = data.get("notes", "")
        warnings: list[str] = data.get("warnings", [])
        return verdict, notes, warnings
    except json.JSONDecodeError:
        # If parsing fails, treat as pass to avoid blocking the pipeline
        return "pass", "Critic response could not be parsed; skipping evaluation.", []


def _build_regeneration_context(
    asset_name: str,
    original_text: str,
    notes: str,
    warnings: list[str],
) -> str:
    """Build an instruction string to guide regeneration using critic feedback."""
    parts = [f"The previous {asset_name} had issues and must be regenerated."]
    parts.append(f"Critic notes: {notes}")
    if warnings:
        parts.append("Specific warnings to address: " + "; ".join(warnings))
    parts.append("Fix all issues while keeping the core message intact.")
    return " ".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────

MAX_RETRIES = 2


def critique_and_refine(
    asset_name: str,
    initial_text: str,
    product: str,
    audience: str,
    tone: str,
    regenerate_fn: Callable[[str], str],
    extra_context: str = "",
) -> CriticResult:
    """Evaluate an asset and regenerate it if the critic returns 'fail'.

    Args:
        asset_name:    Human-readable name, e.g. "Tagline", "Blog Introduction".
        initial_text:  The first-generation text to evaluate.
        product:       Product name (passed through to critic prompt).
        audience:      Target audience (passed through to critic prompt).
        tone:          Brand tone (passed through to critic prompt).
        regenerate_fn: Callable(feedback_context) → new text.
                       Receives a plain-English description of what to fix.
        extra_context: Optional additional context for the critic.

    Returns:
        CriticResult with final text, verdict, notes, warnings, and attempt count.
    """
    current_text = initial_text
    attempts = 1

    for attempt in range(1, MAX_RETRIES + 1):
        verdict, notes, warnings = _run_critic(
            asset_name, current_text, product, audience, tone, extra_context
        )

        if verdict != "fail":
            # pass or warn — use this text
            return CriticResult(
                asset_name=asset_name,
                final_text=current_text,
                verdict=verdict,
                notes=notes,
                attempts=attempt,
                warnings=warnings,
            )

        # verdict == "fail" and we still have retries left
        if attempt < MAX_RETRIES:
            feedback_ctx = _build_regeneration_context(asset_name, current_text, notes, warnings)
            try:
                current_text = regenerate_fn(feedback_ctx)
            except Exception as exc:
                # If regeneration itself fails, return what we have with a warning
                return CriticResult(
                    asset_name=asset_name,
                    final_text=current_text,
                    verdict="warn",
                    notes=f"Critic flagged issues but regeneration failed: {exc}",
                    attempts=attempt,
                    warnings=warnings + [str(exc)],
                )
        attempts = attempt + 1

    # Exhausted retries — return last text with fail verdict
    verdict, notes, warnings = _run_critic(
        asset_name, current_text, product, audience, tone, extra_context
    )
    return CriticResult(
        asset_name=asset_name,
        final_text=current_text,
        verdict=verdict if verdict != "fail" else "warn",  # degrade fail→warn after retries
        notes=notes + " (max retries reached)",
        attempts=MAX_RETRIES,
        warnings=warnings,
    )


def critique_tagline(
    tagline: str,
    product: str,
    audience: str,
    tone: str,
) -> CriticResult:
    """Convenience wrapper: critique a tagline with its dedicated regeneration fn."""
    from text_gen import generate_tagline  # local import to avoid circular dependency

    def regen(feedback: str) -> str:
        return generate_tagline(product, audience, tone, feedback_context=feedback)

    return critique_and_refine(
        "Tagline", tagline, product, audience, tone, regen
    )


def critique_blog(
    blog: str,
    product: str,
    audience: str,
    tone: str,
    tagline: str,
) -> CriticResult:
    """Convenience wrapper: critique a blog intro with its dedicated regeneration fn."""
    from text_gen import generate_blog_intro  # local import

    def regen(feedback: str) -> str:
        return generate_blog_intro(product, audience, tone, tagline, feedback_context=feedback)

    return critique_and_refine(
        "Blog Introduction", blog, product, audience, tone, regen,
        extra_context=f"Campaign tagline: {tagline}",
    )


def critique_social_posts(
    posts: dict,
    product: str,
    audience: str,
    tone: str,
    tagline: str,
) -> CriticResult:
    """Convenience wrapper: critique social posts as a combined asset."""
    from text_gen import generate_social_posts  # local import

    # Flatten posts to a single string for the critic
    combined = (
        f"Twitter: {posts.get('twitter', '')}\n"
        f"Instagram: {posts.get('instagram', '')}\n"
        f"LinkedIn: {posts.get('linkedin', '')}"
    )

    def regen(feedback: str) -> str:  # returns str representation of dict
        new_posts = generate_social_posts(product, audience, tone, tagline, feedback_context=feedback)
        # Return a string for the critic but store dict internally
        regen.last_posts = new_posts  # stash for caller to retrieve
        return (
            f"Twitter: {new_posts.get('twitter', '')}\n"
            f"Instagram: {new_posts.get('instagram', '')}\n"
            f"LinkedIn: {new_posts.get('linkedin', '')}"
        )

    regen.last_posts = posts  # initialise with original

    result = critique_and_refine(
        "Social Posts", combined, product, audience, tone, regen,
        extra_context=f"Campaign tagline: {tagline}",
    )

    # Attach the final dict as an extra attribute for the caller
    result.final_posts_dict = regen.last_posts  # type: ignore[attr-defined]
    return result
