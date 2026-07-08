"""Voiceover Generation — adapts blog intro to a voice script, then synthesises audio.

Pipeline:
    blog_intro → adapt_blog_to_voice_script() → synthesise_voiceover() → MP3 path

Script adapter rules (injected via LLM prompt):
  - Short sentences (≤15 words each)
  - Commas for natural breath pauses
  - Ellipses (…) for dramatic effect
  - Remove all visual references ("as shown above", "see image", etc.)
  - Remove markdown / formatting
  - Aim for ~60–90 seconds of speech at a comfortable pace (~130 wpm → 130–195 words)

TTS backend: gTTS (Google Text-to-Speech, free, no extra API key needed).
The audio is saved as an MP3 in assets/generated/audio/.
"""

from __future__ import annotations

import io
from pathlib import Path

from gtts import gTTS

import config
from text_gen import _chat
from utils import timestamped_filename

# ── Paths ─────────────────────────────────────────────────────────────────────

GENERATED_AUDIO_DIR = "assets/generated/audio"

# ── Script Adapter ────────────────────────────────────────────────────────────

_SCRIPT_ADAPTER_SYSTEM = """\
You are a professional podcast scriptwriter specialising in brand voiceovers.
Transform the given blog introduction into a voice-over script.

Follow these rules strictly:
1. Use SHORT sentences — maximum 15 words each.
2. Use COMMAS to mark natural breathing pauses within longer thoughts.
3. Use ELLIPSES (…) sparingly for dramatic effect, no more than 2 per script.
4. REMOVE all visual references: "as shown", "image", "above", "click", "watch", "see", etc.
5. REMOVE all markdown, asterisks, hashes, bullet points.
6. Keep the tone and brand voice intact.
7. Aim for 130–195 words total (suitable for 60–90 seconds of speech).
8. Return ONLY the final voice script — no headings, no commentary.
"""


def adapt_blog_to_voice_script(
    blog_intro: str,
    product: str,
    tone: str,
) -> str:
    """Use an LLM to convert a blog introduction into a TTS-ready voice script.

    Args:
        blog_intro: The blog introduction text to adapt.
        product:    Product name for context.
        tone:       Brand tone for context.

    Returns:
        Adapted voice script as a plain string.
    """
    user_msg = (
        f"Product: {product}\n"
        f"Brand tone: {tone}\n\n"
        f"Blog introduction to adapt:\n{blog_intro}"
    )
    messages = [
        {"role": "system", "content": _SCRIPT_ADAPTER_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    return _chat(messages, temperature=0.65)


# ── TTS Synthesis ─────────────────────────────────────────────────────────────

# Map brand tones to the most natural-sounding gTTS language/TLD combos.
# gTTS supports `tld` to select accent: 'com' (US), 'co.uk' (UK), 'com.au' (AU), etc.
_TONE_TLD_MAP: dict[str, str] = {
    "Premium": "co.uk",   # British English — authoritative feel
    "Playful": "com.au",  # Australian English — upbeat feel
    "Eco":     "com",     # Standard US English
    "Modern":  "com",     # Standard US English
    "Luxury":  "co.uk",   # British English — refined feel
    "Minimal": "com",     # Standard US English
}


def synthesise_voiceover(
    script: str,
    tone: str,
    filename: str | None = None,
) -> str:
    """Convert a voice script to an MP3 using gTTS and save it locally.

    Args:
        script:   The adapted voice script text.
        tone:     Brand tone — used to select TTS accent.
        filename: Output filename (auto-generated if None).

    Returns:
        Absolute path to the saved MP3 file.

    Raises:
        RuntimeError: On TTS or file I/O failure.
    """
    out_dir = Path(GENERATED_AUDIO_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = timestamped_filename("voiceover", "mp3")

    out_path = out_dir / filename
    tld = _TONE_TLD_MAP.get(tone, "com")

    try:
        tts = gTTS(text=script, lang="en", tld=tld, slow=False)
        tts.save(str(out_path))
        return str(out_path.resolve())
    except Exception as exc:
        raise RuntimeError(f"Voiceover synthesis failed: {exc}") from exc


# ── Convenience entry point ───────────────────────────────────────────────────

def generate_voiceover(
    blog_intro: str,
    product: str,
    tone: str,
) -> tuple[str, str]:
    """Full pipeline: blog → adapted script → synthesised MP3.

    Args:
        blog_intro: Raw blog introduction text.
        product:    Product name.
        tone:       Brand tone.

    Returns:
        (adapted_script, mp3_path) — the script text and path to the audio file.
    """
    script = adapt_blog_to_voice_script(blog_intro, product, tone)
    mp3_path = synthesise_voiceover(script, tone)
    return script, mp3_path
