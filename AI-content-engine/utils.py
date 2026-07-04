"""Shared utility helpers for the AI Content Engine."""

import json
from datetime import datetime
from pathlib import Path


def timestamped_filename(prefix: str, ext: str) -> str:
    """Return a filename like 'prefix_20240101_153045.ext'.

    Args:
        prefix: Short label (e.g. 'hero', 'promo').
        ext: File extension without the dot (e.g. 'png', 'mp4').

    Returns:
        Filename string with timestamp suffix.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{ext}"


def save_campaign(
    product: str,
    tagline: str,
    blog: str,
    posts: dict,
    image_path: str,
    video_path: str,
    out_dir: str = "assets/generated",
) -> str:
    """Persist the full campaign as a JSON file for later retrieval.

    Args:
        product: Product name.
        tagline: Generated campaign tagline.
        blog: Generated blog introduction.
        posts: Dict with twitter/instagram/linkedin keys.
        image_path: Path to saved hero image.
        video_path: Path to saved promo video.
        out_dir: Directory to write the JSON file.

    Returns:
        Path to the written JSON file.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    filename = timestamped_filename("campaign", "json")
    out_path = Path(out_dir) / filename

    payload = {
        "product": product,
        "generated_at": datetime.now().isoformat(),
        "tagline": tagline,
        "blog_intro": blog,
        "social_posts": posts,
        "image_path": image_path,
        "video_path": video_path,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out_path.resolve())


def word_count(text: str) -> int:
    """Return the number of words in a string."""
    return len(text.split())


def truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending '…' if cut.

    Args:
        text: Input string.
        max_chars: Maximum allowed length.

    Returns:
        Original string if short enough, otherwise truncated version.
    """
    return text if len(text) <= max_chars else text[:max_chars - 1] + "…"
