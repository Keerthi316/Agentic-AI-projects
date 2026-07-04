"""Central configuration — everything runs through OpenRouter."""

import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY: str = os.getenv("OPENAI_API_KEY", "")            # text + image generation
OPENROUTER_VIDEO_API_KEY: str = os.getenv("OPENROUTER_VIDEO_API_KEY", "")  # video generation only
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Text
TEXT_MODEL = "openai/gpt-4o-mini"

# Image (Flux — closest to user's "flux1.o" request; flux.2-pro is the flagship Flux model)
IMAGE_MODEL = "black-forest-labs/flux.2-pro"
IMAGE_SIZE = "1024x1024"

# Video (image-to-video via OpenRouter async API)
VIDEO_MODEL = "alibaba/wan-2.6"
VIDEO_DURATION = 5
VIDEO_SIZE = "1280x720"

# Paths
GENERATED_IMAGES_DIR = "assets/generated/images"
GENERATED_VIDEOS_DIR = "assets/generated/videos"

BRAND_TONES = ["Premium", "Playful", "Eco", "Modern", "Luxury", "Minimal"]

TONE_STYLE_MAP = {
    "Premium":  "sleek, high-contrast lighting, dark background, gold accents",
    "Playful":  "bright vivid colors, fun bubbly shapes, cheerful lighting",
    "Eco":      "natural earthy tones, soft green palette, organic textures",
    "Modern":   "minimalist clean lines, neutral palette, geometric shapes",
    "Luxury":   "opulent textures, deep jewel tones, dramatic moody lighting",
    "Minimal":  "pure white background, single subject, negative space",
}


def validate_keys() -> list[str]:
    """Return list of missing required API key names."""
    missing = []
    if not OPENROUTER_API_KEY:
        missing.append("OPENAI_API_KEY (text + image generation)")
    if not OPENROUTER_VIDEO_API_KEY:
        missing.append("OPENROUTER_VIDEO_API_KEY (video generation)")
    return missing
