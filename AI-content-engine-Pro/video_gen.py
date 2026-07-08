"""Video generation — OpenRouter Video API (async image-to-video)."""

import base64
import time
import requests
from pathlib import Path
import config

_HEADERS = lambda: {
    "Authorization": f"Bearer {config.OPENROUTER_VIDEO_API_KEY}",
    "Content-Type": "application/json",
}


def generate_video(image_path: str, prompt_text: str, filename: str = "promo.mp4") -> str:
    """Generate a promo video from the hero image via OpenRouter Video API.

    Args:
        image_path: Path to the locally saved hero image (used as first frame).
        prompt_text: Motion description for the video.
        filename: Output filename under GENERATED_VIDEOS_DIR.

    Returns:
        Absolute path to the downloaded MP4.

    Raises:
        RuntimeError: On API or generation failure.
    """
    out_dir = Path(config.GENERATED_VIDEOS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    try:
        # Encode image as base64 data URI for first-frame reference
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        data_uri = f"data:image/png;base64,{b64}"

        # Submit generation job
        resp = requests.post(
            f"{config.OPENROUTER_BASE}/videos",
            headers=_HEADERS(),
            json={
                "model": config.VIDEO_MODEL,
                "prompt": prompt_text,
                "duration": config.VIDEO_DURATION,
                "size": config.VIDEO_SIZE,
                "frame_images": [
                    {"type": "image_url", "image_url": {"url": data_uri}, "frame_type": "first_frame"}
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        job = resp.json()
        job_id = job["id"]
        polling_url = job["polling_url"]

        # Poll until complete
        while True:
            time.sleep(20)
            poll = requests.get(polling_url, headers=_HEADERS(), timeout=30)
            poll.raise_for_status()
            status = poll.json()

            if status["status"] == "completed":
                video_url = status["unsigned_urls"][0]
                video_resp = requests.get(video_url, headers=_HEADERS(), timeout=120)
                video_resp.raise_for_status()
                out_path.write_bytes(video_resp.content)
                return str(out_path.resolve())

            elif status["status"] == "failed":
                raise RuntimeError(f"Video generation failed: {status.get('error', 'unknown')}")

    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Video generation failed: {exc}") from exc


def build_video_prompt(product: str, tone: str) -> str:
    """Build a motion prompt from product and tone."""
    motion_map = {
        "Premium":  "slow dramatic push-in, cinematic lens flare, dark atmosphere",
        "Playful":  "bouncy zoom, bright pop colours, cheerful particle burst",
        "Eco":      "gentle breeze, leaves swaying, soft natural light fade-in",
        "Modern":   "smooth parallax slide, clean white light sweep",
        "Luxury":   "slow orbit around product, golden bokeh, opulent mood",
        "Minimal":  "subtle float, soft shadow, clean fade transition",
    }
    motion = motion_map.get(tone, "smooth cinematic pan")
    return f"Promotional video for {product}. {motion}. Photorealistic, cinematic quality."
