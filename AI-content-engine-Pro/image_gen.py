"""Image generation — OpenRouter Image API using Flux."""

import base64
import requests
from pathlib import Path
import config

_HEADERS = lambda: {
    "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}


def generate_image(prompt: str, filename: str = "hero.png") -> str:
    """Generate a hero image via OpenRouter Image API (Flux) and save locally.

    Args:
        prompt: Image generation prompt.
        filename: Output filename under GENERATED_IMAGES_DIR.

    Returns:
        Absolute path to the saved PNG.

    Raises:
        RuntimeError: On API failure.
    """
    out_dir = Path(config.GENERATED_IMAGES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    try:
        resp = requests.post(
            f"{config.OPENROUTER_BASE}/images",
            headers=_HEADERS(),
            json={
                "model": config.IMAGE_MODEL,
                "prompt": prompt,
                "size": config.IMAGE_SIZE,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        # OpenRouter image API always returns b64_json in data[0]
        item = data["data"][0]
        if "b64_json" in item and item["b64_json"]:
            out_path.write_bytes(base64.b64decode(item["b64_json"]))
        else:
            # Fallback: download from URL if b64_json is absent
            img_resp = requests.get(item["url"], timeout=60)
            img_resp.raise_for_status()
            out_path.write_bytes(img_resp.content)

        return str(out_path.resolve())
    except Exception as exc:
        raise RuntimeError(f"Image generation failed: {exc}") from exc
