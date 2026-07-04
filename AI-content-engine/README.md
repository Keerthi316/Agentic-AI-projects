# 🚀 AI Content Engine

> **Generate a full marketing campaign from a single product brief — tagline, blog intro, social posts, hero image, and promo video, all powered by OpenRouter AI.**

---

## Overview

The AI Content Engine is a Streamlit application that turns a product name, target audience, and brand tone into a complete marketing asset suite. Each asset is generated sequentially using prompt engineering techniques like few-shot prompting, prompt chaining, and structured output formatting — all through a single API provider (OpenRouter).

---

## Pipeline

Enter a product brief in the sidebar and click **✨ Generate Campaign**. The pipeline runs 5 steps in sequence, where each step can use output from the previous step:

| Step | Asset | Model | Technique |
|------|-------|-------|-----------|
| 1 | 🏷️ **Tagline** | `gpt-4o-mini` | Few-shot prompting with 2 examples |
| 2 | 📝 **Blog Intro** | `gpt-4o-mini` | Role prompting + prompt chaining (tagline → blog) |
| 3 | 📱 **Social Posts** | `gpt-4o-mini` | Structured JSON output (Twitter, Instagram, LinkedIn) |
| 4 | 🖼️ **Hero Image** | `flux.2-pro` | Subject+Style+Composition+Constraints formula |
| 5 | 🎬 **Promo Video** | `wan-2.6` | Image-to-video from hero image, tone-matched motion |

All results are persisted to `assets/generated/` as timestamped files — a campaign JSON, PNG image, and MP4 video.

---

## Setup

```bash
# 1. Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate        # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add API keys
# Copy .env.example → .env and add your OpenRouter keys:
#   OPENAI_API_KEY=sk-or-v1-...          (text + image)
#   OPENROUTER_VIDEO_API_KEY=sk-or-v1-... (video)

# 4. Run the app
streamlit run app.py
```

> **Note:** Video generation (Wan-2.6) requires a separate OpenRouter API key. Some models may also require approved access on your OpenRouter account.

---

## Project Structure

```
AI-content-engine/
├── app.py              # Streamlit UI — form input, orchestrates 5-step generation
├── config.py           # Model IDs, brand tones, tone→style mapping, API key validation
├── text_gen.py         # Text generation: tagline, blog intro, social posts, image prompt
├── image_gen.py        # Hero image via Flux (OpenRouter Images API, base64 response)
├── video_gen.py        # Promo video via Wan-2.6 (async polling, image-to-video)
├── utils.py            # Timestamped filenames, campaign JSON persistence, word count
├── requirements.txt    # streamlit, openai, requests, Pillow, python-dotenv
├── assets/generated/   # Output directory (images, videos, campaign JSONs)
└── .env.example        # API key template
```

---

## Brand Tones

The selected tone controls both the image generation prompt and the video motion description:

| Tone | Image Style | Video Motion |
|------|-------------|-------------|
| **Premium** | Sleek, high-contrast, dark background, gold accents | Slow dramatic push-in, cinematic lens flare |
| **Playful** | Bright vivid colours, bubbly shapes, cheerful lighting | Bouncy zoom, cheerful particle burst |
| **Eco** | Natural earthy tones, soft green palette | Gentle breeze, leaves swaying |
| **Modern** | Minimalist clean lines, neutral palette | Smooth parallax slide, clean light sweep |
| **Luxury** | Opulent textures, deep jewel tones | Slow orbit, golden bokeh |
| **Minimal** | Pure white background, single subject | Subtle float, soft shadow |

---

## Configuration

All settings are in `config.py`:

| Setting | Default | Notes |
|---------|---------|-------|
| `TEXT_MODEL` | `openai/gpt-4o-mini` | Fast, cheap text generation |
| `IMAGE_MODEL` | `black-forest-labs/flux.2-pro` | High-quality Flux model |
| `VIDEO_MODEL` | `alibaba/wan-2.6` | Open-source image-to-video |
| `IMAGE_SIZE` | `1024x1024` | Square format for hero images |
| `VIDEO_SIZE` | `1280x720` | 720p landscape video |

---

## Tech Stack

**Python 3.10+** — **Streamlit** — **OpenRouter** (`gpt-4o-mini`, `flux.2-pro`, `wan-2.6`) — `openai` SDK — `requests` — `Pillow` — `python-dotenv`

---

*Built with ❤️ using Streamlit + OpenRouter*