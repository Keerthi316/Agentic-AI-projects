# 🚀 AI Content Engine Pro

> Generate a full marketing campaign in minutes — complete with AI self-critique, voiceover, hero image, promo video, and multi-channel adaptation.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit&logoColor=white)
![OpenRouter](https://img.shields.io/badge/Powered%20by-OpenRouter-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ What It Does

AI Content Engine Pro is a **Streamlit web app** that takes a product name, target audience, and brand tone — then generates a complete, ready-to-publish marketing campaign using a pipeline of AI models through [OpenRouter](https://openrouter.ai).

| Asset | Model / Method |
|---|---|
| 🏷️ Campaign Tagline | GPT-4o Mini (few-shot prompting) |
| 📝 Blog Introduction | GPT-4o Mini (~200 words, role prompting) |
| 📱 Social Posts | GPT-4o Mini (Twitter, Instagram, LinkedIn) |
| 🖼️ Hero Image | Flux.2-Pro (text-to-image) |
| 🎬 Promo Video | Wan-2.6 (image-to-video, async) |
| 🎙️ Voiceover MP3 | gTTS (Google TTS, no extra key needed) |

---

## 🤖 Pro Features

### AI Self-Critique Loop
Every text asset (tagline, blog, social posts) is automatically evaluated by a second LLM call acting as a **senior brand strategist**. It checks for tone match, audience fit, length, and clarity — then triggers a regeneration pass if the asset fails. No more low-quality first drafts slipping through.

```
Generate → Critic evaluates → Pass/Warn → ship it
                           └─ Fail ──→ Regenerate → ship it
```

### Voiceover Generation
The blog introduction is adapted into a TTS-ready script (short sentences, breath pauses, no visual references) and then synthesised to an **MP3 audio file** using Google Text-to-Speech. The accent is matched to the brand tone (e.g. British English for Premium/Luxury, US English for Modern/Minimal).

### Multi-Channel Adaptation
After the initial campaign is generated, rewrite all text assets for a specific distribution channel with one click:

| Channel | Style |
|---|---|
| **B2B LinkedIn** | Professional, data-driven, C-suite language |
| **Gen-Z TikTok** | Casual, emoji-heavy, ultra-short, trend-aware |
| **Parents Facebook** | Warm, trustworthy, family-centric, conversational |

Hero image and promo video are channel-agnostic and remain unchanged.

---

## 🗂️ Project Structure

```
AI-content-engine-Pro/
├── app.py            # Streamlit entry point — UI, layout, orchestration
├── config.py         # Central config — API keys, models, paths, brand tones
├── text_gen.py       # Text generation — tagline, blog, social posts, image prompts
├── critic.py         # AI self-critique loop — evaluate and regenerate assets
├── adaptation.py     # Multi-channel rewriter — LinkedIn / TikTok / Facebook
├── voice_gen.py      # Voiceover pipeline — blog → voice script → MP3
├── image_gen.py      # Hero image generation via Flux (OpenRouter Images API)
├── video_gen.py      # Promo video generation via Wan-2.6 (OpenRouter Videos API)
├── utils.py          # Helpers — save campaign JSON, word count, timestamped filenames
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
└── assets/
    └── generated/
        ├── images/   # Saved hero images (.png)
        ├── videos/   # Saved promo videos (.mp4)
        └── audio/    # Saved voiceovers (.mp3)
```

---

## ⚡ Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/AI-content-engine-Pro.git
cd AI-content-engine-Pro
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Then edit `.env` and fill in your API keys:

```env
# Powers text (GPT-4o Mini) and image (Flux.2-Pro) generation
OPENAI_API_KEY=sk-or-...

# Powers video generation (Wan-2.6)
OPENROUTER_VIDEO_API_KEY=sk-or-...
```

> Both keys are OpenRouter API keys. You can use the same key for both, or separate keys.  
> Get yours at [openrouter.ai/keys](https://openrouter.ai/keys).

### 4. Run the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## 🎨 Brand Tones

Choose a tone in the sidebar to shape the visual and copy style across the entire campaign:

| Tone | Visual Style |
|---|---|
| **Premium** | Sleek, high-contrast, dark background, gold accents |
| **Playful** | Bright vivid colors, fun shapes, cheerful lighting |
| **Eco** | Natural earthy tones, soft green palette, organic textures |
| **Modern** | Minimalist clean lines, neutral palette, geometric shapes |
| **Luxury** | Opulent textures, deep jewel tones, dramatic moody lighting |
| **Minimal** | Pure white background, single subject, negative space |

---

## 🔧 Configuration

All tunable settings live in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `TEXT_MODEL` | `openai/gpt-4o-mini` | LLM for all text generation and critique |
| `IMAGE_MODEL` | `black-forest-labs/flux.2-pro` | Image generation model |
| `IMAGE_SIZE` | `1024x1024` | Hero image dimensions |
| `VIDEO_MODEL` | `alibaba/wan-2.6` | Image-to-video model |
| `VIDEO_DURATION` | `5` | Promo video length in seconds |
| `VIDEO_SIZE` | `1280x720` | Video resolution |

---

## 📦 Dependencies

```
streamlit>=1.35.0      # Web UI framework
openai>=2.0.0          # OpenAI-compatible SDK (used with OpenRouter)
requests>=2.31.0       # HTTP client for image/video API calls
Pillow>=10.0.0         # Image handling
python-dotenv>=1.0.0   # Load .env variables
gTTS>=2.5.0            # Google Text-to-Speech for voiceover synthesis
```

---

## 💾 Campaign Output

Every generated campaign is automatically saved as a timestamped JSON file in `assets/generated/`:

```json
{
  "product": "AquaFlow Smart Bottle",
  "tagline": "Hydrate Smarter. Live Better.",
  "blog": "...",
  "posts": {
    "twitter": "...",
    "instagram": "...",
    "linkedin": "..."
  },
  "image_path": "assets/generated/images/hero_20260702_112936.png",
  "video_path": "assets/generated/videos/promo_20260702_112936.mp4"
}
```

---

## 🚧 Notes & Limitations

- **Video generation** requires approved API access for Wan-2.6 / Sora on OpenRouter. If the video step fails, all other campaign assets are still generated and usable.
- **Voiceover** uses Google TTS (free, no extra API key), so audio quality is functional but not studio-grade.
- **Critic loop** adds roughly 15 seconds to generation time per asset. It can be disabled in the sidebar with the toggle.
- **Image generation** via Flux.2-Pro typically takes 20–30 seconds.

---

## 📄 License

MIT License — free to use, modify, and distribute.
