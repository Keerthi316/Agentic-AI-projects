# Multimodel App

Query multiple AI models at once via [OpenRouter](https://openrouter.ai) and compare their responses side by side.

## Requirements

- Python 3.11+
- An [OpenRouter API key](https://openrouter.ai/keys)

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key to .env
#    Open .env and replace the placeholder with your real key
OPENROUTER_API_KEY=sk-or-...
```

## Usage

**Interactive mode** — prompts you for input in a loop:
```bash
python main.py
```

**Single-shot mode** — pass a prompt directly:
```bash
python main.py --prompt "Explain quantum entanglement in one sentence"
```

**Override models** — comma-separated OpenRouter IDs:
```bash
python main.py --models openai/gpt-4o,anthropic/claude-sonnet-4-5 --prompt "Hello"
```

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | *(required)* | Your OpenRouter API key |
| `MODELS` | 4 defaults (see spec.md) | Comma-separated model IDs |
| `MAX_TOKENS` | `512` | Max tokens per response |

## Example Output

```
─── openai/gpt-4o (1.23s) ──────────────────────────────────
Quantum entanglement is a phenomenon where two particles...

─── anthropic/claude-sonnet-4-5 (0.98s) ────────────────────
When two particles become entangled, measuring one...
```
