# Bug Report Triage – 4-Stage LLM Prompt Pipeline

Day 2 Homework: The Prompt Pipeline  
GenAI & Agentic AI Engineering

---

## What it does

Runs an incoming bug report through a four-stage LLM pipeline:

| Stage | Role | Output |
|-------|------|--------|
| 1 – Understand | Senior QA Engineer | Structured JSON extraction |
| 2 – Reason | Senior Debugging Engineer (CoT) | Root cause + severity |
| 3 – Produce | Principal Engineer | Developer-ready fix plan |
| 4 – Self-Check | AI Quality Auditor | Quality score + revised summary |

Every stage's input and output is printed to the terminal so the pipeline is fully inspectable. All results are also saved to `pipeline_results.json`.

---

## Project layout

```
bug-report-triage/
├── main.py              # Complete pipeline (single file)
├── app.py               # Streamlit web UI
├── .env                 # API key (you fill this in)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Setup

### 1. Clone / open the project directory

```bash
cd bug-report-triage
```

### 2. Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your OpenRouter API key

Edit `.env` and replace the placeholder value:

```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx
```

Get a free key at <https://openrouter.ai/keys>.

---

## Run

### Option A – Streamlit Web UI (recommended)

```bash
streamlit run app.py
```

Opens a browser at `http://localhost:8501`. You can:
- Paste any bug report into the text area
- Load one of three built-in presets with one click
- Watch each stage complete with a live progress bar
- See a summary dashboard (severity, confidence, quality score)
- Download the full JSON results

### Option B – Terminal (headless)

```bash
python main.py
```

Runs the pipeline on three hardcoded test cases and saves results to `pipeline_results.json`.

---

## Changing the model

Edit `MODEL` near the top of `main.py`:

```python
MODEL = "openai/gpt-4o-mini"   # or "anthropic/claude-3-haiku", etc.
```

Any model available on your OpenRouter plan will work.

---

## Key design decisions

- **`parse_json()` with auto-retry** – if the model returns malformed JSON the helper strips markdown fences and asks the model to regenerate valid JSON, up to `MAX_RETRIES` (default 3) times. If all retries fail the pipeline continues with a structured error placeholder rather than crashing.
- **Low temperature (0.2)** – keeps structured outputs deterministic and consistent across runs.
- **Graceful error handling in `run_pipeline()`** – each stage is wrapped in a `try/except` so a failure in one stage does not block subsequent stages.

---

## Reflection (150–200 words)

The weakest stage in this pipeline is **Stage 2 – Reason**. Its core task is root-cause analysis, yet it operates solely on information already extracted by Stage 1. Because the LLM has no access to actual source code, logs, metrics, deployment history, or runtime telemetry, its chain-of-thought is entirely speculative. The model can identify *plausible* hypotheses, but it cannot rule them in or out with evidence. Confidence percentages are therefore self-reported guesses rather than calibrated estimates, making them potentially misleading for engineering teams under pressure.

RAG (Retrieval-Augmented Generation) could dramatically improve this stage by injecting relevant snippets from the codebase, recent commit diffs, and historical bug tickets directly into the prompt. External tools such as a log-search API (e.g., CloudWatch Insights or Splunk) or a test-results feed could supply runtime evidence the LLM otherwise lacks. With that grounding, Stage 2 could move from educated guessing to evidence-driven diagnosis, raising both accuracy and the practical usefulness of the confidence scores it emits.
