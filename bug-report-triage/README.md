# 🐛 Bug Report Triage Pipeline

>A **4-stage LLM pipeline** that transforms raw bug reports into developer-ready triage tickets with AI-powered analysis, root-cause reasoning, fix recommendations, and quality self-checks.

---

## Pipeline Stages

| Stage | Role | What it does |
|-------|------|-------------|
| **1 – 🔍 Understand** | Senior QA Engineer | Extracts structured fields (summary, steps, errors) from raw bug text |
| **2 – 🧠 Reason** | Senior Debugging Engineer | Chain-of-thought root-cause analysis + severity & confidence |
| **3 – 🛠️ Produce** | Principal Engineer | Generates developer-ready fix recommendation & debugging steps |
| **4 – ✅ Self-Check** | AI Quality Auditor | Scores quality (1–10), flags issues, suggests revisions |

Every stage is wrappped in `try/except` so a single failure never blocks the rest of the pipeline.

---

## Quick Start

```bash
# Setup
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -r requirements.txt
# Edit .env → add your OpenRouter API key

# Run (pick one)
streamlit run app.py   # Web UI with progress bars & summary dashboard
python main.py          # Terminal — runs 3 test cases, saves to JSON
```

---

## Features

- **4 specialised AI personas** — each stage uses a distinct role-specific prompt
- **Graceful error handling** — auto-retries malformed JSON (up to 3 attempts), never crashes on stage failure
- **3 built-in presets** — normal report, complex stack trace, and gibberish input (stress test)
- **Dual interface** — Streamlit UI or headless terminal
- **Downloadable results** — full JSON output with one click

---

## Project Structure

```
bug-report-triage/
├── app.py            # Streamlit web UI
├── main.py           # Pipeline logic + 3 test cases
├── requirements.txt  # requests, python-dotenv, streamlit
└── .env              # OPENROUTER_API_KEY
```

---

## Reflection

Stage 2 (Reason) is the weakest link. It analyses root causes using only the structured data from Stage 1, with no access to source code, logs, or runtime telemetry — its reasoning is entirely speculative. **RAG** could fix this by injecting codebase snippets, commit diffs, and historical bug data directly into the prompt.

---

## Tech Stack

**Python 3.10+** · **Streamlit** · **OpenRouter** (`gpt-4o-mini`) · `requests` · `python-dotenv`

---

*Built with ❤️ for Day 2: The Prompt Pipeline — GenAI & Agentic AI Engineering*