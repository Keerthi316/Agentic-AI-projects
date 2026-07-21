# 🤖 Agentic AI Projects

A collection of AI-powered applications built with **Python**, **Streamlit**, **LangGraph**, and **OpenRouter**. Each project explores a different facet of agentic AI — from multi-agent pipelines and RAG chatbots to content generation engines and prompt engineering tools.

---

## 📦 Projects

### 1. [`multiagent_recruiter`](./multiagent_recruiter/RecruitmentAI) — AI Recruitment Multi-Agent System
> Production-quality multi-agent hiring platform with live workflow visualization.

Five specialized agents (Coordinator → Analyst → Scorer → Verifier → Decider) collaborate through a **LangGraph** state graph to evaluate resumes against a job description. Features a full **Streamlit UI** with a live animated pipeline, human approval checkpoint, KPI dashboard, run history, and optional DeepEval integration.

**Stack:** Streamlit · LangGraph · LangChain · OpenRouter (GPT-4o) · Pydantic v2 · SQLite · PyPDF2 / python-docx

---

### 2. [`recruitment-agent`](./recruitment-agent) — TechVest Recruit AI (Single-Agent)
> Autonomous end-to-end recruitment agent built on a Plan → Act → Observe loop.

A single LangGraph agent that parses resumes (PDF or plain-text), scores candidates against a weighted rubric, ranks the shortlist, checks interview availability, and pauses for **human-in-the-loop** approval before confirming interview slots. Includes a Streamlit dashboard and full trajectory logging.

**Stack:** Streamlit · LangGraph · LangChain · OpenRouter (GPT-4o Mini) · Pydantic v2 · pypdf

---

### 3. [`BVRITH_chatbot`](./BVRITH_chatbot) — BVRIT College FAQ Chatbot
> Production-quality RAG chatbot for a college knowledge base with observability and governance.

Answers student questions using a `.docx` knowledge base via **ChromaDB** vector search and LangChain. Beyond basic RAG, the project ships with full **observability** (LLM call logging, anomaly alerts, A/B testing), **evaluation** (DeepEval test suite), **governance** (Giskard scanner, Promptfoo red-teaming, fairness tests), and a **persistent memory layer** (per-user fact extraction + ChromaDB memory store).

**Stack:** Streamlit · LangChain · ChromaDB · OpenRouter (GPT-4o Mini) · DeepEval · Giskard · Promptfoo

---

### 4. [`AI-content-engine-Pro`](./AI-content-engine-Pro) — AI Content Engine Pro
> Full marketing campaign generator with AI self-critique, voiceover, and multi-channel adaptation.

Takes a product name, target audience, and brand tone, then generates a tagline, blog intro, social posts (Twitter / Instagram / LinkedIn), hero image (Flux.2-Pro), promo video (Wan-2.6), and voiceover MP3 (gTTS). An **AI critic loop** re-evaluates each text asset and triggers regeneration on failure. A **multi-channel adapter** rewrites the campaign for B2B LinkedIn, Gen-Z TikTok, or Parents Facebook.

**Stack:** Streamlit · OpenRouter (GPT-4o Mini, Flux.2-Pro, Wan-2.6) · gTTS · Pillow · python-dotenv

---

### 5. [`AI-content-engine`](./AI-content-engine) — AI Content Engine (v1)
> The original 5-step marketing campaign pipeline without the Pro features.

Generates a tagline, blog intro, social posts, hero image, and promo video from a single product brief. Demonstrates **few-shot prompting**, **prompt chaining**, and **structured JSON output** in a clean sequential pipeline.

**Stack:** Streamlit · OpenRouter (GPT-4o Mini, Flux.2-Pro, Wan-2.6) · Pillow · python-dotenv

---

### 6. [`quiz-generator`](./quiz-generator) — QuizForge — AI Quiz Generator
> Upload a PowerPoint, get an interactive multiple-choice quiz.

Parses `.pptx` files with `python-pptx`, sends slide text to OpenAI to generate MCQs at three difficulty levels (Simple / Medium / Complex), and runs an interactive quiz UI with per-question feedback and distractor explanations. Includes a Flask REST API backend for headless / programmatic usage.

**Stack:** Streamlit · OpenAI (GPT-4o Mini) · python-pptx · Flask · python-dotenv

---

### 7. [`prompt-doctor`](./prompt-doctor) — Prompt Doctor
> Learn prompt engineering through gamified, AI-graded interactive exercises.

A Streamlit learning platform with 5 progressive levels covering role prompting, structured output, few-shot examples, chain-of-thought, and defensive constraints. A **runner** executes your prompt against a sample input; an **examiner** grades it against level-specific principles and returns structured JSON feedback. Pass all principles to unlock the next level.

**Stack:** Streamlit · OpenRouter (GPT-4o Mini) · python-dotenv

---

### 8. [`bug-report-triage`](./bug-report-triage) — Bug Report Triage Pipeline
> 4-stage LLM pipeline that turns raw bug reports into developer-ready triage tickets.

Four AI personas — QA Engineer (Understand) → Debugging Engineer (Reason) → Principal Engineer (Produce) → Quality Auditor (Self-Check) — process each bug report sequentially. Every stage uses a distinct role-specific prompt, retries malformed JSON up to 3 times, and never crashes on stage failure. Ships with a Streamlit UI and a headless terminal mode.

**Stack:** Streamlit · OpenRouter (GPT-4o Mini) · requests · python-dotenv

---

### 9. [`multimodel-app`](./multimodel-app) — Multi-Model Comparison Tool
> Query multiple LLMs at once and compare responses side by side.

Sends a single prompt to several OpenRouter models in parallel and displays each answer with its latency, token counts, and cost. Supports interactive and single-shot CLI modes, and lets you override the model list at runtime.

**Stack:** Python 3.11+ · OpenRouter · python-dotenv

---

## 🗺️ Quick Reference

| Project | Interface | Key Tech | Topic |
|---|---|---|---|
| multiagent_recruiter | Streamlit | LangGraph, LangChain | Multi-agent recruiting |
| recruitment-agent | Streamlit | LangGraph, LangChain | Single-agent recruiting |
| BVRITH_chatbot | Streamlit | LangChain, ChromaDB | RAG + observability |
| AI-content-engine-Pro | Streamlit | OpenRouter, gTTS | Content generation (Pro) |
| AI-content-engine | Streamlit | OpenRouter | Content generation (v1) |
| quiz-generator | Streamlit + Flask | OpenAI, python-pptx | Quiz generation from PPT |
| prompt-doctor | Streamlit | OpenRouter | Prompt engineering education |
| bug-report-triage | Streamlit / CLI | OpenRouter | LLM pipeline / triage |
| multimodel-app | CLI | OpenRouter | Multi-model comparison |

---

## 🔑 Common Prerequisites

All projects share the same basic requirements:

- **Python 3.10+** (3.11 recommended)
- An **[OpenRouter API key](https://openrouter.ai/keys)** (free credits available) — used by most projects
- An **[OpenAI API key](https://platform.openai.com/api-keys)** — required by `quiz-generator`

Each project has its own `requirements.txt` and `.env.example`. The general setup pattern is:

```bash
cd <project-folder>
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # then add your API key(s)
streamlit run app.py   # or: python main.py
```

---

## 🏗️ Repository Structure

```
Agentic-AI-projects/
├── multiagent_recruiter/       # Multi-agent recruiting platform
│   └── RecruitmentAI/
├── recruitment-agent/          # Single-agent recruiting (LangGraph)
├── BVRITH_chatbot/             # College FAQ RAG chatbot
├── AI-content-engine-Pro/      # Marketing campaign generator (Pro)
├── AI-content-engine/          # Marketing campaign generator (v1)
├── quiz-generator/             # PPT → MCQ quiz generator
├── prompt-doctor/              # Prompt engineering learning platform
├── bug-report-triage/          # 4-stage bug triage pipeline
├── multimodel-app/             # Multi-model LLM comparison tool
└── README.md                   # This file
```

---

## 🛠️ Tech Stack Overview

| Category | Technologies Used |
|---|---|
| **UI / Frontend** | Streamlit, Flask |
| **Orchestration** | LangGraph, LangChain |
| **LLM APIs** | OpenRouter, OpenAI |
| **Models** | GPT-4o / GPT-4o Mini, Flux.2-Pro, Wan-2.6 |
| **Vector DBs** | ChromaDB |
| **Data Validation** | Pydantic v2 |
| **TTS** | gTTS (Google TTS) |
| **Document Parsing** | python-pptx, pypdf, python-docx, Docx2txt |
| **Evaluation** | DeepEval, Giskard, Promptfoo |
| **Storage** | SQLite, ChromaDB (persistent), JSON files |
| **Environment** | python-dotenv |

---

## 📄 License

All projects are released under the **MIT License** — free to use, modify, and distribute.
