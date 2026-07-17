# 🤖 AI Recruitment Multi-Agent System

A production-quality multi-agent AI hiring platform built with **Streamlit**, **LangGraph**, **LangChain**, and **OpenRouter**.  
Watch multiple AI agents collaborate in real time to evaluate resumes against a Job Description.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 Multi-Agent Pipeline | 5 specialized agents orchestrated by LangGraph |
| 📄 Resume Parsing | PDF, DOCX, TXT support |
| 🔴 Live Workflow Visualization | Animated node-by-node execution |
| 📋 Real-time Logs | Per-agent execution logs streamed live |
| ✅ Human Approval | Pause-and-approve for Interview/Reject decisions |
| 📊 Dashboard | KPIs, charts, rankings |
| 📜 History | Search, filter, export, delete past runs |
| 📈 Evaluation | DeepEval integration + heuristic metrics |
| 🌙 Theme Toggle | Dark / Light mode |

---

## 🏗️ Architecture

```
User uploads JD + Resumes
         │
    ┌────▼────┐
    │Coordinator│  (deterministic, no LLM)
    └────┬────┘
         │
    ┌────▼────┐
    │ Analyst  │  Extracts structured profile via GPT
    └────┬────┘
         │
    ┌────▼────┐
    │  Scorer  │  Scores candidate against JD (5 dimensions)
    └────┬────┘
         │
    60-80? ──Yes──► ┌──────────┐
                    │ Verifier  │  Checks bias/hallucinations
                    └────┬─────┘
              Rejected+<3? ──► Scorer (retry, max 3)
                    │
    ◄───────────────┘
         │
    ┌────▼────┐
    │  Decider │  Final: Interview | Hold | Reject | Human Review
    └────┬────┘
         │
    Interview/Reject? ──► Human Approval Checkpoint
         │
        END
```

## 🤖 Agents

### 1. Coordinator
- Receives JD and resume files
- Parses text from PDF/DOCX/TXT
- Initializes shared LangGraph state
- **No LLM calls — pure deterministic routing**

### 2. Resume Analyst
- Uses GPT via OpenRouter
- Extracts: skills, experience, education, projects, certifications, summary
- Validates with Pydantic `CandidateProfile`
- Retries up to 3 times on validation failure

### 3. Scorer
- Compares candidate vs JD across 5 dimensions
- Technical, Experience, Education, Projects, Communication
- Returns structured `ScoreBreakdown` with reasons, strengths, weaknesses

### 4. Verifier (conditional — 60–80 score range)
- Audits score for bias, hallucinations, evidence quality
- If rejected: sends back to Scorer (max 3 revisions)
- If accepted or max revisions reached: proceeds to Decider

### 5. Decider
- Final recommendation: **Interview | Hold | Reject | Need Human Review**
- Provides full explanation with confidence score

---

## 📁 Project Structure

```
RecruitmentAI/
├── app.py                    # Main entry point, theme toggle, routing
├── config.py                 # Central config from .env
├── requirements.txt
├── .env                      # API keys
├── .streamlit/
│   └── config.toml           # Streamlit theme
│
├── pages/
│   ├── Dashboard.py          # KPIs, charts, activity
│   ├── Recruitment.py        # Main workflow page
│   ├── Evaluation.py         # DeepEval metrics
│   └── History.py            # Run history
│
├── agents/
│   ├── coordinator.py
│   ├── analyst.py
│   ├── scorer.py
│   ├── verifier.py
│   └── decider.py
│
├── graph/
│   ├── state.py              # RecruitmentState TypedDict
│   ├── router.py             # Deterministic routing functions
│   └── workflow.py           # LangGraph StateGraph assembly
│
├── llm/
│   └── openrouter.py         # ChatOpenAI wrapper for OpenRouter
│
├── tools/
│   ├── parser.py             # PDF/DOCX/TXT extraction
│   ├── scoring.py            # Scoring math utilities
│   └── validator.py          # Pydantic validation helpers
│
├── models/
│   └── profile.py            # Pydantic models
│
├── database/
│   └── db.py                 # SQLite CRUD
│
├── utils/
│   ├── logger.py             # Structured logging + in-memory run log
│   └── helpers.py            # JSON parsing, color helpers
│
├── prompts/
│   ├── analyst.txt
│   ├── scorer.txt
│   ├── verifier.txt
│   └── decider.txt
│
└── history/                  # (reserved for future file exports)
```

---

## 🚀 Installation & Running

### 1. Prerequisites
- Python 3.10+
- OpenRouter API key → https://openrouter.ai

### 2. Install

```bash
cd RecruitmentAI
pip install -r requirements.txt
```

### 3. Configure

Edit `.env`:

```env
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxx
OPENROUTER_MODEL=openai/gpt-4o
```

### 4. Run

```bash
streamlit run app.py
```

Open: http://localhost:8501

---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | ✅ Yes | — | Your OpenRouter API key |
| `OPENROUTER_MODEL` | No | `openai/gpt-4o` | Model to use (e.g. `openai/gpt-4-turbo`) |
| `OPENROUTER_BASE_URL` | No | `https://openrouter.ai/api/v1` | API base URL |
| `MAX_REVISIONS` | No | `3` | Max verifier→scorer revision cycles |
| `DB_PATH` | No | `./database/recruitment.db` | SQLite database path |
| `DEEPEVAL_API_KEY` | No | — | DeepEval API key for LLM-graded metrics |

---

## 📸 Screenshots

> _Add screenshots here after running the app_

- `screenshots/dashboard.png` — Dashboard with KPIs and charts
- `screenshots/recruitment.png` — Live workflow with animated nodes
- `screenshots/evaluation.png` — Metric gauges
- `screenshots/history.png` — Run history table

---

## 🔮 Future Improvements

- [ ] Async LangGraph execution for true parallel multi-resume processing
- [ ] Email notifications on Interview decisions
- [ ] ATS integration (Greenhouse, Lever)
- [ ] Custom scoring rubric editor in UI
- [ ] Resume anonymization before scoring (bias reduction)
- [ ] Candidate comparison view (side-by-side)
- [ ] Batch import from Google Drive / S3
- [ ] Role-based access (HR manager vs recruiter views)
- [ ] Fine-tuned model for domain-specific scoring
- [ ] Audit trail with diff viewer for score revisions

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit 1.35, Plotly 5.22 |
| Orchestration | LangGraph 0.1, LangChain 0.2 |
| LLM | OpenRouter API → GPT-4o |
| Validation | Pydantic v2 |
| Database | SQLite (built-in) |
| Parsing | PyPDF2, python-docx |
| Evaluation | DeepEval (optional) |
| Styling | Glassmorphism CSS, CSS animations |

---

## 📄 License

MIT License — build freely, deploy responsibly.
