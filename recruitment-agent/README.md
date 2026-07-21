# TechVest Recruit AI

An autonomous AI-powered recruitment agent built with LangGraph and OpenRouter. It evaluates candidates end-to-end — parsing resumes, scoring against a job rubric, ranking the shortlist, and scheduling interviews — with a human-in-the-loop approval step before final decisions are sent.

## How It Works

The agent follows a **Plan → Act → Observe** loop implemented as a LangGraph state graph:

```
initialize → plan → parse_resume → score → finalize → check_availability → await_approval
                ↑_______________(loop per candidate)__________________|
```

1. **Initialize** — Analyzes the job description and builds a weighted scoring rubric.
2. **Plan** — An LLM planner decides the next best action based on what's been done.
3. **Parse Resume** — Extracts structured data (skills, experience, education, projects) from raw resume text or PDF.
4. **Score** — Scores each candidate against the rubric criteria (0–5 per criterion, weighted total).
5. **Finalize** — Ranks candidates and produces a shortlist with hire/hold/reject decisions.
6. **Check Availability** — Fetches open interview slots for shortlisted candidates.
7. **Await Approval** — Pauses for human review. The recruiter approves or rejects each candidate before interview proposals are confirmed.

## Features

- **LangGraph state graph** with conditional routing and a 50-step guardrail
- **OpenRouter API** — swap models via an environment variable (default: `openai/gpt-4o-mini`)
- **PDF & plain-text resume support** via `pypdf`
- **Pydantic v2 schemas** for all structured outputs (rubric, scorecards, final ranking)
- **Full trajectory logging** — every thought, tool call, and observation is recorded
- **Human-in-the-loop** approval gate before interview scheduling
- **Streamlit dashboard** for a visual UI to upload resumes and review results

## Project Structure

```
recruitment-agent/
├── app.py                  # Streamlit frontend
├── requirements.txt
├── graph/
│   ├── graph.py            # LangGraph builder + sequential runner
│   ├── nodes.py            # All agent node implementations
│   └── state.py            # AgentState (Pydantic BaseModel)
├── models/
│   └── schemas.py          # Pydantic schemas (JDRequirements, ScoreCard, etc.)
├── prompts/
│   ├── planner_prompt.py
│   ├── jd_prompt.py
│   ├── parser_prompt.py
│   ├── scorer_prompt.py
│   ├── decision_prompt.py
│   ├── guardrail_prompt.py
│   └── schedule_prompt.py
└── tools/
    ├── parse_resume.py
    ├── score_candidate.py
    ├── availability.py
    └── interview.py
```

## Setup

**1. Clone and install dependencies**

```bash
git clone <repo-url>
cd recruitment-agent
pip install -r requirements.txt
```

**2. Configure environment variables**

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
MODEL=openai/gpt-4o-mini   # optional, this is the default
```

Get an API key at [openrouter.ai](https://openrouter.ai).

**3. Run the Streamlit app**

```bash
streamlit run app.py
```

## Usage

### Via the Streamlit UI

1. Paste or type the **job description** in the sidebar.
2. Upload one or more **resumes** (`.txt`, `.md`, or `.pdf`).
3. Click **Run Agent** and watch the agent process each candidate in real time.
4. Review the shortlist and **approve or reject** candidates in the human approval step.
5. Confirmed candidates receive interview slot proposals.

### Via Python API

```python
from graph.graph import run_agent, resume_after_approval

# Provide raw resume text per candidate
candidates = {
    "Alice Smith": "Alice Smith\n5 years Python...",
    "Bob Jones":   "Bob Jones\n3 years Java...",
}

state = run_agent(job_description="Senior Backend Engineer...", candidates=candidates)

# Human reviews state.human_approval_pending, then approves/rejects
decisions = {"Alice Smith": "Approved", "Bob Jones": "Rejected"}
final_state = resume_after_approval(state, decisions)

# Access results
print(final_state.shortlist)
print(final_state.trajectory)
```

## Requirements

| Package | Version |
|---|---|
| `langgraph` | ≥ 1.2.0 |
| `langchain-core` | ≥ 1.4.0 |
| `langchain-openai` | ≥ 1.3.0 |
| `pydantic` | ≥ 2.0.0 |
| `python-dotenv` | ≥ 1.0.0 |
| `streamlit` | ≥ 1.58.0 |
| `pypdf` | ≥ 6.0.0 |

## Output

The agent produces:

- **Scorecards** — per-candidate criterion scores, strengths, gaps, and recommendation
- **Shortlist** — ranked list with hire/hold/reject decisions and interview focus areas
- **Interview proposals** — time slots for approved candidates (pending human confirmation)
- **Trajectory log** — full step-by-step reasoning trace for auditability
