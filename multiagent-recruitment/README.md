# Multi-Agent Recruitment System

A production-quality, multi-agent recruitment workflow built with **LangGraph**, **LangChain**, and **Pydantic**. This system transforms a single recruitment agent into a coordinated team of specialized AI agents that collaborate through a shared state.

## Architecture

```
                    ┌─────────────────┐
                    │   START         │
                    │  (JD + Resumes) │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Resume Analyst  │── Detects prompt injection
                    │                 │── Parses structured profiles
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Scorer       │── Scores candidates against JD
                    │                 │── Flags borderline candidates
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │  (conditional)              │
              ▼                              ▼
    ┌─────────────────┐           ┌─────────────────┐
    │   Verifier      │           │    Decider       │
    │ (blind re-score)│           │ (high-confidence)│
    │ (fairness check)│           └────────┬─────────┘
    └────────┬────────┘                    │
             │ (retry if unfair)           │
             └──────────────┬──────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │    Decider      │── Generates ranked shortlist
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Human Approval  │── Approval gate before scheduling
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   Scheduler     │── Generates interview invites
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │      END        │
                   └─────────────────┘
```

## Key Design Decisions

### 1. Shared State as Contract
The `RecruitmentState` TypedDict is the contract between all agents. Each field documents which agent reads/writes it. Fields with `Annotated[..., operator.add]` support parallel writes for concurrent processing.

### 2. Pydantic Validation at Handoff Boundaries
Every agent-to-agent handoff uses a Pydantic model (`CandidateProfile`, `Scorecard`, `VerifiedScore`, etc.). Invalid data is caught at the boundary before it enters agent logic.

### 3. Conditional Routing
- High-confidence candidates (score > 75) go directly to the Decider.
- Borderline candidates (score 50-75) go to the Verifier for blind re-scoring.
- Verification failure routes back to the Analyst or Scorer (max 3 retries).
- A step budget (50 steps) prevents infinite execution.

### 4. Prompt Injection Detection
The Resume Analyst runs a dedicated injection detection prompt on every resume before parsing. Detected injections are flagged with a confidence score and tracked through the entire workflow.

### 5. Blind Verification
The Verifier strips candidate identity (name, company names) before re-scoring, ensuring a fair, unbiased evaluation.

### 6. Human Approval Gate
The Scheduler only runs after explicit human approval (`human_approved == True`), giving a human-in-the-loop checkpoint before interview scheduling.

## Project Structure

```
multiagent-recruitment/
├── agents/
│   ├── __init__.py
│   ├── resume_analyst.py    # Parses resumes, detects injections
│   ├── scorer.py            # Scores candidates against JD
│   ├── verifier.py          # Blind re-score & fairness check
│   ├── decider.py           # Generates final ranked shortlist
│   └── scheduler.py         # Generates interview invitations
├── graph/
│   ├── __init__.py
│   └── workflow.py          # LangGraph workflow with routing
├── models/
│   ├── __init__.py
│   ├── config.py            # Settings (env vars, defaults)
│   └── state.py             # TypedDict shared state + all Pydantic models
├── prompts/
│   ├── __init__.py
│   └── system_prompts.py    # All LLM prompts (centralized)
├── tools/
│   ├── __init__.py
│   ├── llm.py               # LLM invocation utilities
│   └── logging.py           # Structured logging
├── tests/
│   ├── __init__.py
│   ├── test_models.py       # Pydantic model validation tests
│   └── test_agents.py       # Agent logic & router tests
├── sample_data/             # (optional) Sample resumes and JDs
├── main.py                  # Entry point with sample data
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd multiagent-recruitment

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your keys:
#   OPENAI_API_KEY      — recruitment workflow LLM (LangChain / OpenAI)
#   OPENROUTER_API_KEY  — DeepEval evaluation metrics (OpenRouter)
```

## Usage

### Run the demo workflow

```bash
python main.py
```

This will:
1. Load sample job description and 4 candidate resumes (including one with prompt injection).
2. Run the full LangGraph workflow.
3. Display results at each agent's stage.
4. Pause for human approval before scheduling.

### Run tests

```bash
pytest tests/ -v
```

### Integration tests (unit tests only)

```bash
pytest tests/test_models.py -v
pytest tests/test_agents.py -v
```

## Workflow State Fields

| Field | Type | Reducer | Written By | Read By |
|-------|------|---------|------------|---------|
| `jd` | `JDInput` | - | User input | Analyst, Scorer |
| `candidates` | `list[str]` | `operator.add` | User input | Analyst |
| `parsed_profiles` | `list[CandidateProfile]` | `operator.add` | Analyst | Scorer, Verifier, Decider |
| `scorecards` | `list[Scorecard]` | `operator.add` | Scorer | Verifier, Decider |
| `verified_scores` | `list[VerifiedScore]` | `operator.add` | Verifier | Decider |
| `revision_count` | `int` | - | Router | Router |
| `shortlist` | `list[ShortlistEntry]` | - | Decider | Scheduler, Human |
| `step_count` | `int` | - | All nodes | Router |
| `errors` | `list[str]` | `operator.add` | All nodes | Router, Human |
| `needs_human_escalation` | `bool` | - | Router | Human |
| `human_approved` | `bool` | - | Human | Router |

## Configuration

All settings are managed via `models/config.py` using `pydantic-settings`. Override with environment variables prefixed with `RECRUITMENT_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `RECRUITMENT_LLM_MODEL` | `gpt-4o-mini` | LLM model for agents |
| `RECRUITMENT_LLM_TEMPERATURE` | `0.1` | Low temp for consistent output |
| `RECRUITMENT_MAX_REVISION_COUNT` | `3` | Maximum retry iterations |
| `RECRUITMENT_MAX_STEP_BUDGET` | `50` | Maximum total execution steps |
| `RECRUITMENT_PASSING_SCORE` | `70.0` | Score threshold for passing |
| `RECRUITMENT_BORDERLINE_LOWER` | `50.0` | Lower bound for borderline |
| `RECRUITMENT_BORDERLINE_UPPER` | `75.0` | Upper bound for borderline |

### API Keys

| Variable | Required for | Description |
|----------|-------------|-------------|
| `OPENAI_API_KEY` | Recruitment workflow | LangChain / OpenAI — used by all agents |
| `OPENROUTER_API_KEY` | Evaluation framework | DeepEval metrics via OpenRouter — evaluation traffic never hits `api.openai.com` |
| `OPENROUTER_EVAL_MODEL` | Evaluation framework (optional) | OpenRouter model string for DeepEval (default: `openai/gpt-4o-mini`) |

## Extending the System

### Adding a new agent
1. Create `agents/new_agent.py` with the agent's function.
2. Add any new Pydantic models to `models/state.py`.
3. Add the agent's prompt to `prompts/system_prompts.py`.
4. Add the node and edges in `graph/workflow.py`.
5. Export from `agents/__init__.py`.

### Customizing prompts
Edit `prompts/system_prompts.py` to refine agent behavior. Each prompt instructs the LLM on role, task, guardrails, and output format.

### Using a different LLM
Change `RECRUITMENT_LLM_MODEL` in `.env` or `models/config.py`. The system uses `langchain-openai` by default but can be adapted for other providers.

## Error Handling & Reliability

- **Retry logic**: Failed agent nodes append to `state.errors`. The conditional router checks `revision_count` and re-routes to the appropriate agent.
- **Step budget**: Maximum 50 total execution steps across all nodes prevents infinite loops.
- **Graceful failure**: Missing required state fields (e.g., no `jd`, no `scorecards`) are caught and logged, not crashed.
- **Human escalation**: After max retries, the workflow routes to the Decider with `needs_human_escalation` flag set.