# Prompt Doctor — Architecture

## Overview

Prompt Doctor is a Streamlit application that teaches prompt engineering through
interactive, AI-graded exercises. Users write prompts for progressively harder
levels, and an AI examiner evaluates their prompt against level-specific
principles.

## File Responsibilities

```
prompt_doctor/
├── app.py          # Streamlit entry point — UI, session state, navigation
├── levels.py       # Level definitions — tasks, sample inputs, principles
├── runner.py       # Executes user prompt against sample input via OpenRouter
├── examiner.py     # Evaluates user prompt via OpenRouter, returns structured JSON
└── .env            # OPENROUTER_API_KEY
```

## Data Flow

```
User writes prompt in Streamlit
        │
        ▼
runner.py ──► OpenRouter (user prompt + sample input) ──► raw output
        │
        ▼
examiner.py ──► OpenRouter (examiner system prompt + user prompt + level) ──► JSON
        │
        ▼
Streamlit displays grading
        │
        ▼
All principles pass? ──► Unlock next level
        │
        No ──► Show weaknesses + guiding questions
```

## Level Design

| Level | Principles | Focus |
|-------|-----------|-------|
| 1 | Role, Clear instruction | Basic prompt structure |
| 2 | Structured output (JSON/schema) | Format control |
| 3 | Few-shot examples | In-context learning |
| 4 | Reasoning for multi-step tasks | Chain-of-thought |
| 5 | Defensive constraints | Input validation, edge cases |

## Examiner Design

The examiner uses a strict system prompt that:

1. **Establishes identity**: "You are a strict but fair prompt engineering examiner."
2. **Scopes evaluation**: Judges ONLY the principles for the current level.
3. **Forbids rewriting**: Never improves the student's prompt.
4. **Requires specificity**: Quotes exact weak phrases or identifies what's missing.
5. **Enforces one question per failure**: Exactly one guiding question per failed principle.
6. **Enforces JSON output**: Returns ONLY valid JSON matching the schema.

## JSON Schema (examiner output)

```json
{
  "level": 1,
  "principles": [
    {
      "name": "Role",
      "pass": false,
      "weakness": "No role assigned to the AI.",
      "question": "What persona should the AI adopt to best handle this task?"
    }
  ],
  "ran_ok": true,
  "verdict": "revise"
}
```

## Error Handling Strategy

- **API failures**: Retry once, then return `ran_ok: false` with descriptive error.
- **JSON parse failures**: Try direct parse → regex extraction → fallback default.
- **Schema validation**: Check required keys, types, and value ranges.
- **Streamlit display**: Show errors inline without crashing the app.