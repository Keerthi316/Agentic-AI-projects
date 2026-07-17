"""
Scorer Agent
------------
Compares each candidate profile against the JD using GPT.
Produces a ScoreBreakdown per candidate.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from graph.state import RecruitmentState
from llm.openrouter import get_llm
from tools.validator import validate_scorecard
from utils.helpers import parse_json_from_llm
from utils.logger import log_event

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "scorer.txt")


def _load_prompt() -> str:
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def scorer_node(state: RecruitmentState) -> Dict[str, Any]:
    """LangGraph node: score each candidate against the JD."""
    log_event("Scorer", "Scorer agent started")

    llm = get_llm()
    prompt_template = _load_prompt()

    jd = state.get("job_description", "")
    profiles: List[Dict[str, Any]] = state.get("profiles", [])
    logs: List[str] = list(state.get("logs", []))
    errors: List[str] = list(state.get("errors", []))
    scorecards: List[Dict[str, Any]] = []

    # If re-scoring after verifier feedback, use existing profiles
    verified_scores = state.get("verified_scores", [])
    verifier_feedback = ""
    if verified_scores:
        latest_v = verified_scores[-1]
        if not latest_v.get("accepted", True):
            verifier_feedback = latest_v.get("feedback", "")

    for profile in profiles:
        name = profile.get("candidate_name", "Unknown")
        log_event("Scorer", f"Scoring candidate: {name}")

        profile_str = json.dumps(profile, indent=2)
        prompt = prompt_template.replace("{job_description}", jd).replace(
            "{candidate_profile}", profile_str
        )
        if verifier_feedback:
            prompt += f"\n\nVerifier feedback from previous attempt:\n{verifier_feedback}\nPlease adjust scores accordingly."

        scorecard: Dict[str, Any] = {}
        for attempt in range(3):
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                raw = response.content
                scorecard = parse_json_from_llm(raw)
                model, err = validate_scorecard(scorecard)
                if model:
                    scorecard = model.model_dump()
                    scorecard["_candidate_name"] = name
                    log_event("Scorer", f"Score for {name}: {scorecard.get('overall_score', '?')}/100")
                    break
                else:
                    log_event("Scorer", f"Validation failed (attempt {attempt+1}): {err}", "warning")
            except Exception as exc:
                log_event("Scorer", f"LLM error (attempt {attempt+1}): {exc}", "error")

        if scorecard:
            scorecards.append(scorecard)
        else:
            errors.append(f"[Scorer] Could not score {name}")

    logs.append(f"Scorer: Scored {len(scorecards)} candidate(s)")
    revision_count = state.get("revision_count", 0)
    if verifier_feedback:
        revision_count += 1

    return {"scorecards": scorecards, "revision_count": revision_count, "logs": logs, "errors": errors}
