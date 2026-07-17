"""
Decider Agent
-------------
Makes the final hiring recommendation based on profile + scorecard + verification.
Returns: Interview | Hold | Reject | Need Human Review
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from graph.state import RecruitmentState
from llm.openrouter import get_llm
from tools.validator import validate_decision
from utils.helpers import parse_json_from_llm
from utils.logger import log_event

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "decider.txt")


def _load_prompt() -> str:
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def decider_node(state: RecruitmentState) -> Dict[str, Any]:
    """LangGraph node: produce final decisions."""
    log_event("Decider", "Decider agent started")

    llm = get_llm()
    prompt_template = _load_prompt()

    jd = state.get("job_description", "")
    profiles: List[Dict[str, Any]] = state.get("profiles", [])
    scorecards: List[Dict[str, Any]] = state.get("scorecards", [])
    verified_scores: List[Dict[str, Any]] = state.get("verified_scores", [])
    logs: List[str] = list(state.get("logs", []))
    errors: List[str] = list(state.get("errors", []))
    shortlist: List[Dict[str, Any]] = []

    needs_human_approval = False

    for i, scorecard in enumerate(scorecards):
        profile = profiles[i] if i < len(profiles) else {}
        verified = verified_scores[i] if i < len(verified_scores) else {"accepted": True, "feedback": "N/A"}
        name = scorecard.get("_candidate_name", "Unknown")

        log_event("Decider", f"Deciding for candidate: {name}")

        verification_status = json.dumps(verified, indent=2)
        prompt = (
            prompt_template
            .replace("{job_description}", jd)
            .replace("{candidate_profile}", json.dumps(profile, indent=2))
            .replace("{scorecard}", json.dumps(scorecard, indent=2))
            .replace("{verification_status}", verification_status)
        )

        decision: Dict[str, Any] = {}
        for attempt in range(3):
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                raw = response.content
                decision = parse_json_from_llm(raw)
                model, err = validate_decision(decision)
                if model:
                    decision = model.model_dump()
                    decision["_candidate_name"] = name
                    decision["_filename"] = profile.get("_filename", "")
                    decision["_overall_score"] = scorecard.get("overall_score", 0)
                    decision["_scorecard"] = scorecard
                    decision["_profile"] = profile
                    log_event("Decider", f"Decision for {name}: {decision.get('recommendation')}")
                    if decision.get("recommendation") in ("Interview", "Reject"):
                        needs_human_approval = True
                    break
                else:
                    log_event("Decider", f"Validation failed (attempt {attempt+1}): {err}", "warning")
            except Exception as exc:
                log_event("Decider", f"LLM error (attempt {attempt+1}): {exc}", "error")

        if decision:
            shortlist.append(decision)
        else:
            errors.append(f"[Decider] Could not decide for {name}")

    logs.append(f"Decider: Made {len(shortlist)} decision(s)")
    return {
        "shortlist": shortlist,
        "needs_human_approval": needs_human_approval,
        "logs": logs,
        "errors": errors,
    }
