"""
Verifier Agent
--------------
Triggered only when overall_score is in [60, 80].
Checks for bias, hallucinations, evidence quality, and consistency.
If issues found: returns candidate to scorer (up to MAX_REVISIONS).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from graph.state import RecruitmentState
from llm.openrouter import get_llm
from utils.helpers import parse_json_from_llm
from utils.logger import log_event

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "verifier.txt")


def _load_prompt() -> str:
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def verifier_node(state: RecruitmentState) -> Dict[str, Any]:
    """LangGraph node: verify scores for borderline candidates."""
    log_event("Verifier", "Verifier agent triggered")

    llm = get_llm()
    prompt_template = _load_prompt()

    jd = state.get("job_description", "")
    profiles: List[Dict[str, Any]] = state.get("profiles", [])
    scorecards: List[Dict[str, Any]] = state.get("scorecards", [])
    logs: List[str] = list(state.get("logs", []))
    errors: List[str] = list(state.get("errors", []))
    verified_scores: List[Dict[str, Any]] = []

    for i, scorecard in enumerate(scorecards):
        profile = profiles[i] if i < len(profiles) else {}
        name = scorecard.get("_candidate_name", "Unknown")
        score = scorecard.get("overall_score", 0)

        log_event("Verifier", f"Verifying score {score}/100 for {name}")

        prompt = (
            prompt_template
            .replace("{job_description}", jd)
            .replace("{candidate_profile}", json.dumps(profile, indent=2))
            .replace("{scorecard}", json.dumps(scorecard, indent=2))
        )

        verification: Dict[str, Any] = {}
        for attempt in range(2):
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                verification = parse_json_from_llm(response.content)
                # ensure booleans
                verification.setdefault("accepted", True)
                verification.setdefault("bias_detected", False)
                verification.setdefault("hallucination_detected", False)
                verification.setdefault("evidence_sufficient", True)
                verification.setdefault("consistent", True)
                verification.setdefault("feedback", "Score verified.")
                verification.setdefault("revised_score", None)
                break
            except Exception as exc:
                log_event("Verifier", f"LLM error (attempt {attempt+1}): {exc}", "error")

        if verification.get("accepted"):
            log_event("Verifier", f"Verifier ACCEPTED score for {name}")
        else:
            log_event(
                "Verifier",
                f"Verifier REJECTED score for {name}: {verification.get('feedback', '')}",
                "warning",
            )

        verified_scores.append(verification)

    logs.append(f"Verifier: Verified {len(verified_scores)} scorecard(s)")
    return {"verified_scores": verified_scores, "logs": logs, "errors": errors}
