"""
Resume Analyst Agent
--------------------
Uses GPT via OpenRouter to extract structured profile from resume text.
Validates output with Pydantic CandidateProfile.
Retries up to 2 times on validation failure.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from graph.state import RecruitmentState
from llm.openrouter import get_llm
from tools.validator import validate_profile
from utils.helpers import parse_json_from_llm
from utils.logger import log_event

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "analyst.txt")


def _load_prompt() -> str:
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def analyst_node(state: RecruitmentState) -> Dict[str, Any]:
    """LangGraph node: analyse each resume and build candidate profiles."""
    log_event("Analyst", "Resume Analyst agent started")

    llm = get_llm()
    prompt_template = _load_prompt()

    uploaded_files: List[Dict[str, Any]] = state.get("uploaded_files", [])
    logs: List[str] = list(state.get("logs", []))
    errors: List[str] = list(state.get("errors", []))
    profiles: List[Dict[str, Any]] = []

    for f in uploaded_files:
        filename = f.get("filename", "unknown")
        text = f.get("text", "")
        if not text.strip():
            errors.append(f"[Analyst] Empty text for {filename}")
            continue

        log_event("Analyst", f"Extracting profile from: {filename}")

        prompt = prompt_template.replace("{resume_text}", text[:8000])  # token safety
        profile_dict: Dict[str, Any] = {}
        validation_error = "Not attempted"

        for attempt in range(3):
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                raw = response.content
                profile_dict = parse_json_from_llm(raw)
                profile_dict["raw_text"] = text
                model, validation_error = validate_profile(profile_dict)
                if model:
                    profile_dict = model.model_dump(exclude={"raw_text"})
                    profile_dict["_filename"] = filename
                    log_event("Analyst", f"Profile extracted: {profile_dict.get('candidate_name', 'Unknown')}")
                    break
                else:
                    log_event("Analyst", f"Validation failed (attempt {attempt+1}): {validation_error}", "warning")
            except Exception as exc:
                validation_error = str(exc)
                log_event("Analyst", f"LLM error (attempt {attempt+1}): {exc}", "error")

        if not profile_dict.get("candidate_name"):
            errors.append(f"[Analyst] Could not extract profile from {filename}: {validation_error}")
        else:
            profiles.append(profile_dict)

    logs.append(f"Analyst: Extracted {len(profiles)} profile(s)")
    return {"profiles": profiles, "logs": logs, "errors": errors}
