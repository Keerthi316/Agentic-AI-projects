"""
Coordinator Agent
-----------------
Responsibilities:
- Receive JD and uploaded resume files
- Parse resume text from bytes
- Initialize shared graph state
- Dispatch tasks downstream

No LLM calls — purely deterministic.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List

from graph.state import RecruitmentState
from tools.parser import extract_text
from utils.logger import log_event


def coordinator_node(state: RecruitmentState) -> Dict[str, Any]:
    """LangGraph node function for the Coordinator agent."""
    log_event("Coordinator", "Coordinator initialized workflow")

    run_id = state.get("run_id") or str(uuid.uuid4())
    logs: List[str] = list(state.get("logs", []))
    errors: List[str] = list(state.get("errors", []))

    uploaded_files: List[Dict[str, Any]] = list(state.get("uploaded_files", []))

    # Parse text from each uploaded file if not already parsed
    parsed_files: List[Dict[str, Any]] = []
    for f in uploaded_files:
        if f.get("text"):
            parsed_files.append(f)
            continue
        text, err = extract_text(f["bytes"], f["filename"])
        if err:
            errors.append(f"[Coordinator] Failed to parse {f['filename']}: {err}")
            log_event("Coordinator", f"Parse error for {f['filename']}: {err}", "error")
        else:
            # Check if sanitizer flagged injection (text will contain [REDACTED] markers)
            injection_detected = "[REDACTED]" in text
            if injection_detected:
                errors.append(
                    f"⚠️ SECURITY: Prompt injection attempt detected in '{f['filename']}'. "
                    f"Malicious instructions have been removed. Candidate will be scored on actual content only."
                )
                log_event("Coordinator", f"[SECURITY] Prompt injection stripped from {f['filename']}", "warning")
            log_event("Coordinator", f"Parsed resume: {f['filename']} ({len(text)} chars)")
            parsed_files.append({**f, "text": text, "injection_detected": injection_detected})

    logs.append("Coordinator: Workflow initialized, resumes parsed")

    return {
        "run_id": run_id,
        "uploaded_files": parsed_files,
        "profiles": [],
        "scorecards": [],
        "verified_scores": [],
        "shortlist": [],
        "revision_count": 0,
        "current_candidate_index": 0,
        "needs_verification": False,
        "needs_human_approval": False,
        "human_approved": None,
        "logs": logs,
        "errors": errors,
    }
