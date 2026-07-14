"""
Resume Analyst Agent — first node in the recruitment workflow.

Responsibilities:
1. Parse raw resume text into structured CandidateProfile objects.
2. Detect prompt injection attempts in resume text.
3. Validate parsed profiles against the CandidateProfile Pydantic model.
4. Add parsed profiles to shared state.

Design decisions:
- Uses invoke_llm_structured for automatic Pydantic validation of LLM output.
- Separate injection detection step before parsing to catch malicious resumes early.
- Failed parsing results in an error appended to state.errors rather than crashing.
- Graceful fallback: if LLM returns invalid data, we still log and retry.
"""

import json
import logging
from typing import Any, Dict

from langgraph.types import Send

from models.state import CandidateProfile, RecruitmentState
from prompts.system_prompts import RESUME_ANALYST_PROMPT, INJECTION_DETECTION_PROMPT
from tools.llm import invoke_llm, invoke_llm_structured
from tools.logging import get_agent_logger, log_agent_action

logger = get_agent_logger("ResumeAnalyst")


def parse_resume(state: RecruitmentState) -> Dict[str, Any]:
    """Parse all candidate resumes from the shared state.

    This function is called by the LangGraph node. It processes each
    candidate's raw text, detects injections, extracts structured profiles,
    and appends them to the state.

    Args:
        state: The current RecruitmentState with 'candidates' and 'jd'.

    Returns:
        Dict with updates to 'parsed_profiles', 'errors', and 'step_count'.
    """
    log_agent_action(logger, "Parsing resumes", {"count": len(state.get("candidates", []))})

    parsed_profiles: list[CandidateProfile] = []
    errors: list[str] = []

    for i, candidate_text in enumerate(state.get("candidates", [])):
        candidate_id = f"candidate_{i}"
        log_agent_action(logger, "Processing candidate", {"candidate_id": candidate_id})

        try:
            # Step 1: Detect prompt injection
            injection_check = _detect_injection(candidate_text)
            if injection_check and injection_check.get("is_injected", False):
                log_agent_action(
                    logger,
                    "Injection detected",
                    {
                        "candidate_id": candidate_id,
                        "confidence": injection_check.get("confidence", 0.0),
                    },
                    level="WARNING",
                )

            # Step 2: Parse resume into structured profile
            profile = _extract_profile(candidate_text, candidate_id, injection_check)

            if profile:
                # Step 3: Validate the parsed profile
                if _validate_profile(profile):
                    parsed_profiles.append(profile)
                    log_agent_action(
                        logger,
                        "Profile parsed successfully",
                        {"candidate_id": profile.candidate_id, "name": profile.name},
                    )
                else:
                    err_msg = f"Validation failed for candidate {candidate_id}: missing required fields"
                    errors.append(err_msg)
                    log_agent_action(logger, "Validation failed", {"candidate_id": candidate_id}, level="WARNING")
            else:
                err_msg = f"Failed to parse resume for candidate {candidate_id}"
                errors.append(err_msg)
                log_agent_action(logger, "Parse failed", {"candidate_id": candidate_id}, level="ERROR")

        except Exception as e:
            err_msg = f"Error processing candidate {candidate_id}: {str(e)}"
            errors.append(err_msg)
            log_agent_action(logger, "Unexpected error", {"candidate_id": candidate_id, "error": str(e)}, level="ERROR")

    return {
        "parsed_profiles": parsed_profiles,
        "errors": errors,
        "step_count": state.get("step_count", 0) + 1,
    }


def _detect_injection(text: str) -> Dict[str, Any] | None:
    """Check resume text for prompt injection attempts.

    Args:
        text: Raw resume text.

    Returns:
        Dict with 'is_injected', 'confidence', 'detection_reasoning' or None on failure.
    """
    prompt = INJECTION_DETECTION_PROMPT.format(text=text[:3000])  # Limit to 3000 chars
    try:
        content = invoke_llm(prompt)
        # Parse as plain JSON since injection check doesn't need Pydantic
        data = json.loads(content)
        log_agent_action(
            logger,
            "Injection check complete",
            {"is_injected": data.get("is_injected", False), "confidence": data.get("confidence", 0.0)},
        )
        return data
    except Exception as e:
        log_agent_action(logger, "Injection detection failed", {"error": str(e)}, level="ERROR")
        return {"is_injected": False, "confidence": 0.0, "detection_reasoning": "Detection failed"}


def _extract_profile(
    text: str, candidate_id: str, injection_check: Dict[str, Any] | None
) -> CandidateProfile | None:
    """Extract a structured CandidateProfile from resume text using the LLM.

    Args:
        text: Raw resume text.
        candidate_id: Fallback ID if LLM doesn't provide one.
        injection_check: Result from injection detection (may be None).

    Returns:
        CandidateProfile instance or None if parsing fails.
    """
    prompt = RESUME_ANALYST_PROMPT + f"\n\nRESUME TEXT:\n{text}"
    profile = invoke_llm_structured(prompt, CandidateProfile)

    if profile:
        # Override injection fields from the dedicated detector
        if injection_check:
            profile.is_injection_detected = injection_check.get("is_injected", False)
            profile.injection_confidence = injection_check.get("confidence", 0.0)
        profile.raw_text = text

        # If the LLM didn't assign a candidate_id, use our fallback
        if not profile.candidate_id:
            profile.candidate_id = candidate_id

    return profile


def _validate_profile(profile: CandidateProfile) -> bool:
    """Validate that a parsed profile has the minimum required fields.

    Required: candidate_id, name
    If these are missing, the profile is deemed invalid and should trigger
    the retry loop.

    Args:
        profile: The CandidateProfile to validate.

    Returns:
        True if valid, False otherwise.
    """
    if not profile.candidate_id or not profile.name:
        return False
    # candidate_id must be a non-empty string
    if not isinstance(profile.candidate_id, str) or len(profile.candidate_id.strip()) == 0:
        return False
    if not isinstance(profile.name, str) or len(profile.name.strip()) == 0:
        return False
    return True