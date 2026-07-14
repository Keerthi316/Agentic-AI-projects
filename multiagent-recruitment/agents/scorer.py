"""
Scorer Agent — second node in the recruitment workflow.

Responsibilities:
1. Score each parsed candidate profile against the job description.
2. Generate structured Scorecard objects with per-dimension scores.
3. Flag borderline candidates for verification.
4. Log all scoring decisions.

Design decisions:
- Scores are computed per-dimension (skills, experience, education) for transparency.
- The overall score is a weighted combination: Skills=40%, Experience=40%, Education=20%.
- Borderline candidates (score 50-75) are flagged via the is_borderline field for
  conditional routing to the Verifier.
- Each candidate is scored independently, enabling parallel execution via LangGraph's Send.
"""

from typing import Any, Dict, List

from models.state import CandidateProfile, RecruitmentState, Scorecard
from prompts.system_prompts import SCORER_PROMPT
from tools.llm import invoke_llm_structured
from tools.logging import get_agent_logger, log_agent_action

logger = get_agent_logger("Scorer")


def score_candidates(state: RecruitmentState) -> Dict[str, Any]:
    """Score all parsed candidates against the job description.

    Processes each CandidateProfile in state['parsed_profiles'] and generates
    a Scorecard. Borderline candidates are identified for potential verification.

    Args:
        state: The current RecruitmentState with 'parsed_profiles' and 'jd'.

    Returns:
        Dict with updates to 'scorecards', 'errors', and 'step_count'.
    """
    profiles = state.get("parsed_profiles", [])
    jd = state.get("jd")

    if not jd:
        err_msg = "Job description (jd) is missing from state. Cannot score candidates."
        log_agent_action(logger, "Scoring failed", {"error": err_msg}, level="ERROR")
        return {"errors": [err_msg], "step_count": state.get("step_count", 0) + 1}

    log_agent_action(logger, "Scoring candidates", {"count": len(profiles), "job_title": jd.title})

    scorecards: list[Scorecard] = []
    errors: list[str] = []

    # Format JD for prompt
    jd_str = _format_jd_for_prompt(jd)

    for profile in profiles:
        try:
            scorecard = _score_single_candidate(profile, jd, jd_str)
            if scorecard:
                scorecards.append(scorecard)
                log_agent_action(
                    logger,
                    "Candidate scored",
                    {
                        "candidate_id": profile.candidate_id,
                        "total_score": scorecard.total_score,
                        "is_borderline": scorecard.is_borderline,
                    },
                )
            else:
                err_msg = f"Failed to score candidate {profile.candidate_id}"
                errors.append(err_msg)
                log_agent_action(logger, "Score failed", {"candidate_id": profile.candidate_id}, level="WARNING")
        except Exception as e:
            err_msg = f"Error scoring candidate {profile.candidate_id}: {str(e)}"
            errors.append(err_msg)
            log_agent_action(
                logger, "Unexpected scoring error", {"candidate_id": profile.candidate_id, "error": str(e)}, level="ERROR"
            )

    return {
        "scorecards": scorecards,
        "errors": errors,
        "step_count": state.get("step_count", 0) + 1,
    }


def _score_single_candidate(
    profile: CandidateProfile, jd: Any, jd_str: str
) -> Scorecard | None:
    """Score a single candidate against the job description.

    Args:
        profile: The candidate's structured profile.
        jd: The JDInput object (for candidate_id matching).
        jd_str: Formatted JD string for the prompt.

    Returns:
        Scorecard instance or None on failure.
    """
    # Build profile string for prompt
    profile_str = _format_profile_for_prompt(profile)

    prompt = SCORER_PROMPT.format(jd=jd_str, profile=profile_str)
    scorecard = invoke_llm_structured(prompt, Scorecard)

    if scorecard:
        # Ensure candidate_id is set correctly
        scorecard.candidate_id = profile.candidate_id
        # Determine borderline status from score
        scorecard.is_borderline = 50.0 <= scorecard.total_score <= 75.0

    return scorecard


def _format_jd_for_prompt(jd: Any) -> str:
    """Format job description data into a readable string for prompts.

    Args:
        jd: JDInput object.

    Returns:
        Formatted string.
    """
    lines = [
        f"Title: {jd.title}",
        f"Description: {jd.description}",
        f"Required Skills: {', '.join(jd.required_skills) if jd.required_skills else 'Not specified'}",
        f"Preferred Skills: {', '.join(jd.preferred_skills) if jd.preferred_skills else 'Not specified'}",
        f"Min Experience: {jd.min_experience_years} years",
        f"Education Requirement: {jd.education_requirement or 'Not specified'}",
    ]
    return "\n".join(lines)


def _format_profile_for_prompt(profile: CandidateProfile) -> str:
    """Format a CandidateProfile into a readable string for prompts.

    Args:
        profile: The candidate profile.

    Returns:
        Formatted string with all profile fields.
    """
    lines = [
        f"Candidate ID: {profile.candidate_id}",
        f"Name: {profile.name}",
        f"Skills: {', '.join(profile.skills) if profile.skills else 'None listed'}",
        f"Education: {' | '.join(profile.education) if profile.education else 'None listed'}",
        f"Certifications: {', '.join(profile.certifications) if profile.certifications else 'None listed'}",
        "---",
        "Experience:",
    ]

    for exp in profile.experience:
        lines.append(
            f"  - Role: {exp.get('role', 'N/A')} at {exp.get('company', 'N/A')} "
            f"({exp.get('years', 0)} years): {exp.get('description', '')}"
        )

    lines.append("---")
    lines.append("Projects:")

    for proj in profile.projects:
        techs = ", ".join(proj.get("technologies", []))
        lines.append(f"  - {proj.get('name', 'N/A')}: {proj.get('description', '')} [Technologies: {techs}]")

    return "\n".join(lines)