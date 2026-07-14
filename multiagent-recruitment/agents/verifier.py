"""
Verifier Agent — runs only for borderline candidates.

Responsibilities:
1. Perform a blind re-score by stripping candidate identity from the profile.
2. Compare blind score against the original score.
3. Check for prompt injection effects on scoring.
4. Determine if the score is "fair" (within tolerance).

Design decisions:
- Identity stripping: Removes name, candidate_id, and any identifying information
  from the profile before re-scoring, enforcing blind evaluation.
- Fairness tolerance: A difference of >10 points between original and blind scores
  triggers a fairness concern.
- Injection check: The verifier also looks for signs that prompt injection may have
  influenced the original scoring process.
- This agent ONLY runs for borderline candidates (is_borderline == True).
"""

from typing import Any, Dict

from models.state import CandidateProfile, RecruitmentState, Scorecard, VerifiedScore
from prompts.system_prompts import VERIFIER_PROMPT
from tools.llm import invoke_llm_structured
from tools.logging import get_agent_logger, log_agent_action

logger = get_agent_logger("Verifier")


def verify_scores(state: RecruitmentState) -> Dict[str, Any]:
    """Verify scores for borderline candidates via blind re-scoring.

    Args:
        state: The current RecruitmentState with 'parsed_profiles', 'scorecards', and 'jd'.

    Returns:
        Dict with updates to 'verified_scores', 'errors', 'step_count'.
    """
    profiles = state.get("parsed_profiles", [])
    scorecards = state.get("scorecards", [])
    jd = state.get("jd")

    if not jd:
        err_msg = "Job description (jd) is missing from state. Cannot verify scores."
        log_agent_action(logger, "Verification failed", {"error": err_msg}, level="ERROR")
        return {"errors": [err_msg], "step_count": state.get("step_count", 0) + 1}

    # Build lookup maps
    profile_map: Dict[str, CandidateProfile] = {p.candidate_id: p for p in profiles}
    borderline_scorecards = [s for s in scorecards if s.is_borderline]

    log_agent_action(
        logger,
        "Starting verification",
        {"borderline_count": len(borderline_scorecards), "total_candidates": len(scorecards)},
    )

    verified_scores: list[VerifiedScore] = []
    errors: list[str] = []

    # Format JD for prompt
    jd_str = _format_jd_for_prompt(jd)

    for scorecard in borderline_scorecards:
        try:
            profile = profile_map.get(scorecard.candidate_id)
            if not profile:
                err_msg = f"Candidate profile not found for {scorecard.candidate_id}"
                errors.append(err_msg)
                log_agent_action(logger, "Profile not found", {"candidate_id": scorecard.candidate_id}, level="WARNING")
                continue

            verified = _verify_single_candidate(profile, scorecard, jd_str)
            if verified:
                verified_scores.append(verified)
                log_agent_action(
                    logger,
                    "Verification complete",
                    {
                        "candidate_id": scorecard.candidate_id,
                        "original_score": verified.original_score,
                        "blind_score": verified.blind_score,
                        "is_fair": verified.is_fair,
                    },
                )
            else:
                err_msg = f"Verification failed for candidate {scorecard.candidate_id}"
                errors.append(err_msg)
                log_agent_action(
                    logger, "Verification failed", {"candidate_id": scorecard.candidate_id}, level="WARNING"
                )
        except Exception as e:
            err_msg = f"Error verifying candidate {scorecard.candidate_id}: {str(e)}"
            errors.append(err_msg)
            log_agent_action(
                logger, "Unexpected verification error", {"candidate_id": scorecard.candidate_id, "error": str(e)},
                level="ERROR",
            )

    return {
        "verified_scores": verified_scores,
        "errors": errors,
        "step_count": state.get("step_count", 0) + 1,
    }


def _verify_single_candidate(
    profile: CandidateProfile, scorecard: Scorecard, jd_str: str
) -> VerifiedScore | None:
    """Perform blind re-scoring for a single borderline candidate.

    Strips identity from the profile and re-scores against the JD.

    Args:
        profile: The original candidate profile (full).
        scorecard: The original scorecard.
        jd_str: Formatted job description string.

    Returns:
        VerifiedScore instance or None on failure.
    """
    # Create a blind profile — remove name and candidate_id
    blind_profile = _create_blind_profile(profile)

    prompt = VERIFIER_PROMPT.format(
        candidate_id=scorecard.candidate_id,
        original_score=scorecard.total_score,
        skill_score=scorecard.skill_score,
        experience_score=scorecard.experience_score,
        education_score=scorecard.education_score,
        reasoning=scorecard.reasoning,
        is_borderline=scorecard.is_borderline,
        blind_profile=blind_profile,
        jd=jd_str,
    )

    verified = invoke_llm_structured(prompt, VerifiedScore)

    if verified:
        # Ensure candidate_id matches and compute score difference
        verified.candidate_id = scorecard.candidate_id
        verified.original_score = scorecard.total_score
        verified.score_difference = abs(verified.original_score - verified.blind_score)

    return verified


def _create_blind_profile(profile: CandidateProfile) -> str:
    """Create a blind profile string by removing candidate identity.

    Strips: candidate_id, name, raw_text, injection fields.
    Retains: skills, education, experience, projects, certifications.

    Args:
        profile: The full CandidateProfile.

    Returns:
        A string representation of the profile without identity markers.
    """
    lines = [
        f"Skills: {', '.join(profile.skills) if profile.skills else 'None listed'}",
        f"Education: {' | '.join(profile.education) if profile.education else 'None listed'}",
        f"Certifications: {', '.join(profile.certifications) if profile.certifications else 'None listed'}",
        "---",
        "Experience:",
    ]

    for exp in profile.experience:
        # Remove company name to reduce identifiability
        lines.append(
            f"  - Role: {exp.get('role', 'N/A')} "
            f"({exp.get('years', 0)} years): {exp.get('description', '')}"
        )

    lines.append("---")
    lines.append("Projects:")

    for proj in profile.projects:
        techs = ", ".join(proj.get("technologies", []))
        lines.append(f"  - {proj.get('name', 'N/A')}: {proj.get('description', '')} [Technologies: {techs}]")

    return "\n".join(lines)


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