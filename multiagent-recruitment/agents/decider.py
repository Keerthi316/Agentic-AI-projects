"""
Decider Agent — generates the final ranked shortlist.

Responsibilities:
1. Merge original scorecards with verified scores for borderline candidates.
2. Compute final scores (blind_score if verification found issues, average otherwise).
3. Rank candidates by final score (highest first).
4. Assign status: shortlisted, hold, or rejected.
5. Generate the final ShortlistEntry list.

Design decisions:
- For candidates with verified scores: if |original - blind| > 10, use blind_score.
  Otherwise, use the average of original and blind scores.
- For candidates without verified scores: use the original total_score directly.
- The shortlist is sorted by rank (1 = best) for the human approval step.
"""

from typing import Any, Dict, List

from models.state import RecruitmentState, ShortlistEntry, VerifiedScore
from tools.logging import get_agent_logger, log_agent_action

logger = get_agent_logger("Decider")


def generate_shortlist(state: RecruitmentState) -> Dict[str, Any]:
    """Generate the final ranked shortlist from scorecards and verified scores.

    Args:
        state: The current RecruitmentState with 'scorecards', 'verified_scores',
               'parsed_profiles', and 'jd'.

    Returns:
        Dict with updates to 'shortlist', 'errors', and 'step_count'.
    """
    scorecards = state.get("scorecards", [])
    verified_scores = state.get("verified_scores", [])
    profiles = state.get("parsed_profiles", [])
    jd = state.get("jd")

    if not scorecards:
        err_msg = "No scorecards found in state. Cannot generate shortlist."
        log_agent_action(logger, "Shortlist failed", {"error": err_msg}, level="ERROR")
        return {"errors": [err_msg], "step_count": state.get("step_count", 0) + 1}

    log_agent_action(
        logger,
        "Generating shortlist",
        {"scorecards": len(scorecards), "verified_scores": len(verified_scores)},
    )

    # Build lookup maps
    profile_map: Dict[str, Any] = {p.candidate_id: p for p in profiles}
    verified_map: Dict[str, VerifiedScore] = {v.candidate_id: v for v in verified_scores}

    # Compute final scores
    candidates_with_scores: List[Dict[str, Any]] = []
    errors: list[str] = []

    for sc in scorecards:
        candidate_id = sc.candidate_id
        profile = profile_map.get(candidate_id)

        if not profile:
            err_msg = f"Profile not found for candidate {candidate_id}. Skipping."
            errors.append(err_msg)
            log_agent_action(logger, "Profile missing", {"candidate_id": candidate_id}, level="WARNING")
            continue

        # Determine final score
        if candidate_id in verified_map:
            verified = verified_map[candidate_id]
            if abs(verified.original_score - verified.blind_score) > 10.0:
                # Verification found significant difference — use blind score
                final_score = verified.blind_score
                log_agent_action(
                    logger,
                    "Using blind score",
                    {"candidate_id": candidate_id, "original": verified.original_score, "blind": verified.blind_score},
                )
            else:
                # Scores are close — use average
                final_score = (verified.original_score + verified.blind_score) / 2.0
                log_agent_action(
                    logger,
                    "Using averaged score",
                    {"candidate_id": candidate_id, "average": final_score},
                )
        else:
            # High-confidence candidate — use original score directly
            final_score = sc.total_score
            log_agent_action(
                logger,
                "Using original score",
                {"candidate_id": candidate_id, "score": final_score},
            )

        candidates_with_scores.append({
            "candidate_id": candidate_id,
            "name": profile.name,
            "final_score": round(final_score, 1),
        })

    # Sort by final_score descending
    candidates_with_scores.sort(key=lambda x: x["final_score"], reverse=True)

    # Assign ranks and statuses
    shortlist: list[ShortlistEntry] = []
    max_shortlist = max(1, len(candidates_with_scores) // 2)  # Top 50% shortlisted

    for rank, candidate in enumerate(candidates_with_scores, start=1):
        if rank <= max_shortlist:
            status = "shortlisted"
        elif candidate["final_score"] >= 50.0:
            status = "hold"
        else:
            status = "rejected"

        entry = ShortlistEntry(
            candidate_id=candidate["candidate_id"],
            name=candidate["name"],
            final_score=candidate["final_score"],
            rank=rank,
            status=status,
        )
        shortlist.append(entry)

    log_agent_action(
        logger,
        "Shortlist generated",
        {
            "total": len(shortlist),
            "shortlisted": sum(1 for s in shortlist if s.status == "shortlisted"),
            "hold": sum(1 for s in shortlist if s.status == "hold"),
            "rejected": sum(1 for s in shortlist if s.status == "rejected"),
        },
    )

    return {
        "shortlist": shortlist,
        "errors": errors,
        "step_count": state.get("step_count", 0) + 1,
    }