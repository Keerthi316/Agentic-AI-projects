"""
Deterministic routing functions for the LangGraph workflow.
No LLM calls here — pure logic based on state values.
"""
from __future__ import annotations

from graph.state import RecruitmentState
from config import MAX_REVISIONS


def route_after_scorer(state: RecruitmentState) -> str:
    """
    After scoring, decide whether to run the Verifier.
    Score in [60, 80] → verifier
    Otherwise → decider
    """
    scorecards = state.get("scorecards", [])
    if not scorecards:
        return "decider"
    latest = scorecards[-1]
    score = latest.get("overall_score", 0)
    if 60.0 <= score <= 80.0:
        return "verifier"
    return "decider"


def route_after_verifier(state: RecruitmentState) -> str:
    """
    After verification, decide whether to re-score or proceed to decider.
    If verifier rejected and revisions < MAX_REVISIONS → scorer (re-run)
    Otherwise → decider
    """
    verified = state.get("verified_scores", [])
    revisions = state.get("revision_count", 0)
    if not verified:
        return "decider"
    latest = verified[-1]
    accepted = latest.get("accepted", True)
    if not accepted and revisions < MAX_REVISIONS:
        return "scorer"
    return "decider"


def route_after_decider(state: RecruitmentState) -> str:
    """
    After decider, check if human approval is needed.
    Interview or Reject decisions require human approval.
    """
    shortlist = state.get("shortlist", [])
    if not shortlist:
        return "end"
    latest = shortlist[-1]
    rec = latest.get("recommendation", "")
    if rec in ("Interview", "Reject"):
        return "human_approval"
    return "end"
