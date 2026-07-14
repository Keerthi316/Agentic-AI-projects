"""
LangGraph workflow definition for the Multi-Agent Recruitment System.

This is the core orchestrator that wires all agents together with:
1. Sequential execution (Analyst → Scorer)
2. Conditional routing (Borderline → Verifier, High-Confidence → Decider)
3. Retry loop (Verification failure → Analyst or Scorer)
4. Step budget enforcement (prevents infinite execution)
5. Human approval gate (before Scheduler)

Flow:
  START → ResumeAnalyst → Scorer → conditional_router
    ├──→ Verifier → conditional_router
    │     ├──→ retry (Analyst or Scorer) if verification fails + retries remain
    │     └──→ Decider if verification passes or retries exhausted
    └──→ Decider (high-confidence candidates go directly here)
          ↓
        HumanApproval (human_approved check)
          ↓
        Scheduler → END

Design decisions:
- StateGraph is used for stateful, cyclic graphs (supports retry loops).
- ALL nodes accept and return dict partial updates, merged by LangGraph.
- Conditional edges use a router function that inspects state.
- Step budget and revision_count are checked at every branching point.
"""

import logging
from typing import Any, Dict, Literal

from langgraph.graph import END, StateGraph

from agents.decider import generate_shortlist
from agents.resume_analyst import parse_resume
from agents.scheduler import schedule_interviews
from agents.scorer import score_candidates
from agents.verifier import verify_scores
from models.config import Settings
from models.state import RecruitmentState

logger = logging.getLogger(__name__)
settings = Settings()


# ---------------------------------------------------------------------------
# Conditional Router Functions
# ---------------------------------------------------------------------------


def route_after_scorer(state: RecruitmentState) -> Literal["verifier", "decider", "end"]:
    """Route candidates based on scoring results.

    - If there are borderline scorecards → route to Verifier.
    - If all scorecards are high-confidence (none borderline) → route to Decider.
    - If no scorecards exist at all → end the workflow with error.

    Also checks the step budget and revision count before routing.
    """
    step_count = state.get("step_count", 0)
    revision_count = state.get("revision_count", 0)
    scorecards = state.get("scorecards", [])
    errors = state.get("errors", [])

    logger.info(
        f"route_after_scorer: step={step_count}, revision={revision_count}, "
        f"scorecards={len(scorecards)}, errors={len(errors)}"
    )

    # Step budget check
    if step_count >= settings.max_step_budget:
        logger.warning("Step budget exceeded. Ending workflow.")
        return "end"

    # Revision count check
    if revision_count >= settings.max_revision_count:
        logger.warning(f"Max revisions ({settings.max_revision_count}) reached. Routing to decider.")
        return "decider"

    if not scorecards:
        logger.warning("No scorecards generated. Ending workflow.")
        return "end"

    # Check if any candidates are borderline
    has_borderline = any(sc.is_borderline for sc in scorecards)
    if has_borderline:
        logger.info("Borderline candidates detected. Routing to Verifier.")
        return "verifier"

    logger.info("All candidates high-confidence. Routing directly to Decider.")
    return "decider"


def route_after_verifier(state: RecruitmentState) -> Literal["resume_analyst", "scorer", "decider", "end"]:
    """Route after verification based on results.

    - If verification found unfair scores (injection_affected or not is_fair):
      → Route back to Analyst or Scorer depending on the issue.
    - If verification passed → Route to Decider.
    - If retries exhausted → Escalate (route to Decider with error flag).

    Also checks the step budget and revision count.
    """
    step_count = state.get("step_count", 0)
    revision_count = state.get("revision_count", 0)
    verified_scores = state.get("verified_scores", [])
    errors = state.get("errors", [])

    logger.info(
        f"route_after_verifier: step={step_count}, revision={revision_count}, "
        f"verified_scores={len(verified_scores)}, errors={len(errors)}"
    )

    # Step budget check
    if step_count >= settings.max_step_budget:
        logger.warning("Step budget exceeded. Ending workflow.")
        return "end"

    # Check if verification found issues
    has_unfair = any(not vs.is_fair for vs in verified_scores)
    has_injection_affected = any(vs.injection_affected for vs in verified_scores)

    if has_unfair or has_injection_affected:
        if revision_count < settings.max_revision_count:
            logger.info(
                f"Verification found issues (unfair={has_unfair}, "
                f"injection_affected={has_injection_affected}). "
                f"Retrying (revision {revision_count + 1}/{settings.max_revision_count})."
            )
            # Route back to the appropriate agent
            if has_injection_affected:
                return "resume_analyst"  # Re-parse for injection issues
            else:
                return "scorer"  # Re-score for fairness issues
        else:
            logger.warning(
                f"Max revisions ({settings.max_revision_count}) reached with unresolved issues. "
                "Routing to Decider for final decision."
            )
            return "decider"

    logger.info("Verification passed. Routing to Decider.")
    return "decider"


def route_after_decider(state: RecruitmentState) -> Literal["human_approval", "end"]:
    """Route after the Decider has generated the shortlist.

    - If shortlist was generated → wait for human approval.
    - If errors prevented shortlist generation → end.
    """
    shortlist = state.get("shortlist", [])
    errors = state.get("errors", [])

    if shortlist:
        logger.info(f"Shortlist generated with {len(shortlist)} candidates. Awaiting human approval.")
        return "human_approval"

    logger.warning("No shortlist generated. Ending workflow.")
    return "end"


def route_after_human_approval(state: RecruitmentState) -> Literal["scheduler", "end"]:
    """Route after human approval check.

    - If approved → route to Scheduler.
    - If not approved → end.
    """
    if state.get("human_approved", False):
        logger.info("Human approved. Routing to Scheduler.")
        return "scheduler"

    logger.info("Human did not approve. Ending workflow.")
    return "end"


# ---------------------------------------------------------------------------
# Helper Nodes (for intermediate steps like human approval gate)
# ---------------------------------------------------------------------------


def human_approval_gate(state: RecruitmentState) -> Dict[str, Any]:
    """Human approval gate node.

    This node just logs the current state and returns without changes.
    The actual approval is set externally (e.g., via API or CLI input).
    The conditional edge after this node checks human_approved.

    Args:
        state: Current RecruitmentState.

    Returns:
        Empty dict (no state changes), just increments step_count.
    """
    shortlist = state.get("shortlist", [])
    logger.info(
        "=== HUMAN APPROVAL GATE ==="
        f"\n  Shortlist has {len(shortlist)} candidates."
        "\n  Set 'human_approved' to True to proceed to scheduling."
        "\n  ==========================="
    )
    return {"step_count": state.get("step_count", 0) + 1}


def check_step_budget(state: RecruitmentState) -> Dict[str, Any]:
    """Check if the step budget has been exceeded and set escalation flag.

    Args:
        state: Current RecruitmentState.

    Returns:
        Dict with needs_human_escalation update if budget exceeded.
    """
    if state.get("step_count", 0) >= settings.max_step_budget:
        logger.error(f"Step budget ({settings.max_step_budget}) exceeded. Escalating to human.")
        return {
            "needs_human_escalation": True,
            "errors": [f"Step budget exceeded at step {state.get('step_count', 0)}"],
        }
    return {}


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------


def build_recruitment_graph() -> StateGraph:
    """Build the complete recruitment workflow graph.

    Returns:
        A compiled StateGraph ready for invocation.
    """
    # Initialize the state graph with our RecruitmentState schema
    workflow = StateGraph(RecruitmentState)

    # ── Add Nodes ───────────────────────────────────────────────────────
    workflow.add_node("resume_analyst", parse_resume)
    workflow.add_node("scorer", score_candidates)
    workflow.add_node("verifier", verify_scores)
    workflow.add_node("decider", generate_shortlist)
    workflow.add_node("human_approval_gate", human_approval_gate)
    workflow.add_node("scheduler", schedule_interviews)

    # ── Set Entry Point ─────────────────────────────────────────────────
    workflow.set_entry_point("resume_analyst")

    # ── Add Edges ───────────────────────────────────────────────────────
    # Sequential: Analyst → Scorer
    workflow.add_edge("resume_analyst", "scorer")

    # Conditional: Scorer → Verifier or Decider
    workflow.add_conditional_edges(
        "scorer",
        route_after_scorer,
        {
            "verifier": "verifier",
            "decider": "decider",
            "end": END,
        },
    )

    # Conditional: Verifier → retry, Decider, or end
    workflow.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "resume_analyst": "resume_analyst",
            "scorer": "scorer",
            "decider": "decider",
            "end": END,
        },
    )

    # Decider → Human Approval Gate
    workflow.add_edge("decider", "human_approval_gate")

    # Conditional: Human Approval → Scheduler or end
    workflow.add_conditional_edges(
        "human_approval_gate",
        route_after_human_approval,
        {
            "scheduler": "scheduler",
            "end": END,
        },
    )

    # Scheduler → END
    workflow.add_edge("scheduler", END)

    logger.info("Recruitment graph built successfully.")
    return workflow.compile()