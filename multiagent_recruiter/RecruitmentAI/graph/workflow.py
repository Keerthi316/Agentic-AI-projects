"""
LangGraph Workflow
------------------
Assembles the full multi-agent recruitment pipeline.

START → Coordinator → Analyst → Scorer
  ↓ (60-80)               ↓ (else)
Verifier               Decider → END
  ↓ (rejected, <3)
Scorer (re-run)
  ↓ (accepted or ≥3)
Decider → END
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.coordinator import coordinator_node
from agents.analyst import analyst_node
from agents.scorer import scorer_node
from agents.verifier import verifier_node
from agents.decider import decider_node
from graph.router import route_after_scorer, route_after_verifier
from graph.state import RecruitmentState


def build_workflow() -> StateGraph:
    """Build and compile the recruitment workflow graph."""
    builder = StateGraph(RecruitmentState)

    # ── nodes ──────────────────────────────────────────────────────────────
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("scorer", scorer_node)
    builder.add_node("verifier", verifier_node)
    builder.add_node("decider", decider_node)

    # ── edges ──────────────────────────────────────────────────────────────
    builder.set_entry_point("coordinator")
    builder.add_edge("coordinator", "analyst")
    builder.add_edge("analyst", "scorer")

    # Conditional: after scorer → verifier or decider
    builder.add_conditional_edges(
        "scorer",
        route_after_scorer,
        {"verifier": "verifier", "decider": "decider"},
    )

    # Conditional: after verifier → scorer (retry) or decider
    builder.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"scorer": "scorer", "decider": "decider"},
    )

    builder.add_edge("decider", END)

    return builder.compile()


# Singleton compiled graph
_graph = None


def get_workflow():
    global _graph
    if _graph is None:
        _graph = build_workflow()
    return _graph


def run_workflow(job_description: str, uploaded_files: list) -> RecruitmentState:
    """Run the full recruitment workflow and return final state."""
    from utils.logger import clear_run_log
    clear_run_log()

    graph = get_workflow()
    initial_state: RecruitmentState = {
        "job_description": job_description,
        "uploaded_files": uploaded_files,
        "profiles": [],
        "scorecards": [],
        "verified_scores": [],
        "shortlist": [],
        "revision_count": 0,
        "current_candidate_index": 0,
        "needs_verification": False,
        "needs_human_approval": False,
        "human_approved": None,
        "run_id": None,
        "logs": [],
        "errors": [],
    }
    final_state = graph.invoke(initial_state)
    return final_state
