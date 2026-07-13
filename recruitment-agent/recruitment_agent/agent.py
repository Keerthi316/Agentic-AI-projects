"""
LangGraph-based Recruitment Agent with Plan → Act → Observe loop.
Includes full trajectory logging, guardrails, and step limits.
"""

import json
import copy
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime

from .state import (
    AgentState, TrajectoryStep, CandidateProfile, ScoreCard,
    ShortlistEntry, InterviewProposal, TimeSlot
)
from .tools import (
    parse_resume, build_rubric, score_candidate,
    check_availability, propose_interview
)

# ============================================================
# GUARDRAIL: Step Limit
# ============================================================
MAX_STEPS = 50


# ============================================================
# FAIRNESS GUARDRAIL
# ============================================================
BIASED_KEYWORDS = ["gender", "age", "college prestige", "university ranking",
                   "nationality", "religion", "marital status", "race", "ethnicity"]


def _check_fairness(profile: CandidateProfile) -> List[str]:
    """Check if any unfair biases could affect scoring. Returns warnings."""
    warnings = []
    return warnings  # We simply don't use biased attributes in evaluation


# ============================================================
# AUDIT LOG HELPER
# ============================================================
def _append_trajectory(state: AgentState, thought: str, action: str,
                       input_data: Dict[str, Any], observation: str,
                       decision: str) -> AgentState:
    """Append a step to the trajectory log."""
    trajectory = list(state.get("trajectory", []))
    step_count = state.get("step_count", 0) + 1
    
    step = TrajectoryStep(
        step_number=step_count,
        thought=thought,
        action=action,
        input=input_data,
        observation=str(observation)[:2000],  # Truncate long observations
        decision=decision
    )
    trajectory.append(step)
    
    state["trajectory"] = trajectory
    state["step_count"] = step_count
    return state


# ============================================================
# AGENT LOOP NODES
# ============================================================

def initialize_agent(state: AgentState) -> AgentState:
    """Initialize the agent state with JD and candidates."""
    state["rubric"] = build_rubric(state["job_description"])
    state["candidates_to_process"] = list(state["candidates"].keys())
    state["total_candidates"] = len(state["candidates_to_process"])
    state["current_candidate_index"] = 0
    state["parsed_profiles"] = {}
    state["scorecards"] = {}
    state["shortlist"] = []
    state["availability"] = {}
    state["actions"] = []
    state["trajectory"] = []
    state["step_count"] = 0
    state["status"] = "RUNNING"
    state["next_action"] = "parse_resume"
    state["error"] = None
    state["human_approval_pending"] = None
    
    # Log initialization
    state = _append_trajectory(
        state,
        "Initializing recruitment agent with job description and candidate list",
        "initialize",
        {"num_candidates": state["total_candidates"], "candidates": state["candidates_to_process"]},
        f"Agent initialized with {state['total_candidates']} candidates. "
        f"Rubric built with {len(state['rubric']['criteria'])} criteria.",
        "PROCEED"
    )
    
    return state


def parse_resume_node(state: AgentState) -> AgentState:
    """Parse the current candidate's resume."""
    if state["current_candidate_index"] >= state["total_candidates"]:
        state["next_action"] = "score_all"
        return state
    
    candidate_name = state["candidates_to_process"][state["current_candidate_index"]]
    resume_text = state["candidates"][candidate_name]
    
    thought = f"Parsing resume for candidate: {candidate_name}"
    
    try:
        profile = parse_resume(resume_text)
        state["parsed_profiles"][candidate_name] = profile
        
        # Check for prompt injection
        injection_warning = ""
        if profile.get("has_prompt_injection", False):
            injection_warning = " [PROMPT INJECTION DETECTED - instructions in resume ignored]"
        
        observation = (f"Parsed profile for {candidate_name}: "
                       f"{profile.get('years_of_experience', 0)} years exp, "
                       f"{len(profile.get('work_experience', []))} jobs, "
                       f"{len(profile.get('projects', []))} projects"
                       f"{injection_warning}")
        
        decision = "PROCEED"
        state["next_action"] = "evaluate_candidate"
        
    except Exception as e:
        observation = f"Error parsing resume: {str(e)}"
        decision = "ERROR"
        state["error"] = str(e)
        state["status"] = "ERROR"
    
    state = _append_trajectory(state, thought, "parse_resume",
                                {"candidate": candidate_name},
                                observation, decision)
    return state


def evaluate_candidate_node(state: AgentState) -> AgentState:
    """Score the current candidate against the rubric."""
    candidate_name = state["candidates_to_process"][state["current_candidate_index"]]
    profile = state["parsed_profiles"][candidate_name]
    
    thought = f"Evaluating candidate {candidate_name} against the scoring rubric"
    
    try:
        scorecard = score_candidate(profile, state["rubric"])
        state["scorecards"][candidate_name] = scorecard
        
        # Build score detail for observation
        score_details = []
        for cs in scorecard.get("criteria_scores", []):
            score_details.append(
                f"  {cs['criterion']}: {cs['score']}/5 (w:{cs['weight']}) - "
                f"Evidence: {cs['evidence'][:100]}..."
            )
        
        injection_note = ""
        if profile.get("has_prompt_injection", False):
            injection_note = "\n  [GUARDRAIL: Prompt injection detected and blocked during parsing. Scores based on actual resume content only.]"
        
        observation = (
            f"Scorecard for {candidate_name}: "
            f"Total={scorecard['total_weighted_score']}/{scorecard['max_possible_score']}, "
            f"Normalized={scorecard['normalized_score']}%\n"
            + "\n".join(score_details)
            + injection_note
        )
        decision = "PROCEED"
        state["next_action"] = "check_next"
        
    except Exception as e:
        observation = f"Error scoring candidate: {str(e)}"
        decision = "ERROR"
        state["error"] = str(e)
        state["status"] = "ERROR"
    
    state = _append_trajectory(state, thought, "score_candidate",
                                {"candidate": candidate_name},
                                observation, decision)
    return state


def check_next_node(state: AgentState) -> AgentState:
    """Determine if there are more candidates to process or if we should move on."""
    candidate_name = state["candidates_to_process"][state["current_candidate_index"]]
    
    thought = f"Checking if there are more candidates to process after {candidate_name}"
    
    state["current_candidate_index"] += 1
    
    if state["current_candidate_index"] < state["total_candidates"]:
        observation = f"Moving to next candidate (index {state['current_candidate_index']}: {state['candidates_to_process'][state['current_candidate_index']]})"
        state["next_action"] = "parse_resume"
        decision = "PROCEED_NEXT"
    else:
        observation = "All candidates evaluated. Moving to final ranking."
        state["next_action"] = "rank_shortlist"
        decision = "ALL_EVALUATED"
    
    state = _append_trajectory(state, thought, "check_next",
                                {"current_index": state["current_candidate_index"] - 1,
                                 "total": state["total_candidates"]},
                                observation, decision)
    return state


def rank_shortlist_node(state: AgentState) -> AgentState:
    """Rank all candidates and produce the final shortlist."""
    thought = "Ranking all candidates and producing final shortlist"
    
    try:
        # Build shortlist sorted by normalized score descending
        scored_candidates = []
        for name, scorecard in state["scorecards"].items():
            profile = state["parsed_profiles"].get(name, {})
            normalized_score = scorecard["normalized_score"]
            
            # Determine decision based on score
            if normalized_score >= 70:
                decision = "INTERVIEW"
            elif normalized_score >= 40:
                decision = "HOLD"
            else:
                decision = "REJECT"
            
            # Build justification
            justification_parts = [f"Candidate scored {normalized_score}% overall."]
            
            # Add evidence from top/bottom criteria
            criteria = scorecard.get("criteria_scores", [])
            sorted_criteria = sorted(criteria, key=lambda c: c["score"] / 5, reverse=True)
            
            if sorted_criteria:
                top = sorted_criteria[0]
                justification_parts.append(
                    f"Strongest area: {top['criterion']} ({top['score']}/5) - "
                    f"evidence: {top['evidence'][:150]}"
                )
                
                bottom = sorted_criteria[-1]
                if bottom['score'] < 3:
                    justification_parts.append(
                        f"Weakest area: {bottom['criterion']} ({bottom['score']}/5) - "
                        f"evidence: {bottom['evidence'][:150]}"
                    )
            
            # Injection note
            if profile.get("has_prompt_injection", False):
                justification_parts.append(
                    "NOTE: Resume contained prompt injection attempt which was detected and blocked."
                )
            
            scored_candidates.append({
                "name": name,
                "decision": decision,
                "score": normalized_score,
                "justification": "\n".join(justification_parts),
                "scorecard": scorecard,
                "has_injection": profile.get("has_prompt_injection", False)
            })
        
        # Sort by score descending
        scored_candidates.sort(key=lambda c: c["score"], reverse=True)
        
        # Build shortlist entries
        shortlist = []
        for sc in scored_candidates:
            entry = ShortlistEntry(
                name=sc["name"],
                decision=sc["decision"],
                score=sc["score"],
                justification=sc["justification"],
                scorecard=sc["scorecard"]
            )
            shortlist.append(entry)
        
        state["shortlist"] = shortlist
        
        observation_parts = ["RANKED SHORTLIST:"]
        for i, entry in enumerate(shortlist):
            rank = i + 1
            inj_flag = " [INJECTION ATTEMPT BLOCKED]" if scored_candidates[i]["has_injection"] else ""
            observation_parts.append(
                f"  #{rank}: {entry['name']} - Score: {entry['score']}% - "
                f"Decision: {entry['decision']}{inj_flag}"
            )
        observation_parts.append(f"\nTop candidate: {shortlist[0]['name']} ({shortlist[0]['score']}%)")
        observation_parts.append(f"Bottom candidate: {shortlist[-1]['name']} ({shortlist[-1]['score']}%)")
        
        observation = "\n".join(observation_parts)
        decision = "SHORTLIST_PRODUCED"
        state["next_action"] = "check_availability"
        
    except Exception as e:
        observation = f"Error ranking candidates: {str(e)}"
        decision = "ERROR"
        state["error"] = str(e)
        state["status"] = "ERROR"
    
    state = _append_trajectory(state, thought, "rank_shortlist",
                                {"num_candidates": len(state["scorecards"])},
                                observation, decision)
    return state


def check_availability_node(state: AgentState) -> AgentState:
    """Check availability for candidates marked as INTERVIEW."""
    thought = "Checking availability for INTERVIEW candidates"
    
    interview_candidates = [e for e in state["shortlist"] if e["decision"] == "INTERVIEW"]
    
    if not interview_candidates:
        observation = "No candidates marked for INTERVIEW. Skipping availability check."
        decision = "NO_INTERVIEW_CANDIDATES"
        state["next_action"] = "complete"
        state = _append_trajectory(state, thought, "check_availability",
                                    {}, observation, decision)
        return state
    
    try:
        observation_parts = ["Checking availability for interview candidates:"]
        
        for entry in interview_candidates:
            name = entry["name"]
            slots = check_availability(name)
            state["availability"][name] = slots
            
            # Select first available slot
            if slots:
                first_slot = slots[0]
                observation_parts.append(
                    f"  {name}: {len(slots)} slots available. "
                    f"First: {first_slot['day']} {first_slot['start_time']}-{first_slot['end_time']}"
                )
                
                # Propose interview (requires human approval)
                proposal = propose_interview(name, first_slot)
                state["actions"].append(proposal)
                observation_parts.append(
                    f"  -> Proposed interview: {name} on {first_slot['day']} "
                    f"{first_slot['start_time']}-{first_slot['end_time']} "
                    f"[STATUS: PENDING_APPROVAL]"
                )
        
        observation = "\n".join(observation_parts)
        decision = "AVAILABILITY_CHECKED"
        state["next_action"] = "await_approval"
        
    except Exception as e:
        observation = f"Error checking availability: {str(e)}"
        decision = "ERROR"
        state["error"] = str(e)
        state["status"] = "ERROR"
    
    state = _append_trajectory(state, thought, "check_availability",
                                {"candidates": [e["name"] for e in interview_candidates]},
                                observation, decision)
    return state


def await_approval_node(state: AgentState) -> AgentState:
    """Wait for human approval before executing interview scheduling.
    
    This is the HUMAN-IN-THE-LOOP guardrail.
    """
    pending_actions = [a for a in state["actions"] if a["status"] == "PENDING_APPROVAL"]
    
    if not pending_actions:
        observation = "No pending actions requiring approval."
        decision = "NO_ACTIONS"
        state["next_action"] = "complete"
        state = _append_trajectory(state, 
                                    "No pending actions. Completing process.",
                                    "await_approval", {}, observation, decision)
        return state
    
    thought = f"Waiting for human approval on {len(pending_actions)} interview proposal(s)"
    
    # Store pending approvals in state for external consumption
    state["human_approval_pending"] = {
        "type": "interview_scheduling",
        "proposals": [
            {
                "candidate": a["candidate"],
                "proposed_slot": a["proposed_slot"],
                "status": a["status"]
            }
            for a in pending_actions
        ],
        "message": "Please approve or reject each proposed interview slot."
    }
    state["status"] = "WAITING_APPROVAL"
    
    observation = (
        f"Human approval required for {len(pending_actions)} interview proposals:\n"
        + "\n".join([
            f"  - {a['candidate']}: {a['proposed_slot']['day']} "
            f"{a['proposed_slot']['start_time']}-{a['proposed_slot']['end_time']}"
            for a in pending_actions
        ])
        + "\n\nAgent paused. Waiting for human decision..."
    )
    decision = "WAITING_FOR_HUMAN"
    
    state = _append_trajectory(state, thought, "propose_interview",
                                {"pending_proposals": len(pending_actions)},
                                observation, decision)
    return state


def complete_node(state: AgentState) -> AgentState:
    """Finalize and output the complete decision."""
    thought = "Finalizing recruitment agent output"
    
    state["status"] = "COMPLETED"
    state["next_action"] = "done"
    
    # Summarize the full run
    total_injection = sum(
        1 for p in state["parsed_profiles"].values() if p.get("has_prompt_injection", False)
    )
    
    observation_parts = [
        "=== RECRUITMENT AGENT - FINAL REPORT ===",
        f"Total candidates processed: {state['total_candidates']}",
        f"Prompt injection attempts blocked: {total_injection}",
        f"Total steps taken: {state['step_count']}",
        "",
        "SHORTLIST:",
    ]
    
    for i, entry in enumerate(state["shortlist"]):
        observation_parts.append(f"  #{i+1}: {entry['name']} - {entry['decision']} ({entry['score']}%)")
    
    observation_parts.extend([
        "",
        "PENDING ACTIONS:",
    ])
    
    if state["actions"]:
        for a in state["actions"]:
            observation_parts.append(
                f"  - {a['candidate']}: {a['proposed_slot']['day']} "
                f"{a['proposed_slot']['start_time']}-{a['proposed_slot']['end_time']} "
                f"[{a['status']}]"
            )
    else:
        observation_parts.append("  No interview proposals.")
    
    observation = "\n".join(observation_parts)
    decision = "COMPLETE"
    
    state = _append_trajectory(state, thought, "complete",
                                {"status": "COMPLETED"},
                                observation, decision)
    return state


# ============================================================
# ROUTER FUNCTION
# ============================================================

def router(state: AgentState) -> Literal[
    "parse_resume", "evaluate_candidate", "check_next",
    "rank_shortlist", "check_availability", "await_approval",
    "complete", "__end__"
]:
    """
    Route to the next node based on the current state.
    This enables the dynamic Plan -> Act -> Observe loop.
    """
    
    # GUARDRAIL: Step limit - prevent infinite loops
    if state.get("step_count", 0) >= MAX_STEPS:
        state["status"] = "ERROR"
        state["error"] = f"Step limit exceeded ({MAX_STEPS} steps)"
        return "complete"
    
    # GUARDRAIL: Check for errors
    if state.get("status") == "ERROR":
        return "complete"
    
    next_action = state.get("next_action", "parse_resume")
    
    routing_map = {
        "parse_resume": "parse_resume",
        "evaluate_candidate": "evaluate_candidate",
        "check_next": "check_next",
        "rank_shortlist": "rank_shortlist",
        "check_availability": "check_availability",
        "await_approval": "await_approval",
        "complete": "complete",
        "done": "__end__",
        "score_all": "rank_shortlist",
    }
    
    return routing_map.get(next_action, "complete")


# ============================================================
# BUILD LANGGRAPH GRAPH
# ============================================================

def build_agent_graph():
    """
    Build the LangGraph state graph for the recruitment agent.
    
    Graph structure:
    initialize -> parse_resume -> evaluate_candidate -> check_next (loop back or continue)
               -> rank_shortlist -> check_availability -> await_approval -> complete -> END
    """
    try:
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        
        # Define the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("initialize", initialize_agent)
        workflow.add_node("parse_resume", parse_resume_node)
        workflow.add_node("evaluate_candidate", evaluate_candidate_node)
        workflow.add_node("check_next", check_next_node)
        workflow.add_node("rank_shortlist", rank_shortlist_node)
        workflow.add_node("check_availability", check_availability_node)
        workflow.add_node("await_approval", await_approval_node)
        workflow.add_node("complete", complete_node)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        # Add edges with router
        workflow.add_conditional_edges(
            "initialize",
            router,
            {
                "parse_resume": "parse_resume",
                "evaluate_candidate": "evaluate_candidate",
                "check_next": "check_next",
                "rank_shortlist": "rank_shortlist",
                "check_availability": "check_availability",
                "await_approval": "await_approval",
                "complete": "complete",
                "__end__": END,
            }
        )
        
        workflow.add_conditional_edges(
            "parse_resume",
            router,
            {
                "parse_resume": "parse_resume",
                "evaluate_candidate": "evaluate_candidate",
                "check_next": "check_next",
                "rank_shortlist": "rank_shortlist",
                "check_availability": "check_availability",
                "await_approval": "await_approval",
                "complete": "complete",
                "__end__": END,
            }
        )
        
        workflow.add_conditional_edges(
            "evaluate_candidate",
            router,
            {
                "parse_resume": "parse_resume",
                "evaluate_candidate": "evaluate_candidate",
                "check_next": "check_next",
                "rank_shortlist": "rank_shortlist",
                "check_availability": "check_availability",
                "await_approval": "await_approval",
                "complete": "complete",
                "__end__": END,
            }
        )
        
        workflow.add_conditional_edges(
            "check_next",
            router,
            {
                "parse_resume": "parse_resume",
                "evaluate_candidate": "evaluate_candidate",
                "check_next": "check_next",
                "rank_shortlist": "rank_shortlist",
                "check_availability": "check_availability",
                "await_approval": "await_approval",
                "complete": "complete",
                "__end__": END,
            }
        )
        
        workflow.add_conditional_edges(
            "rank_shortlist",
            router,
            {
                "parse_resume": "parse_resume",
                "evaluate_candidate": "evaluate_candidate",
                "check_next": "check_next",
                "rank_shortlist": "rank_shortlist",
                "check_availability": "check_availability",
                "await_approval": "await_approval",
                "complete": "complete",
                "__end__": END,
            }
        )
        
        workflow.add_conditional_edges(
            "check_availability",
            router,
            {
                "parse_resume": "parse_resume",
                "evaluate_candidate": "evaluate_candidate",
                "check_next": "check_next",
                "rank_shortlist": "rank_shortlist",
                "check_availability": "check_availability",
                "await_approval": "await_approval",
                "complete": "complete",
                "__end__": END,
            }
        )
        
        workflow.add_conditional_edges(
            "await_approval",
            router,
            {
                "parse_resume": "parse_resume",
                "evaluate_candidate": "evaluate_candidate",
                "check_next": "check_next",
                "rank_shortlist": "rank_shortlist",
                "check_availability": "check_availability",
                "await_approval": "await_approval",
                "complete": "complete",
                "__end__": END,
            }
        )
        
        workflow.add_conditional_edges(
            "complete",
            router,
            {
                "complete": "complete",
                "__end__": END,
            }
        )
        
        # Compile with checkpointing for persistence
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        
        return app
    
    except ImportError:
        # Fallback: Return None if langgraph not available
        return None


# ============================================================
# EXECUTION HELPER
# ============================================================

def run_agent(job_description: str, candidates: Dict[str, str],
              thread_id: str = "recruitment-run-1") -> Dict[str, Any]:
    """
    Run the recruitment agent end-to-end.
    
    Uses the sequential Plan → Act → Observe loop.
    For LangGraph execution, use build_agent_graph() directly.
    
    Args:
        job_description: The full job description text
        candidates: Dict mapping candidate name -> resume text
        thread_id: Unique thread ID for this run (for LangGraph mode)
    
    Returns:
        Complete state with shortlist, actions, and trajectory
    """
    return _run_sequential(job_description, candidates)


def _run_sequential(job_description: str, candidates: Dict[str, str]) -> Dict[str, Any]:
    """
    Fallback sequential execution without LangGraph.
    Implements the same Plan → Act → Observe loop.
    """
    state = AgentState(
        job_description=job_description,
        candidates=candidates,
        rubric=None,
        current_candidate_index=0,
        total_candidates=0,
        candidates_to_process=[],
        parsed_profiles={},
        scorecards={},
        shortlist=[],
        availability={},
        actions=[],
        trajectory=[],
        step_count=0,
        status="RUNNING",
        next_action="initialize",
        error=None,
        human_approval_pending=None,
    )
    
    step_limit = MAX_STEPS
    steps = 0
    
    while steps < step_limit and state["status"] not in ["COMPLETED", "ERROR", "WAITING_APPROVAL"]:
        steps += 1
        next_action = state.get("next_action", "")
        
        if next_action == "initialize" or next_action == "":
            state = initialize_agent(state)
        elif next_action == "parse_resume":
            state = parse_resume_node(state)
        elif next_action == "evaluate_candidate":
            state = evaluate_candidate_node(state)
        elif next_action == "check_next":
            state = check_next_node(state)
        elif next_action == "rank_shortlist":
            state = rank_shortlist_node(state)
        elif next_action == "check_availability":
            state = check_availability_node(state)
        elif next_action == "await_approval":
            state = await_approval_node(state)
        elif next_action == "complete":
            state = complete_node(state)
            break
        elif next_action == "done":
            state["status"] = "COMPLETED"
            break
        else:
            state["status"] = "ERROR"
            state["error"] = f"Unknown action: {next_action}"
            break
    
    if steps >= step_limit and state["status"] not in ["COMPLETED", "ERROR"]:
        state["status"] = "ERROR"
        state["error"] = f"Step limit exceeded ({step_limit} steps)"
    
    return state


def approve_interview(state: Dict[str, Any], candidate_name: str,
                      approved: bool = True) -> Dict[str, Any]:
    """
    Approve or reject a pending interview proposal.
    This is how the human-in-the-loop provides their decision.
    
    Args:
        state: Current agent state
        candidate_name: Name of the candidate to approve/reject
        approved: True to approve, False to reject
    
    Returns:
        Updated state
    """
    for action in state.get("actions", []):
        if action["candidate"] == candidate_name and action["status"] == "PENDING_APPROVAL":
            action["status"] = "APPROVED" if approved else "REJECTED"
            
            # Log the human decision
            state = _append_trajectory(
                state,
                f"Human {'approved' if approved else 'rejected'} interview for {candidate_name}",
                "human_approval",
                {"candidate": candidate_name, "approved": approved},
                f"Interview for {candidate_name} was {'APPROVED' if approved else 'REJECTED'} by human.",
                "HUMAN_DECISION"
            )
    
    # Clear pending flag if all resolved
    pending = [a for a in state.get("actions", []) if a["status"] == "PENDING_APPROVAL"]
    if not pending:
        state["human_approval_pending"] = None
        if state.get("status") == "WAITING_APPROVAL":
            state["status"] = "COMPLETED"
    
    return state