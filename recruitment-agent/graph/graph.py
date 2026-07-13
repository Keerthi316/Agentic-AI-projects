"""
LangGraph graph builder for the Recruitment Agent.
Defines the Plan → Act → Observe state graph.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.state import AgentState
from graph.nodes import (
    initialize_node, plan_node, parse_resume_node, score_node,
    finalize_node, check_availability_node, await_approval_node, complete_node
)

MAX_STEPS = 50


def router(state: AgentState) -> str:
    """Route to the next node based on current state."""
    
    # Guardrail: Step limit
    if state.step_count >= state.max_steps:
        state.status = "ERROR"
        state.error = f"Step limit exceeded ({state.max_steps} steps)"
        return "complete"
    
    # Guardrail: Error check
    if state.status == "ERROR":
        return "complete"
    
    next_action = state.next_action
    
    routing = {
        "initialize": "initialize",
        "plan": "plan",
        "parse_resume": "parse_resume",
        "score": "score",
        "finalize": "finalize",
        "check_availability": "check_availability",
        "await_approval": "await_approval",
        "complete": "complete",
        "done": "__end__",
    }
    
    return routing.get(next_action, "complete")


def build_graph():
    """Build the LangGraph state graph."""
    from langgraph.graph import StateGraph, END
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("parse_resume", parse_resume_node)
    workflow.add_node("score", score_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("check_availability", check_availability_node)
    workflow.add_node("await_approval", await_approval_node)
    workflow.add_node("complete", complete_node)
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    # Add conditional edges from each node
    for node in ["initialize", "plan", "parse_resume", "score", "finalize",
                 "check_availability", "await_approval", "complete"]:
        workflow.add_conditional_edges(
            node,
            router,
            {
                "initialize": "initialize",
                "plan": "plan",
                "parse_resume": "parse_resume",
                "score": "score",
                "finalize": "finalize",
                "check_availability": "check_availability",
                "await_approval": "await_approval",
                "complete": "complete",
                "__end__": END,
            }
        )
    
    app = workflow.compile()
    return app


def run_agent(job_description: str, candidates: dict) -> AgentState:
    """Run the recruitment agent end-to-end using sequential execution."""
    # Use the sequential runner since LangGraph's invoke() may return dicts
    # that don't cleanly convert back to AgentState with nested types
    return run_sequential(job_description, candidates)


def run_sequential(job_description: str, candidates: dict) -> AgentState:
    """Run the agent sequentially (fallback if LangGraph fails)."""
    state = AgentState(
        job_description=job_description,
        candidates=candidates,
    )
    
    steps = 0
    while steps < MAX_STEPS and state.status not in ["COMPLETED", "ERROR", "WAITING_APPROVAL"]:
        steps += 1
        next_action = state.next_action
        
        if next_action == "initialize":
            state = initialize_node(state)
        elif next_action == "plan":
            state = plan_node(state)
        elif next_action == "parse_resume":
            state = parse_resume_node(state)
        elif next_action == "score":
            state = score_node(state)
        elif next_action == "finalize":
            state = finalize_node(state)
        elif next_action == "check_availability":
            state = check_availability_node(state)
        elif next_action == "await_approval":
            state = await_approval_node(state)
        elif next_action == "complete":
            state = complete_node(state)
            break
        elif next_action == "done":
            state.status = "COMPLETED"
            break
        else:
            state.status = "ERROR"
            state.error = f"Unknown action: {next_action}"
            break
    
    if steps >= MAX_STEPS and state.status not in ["COMPLETED", "ERROR"]:
        state.status = "ERROR"
        state.error = f"Step limit exceeded ({MAX_STEPS} steps)"
    
    return state