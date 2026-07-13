"""
Graph nodes for the LangGraph recruitment agent.
Each node implements a step in the Plan → Act → Observe loop.
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from graph.state import AgentState
from prompts.jd_prompt import JD_ANALYZER_PROMPT
from prompts.planner_prompt import PLANNER_PROMPT
from prompts.decision_prompt import DECISION_PROMPT
from prompts.guardrail_prompt import GUARDRAIL_PROMPT
from tools.parse_resume import parse_resume
from tools.score_candidate import score_candidate
from tools.availability import check_availability
from tools.interview import propose_interview


def get_llm():
    """Get the LLM configured via environment variables."""
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model = os.getenv("MODEL", "openai/gpt-4o-mini")
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.1
    )


def _append_trajectory(state: AgentState, thought: str, tool: str,
                       arguments: dict, observation: str, decision: str) -> AgentState:
    """Append a step to the trajectory log."""
    state.step_count += 1
    state.trajectory.append({
        "step_number": state.step_count,
        "thought": thought,
        "tool": tool,
        "arguments": arguments,
        "observation": str(observation)[:2000],
        "state_changes": {"next_action": state.next_action, "status": state.status},
        "decision": decision
    })
    return state


def initialize_node(state: AgentState) -> AgentState:
    """Initialize the agent by analyzing the JD and building the rubric."""
    llm = get_llm()
    
    # Analyze JD
    jd_prompt = JD_ANALYZER_PROMPT.format(jd=state.job_description)
    response = llm.invoke([HumanMessage(content=jd_prompt)])
    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("\n", 1)[0]
        if content.endswith("```"):
            content = content[:-3]
    
    state.job_requirements = json.loads(content)
    
    # Build rubric from JD requirements
    req = state.job_requirements
    criteria = []
    weights = [25, 20, 15, 15, 10, 10, 5]  # default weights
    
    for i, skill in enumerate(req.get("required_skills", [])[:5]):
        w = weights[i] if i < len(weights) else 10
        criteria.append({
            "name": skill,
            "weight": w,
            "description": f"Required: {skill}",
            "levels": {0: "No evidence", 1: "Basic", 2: "Some", 3: "Good", 4: "Strong", 5: "Expert"}
        })
    
    for i, skill in enumerate(req.get("preferred_skills", [])[:3]):
        w = 5
        criteria.append({
            "name": f"{skill} (Bonus)",
            "weight": w,
            "description": f"Preferred: {skill}",
            "levels": {0: "No evidence", 1: "Basic", 2: "Some", 3: "Good", 4: "Strong", 5: "Expert"}
        })
    
    # Add experience and education criteria
    if req.get("minimum_experience"):
        criteria.append({
            "name": "Relevant Experience",
            "weight": 15,
            "description": f"Minimum {req['minimum_experience']}",
            "levels": {0: "None", 1: "Minimal", 2: "Some", 3: "Good", 4: "Strong", 5: "Exceptional"}
        })
    
    if req.get("minimum_education"):
        criteria.append({
            "name": "Education",
            "weight": 10,
            "description": f"Minimum {req['minimum_education']}",
            "levels": {0: "None", 1: "Some", 2: "Relevant", 3: "Meets", 4: "Exceeds", 5: "Advanced"}
        })
    
    # Normalize weights
    total = sum(c["weight"] for c in criteria)
    if total > 0:
        for c in criteria:
            c["weight"] = round(c["weight"] / total * 100, 1)
    
    state.rubric = {"criteria": criteria}
    
    # Set up candidate processing order
    state.candidates_to_process = list(state.candidates.keys())
    state.current_candidate_index = 0
    
    state = _append_trajectory(
        state,
        "Analyzing job description and building scoring rubric",
        "initialize",
        {"num_candidates": len(state.candidates_to_process)},
        f"Job: {req.get('job_title', 'Unknown')}. "
        f"Required skills: {req.get('required_skills', [])}. "
        f"Preferred skills: {req.get('preferred_skills', [])}. "
        f"Rubric built with {len(criteria)} criteria.",
        "PROCEED"
    )
    
    state.next_action = "plan"
    return state


def plan_node(state: AgentState) -> AgentState:
    """Use the planner LLM to decide the next action."""
    llm = get_llm()
    
    # Build summary of current state
    jd_summary = json.dumps(state.job_requirements, indent=2)[:500] if state.job_requirements else "Not analyzed"
    rubric_summary = json.dumps(state.rubric, indent=2)[:500] if state.rubric else "Not built"
    
    remaining = state.candidates_to_process[state.current_candidate_index:]
    remaining_str = str(remaining) if remaining else "None"
    processed = list(state.parsed_profiles.keys())
    processed_str = str(processed) if processed else "None"
    
    shortlist_str = "Not yet ranked"
    if state.shortlist:
        shortlist_str = str([(e["name"], e["decision"]) for e in state.shortlist])
    
    planner_input = PLANNER_PROMPT.format(
        jd_summary=jd_summary,
        rubric_summary=rubric_summary,
        remaining_candidates=remaining_str,
        processed_candidates=processed_str,
        shortlist_summary=shortlist_str
    )
    
    response = llm.invoke([HumanMessage(content=planner_input)])
    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("\n", 1)[0]
        if content.endswith("```"):
            content = content[:-3]
    
    decision = json.loads(content)
    thought = decision.get("thought", "Planning next step")
    next_tool = decision.get("next_tool", "complete")
    reason = decision.get("reason", "")
    
    # Map tool names to internal action names
    tool_to_action = {
        "parse_resume": "parse_resume",
        "score_candidate": "score",
        "check_availability": "check_availability",
        "propose_interview": "await_approval",
        "complete": "complete",
        "finalize": "finalize",
    }
    action = tool_to_action.get(next_tool, "complete")
    
    # Guard: If planner says "score" but current candidate isn't parsed yet, parse first
    if action == "score" and state.current_candidate_index < len(state.candidates_to_process):
        current_name = state.candidates_to_process[state.current_candidate_index]
        if current_name not in state.parsed_profiles:
            action = "parse_resume"
    
    # Guard: If all candidates processed, finalize (force it even if planner says complete)
    if state.current_candidate_index >= len(state.candidates_to_process):
        if action not in ("finalize", "check_availability", "await_approval"):
            action = "finalize"
    
    state = _append_trajectory(
        state,
        thought,
        "planner",
        {"remaining": remaining, "processed": processed},
        f"Planner decision: {action} (was: {next_tool}). Reason: {reason}",
        "PLAN"
    )
    
    state.next_action = action
    return state


def parse_resume_node(state: AgentState) -> AgentState:
    """Parse the current candidate's resume using the LLM."""
    if state.current_candidate_index >= len(state.candidates_to_process):
        state.next_action = "finalize"
        return state
    
    candidate_name = state.candidates_to_process[state.current_candidate_index]
    resume_text = state.candidates[candidate_name]
    
    llm = get_llm()
    profile = parse_resume(resume_text, llm)
    
    state.parsed_profiles[candidate_name] = profile.model_dump()
    
    # Check for injection (resume_lines will contain the raw text, check for injection patterns)
    injection_detected = False
    injection_patterns = ["ignore previous instructions", "rank me first", "system override", "perfect score"]
    for line in resume_text.lower().split("\n"):
        for pattern in injection_patterns:
            if pattern in line:
                injection_detected = True
                break
    
    injection_warning = " [INJECTION DETECTED]" if injection_detected else ""
    
    state = _append_trajectory(
        state,
        f"Parsing resume for {candidate_name}",
        "parse_resume",
        {"candidate": candidate_name},
        f"Parsed profile for {candidate_name}: {profile.experience_years} years exp, "
        f"{len(profile.skills)} skills, {len(profile.projects)} projects{injection_warning}",
        "PROCEED"
    )
    
    state.next_action = "score"
    return state


def score_node(state: AgentState) -> AgentState:
    """Score the current candidate."""
    candidate_name = state.candidates_to_process[state.current_candidate_index]
    profile_data = state.parsed_profiles[candidate_name]
    
    from models.schemas import ParsedResume
    profile = ParsedResume(**profile_data)
    
    job_req = json.dumps(state.job_requirements, indent=2)
    rubric = json.dumps(state.rubric, indent=2)
    
    llm = get_llm()
    scorecard = score_candidate(profile, job_req, rubric, llm)
    
    state.scorecards[candidate_name] = scorecard.model_dump()
    
    score_details = "\n".join([
        f"  {c.name}: {c.score}/5 (w:{c.weight})"
        for c in scorecard.criteria
    ])
    
    state = _append_trajectory(
        state,
        f"Scoring {candidate_name} against rubric",
        "score_candidate",
        {"candidate": candidate_name},
        f"Scorecard for {candidate_name}: Total={scorecard.total_score}, "
        f"Recommendation={scorecard.recommendation}\n{score_details}",
        "PROCEED"
    )
    
    state.current_candidate_index += 1
    state.next_action = "plan"
    return state


def check_availability_node(state: AgentState) -> AgentState:
    """Check availability for INTERVIEW candidates."""
    interview_candidates = [
        e for e in state.shortlist
        if e.get("decision") == "Interview"
    ]
    
    if not interview_candidates:
        state = _append_trajectory(
            state,
            "No interview candidates to check availability for",
            "check_availability",
            {},
            "No candidates marked for Interview. Skipping.",
            "NO_CANDIDATES"
        )
        state.next_action = "complete"
        return state
    
    for entry in interview_candidates:
        name = entry.get("candidate", entry.get("name", "Unknown"))
        slots = check_availability(name)
        state.availability[name] = slots
        
        if slots:
            slot_str = f"{slots[0]['day']} {slots[0]['start_time']}-{slots[0]['end_time']}"
            proposal = propose_interview(name, slot_str)
            state.actions.append(proposal.model_dump())
    
    state = _append_trajectory(
        state,
        f"Checking availability for {len(interview_candidates)} interview candidates",
        "check_availability",
        {"candidates": [e.get("candidate", e.get("name", "")) for e in interview_candidates]},
        f"Availability checked. Proposals: {len(state.actions)}",
        "AVAILABILITY_CHECKED"
    )
    
    state.next_action = "await_approval"
    return state


def await_approval_node(state: AgentState) -> AgentState:
    """Wait for human approval."""
    pending = [a for a in state.actions if a.get("status") == "Pending Human Approval"]
    
    if not pending:
        state.next_action = "complete"
        return state
    
    state.human_approval_pending = {
        "type": "interview_scheduling",
        "proposals": pending,
        "message": "Please approve or reject each proposed interview slot."
    }
    state.status = "WAITING_APPROVAL"
    
    state = _append_trajectory(
        state,
        f"Waiting for human approval on {len(pending)} interview proposals",
        "propose_interview",
        {"pending": len(pending)},
        f"Agent paused. {len(pending)} proposals awaiting human decision.",
        "WAITING_FOR_HUMAN"
    )
    
    state.next_action = "complete"
    return state


def finalize_node(state: AgentState) -> AgentState:
    """Produce the final ranked shortlist using the BUILT scores from scorecards."""
    # Sort candidates by their scorecards' total_score (descending)
    scored = []
    for name, sc in state.scorecards.items():
        total = sc.get("total_score", 0)
        rec = sc.get("recommendation", "Hold")
        strengths = sc.get("strengths", [])
        gaps = sc.get("gaps", [])
        
        # Apply decision rules directly
        if total >= 70:
            rec = "Interview"
        elif total >= 50:
            rec = "Hold"
        else:
            rec = "Reject"
        
        scored.append({
            "candidate": name,
            "score": round(total, 1),
            "decision": rec,
            "evidence": strengths[:3] if strengths else [],
            "gaps": gaps[:3] if gaps else [],
        })
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    
    # Build final shortlist with ranks
    shortlist = []
    for i, entry in enumerate(scored):
        shortlist.append({
            "candidate": entry["candidate"],
            "rank": i + 1,
            "decision": entry["decision"],
            "score": entry["score"],
            "summary": f"Score: {entry['score']}. {'Evidence: ' + '; '.join(entry['evidence'][:2]) if entry['evidence'] else 'No evidence found.'}"[:200],
            "evidence": entry["evidence"],
            "interview_focus": entry["gaps"] if entry["decision"] == "Interview" else [],
            "slot": "N/A"
        })
    
    state.shortlist = shortlist
    
    state = _append_trajectory(
        state,
        "Producing final ranked shortlist from computed scorecard scores",
        "finalize",
        {"num_candidates": len(state.scorecards)},
        f"Shortlist produced with {len(shortlist)} candidates. "
        f"Decisions: Interview={sum(1 for e in shortlist if e['decision']=='Interview')}, "
        f"Hold={sum(1 for e in shortlist if e['decision']=='Hold')}, "
        f"Reject={sum(1 for e in shortlist if e['decision']=='Reject')}",
        "SHORTLIST_PRODUCED"
    )
    
    state.next_action = "check_availability"
    return state


def resume_approval_node(state: AgentState) -> AgentState:
    """
    Process human approval decisions and update action statuses.
    
    Expects state.human_approval_decisions to be a dict mapping
    candidate name -> "Approved" | "Rejected".
    """
    decisions = getattr(state, "human_approval_decisions", None) or {}
    
    approved_count = 0
    rejected_count = 0
    
    for action in state.actions:
        candidate = action.get("candidate", "")
        decision = decisions.get(candidate)
        if decision == "Approved":
            action["status"] = "Approved"
            approved_count += 1
        elif decision == "Rejected":
            action["status"] = "Rejected"
            rejected_count += 1
        # Leave untouched if no decision was provided
    
    # Clear the pending approval gate
    state.human_approval_pending = None
    state.status = "RUNNING"
    
    state = _append_trajectory(
        state,
        f"Processing human approval decisions: {approved_count} approved, {rejected_count} rejected",
        "resume_approval",
        {"decisions": decisions},
        f"Actions updated. Approved={approved_count}, Rejected={rejected_count}, "
        f"Unchanged={len(state.actions) - approved_count - rejected_count}.",
        "APPROVAL_PROCESSED"
    )
    
    state.next_action = "complete"
    return state


def complete_node(state: AgentState) -> AgentState:
    """Finalize the agent execution."""
    state.status = "COMPLETED" if state.status != "WAITING_APPROVAL" else state.status
    state.next_action = "done"
    
    state = _append_trajectory(
        state,
        "Finalizing recruitment agent execution",
        "complete",
        {"status": state.status},
        f"Agent completed. Candidates processed: {len(state.parsed_profiles)}. "
        f"Shortlist: {len(state.shortlist)}. Actions: {len(state.actions)}.",
        "COMPLETE"
    )
    
    return state