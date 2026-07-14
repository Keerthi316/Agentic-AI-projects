"""
Scheduler Agent — final node in the recruitment workflow (runs after human approval).

Responsibilities:
1. Generate interview invitations for shortlisted candidates.
2. Generate polite rejection templates for non-shortlisted candidates.
3. Suggest interview format (Technical, Behavioral, Mixed) and duration.

Design decisions:
- This agent ONLY runs after human approval (human_approved == True).
- It uses the LLM to generate personalized, context-aware emails.
- The scheduler does NOT send emails — it only generates the templates.
  Email delivery is a separate concern handled outside the workflow.
"""

from typing import Any, Dict

from models.state import RecruitmentState
from prompts.system_prompts import SCHEDULER_PROMPT
from tools.llm import invoke_llm
from tools.logging import get_agent_logger, log_agent_action
import json

logger = get_agent_logger("Scheduler")


def schedule_interviews(state: RecruitmentState) -> Dict[str, Any]:
    """Generate interview schedules for shortlisted candidates.

    Only executes if human_approved is True. Otherwise, logs a warning
    and returns without changes.

    Args:
        state: The current RecruitmentState with 'shortlist' and 'jd'.

    Returns:
        Dict with scheduling results and step_count update.
    """
    if not state.get("human_approved", False):
        log_agent_action(
            logger,
            "Scheduling skipped",
            {"reason": "Human approval not granted"},
            level="WARNING",
        )
        return {"step_count": state.get("step_count", 0) + 1}

    shortlist = state.get("shortlist", [])
    jd = state.get("jd")

    if not shortlist:
        err_msg = "No shortlist found in state. Cannot schedule interviews."
        log_agent_action(logger, "Scheduling failed", {"error": err_msg}, level="ERROR")
        return {"errors": [err_msg], "step_count": state.get("step_count", 0) + 1}

    if not jd:
        err_msg = "Job description missing. Cannot schedule interviews."
        log_agent_action(logger, "Scheduling failed", {"error": err_msg}, level="ERROR")
        return {"errors": [err_msg], "step_count": state.get("step_count", 0) + 1}

    log_agent_action(
        logger,
        "Generating interview schedules",
        {"shortlisted_count": sum(1 for s in shortlist if s.status == "shortlisted")},
    )

    # Format shortlist for prompt
    shortlist_str = _format_shortlist_for_prompt(shortlist)

    prompt = SCHEDULER_PROMPT.format(
        shortlist=shortlist_str,
        job_title=jd.title,
    )

    try:
        content = invoke_llm(prompt)
        # Parse the JSON response
        schedules = json.loads(_clean_json_response(content))

        if not isinstance(schedules, list):
            schedules = [schedules]

        log_agent_action(
            logger,
            "Schedules generated",
            {"count": len(schedules)},
        )

        return {
            "schedules": schedules,
            "step_count": state.get("step_count", 0) + 1,
        }

    except Exception as e:
        err_msg = f"Failed to generate interview schedules: {str(e)}"
        log_agent_action(logger, "Scheduling error", {"error": err_msg}, level="ERROR")
        return {"errors": [err_msg], "schedules": [], "step_count": state.get("step_count", 0) + 1}


def _format_shortlist_for_prompt(shortlist: list) -> str:
    """Format the shortlist into a readable string for the LLM prompt.

    Args:
        shortlist: List of ShortlistEntry objects.

    Returns:
        Formatted string.
    """
    lines = ["Rank | Candidate ID | Name | Score | Status"]
    lines.append("-" * 60)
    for entry in shortlist:
        lines.append(f"{entry.rank:4d} | {entry.candidate_id:20s} | {entry.name:20s} | {entry.final_score:5.1f} | {entry.status}")
    return "\n".join(lines)


def _clean_json_response(content: str) -> str:
    """Remove markdown code fences and leading/trailing whitespace from LLM output.

    Args:
        content: Raw LLM response string.

    Returns:
        Cleaned JSON string.
    """
    content = content.strip()
    if content.startswith("```json"):
        content = content[len("```json"):]
        if content.endswith("```"):
            content = content[:-3]
    elif content.startswith("```"):
        content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
    return content.strip()