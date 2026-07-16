"""
Tool definitions and execution for the College FAQ Chatbot.

Defines three tools (fee_calculator, date_checker, percentage_calculator)
with JSON schemas for OpenAI/OpenRouter Function Calling, plus
validation and execution logic.
"""

import json
import logging
from datetime import datetime, date
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Tool Definitions (JSON Schema for Function Calling)
# ──────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fee_calculator",
            "description": (
                "Calculate total tuition fees, hostel fees, scholarship discounts, "
                "and combined fee calculations for BVRIT college. Use this tool ANY time "
                "the user asks about fee totals, scholarship reductions, combined costs, "
                "or fee calculations. The tool will compute: total tuition (annual_fee * years), "
                "scholarship discount, hostel fees, and grand total. "
                "Always use this tool for fee-related math instead of doing it yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "annual_fee": {
                        "type": "number",
                        "description": "Annual tuition fee amount in rupees (e.g., 110000)."
                    },
                    "years": {
                        "type": "integer",
                        "description": "Number of years for the course (e.g., 4 for B.Tech)."
                    },
                    "scholarship_percent": {
                        "type": "number",
                        "description": "Scholarship percentage discount (e.g., 20 for 20%%). Optional.",
                    },
                    "hostel_fee": {
                        "type": "number",
                        "description": "Annual hostel fee in rupees. Optional.",
                    },
                },
                "required": ["annual_fee", "years"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "date_checker",
            "description": (
                "Compare a date from the knowledge base with today's date and report "
                "whether the event is upcoming, past, or how many days remain. Use this "
                "when the user asks about deadlines, admission dates, exam dates, or events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_name": {
                        "type": "string",
                        "description": "Name of the event or deadline (e.g., 'Admission Deadline', 'Exam Start')."
                    },
                    "event_date": {
                        "type": "string",
                        "description": "Date of the event in YYYY-MM-DD format (e.g., '2026-08-15')."
                    },
                },
                "required": ["event_name", "event_date"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "percentage_calculator",
            "description": (
                "Calculate percentages for scholarships, placement rates, admission rates, "
                "and similar college-related percentage calculations. Use this when the user "
                "asks 'what is X%% of Y' or 'what percentage is X of Y'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "The base value or total amount (e.g., total fee, total students)."
                    },
                    "percentage": {
                        "type": "number",
                        "description": "The percentage to calculate (e.g., 20 for 20%%)."
                    },
                },
                "required": ["value", "percentage"],
            },
        }
    },
]


# ──────────────────────────────────────────────
# Argument Validation
# ──────────────────────────────────────────────

def validate_fee_args(args: Dict[str, Any]) -> Optional[str]:
    """Validate fee_calculator arguments. Returns error message or None."""
    annual_fee = args.get("annual_fee")
    years = args.get("years")
    scholarship = args.get("scholarship_percent")
    hostel_fee = args.get("hostel_fee")

    if annual_fee is not None and (not isinstance(annual_fee, (int, float)) or annual_fee <= 0):
        return "annual_fee must be a positive number."
    if years is not None and (not isinstance(years, int) or years <= 0):
        return "years must be a positive integer."
    if scholarship is not None:
        if not isinstance(scholarship, (int, float)):
            return "scholarship_percent must be a number."
        if scholarship < 0:
            return "scholarship_percent cannot be negative."
        if scholarship > 100:
            return "scholarship_percent cannot exceed 100%."
    if hostel_fee is not None and (not isinstance(hostel_fee, (int, float)) or hostel_fee < 0):
        return "hostel_fee must be a non-negative number."
    return None


def validate_date_args(args: Dict[str, Any]) -> Optional[str]:
    """Validate date_checker arguments. Returns error message or None."""
    event_date = args.get("event_date")
    if not event_date or not isinstance(event_date, str):
        return "event_date must be a string in YYYY-MM-DD format."
    try:
        datetime.strptime(event_date, "%Y-%m-%d")
    except ValueError:
        return f"Invalid date format: '{event_date}'. Expected YYYY-MM-DD."
    event_name = args.get("event_name")
    if not event_name or not isinstance(event_name, str) or not event_name.strip():
        return "event_name must be a non-empty string."
    return None


def validate_percentage_args(args: Dict[str, Any]) -> Optional[str]:
    """Validate percentage_calculator arguments. Returns error message or None."""
    value = args.get("value")
    percentage = args.get("percentage")

    if value is not None and (not isinstance(value, (int, float)) or value <= 0):
        return "value must be a positive number."
    if percentage is not None and (not isinstance(percentage, (int, float)) or percentage < 0):
        return "percentage must be a non-negative number."
    if percentage is not None and percentage > 100:
        return "percentage cannot exceed 100%."
    return None


# ──────────────────────────────────────────────
# Tool Execution
# ──────────────────────────────────────────────

def execute_fee_calculator(args: Dict[str, Any]) -> str:
    """
    Execute fee_calculator tool.

    Returns a formatted string with the calculation results.
    """
    error = validate_fee_args(args)
    if error:
        return f"⚠️ Validation error: {error}"

    annual_fee = args["annual_fee"]
    years = args["years"]
    scholarship_percent = args.get("scholarship_percent", 0)
    hostel_fee = args.get("hostel_fee", 0)

    total_tuition = annual_fee * years
    total_hostel = hostel_fee * years if hostel_fee else 0
    scholarship_amount = total_tuition * (scholarship_percent / 100) if scholarship_percent else 0
    total_after_scholarship = total_tuition - scholarship_amount
    grand_total = total_after_scholarship + total_hostel

    lines = [f"📊 Fee Calculation Result:"]
    lines.append(f"  Annual Tuition Fee: Rs. {annual_fee:,.2f}")
    lines.append(f"  Duration: {years} year(s)")
    lines.append(f"  Total Tuition ({years} years): Rs. {total_tuition:,.2f}")

    if hostel_fee:
        lines.append(f"  Annual Hostel Fee: Rs. {hostel_fee:,.2f}")
        lines.append(f"  Total Hostel Fee ({years} years): Rs. {total_hostel:,.2f}")

    if scholarship_percent:
        lines.append(f"  Scholarship: {scholarship_percent}%")
        lines.append(f"  Scholarship Amount: Rs. {scholarship_amount:,.2f}")
        lines.append(f"  Tuition After Scholarship: Rs. {total_after_scholarship:,.2f}")

    lines.append(f"  Grand Total: Rs. {grand_total:,.2f}")

    return "\n".join(lines)


def execute_date_checker(args: Dict[str, Any]) -> str:
    """
    Execute date_checker tool.

    Compares the event date with today's date and returns a human-readable result.
    """
    error = validate_date_args(args)
    if error:
        return f"⚠️ Validation error: {error}"

    event_name = args["event_name"].strip()
    event_date_str = args["event_date"]
    event_date = datetime.strptime(event_date_str, "%Y-%m-%d").date()
    today = date.today()

    diff = (event_date - today).days

    if diff < 0:
        days_ago = abs(diff)
        if days_ago == 0:
            return f"📅 '{event_name}' is today!"
        return (
            f"📅 '{event_name}' was {days_ago} day(s) ago (on {event_date_str}). "
            f"This date has already passed."
        )
    elif diff == 0:
        return f"📅 '{event_name}' is today! ({event_date_str})"
    else:
        return (
            f"📅 '{event_name}' is in {diff} day(s) (on {event_date_str}). "
            f"This date is upcoming."
        )


def execute_percentage_calculator(args: Dict[str, Any]) -> str:
    """
    Execute percentage_calculator tool.

    Calculates the percentage of a value and returns a formatted result.
    """
    error = validate_percentage_args(args)
    if error:
        return f"⚠️ Validation error: {error}"

    value = args["value"]
    percentage = args["percentage"]

    result = value * (percentage / 100)

    return (
        f"📊 Percentage Calculation Result:\n"
        f"  {percentage}% of {value:,.2f} = {result:,.2f}\n"
        f"  (Value: {value:,.2f}, Percentage: {percentage}%)"
    )


# ──────────────────────────────────────────────
# Tool Router
# ──────────────────────────────────────────────

TOOL_EXECUTORS = {
    "fee_calculator": execute_fee_calculator,
    "date_checker": execute_date_checker,
    "percentage_calculator": execute_percentage_calculator,
}


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Execute a tool by name with the given arguments.

    Args:
        tool_name (str): The name of the tool to execute.
        arguments (Dict[str, Any]): The arguments for the tool.

    Returns:
        str: The tool execution result as a formatted string.

    Raises:
        ValueError: If the tool name is unknown.
    """
    if tool_name not in TOOL_EXECUTORS:
        raise ValueError(f"Unknown tool: {tool_name}")

    executor = TOOL_EXECUTORS[tool_name]
    logger.info(f"Executing tool: {tool_name} with args: {arguments}")
    return executor(arguments)