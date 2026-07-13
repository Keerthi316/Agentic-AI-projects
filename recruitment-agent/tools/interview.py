"""
Interview proposal tool.
Returns a proposal that requires human approval.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.schemas import InterviewProposal


def propose_interview(candidate: str, slot: str) -> InterviewProposal:
    """
    Propose an interview slot for a candidate.
    This is an ACTION TOOL - requires human approval before execution.
    """
    return InterviewProposal(
        candidate=candidate,
        slot=slot,
        status="Pending Human Approval"
    )