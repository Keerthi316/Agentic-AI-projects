"""
Agent state definition for the LangGraph recruitment agent.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """State that persists across the agent's execution loop."""
    
    # Input
    job_description: str = ""
    job_requirements: Optional[Dict[str, Any]] = None
    rubric: Optional[Dict[str, Any]] = None
    
    # Candidates
    candidates: Dict[str, str] = Field(default_factory=dict)  # name -> raw text
    candidates_to_process: List[str] = Field(default_factory=list)
    current_candidate_index: int = 0
    
    # Results
    parsed_profiles: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    scorecards: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    shortlist: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Availability
    availability: Dict[str, List[Dict[str, str]]] = Field(default_factory=dict)
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Trajectory
    trajectory: List[Dict[str, Any]] = Field(default_factory=list)
    step_count: int = 0
    max_steps: int = 50
    
    # Agent control
    status: str = "RUNNING"  # RUNNING, COMPLETED, ERROR, WAITING_APPROVAL
    next_action: str = "initialize"
    error: Optional[str] = None
    human_approval_pending: Optional[Dict[str, Any]] = None
    human_approval_decisions: Optional[Dict[str, str]] = None  # candidate -> "Approved" | "Rejected"