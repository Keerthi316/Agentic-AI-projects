"""
State Schema for the Recruitment Agent.
Defines the structured state that persists across agent steps.
"""

from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime


class TrajectoryStep(TypedDict):
    """A single step in the agent's reasoning trajectory."""
    step_number: int
    thought: str
    action: str
    input: Dict[str, Any]
    observation: str
    decision: str


class CandidateProfile(TypedDict, total=False):
    """Parsed structured profile of a candidate."""
    name: str
    email: str
    education: List[Dict[str, str]]
    work_experience: List[Dict[str, str]]
    projects: List[Dict[str, str]]
    skills: Dict[str, List[str]]  # e.g., {"languages": [...], "frameworks": [...], ...}
    certifications: List[str]
    achievements: List[str]
    years_of_experience: float
    has_prompt_injection: bool  # Guardrail flag


class CriterionScore(TypedDict):
    """Score for a single rubric criterion."""
    criterion: str
    weight: float
    score: int  # 0-5
    evidence: str  # Must cite resume text
    max_score: int


class ScoreCard(TypedDict):
    """Complete scorecard for a candidate."""
    candidate_name: str
    criteria_scores: List[CriterionScore]
    total_weighted_score: float
    max_possible_score: float
    normalized_score: float  # 0-100


class TimeSlot(TypedDict):
    """Available time slot for interview."""
    day: str
    start_time: str
    end_time: str


class InterviewProposal(TypedDict):
    """Proposed interview slot for a candidate."""
    candidate: str
    proposed_slot: TimeSlot
    status: str  # "PENDING_APPROVAL" | "APPROVED" | "REJECTED"


class ShortlistEntry(TypedDict):
    """Entry in the final shortlist."""
    name: str
    decision: str  # "INTERVIEW" | "HOLD" | "REJECT"
    score: float
    justification: str
    scorecard: ScoreCard


class AgentState(TypedDict):
    """Main state that persists across the agent's execution loop."""
    # Input
    job_description: str
    candidates: Dict[str, str]  # name -> raw resume text
    
    # Rubric
    rubric: Optional[Dict[str, Any]]
    
    # Processing state
    current_candidate_index: int
    total_candidates: int
    candidates_to_process: List[str]  # Ordered list of candidate names
    
    # Results
    parsed_profiles: Dict[str, CandidateProfile]  # name -> profile
    scorecards: Dict[str, ScoreCard]  # name -> scorecard
    shortlist: List[ShortlistEntry]
    
    # Availability
    availability: Dict[str, List[TimeSlot]]  # name -> slots
    
    # Actions
    actions: List[InterviewProposal]
    
    # Trajectory
    trajectory: List[TrajectoryStep]
    step_count: int
    
    # Agent control
    status: str  # "RUNNING" | "COMPLETED" | "ERROR" | "WAITING_APPROVAL"
    next_action: str  # What the agent plans to do next
    error: Optional[str]
    human_approval_pending: Optional[Dict[str, Any]]
