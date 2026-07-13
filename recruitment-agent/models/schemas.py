"""
Pydantic schemas for the Recruitment Agent.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class JDRequirements(BaseModel):
    job_title: str = ""
    required_skills: List[str] = []
    preferred_skills: List[str] = []
    minimum_education: str = ""
    minimum_experience: str = ""
    responsibilities: List[str] = []
    communication_required: bool = True
    weight_suggestions: List[str] = []


class ParsedResume(BaseModel):
    name: str = "Unknown"
    education: List[Any] = []
    experience_years: float = 0.0
    skills: List[str] = []
    projects: List[Any] = []
    certifications: List[Any] = []
    communication_evidence: str = ""
    resume_lines: List[str] = []

    def model_post_init(self, __context):
        """Normalize education, projects, certifications to strings."""
        def normalize(items):
            result = []
            for item in items:
                if isinstance(item, dict):
                    # Convert dict to readable string
                    parts = []
                    for k, v in item.items():
                        if v and k != "resume_lines":
                            parts.append(str(v))
                    result.append(" - ".join(parts) if parts else str(item))
                else:
                    result.append(str(item))
            return result
        self.education = normalize(self.education)
        self.projects = normalize(self.projects)
        self.certifications = normalize(self.certifications)


class CriterionScore(BaseModel):
    name: str
    score: int = Field(ge=0, le=5)
    weight: float = Field(ge=0.0)
    evidence: str = ""


class ScoreCard(BaseModel):
    candidate: str
    criteria: List[CriterionScore] = []
    total_score: float = 0.0
    strengths: List[str] = []
    gaps: List[str] = []
    recommendation: str = "Hold"


class TimeSlot(BaseModel):
    day: str = ""
    start_time: str = ""
    end_time: str = ""


class InterviewProposal(BaseModel):
    candidate: str = ""
    slot: str = ""
    status: str = "Pending Human Approval"


class PlannerDecision(BaseModel):
    thought: str = ""
    next_tool: str = ""
    reason: str = ""


class RankingEntry(BaseModel):
    candidate: str = ""
    rank: int = 0
    decision: str = ""
    score: float = 0.0
    summary: str = ""
    evidence: List[str] = []
    interview_focus: List[str] = []
    slot: str = ""


class FinalDecision(BaseModel):
    ranking: List[RankingEntry] = []


class TrajectoryStep(BaseModel):
    step_number: int = 0
    thought: str = ""
    tool: str = ""
    arguments: Dict[str, Any] = {}
    observation: str = ""
    state_changes: Dict[str, Any] = {}
    decision: str = ""