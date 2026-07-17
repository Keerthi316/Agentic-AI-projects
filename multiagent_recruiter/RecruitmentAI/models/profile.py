"""
Pydantic models for candidate profiles, scorecards and decisions.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class CandidateProfile(BaseModel):
    """Structured profile extracted by the Analyst agent."""
    candidate_name: str = Field(default="Unknown", description="Full name of the candidate")
    skills: List[str] = Field(default_factory=list, description="Technical and soft skills")
    experience: List[str] = Field(default_factory=list, description="Work experience entries")
    education: List[str] = Field(default_factory=list, description="Education entries")
    projects: List[str] = Field(default_factory=list, description="Notable projects")
    certifications: List[str] = Field(default_factory=list, description="Certifications and courses")
    summary: str = Field(default="", description="Professional summary")
    years_of_experience: float = Field(default=0.0, description="Total years of experience")
    raw_text: Optional[str] = Field(default=None, exclude=True)

    @validator("skills", "experience", "education", "projects", "certifications", pre=True)
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v or []


class ScoreBreakdown(BaseModel):
    """Detailed scoring produced by the Scorer agent."""
    technical_score: float = Field(ge=0, le=100)
    experience_score: float = Field(ge=0, le=100)
    education_score: float = Field(ge=0, le=100)
    projects_score: float = Field(ge=0, le=100)
    communication_score: float = Field(ge=0, le=100)
    overall_score: float = Field(ge=0, le=100)
    reasons: str = Field(default="")
    missing_skills: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """Result produced by the Verifier agent."""
    accepted: bool = Field(description="Whether the score is accepted as-is")
    bias_detected: bool = Field(default=False)
    hallucination_detected: bool = Field(default=False)
    evidence_sufficient: bool = Field(default=True)
    consistent: bool = Field(default=True)
    feedback: str = Field(default="")
    revised_score: Optional[float] = Field(default=None)


class Decision(BaseModel):
    """Final recommendation produced by the Decider agent."""
    recommendation: str = Field(
        description="One of: Interview, Hold, Reject, Need Human Review"
    )
    explanation: str = Field(description="Detailed explanation of the decision")
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)

    @validator("recommendation")
    def validate_recommendation(cls, v):
        allowed = {"Interview", "Hold", "Reject", "Need Human Review"}
        if v not in allowed:
            raise ValueError(f"recommendation must be one of {allowed}")
        return v


class RunRecord(BaseModel):
    """Stored in SQLite for history."""
    run_id: str
    filename: str
    candidate_name: str
    overall_score: float
    recommendation: str
    explanation: str
    skills: List[str]
    experience: List[str]
    education: List[str]
    timestamp: str
    approved: Optional[bool] = None
