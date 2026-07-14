"""
Shared state definitions for the recruitment workflow.

Design decisions:
1. TypedDict with Annotated reducers: LangGraph uses TypedDict for state,
   and Annotated[..., operator.add] enables parallel writes to accumulate
   items into a list. This is essential because the Scorer may process
   candidates in parallel.

2. Pydantic models for handoffs: Between agents, we use Pydantic models
   (CandidateProfile, Scorecard, etc.) so that invalid data is caught
   at validation boundaries — not deep in agent logic.

3. Field-level documentation: Each field documents WHICH agent reads/writes it.
   This makes the state a living contract between agents.

4. revision_count with step budget: Prevents infinite loops by limiting
   retries AND total execution steps.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Pydantic Models — validated handoff contracts between agents
# ---------------------------------------------------------------------------


class CandidateProfile(BaseModel):
    """Structured profile extracted from a resume by the Resume Analyst.

    Fields marked with `...` are required; missing required fields
    should trigger the retry loop rather than produce incorrect results.

    Written by: ResumeAnalystAgent
    Read by: ScorerAgent, VerifierAgent, DeciderAgent
    """

    candidate_id: str = Field(..., description="Unique identifier (e.g., email hash)")
    name: str = Field(..., description="Full name of the candidate")
    skills: list[str] = Field(default_factory=list, description="Technical and soft skills")
    education: list[str] = Field(default_factory=list, description="Degrees and institutions")
    experience: list[dict] = Field(
        default_factory=list,
        description="List of work experiences: each dict has 'role', 'company', 'years', 'description'",
    )
    projects: list[dict] = Field(
        default_factory=list,
        description="List of projects: each dict has 'name', 'description', 'technologies'",
    )
    certifications: list[str] = Field(default_factory=list, description="Professional certifications")
    raw_text: str = Field(default="", description="Original resume text, used for verification")
    is_injection_detected: bool = Field(
        default=False,
        description="True if the analyst detected a prompt injection in the resume",
    )
    injection_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of injection detection (0 = clean, 1 = definitely injected)",
    )


class Scorecard(BaseModel):
    """Scoring result for a single candidate against the job description.

    Written by: ScorerAgent
    Read by: VerifierAgent (for blind verification), DeciderAgent
    """

    candidate_id: str = Field(..., description="Matches CandidateProfile.candidate_id")
    total_score: float = Field(..., ge=0.0, le=100.0, description="Overall score (0-100)")
    skill_score: float = Field(default=0.0, ge=0.0, le=100.0)
    experience_score: float = Field(default=0.0, ge=0.0, le=100.0)
    education_score: float = Field(default=0.0, ge=0.0, le=100.0)
    reasoning: str = Field(default="", description="Explanation of the score")
    is_borderline: bool = Field(
        default=False,
        description="True if score falls in the borderline range (50-75 by default)",
    )


class VerifiedScore(BaseModel):
    """Output of the Verifier agent — blind re-score with fairness check.

    Written by: VerifierAgent
    Read by: DeciderAgent (replaces original Scorecard if verification fails)
    """

    candidate_id: str = Field(...)
    original_score: float = Field(...)
    blind_score: float = Field(..., ge=0.0, le=100.0, description="Score computed without candidate identity")
    score_difference: float = Field(default=0.0, description="|original_score - blind_score|")
    is_fair: bool = Field(default=True, description="True if the score difference is within tolerance")
    fairness_notes: str = Field(default="", description="Explanation of fairness check results")
    injection_affected: bool = Field(
        default=False,
        description="True if verification confirms injection affected the score",
    )


class ShortlistEntry(BaseModel):
    """A single entry in the final shortlist.

    Written by: DeciderAgent
    Read by: SchedulerAgent
    """

    candidate_id: str = Field(...)
    name: str = Field(...)
    final_score: float = Field(..., ge=0.0, le=100.0)
    rank: int = Field(..., ge=1, description="Rank in the shortlist (1 = best)")
    status: str = Field(default="shortlisted", pattern=r"^(shortlisted|rejected|hold)$")


class JDInput(BaseModel):
    """Validated job description input.

    Written by: User / system entry point
    Read by: ResumeAnalystAgent, ScorerAgent
    """

    title: str = Field(..., min_length=1, description="Job title")
    description: str = Field(..., min_length=10, description="Full job description text")
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_experience_years: int = Field(default=0, ge=0)
    education_requirement: str = Field(default="", description="Minimum education level required")


# ---------------------------------------------------------------------------
# LangGraph Shared State (TypedDict with reducers)
# ---------------------------------------------------------------------------


class RecruitmentState(TypedDict):
    """TypedDict shared state for the LangGraph workflow.

    Each field documents:
    - Initializer: the agent that first writes the field
    - Readers: agents that may read it
    - Mutators: agents that may modify it after initialization
    - Reducer: special annotations for parallel-write-safe fields

    Fields annotated with `Annotated[list[T], operator.add]` can receive
    writes from multiple nodes in parallel — LangGraph will accumulate them.
    """

    # ── Job Definition ──────────────────────────────────────────────────
    # Written by: User input (entry point)
    # Read by: ResumeAnalystAgent, ScorerAgent
    jd: JDInput

    # ── Candidate Input ─────────────────────────────────────────────────
    # Written by: User input (entry point)
    # Read by: ResumeAnalystAgent
    candidates: Annotated[list[str], operator.add]

    # ── Parsed Profiles ─────────────────────────────────────────────────
    # Written by: ResumeAnalystAgent
    # Read by: ScorerAgent, VerifierAgent, DeciderAgent
    # Reducer: operator.add allows parallel writing from multiple analyst nodes
    parsed_profiles: Annotated[list[CandidateProfile], operator.add]

    # ── Scorecards ──────────────────────────────────────────────────────
    # Written by: ScorerAgent
    # Read by: VerifierAgent (filters borderline), DeciderAgent
    # Reducer: operator.add allows parallel scoring
    scorecards: Annotated[list[Scorecard], operator.add]

    # ── Verified Scores ─────────────────────────────────────────────────
    # Written by: VerifierAgent
    # Read by: DeciderAgent (replaces scorecards for borderline candidates)
    verified_scores: Annotated[list[VerifiedScore], operator.add]

    # ── Retry Tracking ──────────────────────────────────────────────────
    # Written by: Conditional router (increments on retry)
    # Read by: Conditional router (checks limit)
    revision_count: int

    # ── Final Shortlist ─────────────────────────────────────────────────
    # Written by: DeciderAgent
    # Read by: SchedulerAgent, Human (for approval)
    shortlist: list[ShortlistEntry]

    # ── Execution Control ───────────────────────────────────────────────
    # Written by: Every node (incremented on each step)
    # Read by: Conditional router (checks step budget)
    step_count: int

    # ── Error / Escalation ──────────────────────────────────────────────
    # Written by: Any agent on failure
    # Read by: Conditional router, Human (escalation path)
    errors: Annotated[list[str], operator.add]
    needs_human_escalation: bool

    # ── Human Approval Gate ─────────────────────────────────────────────
    # Written by: Human (approval/rejection)
    # Read by: Conditional router (decides whether to proceed to Scheduler)
    human_approved: bool

    # ── Interview Schedules ──────────────────────────────────────────────
    # Written by: SchedulerAgent
    # Read by: Frontend (Interview Scheduler page)
    schedules: list[dict]