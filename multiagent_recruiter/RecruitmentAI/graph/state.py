"""
Shared LangGraph state — every agent reads from and writes to this TypedDict.
Each agent writes ONLY its own fields.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class RecruitmentState(TypedDict, total=False):
    # ── inputs set by Coordinator ──────────────────────────────────────────
    job_description: str
    uploaded_files: List[Dict[str, Any]]   # [{"filename": ..., "bytes": ..., "text": ...}]

    # ── analyst output ────────────────────────────────────────────────────
    profiles: List[Dict[str, Any]]         # list of CandidateProfile dicts

    # ── scorer output ─────────────────────────────────────────────────────
    scorecards: List[Dict[str, Any]]       # list of ScoreBreakdown dicts

    # ── verifier output ───────────────────────────────────────────────────
    verified_scores: List[Dict[str, Any]]  # list of VerificationResult dicts
    revision_count: int

    # ── decider output ────────────────────────────────────────────────────
    shortlist: List[Dict[str, Any]]        # list of Decision dicts

    # ── routing / meta ────────────────────────────────────────────────────
    current_candidate_index: int
    logs: List[str]
    errors: List[str]
    needs_verification: bool               # set by router after scoring
    needs_human_approval: bool
    human_approved: Optional[bool]
    run_id: Optional[str]
