"""
Scoring helpers — pure Python, no LLM calls.
These are used by the Scorer agent to normalise / aggregate numbers.
"""
from __future__ import annotations

from typing import Dict, List


WEIGHTS: Dict[str, float] = {
    "technical": 0.35,
    "experience": 0.25,
    "education": 0.15,
    "projects": 0.15,
    "communication": 0.10,
}


def weighted_overall(scores: Dict[str, float]) -> float:
    """Compute weighted overall score from individual dimension scores."""
    total = (
        scores.get("technical_score", 0) * WEIGHTS["technical"]
        + scores.get("experience_score", 0) * WEIGHTS["experience"]
        + scores.get("education_score", 0) * WEIGHTS["education"]
        + scores.get("projects_score", 0) * WEIGHTS["projects"]
        + scores.get("communication_score", 0) * WEIGHTS["communication"]
    )
    return round(min(max(total, 0), 100), 2)


def skill_overlap(candidate_skills: List[str], required_skills: List[str]) -> float:
    """Return percentage of required skills covered by candidate (0-100)."""
    if not required_skills:
        return 100.0
    req = {s.lower().strip() for s in required_skills}
    cand = {s.lower().strip() for s in candidate_skills}
    matched = req & cand
    return round(len(matched) / len(req) * 100, 1)


def missing_skills(candidate_skills: List[str], required_skills: List[str]) -> List[str]:
    """Return required skills not found in candidate profile."""
    req = {s.lower().strip(): s for s in required_skills}
    cand = {s.lower().strip() for s in candidate_skills}
    return [orig for key, orig in req.items() if key not in cand]
