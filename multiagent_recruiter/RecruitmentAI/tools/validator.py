"""
Pydantic-based validation helpers for LLM outputs.
"""
from __future__ import annotations

from typing import Tuple

from pydantic import ValidationError

from models.profile import CandidateProfile, ScoreBreakdown, Decision
from utils.logger import get_logger

logger = get_logger("validator")


def validate_profile(data: dict) -> Tuple[CandidateProfile | None, str]:
    """Validate raw dict into CandidateProfile. Returns (model, error)."""
    try:
        return CandidateProfile(**data), ""
    except ValidationError as e:
        msg = f"Profile validation failed: {e}"
        logger.warning(msg)
        return None, msg


def validate_scorecard(data: dict) -> Tuple[ScoreBreakdown | None, str]:
    try:
        return ScoreBreakdown(**data), ""
    except ValidationError as e:
        msg = f"Scorecard validation failed: {e}"
        logger.warning(msg)
        return None, msg


def validate_decision(data: dict) -> Tuple[Decision | None, str]:
    try:
        return Decision(**data), ""
    except ValidationError as e:
        msg = f"Decision validation failed: {e}"
        logger.warning(msg)
        return None, msg
