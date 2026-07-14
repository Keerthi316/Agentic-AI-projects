"""
Trace capture and validation package for the Multi-Agent Recruitment System.

Provides:
- capture_trace(): wraps graph.stream() and records all node execution events
- TraceCapture:    dataclass holding the full execution trace
- NodeEvent:       a single node execution event
- TraceValidator:  validates workflow invariants against a TraceCapture
- TraceValidationResult: aggregate result with per-invariant pass/fail
- InvariantResult: result for a single invariant check
"""

from .capture import (
    capture_trace,
    TraceCapture,
    NodeEvent,
)
from .validator import (
    TraceValidator,
    TraceValidationResult,
    InvariantResult,
    InvariantSeverity,
)

__all__ = [
    "capture_trace",
    "TraceCapture",
    "NodeEvent",
    "TraceValidator",
    "TraceValidationResult",
    "InvariantResult",
    "InvariantSeverity",
]
