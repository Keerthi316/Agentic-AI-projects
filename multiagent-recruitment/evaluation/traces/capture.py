"""
Trace capture for the Multi-Agent Recruitment System.

Wraps graph.stream() and records every node execution event into a
structured TraceCapture object. The capture is the single input to
TraceValidator — nothing else in the evaluation framework reads the
graph stream directly.

Design decisions:
- TraceCapture is a plain dataclass (not Pydantic) because it holds
  live Python objects (Scorecard, CandidateProfile, etc.) that aren't
  JSON-serialisable without custom encoders. Pydantic validation is
  done on the dataset schema, not on the runtime trace.
- Each NodeEvent records the node name, the dict it returned, and
  the cumulative step_count so validators can check ordering and
  step-budget compliance without re-scanning the full state.
- capture_trace() accepts either a pre-built RecruitmentState or an
  EvalTask so callers can use it from tests or the run_evaluation script.
- Errors raised inside the graph are caught and stored in
  TraceCapture.runtime_error so tests can assert on them without
  crashing the pytest session.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Force demo mode before importing workflow modules
os.environ.setdefault("RECRUITMENT_DEMO_MODE", "true")

from models.state import (
    CandidateProfile,
    JDInput,
    RecruitmentState,
    Scorecard,
    ShortlistEntry,
    VerifiedScore,
)
from graph.workflow import build_recruitment_graph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NodeEvent:
    """A single node execution event captured from graph.stream()."""

    node_name: str
    """Name of the LangGraph node that produced this event."""

    output: Dict[str, Any]
    """The dict returned by the node (partial state update)."""

    step_index: int
    """Position of this event in the trace (0-based)."""

    step_count_after: Optional[int] = None
    """The step_count value in the state after this node ran, if present."""


@dataclass
class TraceCapture:
    """Complete execution trace for a single workflow run.

    Produced by capture_trace() and consumed by TraceValidator.
    """

    # Ordered list of node events (same order as graph.stream() emitted them)
    events: List[NodeEvent] = field(default_factory=list)

    # Convenience: ordered list of node names only
    node_sequence: List[str] = field(default_factory=list)

    # Merged final state (all event outputs folded together)
    final_state: Dict[str, Any] = field(default_factory=dict)

    # Wall-clock time the workflow took (seconds)
    elapsed_seconds: float = 0.0

    # Set if graph.stream() raised an unhandled exception
    runtime_error: Optional[str] = None

    # Whether the workflow was run with human_approved=True
    human_approved: bool = False

    # ----------------------------------------------------------------
    # Convenience accessors used by TraceValidator
    # ----------------------------------------------------------------

    def nodes_run(self) -> List[str]:
        """Ordered list of node names that executed."""
        return list(self.node_sequence)

    def ran(self, node_name: str) -> bool:
        """Return True if the given node appeared in the trace."""
        return node_name in self.node_sequence

    def index_of(self, node_name: str) -> int:
        """Return the 0-based position of the first occurrence of node_name.

        Returns -1 if not present.
        """
        try:
            return self.node_sequence.index(node_name)
        except ValueError:
            return -1

    def ran_before(self, first: str, second: str) -> bool:
        """Return True if `first` ran before `second` in the trace.

        Returns False if either node did not run.
        """
        i = self.index_of(first)
        j = self.index_of(second)
        if i == -1 or j == -1:
            return False
        return i < j

    def parsed_profiles(self) -> List[CandidateProfile]:
        """All CandidateProfile objects emitted by resume_analyst."""
        profiles: List[CandidateProfile] = []
        for ev in self.events:
            if ev.node_name == "resume_analyst":
                profiles.extend(ev.output.get("parsed_profiles", []))
        return profiles

    def scorecards(self) -> List[Scorecard]:
        """All Scorecard objects emitted by scorer."""
        cards: List[Scorecard] = []
        for ev in self.events:
            if ev.node_name == "scorer":
                cards.extend(ev.output.get("scorecards", []))
        return cards

    def verified_scores(self) -> List[VerifiedScore]:
        """All VerifiedScore objects emitted by verifier."""
        scores: List[VerifiedScore] = []
        for ev in self.events:
            if ev.node_name == "verifier":
                scores.extend(ev.output.get("verified_scores", []))
        return scores

    def shortlist(self) -> List[ShortlistEntry]:
        """ShortlistEntry objects emitted by decider."""
        for ev in self.events:
            if ev.node_name == "decider":
                return ev.output.get("shortlist", [])
        return []

    def errors(self) -> List[str]:
        """All error strings accumulated across all node outputs."""
        errs: List[str] = []
        for ev in self.events:
            errs.extend(ev.output.get("errors", []))
        return errs

    def final_step_count(self) -> int:
        """The highest step_count seen across all events."""
        counts = [
            ev.step_count_after
            for ev in self.events
            if ev.step_count_after is not None
        ]
        return max(counts, default=0)

    def borderline_candidate_ids(self) -> List[str]:
        """IDs of candidates that had is_borderline=True in their scorecard."""
        return [sc.candidate_id for sc in self.scorecards() if sc.is_borderline]

    def injection_detected_ids(self) -> List[str]:
        """IDs of candidates where is_injection_detected=True."""
        return [p.candidate_id for p in self.parsed_profiles() if p.is_injection_detected]


# ---------------------------------------------------------------------------
# Capture function
# ---------------------------------------------------------------------------


def _merge_output(merged: Dict[str, Any], output: Dict[str, Any]) -> None:
    """Fold a node output dict into the merged state.

    Lists are accumulated (mirrors operator.add reducer behaviour).
    Scalars are overwritten.
    """
    for key, value in output.items():
        if isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = merged[key] + value
        else:
            merged[key] = value


def capture_trace(
    state: RecruitmentState,
    graph=None,
) -> TraceCapture:
    """Run the recruitment graph on `state` and capture the full execution trace.

    Args:
        state:  A RecruitmentState dict ready for graph.stream().
        graph:  Optional pre-built compiled graph. If None, one is built
                automatically. Pass a cached graph from fixtures for speed.

    Returns:
        A TraceCapture with all events, the merged final state, and
        timing information. Never raises — runtime errors are stored in
        TraceCapture.runtime_error.
    """
    if graph is None:
        graph = build_recruitment_graph()

    capture = TraceCapture(human_approved=state.get("human_approved", False))
    start = time.perf_counter()

    try:
        for step_index, event in enumerate(graph.stream(state)):
            for node_name, node_output in event.items():
                if not isinstance(node_output, dict):
                    node_output = {}

                step_count_after = node_output.get("step_count")

                node_event = NodeEvent(
                    node_name=node_name,
                    output=node_output,
                    step_index=step_index,
                    step_count_after=step_count_after,
                )
                capture.events.append(node_event)
                capture.node_sequence.append(node_name)
                _merge_output(capture.final_state, node_output)

                logger.debug(
                    "Captured node '%s' at step %d (step_count=%s)",
                    node_name,
                    step_index,
                    step_count_after,
                )

    except Exception as exc:
        capture.runtime_error = f"{type(exc).__name__}: {exc}"
        logger.error("Graph execution raised an error: %s", capture.runtime_error)

    capture.elapsed_seconds = time.perf_counter() - start
    logger.info(
        "Trace captured: %d nodes in %.3fs — %s",
        len(capture.events),
        capture.elapsed_seconds,
        capture.node_sequence,
    )
    return capture
