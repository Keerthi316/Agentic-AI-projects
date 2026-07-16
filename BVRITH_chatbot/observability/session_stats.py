"""
observability/session_stats.py — Session-Level Statistics Tracker

Tracks running statistics for a single Streamlit user session:
  - Total queries answered
  - Average latency
  - P95 latency (95th-percentile)
  - Total estimated cost (USD)
  - Total tokens consumed
  - Error count

All state is kept in memory (no persistence needed for session-level data).
This object is stored in st.session_state so it survives Streamlit reruns.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class SessionStats:
    """
    Accumulates per-session LLM call statistics.

    Designed to be instantiated once per Streamlit session and stored
    in ``st.session_state.session_stats``.

    Attributes:
        total_queries:   Number of successful + failed LLM calls.
        total_tokens:    Cumulative token count (input + output).
        total_cost_usd:  Cumulative estimated cost in USD.
        error_count:     Number of failed LLM calls.
        latencies:       Ordered list of per-call latency values (seconds).
                         Used to compute average and P95.
    """

    total_queries: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    error_count: int = 0
    latencies: List[float] = field(default_factory=list)

    # ── Update ─────────────────────────────────────────────────────────────

    def record(
        self,
        latency: float,
        tokens: int,
        cost: float,
        success: bool,
    ) -> None:
        """
        Record a completed LLM call.

        Args:
            latency: Call latency in seconds.
            tokens:  Total tokens (input + output) for this call.
            cost:    Estimated cost in USD for this call.
            success: Whether the call completed without error.
        """
        self.total_queries += 1
        self.total_tokens += tokens
        self.total_cost_usd += cost
        self.latencies.append(latency)
        if not success:
            self.error_count += 1

    # ── Computed properties ────────────────────────────────────────────────

    @property
    def avg_latency(self) -> float:
        """Average latency across all calls (seconds)."""
        if not self.latencies:
            return 0.0
        return round(sum(self.latencies) / len(self.latencies), 3)

    @property
    def p95_latency(self) -> float:
        """
        95th-percentile latency (seconds).

        Uses the nearest-rank method so it works without numpy.
        """
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        # Nearest-rank: ceil(95/100 * N) → 1-indexed
        idx = math.ceil(0.95 * len(sorted_lat)) - 1
        idx = max(0, min(idx, len(sorted_lat) - 1))
        return round(sorted_lat[idx], 3)

    @property
    def error_rate(self) -> float:
        """
        Error rate as a fraction (0–1).

        Returns 0 when no queries have been made.
        """
        if self.total_queries == 0:
            return 0.0
        return round(self.error_count / self.total_queries, 4)

    @property
    def success_count(self) -> int:
        """Number of successful calls."""
        return max(0, self.total_queries - self.error_count)

    # ── Display helpers ────────────────────────────────────────────────────

    def as_dict(self) -> dict:
        """
        Serialise current stats to a plain dict for display / logging.

        Returns:
            Dict with all computed + raw stats.
        """
        return {
            "total_queries": self.total_queries,
            "avg_latency_s": self.avg_latency,
            "p95_latency_s": self.p95_latency,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "error_count": self.error_count,
            "error_rate_pct": round(self.error_rate * 100, 2),
        }

    def reset(self) -> None:
        """Reset all counters (useful for "Clear Stats" button)."""
        self.total_queries = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.error_count = 0
        self.latencies = []
        logger.info("SessionStats reset.")
