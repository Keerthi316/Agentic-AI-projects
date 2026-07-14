"""
observability/alerts.py — Threshold Alert Engine & Input Validator

Raises structured alerts when:
  - Latency exceeds 10 seconds per call
  - Estimated cost per call exceeds $0.10
  - Session error rate exceeds 5 %

Rejects inputs that exceed 2 000 characters and logs the event.

Alerts are:
  1. Logged to the same JSONL log file used by LLMLogger.
  2. Returned to the caller so the Streamlit UI can display them.
"""

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
LATENCY_THRESHOLD_S: float = 10.0     # seconds
COST_THRESHOLD_USD: float = 0.10      # USD per single call
ERROR_RATE_THRESHOLD: float = 0.05    # 5 % of queries in session
MAX_INPUT_CHARS: int = 2_000          # characters in user input

# Alert log (separate file so it's easy to tail)
LOG_DIR = Path("observability_logs")
ALERT_LOG_FILE = LOG_DIR / "alerts.jsonl"

_write_lock = threading.Lock()


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Alert:
    """A single threshold violation event."""
    alert_type: str           # "latency" | "cost" | "error_rate" | "input_rejected"
    severity: str             # "warning" | "critical"
    message: str              # Human-readable description
    value: float              # The observed value that triggered the alert
    threshold: float          # The threshold that was exceeded
    timestamp: str = ""       # ISO-8601 UTC, set automatically on creation

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class AlertEngine:
    """
    Evaluates observability metrics against defined thresholds and
    emits structured Alert objects.

    Usage::

        engine = AlertEngine()

        # After every LLM call:
        alerts = engine.check_call(latency=12.3, cost=0.002)

        # After updating session stats:
        alerts += engine.check_session(error_rate=0.08)

        # Before processing user input:
        ok, alert = engine.validate_input(user_text)
        if not ok:
            st.error(alert.message)
    """

    def __init__(self, log_file: Optional[Path] = None) -> None:
        self._log_file = Path(log_file) if log_file else ALERT_LOG_FILE
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Threshold checks ──────────────────────────────────────────────────────

    def check_call(self, latency: float, cost: float) -> List[Alert]:
        """
        Check per-call thresholds (latency and cost).

        Args:
            latency: Call latency in seconds.
            cost:    Estimated cost for this call in USD.

        Returns:
            List of Alert objects (may be empty).
        """
        alerts: List[Alert] = []

        if latency > LATENCY_THRESHOLD_S:
            alert = Alert(
                alert_type="latency",
                severity="critical",
                message=(
                    f"Latency {latency:.2f}s exceeded threshold "
                    f"{LATENCY_THRESHOLD_S}s."
                ),
                value=latency,
                threshold=LATENCY_THRESHOLD_S,
            )
            alerts.append(alert)
            self._persist(alert)
            logger.warning(alert.message)

        if cost > COST_THRESHOLD_USD:
            alert = Alert(
                alert_type="cost",
                severity="warning",
                message=(
                    f"Per-call cost ${cost:.6f} exceeded threshold "
                    f"${COST_THRESHOLD_USD:.2f}."
                ),
                value=cost,
                threshold=COST_THRESHOLD_USD,
            )
            alerts.append(alert)
            self._persist(alert)
            logger.warning(alert.message)

        return alerts

    def check_session(self, error_rate: float) -> List[Alert]:
        """
        Check session-level error rate threshold.

        Args:
            error_rate: Current session error rate (0–1 fraction).

        Returns:
            List of Alert objects (may be empty).
        """
        alerts: List[Alert] = []

        if error_rate > ERROR_RATE_THRESHOLD:
            alert = Alert(
                alert_type="error_rate",
                severity="critical",
                message=(
                    f"Session error rate {error_rate:.1%} exceeded threshold "
                    f"{ERROR_RATE_THRESHOLD:.1%}."
                ),
                value=error_rate,
                threshold=ERROR_RATE_THRESHOLD,
            )
            alerts.append(alert)
            self._persist(alert)
            logger.warning(alert.message)

        return alerts

    def validate_input(self, user_input: str) -> tuple[bool, Optional[Alert]]:
        """
        Reject inputs that exceed MAX_INPUT_CHARS characters.

        Args:
            user_input: The raw text submitted by the user.

        Returns:
            Tuple of (is_valid: bool, alert: Optional[Alert]).
            ``is_valid`` is False when the input is rejected.
        """
        if len(user_input) > MAX_INPUT_CHARS:
            alert = Alert(
                alert_type="input_rejected",
                severity="warning",
                message=(
                    f"Input rejected: {len(user_input)} characters exceeds "
                    f"the {MAX_INPUT_CHARS}-character limit."
                ),
                value=float(len(user_input)),
                threshold=float(MAX_INPUT_CHARS),
            )
            self._persist(alert)
            logger.warning(alert.message)
            return False, alert

        return True, None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _persist(self, alert: Alert) -> None:
        """Append an alert record to the JSONL alert log (thread-safe)."""
        record = asdict(alert)
        with _write_lock:
            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.error(f"AlertEngine: could not write alert log: {exc}")

    def read_recent_alerts(self, n: int = 20) -> List[dict]:
        """
        Read the most recent *n* alerts from the alert log.

        Returns:
            List of alert dicts (most recent last).
        """
        records = []
        if not self._log_file.exists():
            return records
        with open(self._log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records[-n:]


# Module-level singleton
alert_engine = AlertEngine()
