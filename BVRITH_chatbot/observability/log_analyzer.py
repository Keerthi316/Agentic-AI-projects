"""
observability/log_analyzer.py — Production Log Analyzer

Reads the JSONL call log produced by LLMLogger and:
  1. Computes overall statistics
  2. Detects anomalies (latency spikes, cost spikes, error bursts,
     refusal bursts, low confidence)
  3. Suggests root causes and remediation steps for each anomaly
  4. Returns a structured report dict ready to render in Streamlit
     or print to the console

Run standalone::

    python -m observability.log_analyzer

Or import and call from code::

    from observability.log_analyzer import LogAnalyzer
    report = LogAnalyzer().analyze()
    print(report["summary"])
"""

import json
import logging
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LOG_FILE = Path("observability_logs") / "llm_calls.jsonl"

# ── Anomaly thresholds ────────────────────────────────────────────────────────
LATENCY_SPIKE_MULTIPLIER = 2.5   # flag if latency > 2.5× median
COST_SPIKE_MULTIPLIER = 3.0      # flag if cost > 3× median
ERROR_BURST_WINDOW = 5           # consecutive calls
ERROR_BURST_THRESHOLD = 0.6      # 60 % errors in that window
REFUSAL_RATE_THRESHOLD = 0.30    # flag if >30 % of calls are refusals
LOW_CONFIDENCE_THRESHOLD = 0.35  # flag if avg confidence < 0.35


class LogAnalyzer:
    """
    Analyzes the JSONL LLM call log for anomalies and root causes.

    Attributes:
        log_file: Path to the JSONL log file (defaults to module constant).
    """

    def __init__(self, log_file: Optional[Path] = None) -> None:
        self._log_file = Path(log_file) if log_file else LOG_FILE

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        """
        Run the full analysis pipeline.

        Returns:
            Dict with keys:
              - total_records (int)
              - summary (dict)          — aggregate statistics
              - anomalies (list[dict])  — detected anomalies
              - suggestions (list[str]) — deduplicated root-cause suggestions
              - version_comparison (dict) — stats broken down by prompt version
        """
        records = self._load_records()
        if not records:
            return {
                "total_records": 0,
                "summary": {},
                "anomalies": [],
                "suggestions": ["No log records found. Run the chatbot first."],
                "version_comparison": {},
            }

        summary = self._compute_summary(records)
        anomalies = self._detect_anomalies(records, summary)
        suggestions = self._generate_suggestions(anomalies, summary)
        version_comparison = self._compare_versions(records)

        return {
            "total_records": len(records),
            "summary": summary,
            "anomalies": anomalies,
            "suggestions": suggestions,
            "version_comparison": version_comparison,
        }

    # ── Helpers: loading ──────────────────────────────────────────────────────

    def _load_records(self) -> List[Dict[str, Any]]:
        """Load and parse all JSONL records."""
        records: List[Dict[str, Any]] = []
        if not self._log_file.exists():
            logger.warning(f"Log file not found: {self._log_file}")
            return records
        with open(self._log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    # ── Helpers: summary ─────────────────────────────────────────────────────

    def _compute_summary(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate statistics across all records."""
        latencies = [r.get("latency_seconds", 0) for r in records if r.get("latency_seconds")]
        costs = [r.get("estimated_cost_usd", 0) for r in records if r.get("estimated_cost_usd")]
        tokens = [r.get("total_tokens", 0) for r in records]
        successes = [r.get("success", True) for r in records]
        confidences = [r.get("confidence", 0) for r in records if r.get("confidence")]
        refusals = [r.get("refusal", False) for r in records]

        n = len(records)

        def safe_median(lst: list) -> float:
            return round(statistics.median(lst), 4) if lst else 0.0

        def safe_mean(lst: list) -> float:
            return round(statistics.mean(lst), 4) if lst else 0.0

        def p95(lst: list) -> float:
            if not lst:
                return 0.0
            s = sorted(lst)
            idx = max(0, math.ceil(0.95 * len(s)) - 1)
            return round(s[idx], 4)

        return {
            "total_calls": n,
            "success_count": sum(successes),
            "error_count": n - sum(successes),
            "error_rate": round((n - sum(successes)) / n, 4) if n else 0,
            "avg_latency_s": safe_mean(latencies),
            "median_latency_s": safe_median(latencies),
            "p95_latency_s": p95(latencies),
            "max_latency_s": round(max(latencies), 3) if latencies else 0,
            "avg_cost_usd": safe_mean(costs),
            "total_cost_usd": round(sum(costs), 6),
            "avg_tokens": safe_mean(tokens),
            "total_tokens": sum(tokens),
            "avg_confidence": safe_mean(confidences),
            "refusal_rate": round(sum(refusals) / n, 4) if n else 0,
        }

    # ── Helpers: anomaly detection ────────────────────────────────────────────

    def _detect_anomalies(
        self,
        records: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Detect anomalous records and patterns."""
        anomalies: List[Dict[str, Any]] = []

        med_lat = summary["median_latency_s"]
        med_cost = summary["avg_cost_usd"]  # use avg as baseline for cost

        for i, r in enumerate(records):
            lat = r.get("latency_seconds", 0)
            cost = r.get("estimated_cost_usd", 0)
            ts = r.get("timestamp", "")

            # Latency spike
            if med_lat > 0 and lat > LATENCY_SPIKE_MULTIPLIER * med_lat:
                anomalies.append(
                    {
                        "type": "latency_spike",
                        "index": i,
                        "timestamp": ts,
                        "value": lat,
                        "baseline": med_lat,
                        "detail": f"Latency {lat:.2f}s is {lat/med_lat:.1f}× the median ({med_lat:.2f}s).",
                    }
                )

            # Cost spike
            if med_cost > 0 and cost > COST_SPIKE_MULTIPLIER * med_cost:
                anomalies.append(
                    {
                        "type": "cost_spike",
                        "index": i,
                        "timestamp": ts,
                        "value": cost,
                        "baseline": med_cost,
                        "detail": f"Cost ${cost:.6f} is {cost/med_cost:.1f}× the average (${med_cost:.6f}).",
                    }
                )

            # Individual call failure
            if not r.get("success", True):
                anomalies.append(
                    {
                        "type": "call_failure",
                        "index": i,
                        "timestamp": ts,
                        "value": 1,
                        "baseline": 0,
                        "detail": f"Call failed: {r.get('error_message', 'unknown error')}",
                    }
                )

        # Error burst detection
        for start in range(len(records) - ERROR_BURST_WINDOW + 1):
            window = records[start : start + ERROR_BURST_WINDOW]
            errors_in_window = sum(1 for r in window if not r.get("success", True))
            if errors_in_window / ERROR_BURST_WINDOW >= ERROR_BURST_THRESHOLD:
                anomalies.append(
                    {
                        "type": "error_burst",
                        "index": start,
                        "timestamp": window[0].get("timestamp", ""),
                        "value": errors_in_window,
                        "baseline": ERROR_BURST_WINDOW * ERROR_BURST_THRESHOLD,
                        "detail": (
                            f"{errors_in_window}/{ERROR_BURST_WINDOW} consecutive calls failed "
                            f"starting at record #{start}."
                        ),
                    }
                )

        # High refusal rate
        if summary["refusal_rate"] > REFUSAL_RATE_THRESHOLD:
            anomalies.append(
                {
                    "type": "high_refusal_rate",
                    "index": -1,
                    "timestamp": "",
                    "value": summary["refusal_rate"],
                    "baseline": REFUSAL_RATE_THRESHOLD,
                    "detail": (
                        f"Refusal rate {summary['refusal_rate']:.1%} exceeds threshold "
                        f"{REFUSAL_RATE_THRESHOLD:.0%}. Possibly over-constrained prompt."
                    ),
                }
            )

        # Low average confidence
        if summary["avg_confidence"] > 0 and summary["avg_confidence"] < LOW_CONFIDENCE_THRESHOLD:
            anomalies.append(
                {
                    "type": "low_confidence",
                    "index": -1,
                    "timestamp": "",
                    "value": summary["avg_confidence"],
                    "baseline": LOW_CONFIDENCE_THRESHOLD,
                    "detail": (
                        f"Average confidence {summary['avg_confidence']:.2%} is below "
                        f"threshold {LOW_CONFIDENCE_THRESHOLD:.0%}. KB coverage may be poor."
                    ),
                }
            )

        return anomalies

    # ── Helpers: root-cause suggestions ──────────────────────────────────────

    def _generate_suggestions(
        self,
        anomalies: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> List[str]:
        """Map anomaly types to actionable root-cause suggestions."""
        seen: set = set()
        suggestions: List[str] = []

        def add(s: str) -> None:
            if s not in seen:
                seen.add(s)
                suggestions.append(s)

        for a in anomalies:
            atype = a["type"]

            if atype == "latency_spike":
                add("⏱️ Latency spike detected. Possible causes: LLM provider overload, large context window, slow network. Consider reducing TOP_K or enabling streaming.")
            elif atype == "cost_spike":
                add("💸 Cost spike detected. Check for unusually long user inputs or verbose LLM responses. Consider max_tokens capping or input truncation.")
            elif atype == "call_failure":
                add("❌ Call failures found. Verify OPENROUTER_API_KEY is valid, check API rate limits, and inspect error messages in the log.")
            elif atype == "error_burst":
                add("🔴 Error burst: multiple consecutive failures. This could indicate a network outage, invalid API key, or a quota exhaustion event. Check OpenRouter status.")
            elif atype == "high_refusal_rate":
                add("🚫 High refusal rate: the LLM is frequently declining to answer. Consider loosening the system prompt (switch from v2 to v1) or expanding the knowledge base.")
            elif atype == "low_confidence":
                add("📉 Low retrieval confidence: queries are not matching well in the vector store. Consider re-chunking the KB document or improving query expansion.")

        if not suggestions:
            add("✅ No significant anomalies detected. The system is operating within normal parameters.")

        # General recommendations based on summary
        if summary.get("error_rate", 0) > 0.05:
            add("⚠️ Overall error rate exceeds 5%. Investigate API connectivity and model availability.")
        if summary.get("avg_latency_s", 0) > 5:
            add("⚠️ Average latency is above 5s. Consider model caching or switching to a faster model.")

        return suggestions

    # ── Helpers: version comparison ───────────────────────────────────────────

    def _compare_versions(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Break down key metrics by prompt version (v1 vs v2)."""
        buckets: Dict[str, List[Dict]] = {}
        for r in records:
            v = r.get("prompt_version", "v1")
            buckets.setdefault(v, []).append(r)

        result: Dict[str, Any] = {}
        for v, recs in buckets.items():
            lats = [r.get("latency_seconds", 0) for r in recs]
            costs = [r.get("estimated_cost_usd", 0) for r in recs]
            refusals = sum(1 for r in recs if r.get("refusal", False))
            errors = sum(1 for r in recs if not r.get("success", True))
            cits = [len(r.get("citations", [])) for r in recs]
            n = len(recs) or 1

            result[v] = {
                "calls": len(recs),
                "avg_latency_s": round(sum(lats) / n, 3),
                "avg_cost_usd": round(sum(costs) / n, 6),
                "refusal_rate": round(refusals / n, 4),
                "avg_citations": round(sum(cits) / n, 2),
                "error_rate": round(errors / n, 4),
            }
        return result


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint
    report = LogAnalyzer().analyze()
    print(f"\n=== Log Analysis Report ({report['total_records']} records) ===\n")
    print("── Summary ──")
    pprint.pprint(report["summary"])
    print("\n── Anomalies ──")
    for a in report["anomalies"]:
        print(f"  [{a['type']}] {a['detail']}")
    print("\n── Suggestions ──")
    for s in report["suggestions"]:
        print(f"  {s}")
    if report["version_comparison"]:
        print("\n── A/B Version Comparison ──")
        pprint.pprint(report["version_comparison"])
