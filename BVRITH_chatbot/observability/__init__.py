"""
observability/ — Production monitoring and observability for the BVRIT RAG Chatbot.

Modules:
    llm_logger    — JSONL-persisted LLM call logger (latency, tokens, cost, success/failure)
    session_stats — Per-session aggregate stats for the Streamlit sidebar
    alerts        — Threshold-based alert engine and input length validator
    ab_testing    — A/B prompt-variant manager (prompt_v1 vs prompt_v2)
    log_analyzer  — Offline log analysis: anomaly detection + root-cause suggestions
"""

from observability.llm_logger import LLMLogger
from observability.session_stats import SessionStats
from observability.alerts import AlertEngine
from observability.ab_testing import ABTestManager
from observability.log_analyzer import LogAnalyzer

__all__ = [
    "LLMLogger",
    "SessionStats",
    "AlertEngine",
    "ABTestManager",
    "LogAnalyzer",
]
