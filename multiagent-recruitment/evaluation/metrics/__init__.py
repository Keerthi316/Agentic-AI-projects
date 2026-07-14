"""
Evaluation metrics package.

Provides three custom evaluation metrics for the Multi-Agent Recruitment System:

  Layer 2 — Tool-Call Accuracy:
    ToolCallMetrics  — compares actual vs expected node execution sequences,
                       validates Pydantic arguments.

  Layer 3 — Fairness:
    FairnessMetric   — name-swap test to detect demographic bias in rankings.

  Layer 2 — Human Gate:
    HumanGateMetric  — verifies scheduler never runs without human approval.
"""

from .tool_call import ToolCallMetrics, ToolCallResult, ToolCallReport
from .fairness import FairnessMetric, FairnessResult, FairnessReport, NameSwapVariant
from .human_gate import HumanGateMetric, HumanGateResult, HumanGateReport, HumanGateCheckResult

__all__ = [
    # Tool-call
    "ToolCallMetrics",
    "ToolCallResult",
    "ToolCallReport",
    # Fairness
    "FairnessMetric",
    "FairnessResult",
    "FairnessReport",
    "NameSwapVariant",
    # Human gate
    "HumanGateMetric",
    "HumanGateResult",
    "HumanGateReport",
    "HumanGateCheckResult",
]
