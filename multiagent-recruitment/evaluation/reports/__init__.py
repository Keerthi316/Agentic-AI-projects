"""
Report generation package for the Multi-Agent Recruitment System.

Aggregates results from all evaluation layers (trace, tool-call, output,
fairness, human-gate, red-team) into a structured EvaluationReport.

Supports JSON and plain-text output. Rich console output used when
the `rich` package is installed.
"""

from .generator import (
    ReportGenerator,
    EvaluationReport,
    TraceLayerSummary,
    ToolCallLayerSummary,
    OutputLayerSummary,
    FairnessLayerSummary,
    HumanGateLayerSummary,
    RedTeamLayerSummary,
    TaskSummaryRow,
)

__all__ = [
    "ReportGenerator",
    "EvaluationReport",
    "TraceLayerSummary",
    "ToolCallLayerSummary",
    "OutputLayerSummary",
    "FairnessLayerSummary",
    "HumanGateLayerSummary",
    "RedTeamLayerSummary",
    "TaskSummaryRow",
]
