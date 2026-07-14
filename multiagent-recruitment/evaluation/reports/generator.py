"""
Report Generator for the Multi-Agent Recruitment System Evaluation Framework.

Aggregates results from all evaluation layers into a structured
EvaluationReport and serialises it to JSON and plain-text formats.

Layers aggregated:
  Layer 1 — Trace Evaluation       (TraceValidationResult per task)
  Layer 2 — Tool-Call Evaluation   (ToolCallReport)
  Layer 3a — DeepEval Output       (list[MetricResult] per test case)
  Layer 3b — Fairness              (FairnessReport)
  Layer 3c — Human Gate            (HumanGateReport)
  Red Team — Giskard Scan          (GiskardScanResult)

Design decisions:
- EvaluationReport is a plain dataclass (not Pydantic) so it can hold
  arbitrary result objects without a rigid schema.
- JSON serialisation uses a custom encoder for dataclasses and Enums.
- The overall pass/fail verdict requires ALL layers to pass.
- Rich console output uses simple ASCII tables when rich is not installed.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional rich import
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None

# ---------------------------------------------------------------------------
# Layer result containers
# ---------------------------------------------------------------------------


@dataclass
class TraceLayerSummary:
    total_tasks: int
    passed_tasks: int
    total_invariants: int
    passed_invariants: int
    critical_failures: int
    pass_rate: float

    @property
    def passed(self) -> bool:
        return self.pass_rate >= 0.85 and self.critical_failures == 0


@dataclass
class ToolCallLayerSummary:
    total_tasks: int
    passed_tasks: int
    overall_accuracy: float
    pass_rate: float

    @property
    def passed(self) -> bool:
        return self.overall_accuracy >= 0.85


@dataclass
class OutputLayerSummary:
    total_test_cases: int
    passed_test_cases: int
    pass_rate: float
    deepeval_available: bool

    @property
    def passed(self) -> bool:
        return self.pass_rate >= 0.80


@dataclass
class FairnessLayerSummary:
    total_tasks: int
    passed_tasks: int
    overall_fairness_score: float

    @property
    def passed(self) -> bool:
        return self.overall_fairness_score >= 0.90


@dataclass
class HumanGateLayerSummary:
    total_tasks: int
    passed_tasks: int
    critical_failures: int
    pass_rate: float

    @property
    def passed(self) -> bool:
        return self.critical_failures == 0 and self.pass_rate >= 0.95


@dataclass
class RedTeamLayerSummary:
    total_findings: int
    critical_findings: int
    high_findings: int
    used_giskard: bool

    @property
    def passed(self) -> bool:
        return self.critical_findings == 0


@dataclass
class TaskSummaryRow:
    task_id: str
    category: str
    trace_pass: bool
    tool_call_pass: bool
    output_pass: bool
    fairness_pass: bool
    gate_pass: bool


# ---------------------------------------------------------------------------
# Main report object
# ---------------------------------------------------------------------------


@dataclass
class EvaluationReport:
    """Complete evaluation report for one run of the evaluation pipeline."""

    run_id: str
    timestamp: str
    demo_mode: bool

    # Per-layer summaries
    trace: Optional[TraceLayerSummary] = None
    tool_call: Optional[ToolCallLayerSummary] = None
    output: Optional[OutputLayerSummary] = None
    fairness: Optional[FairnessLayerSummary] = None
    human_gate: Optional[HumanGateLayerSummary] = None
    red_team: Optional[RedTeamLayerSummary] = None

    # Per-task breakdown (one row per task)
    task_rows: List[TaskSummaryRow] = field(default_factory=list)

    # Raw detail strings for the text report
    details: List[str] = field(default_factory=list)

    @property
    def overall_passed(self) -> bool:
        """True only if all available layers passed."""
        checks = []
        if self.trace:
            checks.append(self.trace.passed)
        if self.tool_call:
            checks.append(self.tool_call.passed)
        if self.output:
            checks.append(self.output.passed)
        if self.fairness:
            checks.append(self.fairness.passed)
        if self.human_gate:
            checks.append(self.human_gate.passed)
        if self.red_team:
            checks.append(self.red_team.passed)
        return all(checks) if checks else False

    def verdict(self) -> str:
        return "PASS" if self.overall_passed else "FAIL"


# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------


class _ReportEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _build_trace_summary(trace_results: list) -> TraceLayerSummary:
    """Build TraceLayerSummary from a list of TraceValidationResult objects."""
    total_tasks = len(trace_results)
    passed_tasks = sum(1 for r in trace_results if r.passed)
    total_inv = sum(len(r.invariant_results) for r in trace_results)
    passed_inv = sum(
        sum(1 for inv in r.invariant_results if inv.passed)
        for r in trace_results
    )
    critical = sum(len(r.critical_failures) for r in trace_results)
    rate = passed_inv / total_inv if total_inv else 0.0
    return TraceLayerSummary(
        total_tasks=total_tasks,
        passed_tasks=passed_tasks,
        total_invariants=total_inv,
        passed_invariants=passed_inv,
        critical_failures=critical,
        pass_rate=rate,
    )


def _build_tool_call_summary(tool_report) -> ToolCallLayerSummary:
    return ToolCallLayerSummary(
        total_tasks=len(tool_report.results),
        passed_tasks=sum(1 for r in tool_report.results if r.passed),
        overall_accuracy=tool_report.overall_accuracy,
        pass_rate=tool_report.pass_rate,
    )


def _build_output_summary(
    total_cases: int, passed_cases: int, deepeval_available: bool
) -> OutputLayerSummary:
    rate = passed_cases / total_cases if total_cases else 1.0
    return OutputLayerSummary(
        total_test_cases=total_cases,
        passed_test_cases=passed_cases,
        pass_rate=rate,
        deepeval_available=deepeval_available,
    )


def _build_fairness_summary(fairness_report) -> FairnessLayerSummary:
    return FairnessLayerSummary(
        total_tasks=len(fairness_report.results),
        passed_tasks=sum(1 for r in fairness_report.results if r.passed),
        overall_fairness_score=fairness_report.overall_fairness_score,
    )


def _build_gate_summary(gate_report) -> HumanGateLayerSummary:
    return HumanGateLayerSummary(
        total_tasks=len(gate_report.results),
        passed_tasks=sum(1 for r in gate_report.results if r.passed),
        critical_failures=len(gate_report.critical_failures),
        pass_rate=gate_report.pass_rate,
    )


def _build_red_team_summary(scan_result) -> RedTeamLayerSummary:
    return RedTeamLayerSummary(
        total_findings=scan_result.total_findings,
        critical_findings=len(scan_result.critical_findings),
        high_findings=len(scan_result.high_findings),
        used_giskard=scan_result.used_giskard,
    )


# ---------------------------------------------------------------------------
# Main ReportGenerator class
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Aggregates evaluation results and generates structured reports.

    Usage:
        gen = ReportGenerator()
        gen.add_trace_results(trace_results)
        gen.add_tool_call_report(tool_report)
        gen.add_output_results(total, passed)
        gen.add_fairness_report(fairness_report)
        gen.add_human_gate_report(gate_report)
        gen.add_red_team_results(scan_result)
        report = gen.build()
        gen.print_console(report)
        gen.save_json(report, "evaluation/reports/report.json")
        gen.save_text(report, "evaluation/reports/report.txt")
    """

    def __init__(self, run_id: Optional[str] = None, demo_mode: bool = True):
        self._run_id = run_id or _generate_run_id()
        self._demo_mode = demo_mode
        self._trace_results: list = []
        self._tool_report: Any = None
        self._output_total: int = 0
        self._output_passed: int = 0
        self._deepeval_available: bool = False
        self._fairness_report: Any = None
        self._gate_report: Any = None
        self._scan_result: Any = None
        self._task_rows: List[TaskSummaryRow] = []
        self._details: List[str] = []

    def add_trace_results(self, results: list) -> "ReportGenerator":
        self._trace_results = results
        return self

    def add_tool_call_report(self, report: Any) -> "ReportGenerator":
        self._tool_report = report
        return self

    def add_output_results(
        self, total: int, passed: int, deepeval_available: bool = False
    ) -> "ReportGenerator":
        self._output_total = total
        self._output_passed = passed
        self._deepeval_available = deepeval_available
        return self

    def add_fairness_report(self, report: Any) -> "ReportGenerator":
        self._fairness_report = report
        return self

    def add_human_gate_report(self, report: Any) -> "ReportGenerator":
        self._gate_report = report
        return self

    def add_red_team_results(self, scan_result: Any) -> "ReportGenerator":
        self._scan_result = scan_result
        return self

    def add_task_rows(self, rows: List[TaskSummaryRow]) -> "ReportGenerator":
        self._task_rows = rows
        return self

    def add_detail(self, line: str) -> "ReportGenerator":
        self._details.append(line)
        return self

    def build(self) -> EvaluationReport:
        """Build the EvaluationReport from accumulated data."""
        report = EvaluationReport(
            run_id=self._run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            demo_mode=self._demo_mode,
            task_rows=self._task_rows,
            details=self._details,
        )

        if self._trace_results:
            report.trace = _build_trace_summary(self._trace_results)

        if self._tool_report is not None:
            report.tool_call = _build_tool_call_summary(self._tool_report)

        if self._output_total > 0 or self._output_passed > 0:
            report.output = _build_output_summary(
                self._output_total, self._output_passed, self._deepeval_available
            )

        if self._fairness_report is not None:
            report.fairness = _build_fairness_summary(self._fairness_report)

        if self._gate_report is not None:
            report.human_gate = _build_gate_summary(self._gate_report)

        if self._scan_result is not None:
            report.red_team = _build_red_team_summary(self._scan_result)

        return report

    # ----------------------------------------------------------------
    # Output methods
    # ----------------------------------------------------------------

    def print_console(self, report: EvaluationReport) -> None:
        """Print a formatted evaluation report to the console."""
        if _RICH:
            self._print_rich(report)
        else:
            self._print_plain(report)

    def _print_rich(self, report: EvaluationReport) -> None:
        """Print using Rich tables."""
        from rich.panel import Panel
        from rich.text import Text

        verdict_color = "green" if report.overall_passed else "red"
        _console.print(
            Panel(
                Text(f"  Evaluation Report — {report.verdict()}  ", style=f"bold {verdict_color}"),
                subtitle=f"Run: {report.run_id} | {report.timestamp[:19]} UTC | demo={report.demo_mode}",
            )
        )

        # Layer summary table
        tbl = Table(title="Evaluation Layers", box=box.SIMPLE_HEAVY, show_lines=True)
        tbl.add_column("Layer", style="cyan", no_wrap=True)
        tbl.add_column("Metric", justify="right")
        tbl.add_column("Status", justify="center")

        def _status(b: bool) -> str:
            return "[green]PASS[/green]" if b else "[red]FAIL[/red]"

        if report.trace:
            t = report.trace
            tbl.add_row(
                "Trace Invariants",
                f"{t.passed_invariants}/{t.total_invariants} ({t.pass_rate:.0%})",
                _status(t.passed),
            )
        if report.tool_call:
            tc = report.tool_call
            tbl.add_row(
                "Tool-Call Accuracy",
                f"{tc.overall_accuracy:.1%}",
                _status(tc.passed),
            )
        if report.output:
            op = report.output
            mode = "DeepEval" if op.deepeval_available else "Structural"
            tbl.add_row(
                f"Output Quality ({mode})",
                f"{op.passed_test_cases}/{op.total_test_cases} ({op.pass_rate:.0%})",
                _status(op.passed),
            )
        if report.fairness:
            f = report.fairness
            tbl.add_row(
                "Fairness (Name-Swap)",
                f"{f.overall_fairness_score:.1%}",
                _status(f.passed),
            )
        if report.human_gate:
            g = report.human_gate
            tbl.add_row(
                "Human Gate",
                f"{g.passed_tasks}/{g.total_tasks} tasks, {g.critical_failures} critical",
                _status(g.passed),
            )
        if report.red_team:
            r = report.red_team
            mode = "Giskard" if r.used_giskard else "Structural"
            tbl.add_row(
                f"Red Team ({mode})",
                f"{r.total_findings} findings ({r.critical_findings} critical)",
                _status(r.passed),
            )

        _console.print(tbl)

        # Task breakdown
        if report.task_rows:
            task_tbl = Table(title="Per-Task Breakdown", box=box.MINIMAL_DOUBLE_HEAD)
            task_tbl.add_column("Task ID", style="dim")
            task_tbl.add_column("Category")
            task_tbl.add_column("Trace", justify="center")
            task_tbl.add_column("Tool-Call", justify="center")
            task_tbl.add_column("Output", justify="center")
            task_tbl.add_column("Fairness", justify="center")
            task_tbl.add_column("Gate", justify="center")

            for row in report.task_rows:
                task_tbl.add_row(
                    row.task_id,
                    row.category,
                    _status(row.trace_pass),
                    _status(row.tool_call_pass),
                    _status(row.output_pass),
                    _status(row.fairness_pass),
                    _status(row.gate_pass),
                )
            _console.print(task_tbl)

        # Overall verdict
        _console.print(
            f"\n[bold {verdict_color}]Overall Verdict: {report.verdict()}[/bold {verdict_color}]"
        )

    def _print_plain(self, report: EvaluationReport) -> None:
        """Print plain-text report (no Rich)."""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  EVALUATION REPORT — {report.verdict()}")
        print(f"  Run: {report.run_id}")
        print(f"  Time: {report.timestamp[:19]} UTC")
        print(f"  Demo Mode: {report.demo_mode}")
        print(sep)

        def _s(b: bool) -> str:
            return "PASS" if b else "FAIL"

        if report.trace:
            t = report.trace
            print(f"  Trace Invariants:      {t.passed_invariants}/{t.total_invariants} "
                  f"({t.pass_rate:.0%}) critical={t.critical_failures} [{_s(t.passed)}]")
        if report.tool_call:
            tc = report.tool_call
            print(f"  Tool-Call Accuracy:    {tc.overall_accuracy:.1%}            [{_s(tc.passed)}]")
        if report.output:
            op = report.output
            mode = "DeepEval" if op.deepeval_available else "Structural"
            print(f"  Output Quality({mode}): {op.passed_test_cases}/{op.total_test_cases} "
                  f"({op.pass_rate:.0%})          [{_s(op.passed)}]")
        if report.fairness:
            f = report.fairness
            print(f"  Fairness (Name-Swap):  {f.overall_fairness_score:.1%}            [{_s(f.passed)}]")
        if report.human_gate:
            g = report.human_gate
            print(f"  Human Gate:            {g.passed_tasks}/{g.total_tasks} tasks "
                  f"({g.critical_failures} critical)    [{_s(g.passed)}]")
        if report.red_team:
            r = report.red_team
            mode = "Giskard" if r.used_giskard else "Structural"
            print(f"  Red Team ({mode}):    {r.total_findings} findings "
                  f"({r.critical_findings} critical) [{_s(r.passed)}]")

        print(sep)
        print(f"  OVERALL: {report.verdict()}")
        print(f"{sep}\n")

    def save_json(self, report: EvaluationReport, path: str) -> None:
        """Save the report as JSON to the given path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, cls=_ReportEncoder)
        logger.info("JSON report saved to %s", out)

    def save_text(self, report: EvaluationReport, path: str) -> None:
        """Save a plain-text report to the given path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "=" * 60,
            f"EVALUATION REPORT — {report.verdict()}",
            f"Run: {report.run_id}",
            f"Timestamp: {report.timestamp}",
            f"Demo Mode: {report.demo_mode}",
            "=" * 60,
        ]

        def _s(b: bool) -> str:
            return "PASS" if b else "FAIL"

        if report.trace:
            t = report.trace
            lines.append(f"[{_s(t.passed)}] Trace Invariants: "
                         f"{t.passed_invariants}/{t.total_invariants} ({t.pass_rate:.0%}), "
                         f"critical_failures={t.critical_failures}")
        if report.tool_call:
            tc = report.tool_call
            lines.append(f"[{_s(tc.passed)}] Tool-Call Accuracy: {tc.overall_accuracy:.1%}")
        if report.output:
            op = report.output
            lines.append(f"[{_s(op.passed)}] Output Quality: "
                         f"{op.passed_test_cases}/{op.total_test_cases} ({op.pass_rate:.0%})")
        if report.fairness:
            f = report.fairness
            lines.append(f"[{_s(f.passed)}] Fairness: {f.overall_fairness_score:.1%}")
        if report.human_gate:
            g = report.human_gate
            lines.append(f"[{_s(g.passed)}] Human Gate: "
                         f"{g.passed_tasks}/{g.total_tasks}, critical={g.critical_failures}")
        if report.red_team:
            r = report.red_team
            lines.append(f"[{_s(r.passed)}] Red Team: "
                         f"{r.total_findings} findings, critical={r.critical_findings}")

        if report.task_rows:
            lines.append("-" * 60)
            lines.append("Per-Task Breakdown:")
            header = f"{'Task ID':<12} {'Category':<22} {'Trace':^5} {'ToolCall':^8} {'Output':^6} {'Fair':^4} {'Gate':^4}"
            lines.append(header)
            lines.append("-" * len(header))
            for row in report.task_rows:
                lines.append(
                    f"{row.task_id:<12} {row.category:<22} "
                    f"{'P' if row.trace_pass else 'F':^5} "
                    f"{'P' if row.tool_call_pass else 'F':^8} "
                    f"{'P' if row.output_pass else 'F':^6} "
                    f"{'P' if row.fairness_pass else 'F':^4} "
                    f"{'P' if row.gate_pass else 'F':^4}"
                )

        if report.details:
            lines.append("-" * 60)
            lines.append("Details:")
            lines.extend(report.details)

        lines.append("=" * 60)
        lines.append(f"OVERALL VERDICT: {report.verdict()}")
        lines.append("=" * 60)

        with out.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("Text report saved to %s", out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_run_id() -> str:
    """Generate a short unique run ID."""
    import uuid
    return "run-" + str(uuid.uuid4())[:8]
