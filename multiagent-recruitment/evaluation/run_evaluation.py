"""
Full evaluation pipeline entry point for the Multi-Agent Recruitment System.

Runs all evaluation layers in sequence:
  1. Trace Evaluation      — capture and validate LangGraph execution traces
  2. Tool-Call Evaluation  — verify node sequences and Pydantic argument validity
  3. Output Evaluation     — DeepEval (or structural) output quality metrics
  4. Fairness Evaluation   — name-swap bias test
  5. Human Gate Evaluation — verify approval gate invariant
  6. Red Team Scan         — structural vulnerability scan (+ Giskard if available)

Generates:
  - Console report (Rich if installed, plain-text otherwise)
  - evaluation/reports/latest_report.json
  - evaluation/reports/latest_report.txt

Usage:
    python evaluation/run_evaluation.py [options]

Options:
    --tasks TASK_ID,...   Run only the specified task IDs (comma-separated)
    --layers LAYER,...    Run only the specified layers (trace,tool,output,fairness,gate,redteam)
    --no-giskard          Skip Giskard scan even if giskard is installed
    --report-dir PATH     Output directory for reports (default: evaluation/reports)
    --dry-run             Run without saving reports

Exit codes:
    0 — All layers passed
    1 — One or more layers failed
    2 — Unexpected error during evaluation
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Set

# ---------------------------------------------------------------------------
# Path setup — must come before any project imports
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Force demo mode by default (override with RECRUITMENT_DEMO_MODE=false)
if "RECRUITMENT_DEMO_MODE" not in os.environ:
    os.environ["RECRUITMENT_DEMO_MODE"] = "true"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("run_evaluation")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from evaluation.datasets.loader import load_dataset
from evaluation.datasets.schema import EvalDataset, EvalTask, TaskCategory
from evaluation.traces.capture import capture_trace, TraceCapture
from evaluation.traces.validator import TraceValidator, TraceValidationResult
from evaluation.metrics.tool_call import ToolCallMetrics, ToolCallReport
from evaluation.metrics.fairness import FairnessMetric, FairnessReport
from evaluation.metrics.human_gate import HumanGateMetric, HumanGateReport
from evaluation.deepeval_suite.test_cases import build_deepeval_test_cases, DEEPEVAL_AVAILABLE
from evaluation.deepeval_suite.metrics import evaluate_test_case
from evaluation.giskard_suite.scanner import GiskardScanner, GiskardScanResult
from evaluation.reports.generator import (
    ReportGenerator,
    EvaluationReport,
    TaskSummaryRow,
)
from models.state import JDInput, RecruitmentState
from graph.workflow import build_recruitment_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_to_state(task: EvalTask, human_approved: bool = False) -> RecruitmentState:
    jd = JDInput(**task.input.jd.model_dump())
    return RecruitmentState(
        jd=jd,
        candidates=list(task.input.candidates),
        parsed_profiles=[],
        scorecards=[],
        verified_scores=[],
        revision_count=task.input.override_revision_count or 0,
        shortlist=[],
        step_count=0,
        errors=[],
        needs_human_escalation=False,
        human_approved=human_approved,
    )


def _print_progress(msg: str) -> None:
    """Print a progress message to stdout."""
    print(f"  >> {msg}")


# ---------------------------------------------------------------------------
# Layer runners
# ---------------------------------------------------------------------------


def run_trace_layer(
    tasks: List[EvalTask],
    graph,
) -> tuple[list, dict]:
    """Run Layer 1: Trace Evaluation.

    Returns:
        (trace_results, traces_by_id) where trace_results is a list of
        TraceValidationResult and traces_by_id maps task_id → TraceCapture.
    """
    _print_progress(f"Trace Evaluation — {len(tasks)} tasks")
    validator = TraceValidator()
    results = []
    traces: dict = {}

    for task in tasks:
        state = _task_to_state(task, human_approved=task.input.human_approved)
        trace = capture_trace(state, graph=graph)
        traces[task.id] = trace
        result = validator.validate_all(trace)
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    _print_progress(f"  Trace: {passed}/{len(results)} tasks passed")
    return results, traces


def run_tool_call_layer(
    tasks: List[EvalTask],
    traces: dict,
) -> ToolCallReport:
    """Run Layer 2: Tool-Call Evaluation."""
    _print_progress(f"Tool-Call Evaluation — {len(tasks)} tasks")
    metrics = ToolCallMetrics()
    pairs = [(task, traces[task.id]) for task in tasks if task.id in traces]
    report = metrics.evaluate_all(pairs)
    _print_progress(f"  Tool-Call: accuracy={report.overall_accuracy:.1%}")
    return report


def run_output_layer(
    tasks: List[EvalTask],
    traces: dict,
) -> tuple[int, int]:
    """Run Layer 3a: Output Quality Evaluation.

    Returns:
        (total_cases, passed_cases)
    """
    _print_progress(f"Output Quality Evaluation — {len(tasks)} tasks")
    total = 0
    passed = 0

    for task in tasks:
        trace = traces.get(task.id)
        if trace is None:
            continue
        cases = build_deepeval_test_cases(task, trace)
        for case in cases:
            total += 1
            if DEEPEVAL_AVAILABLE:
                results = evaluate_test_case(case)
                if all(r.passed for r in results):
                    passed += 1
            else:
                # Structural quality check
                actual = getattr(case, "actual_output", "")
                if actual and len(actual.strip()) >= 20:
                    passed += 1

    mode = "DeepEval" if DEEPEVAL_AVAILABLE else "structural"
    _print_progress(f"  Output ({mode}): {passed}/{total} test cases passed")
    return total, passed


def run_fairness_layer(
    tasks: List[EvalTask],
    traces: dict,
    graph,
) -> FairnessReport:
    """Run Layer 3b: Fairness Evaluation."""
    _print_progress(f"Fairness Evaluation — {len(tasks)} tasks")
    metric = FairnessMetric(graph=graph, max_swaps_per_task=2)
    pairs = [(task, traces[task.id]) for task in tasks if task.id in traces]
    report = metric.evaluate_all(pairs)
    _print_progress(f"  Fairness: {report.overall_fairness_score:.1%}")
    return report


def run_human_gate_layer(
    tasks: List[EvalTask],
    graph,
) -> HumanGateReport:
    """Run Layer 3c: Human Gate Evaluation."""
    _print_progress(f"Human Gate Evaluation — {len(tasks)} tasks")
    metric = HumanGateMetric(graph=graph)
    report = metric.evaluate_all(tasks)
    _print_progress(
        f"  Human Gate: {report.pass_rate:.1%} pass, "
        f"{len(report.critical_failures)} critical"
    )
    return report


def run_red_team_layer(
    tasks: List[EvalTask],
    graph,
    use_giskard: bool = True,
) -> GiskardScanResult:
    """Run Red Team Evaluation."""
    _print_progress(f"Red Team Scan — {len(tasks)} tasks")
    scanner = GiskardScanner(graph=graph, use_giskard=use_giskard)
    result = scanner.scan(tasks)
    _print_progress(
        f"  Red Team: {result.total_findings} findings "
        f"({len(result.critical_findings)} critical)"
    )
    return result


# ---------------------------------------------------------------------------
# Per-task row builder
# ---------------------------------------------------------------------------


def _build_task_rows(
    tasks: List[EvalTask],
    trace_results: list,
    tool_report: ToolCallReport,
    output_total: int,
    output_passed: int,
    fairness_report: FairnessReport,
    gate_report: HumanGateReport,
    traces: dict,
) -> List[TaskSummaryRow]:
    """Build per-task summary rows for the report."""
    rows = []

    trace_map = {tasks[i].id: trace_results[i] for i in range(min(len(tasks), len(trace_results)))}
    tool_map = {r.task_id: r for r in tool_report.results}
    fair_map = {r.task_id: r for r in fairness_report.results}
    gate_map = {r.task_id: r for r in gate_report.results}

    for task in tasks:
        trace_pass = trace_map[task.id].passed if task.id in trace_map else True
        tool_pass = tool_map[task.id].passed if task.id in tool_map else True

        # Output: approximate per-task pass (all cases for this task passed)
        trace = traces.get(task.id)
        if trace is not None and DEEPEVAL_AVAILABLE:
            cases = build_deepeval_test_cases(task, trace)
            output_pass = all(
                all(r.passed for r in evaluate_test_case(c))
                for c in cases
            ) if cases else True
        else:
            output_pass = True  # structural checks done at aggregate level

        fair_pass = fair_map[task.id].passed if task.id in fair_map else True
        gate_pass = gate_map[task.id].passed if task.id in gate_map else True

        rows.append(TaskSummaryRow(
            task_id=task.id,
            category=task.category.value,
            trace_pass=trace_pass,
            tool_call_pass=tool_pass,
            output_pass=output_pass,
            fairness_pass=fair_pass,
            gate_pass=gate_pass,
        ))

    return rows


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_evaluation(
    task_filter: Optional[Set[str]] = None,
    layer_filter: Optional[Set[str]] = None,
    use_giskard: bool = True,
    report_dir: str = "evaluation/reports",
    dry_run: bool = False,
) -> EvaluationReport:
    """Run the complete evaluation pipeline.

    Args:
        task_filter:   Optional set of task IDs to include. None = all tasks.
        layer_filter:  Optional set of layer names to run.
                       Values: "trace", "tool", "output", "fairness", "gate", "redteam"
                       None = all layers.
        use_giskard:   If True, attempt Giskard scan. If False, structural only.
        report_dir:    Directory to write reports to.
        dry_run:       If True, don't write reports to disk.

    Returns:
        EvaluationReport with all layer results.

    Exit codes are set via sys.exit() after the function returns in __main__.
    """
    demo_mode = os.environ.get("RECRUITMENT_DEMO_MODE", "true").lower() == "true"
    ALL_LAYERS = {"trace", "tool", "output", "fairness", "gate", "redteam"}
    active_layers = layer_filter or ALL_LAYERS

    print("\n" + "=" * 60)
    print("  Multi-Agent Recruitment System — Evaluation Pipeline")
    print("=" * 60)
    print(f"  Demo mode:  {demo_mode}")
    print(f"  DeepEval:   {'available' if DEEPEVAL_AVAILABLE else 'not installed'}")
    print(f"  Layers:     {', '.join(sorted(active_layers))}")
    print("=" * 60 + "\n")

    # ── Load dataset ────────────────────────────────────────────
    _print_progress("Loading evaluation dataset…")
    dataset = load_dataset()
    tasks = [t for t in dataset.tasks if task_filter is None or t.id in task_filter]
    _print_progress(f"  {len(tasks)} tasks loaded (dataset v{dataset.version})")

    # ── Build graph once ────────────────────────────────────────
    _print_progress("Building LangGraph workflow…")
    t0 = time.perf_counter()
    graph = build_recruitment_graph()
    _print_progress(f"  Graph built in {time.perf_counter() - t0:.2f}s")

    # ── Run layers ──────────────────────────────────────────────
    gen = ReportGenerator(demo_mode=demo_mode)
    traces: dict = {}
    trace_results: list = []
    tool_report = ToolCallReport()
    fairness_report = FairnessReport()
    gate_report = HumanGateReport()

    if "trace" in active_layers:
        print("\n[Layer 1] Trace Evaluation")
        trace_results, traces = run_trace_layer(tasks, graph)
        gen.add_trace_results(trace_results)

    if "tool" in active_layers:
        print("\n[Layer 2] Tool-Call Evaluation")
        if not traces:
            # Need traces — run them now
            _print_progress("Running traces for tool-call evaluation…")
            for task in tasks:
                state = _task_to_state(task, human_approved=task.input.human_approved)
                traces[task.id] = capture_trace(state, graph=graph)
        tool_report = run_tool_call_layer(tasks, traces)
        gen.add_tool_call_report(tool_report)

    if "output" in active_layers:
        print("\n[Layer 3a] Output Quality Evaluation")
        if not traces:
            for task in tasks:
                state = _task_to_state(task, human_approved=task.input.human_approved)
                traces[task.id] = capture_trace(state, graph=graph)
        total, passed = run_output_layer(tasks, traces)
        gen.add_output_results(total, passed, DEEPEVAL_AVAILABLE)

    if "fairness" in active_layers:
        print("\n[Layer 3b] Fairness Evaluation")
        if not traces:
            for task in tasks:
                state = _task_to_state(task, human_approved=task.input.human_approved)
                traces[task.id] = capture_trace(state, graph=graph)
        fairness_report = run_fairness_layer(tasks, traces, graph)
        gen.add_fairness_report(fairness_report)

    if "gate" in active_layers:
        print("\n[Layer 3c] Human Gate Evaluation")
        gate_report = run_human_gate_layer(tasks, graph)
        gen.add_human_gate_report(gate_report)

    if "redteam" in active_layers:
        print("\n[Red Team] Vulnerability Scan")
        scan_result = run_red_team_layer(tasks, graph, use_giskard=use_giskard)
        gen.add_red_team_results(scan_result)

    # ── Build per-task rows ──────────────────────────────────────
    if traces and trace_results:
        rows = _build_task_rows(
            tasks, trace_results, tool_report,
            0, 0, fairness_report, gate_report, traces,
        )
        gen.add_task_rows(rows)

    # ── Build and print report ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  Building report…")
    report = gen.build()
    gen.print_console(report)

    # ── Save reports ─────────────────────────────────────────────
    if not dry_run:
        rep_dir = Path(report_dir)
        json_path = rep_dir / "latest_report.json"
        txt_path = rep_dir / "latest_report.txt"
        # Also save a timestamped copy
        ts = report.timestamp[:19].replace(":", "-").replace("T", "_")
        ts_json = rep_dir / f"report_{ts}.json"

        gen.save_json(report, str(json_path))
        gen.save_json(report, str(ts_json))
        gen.save_text(report, str(txt_path))
        print(f"\n  Reports saved to {rep_dir}/")

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete evaluation pipeline for the recruitment system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        metavar="TASK_IDS",
        default=None,
        help="Comma-separated task IDs to run (default: all 12 tasks)",
    )
    parser.add_argument(
        "--layers",
        metavar="LAYERS",
        default=None,
        help="Comma-separated layers to run: trace,tool,output,fairness,gate,redteam",
    )
    parser.add_argument(
        "--no-giskard",
        action="store_true",
        default=False,
        help="Skip Giskard scan — use structural checks only",
    )
    parser.add_argument(
        "--report-dir",
        metavar="PATH",
        default="evaluation/reports",
        help="Output directory for reports (default: evaluation/reports)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run evaluation but do not save reports to disk",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    task_filter: Optional[Set[str]] = None
    if args.tasks:
        task_filter = set(t.strip() for t in args.tasks.split(","))

    layer_filter: Optional[Set[str]] = None
    if args.layers:
        layer_filter = set(l.strip() for l in args.layers.split(","))

    try:
        report = run_evaluation(
            task_filter=task_filter,
            layer_filter=layer_filter,
            use_giskard=not args.no_giskard,
            report_dir=args.report_dir,
            dry_run=args.dry_run,
        )
        sys.exit(0 if report.overall_passed else 1)
    except Exception as exc:
        logger.exception("Unexpected error during evaluation: %s", exc)
        sys.exit(2)
