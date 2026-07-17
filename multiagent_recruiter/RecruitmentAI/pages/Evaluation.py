"""
Evaluation Page
---------------
DeepEval integration for measuring agent quality.
Shows Faithfulness, Answer Relevancy, Task Completion metrics.
Falls back gracefully if DeepEval not configured.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from database.db import get_all_runs
import plotly.graph_objects as go


def _gauge(value: float, title: str, dark: bool) -> go.Figure:
    paper = "#1a1a2e" if dark else "#ffffff"
    font_color = "#e2e8f0" if dark else "#0f172a"
    color = "#22c55e" if value >= 0.75 else "#f59e0b" if value >= 0.5 else "#ef4444"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(value * 100, 1),
            number={"suffix": "%", "font": {"color": font_color}},
            title={"text": title, "font": {"color": font_color, "size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": font_color},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(239,68,68,0.15)"},
                    {"range": [50, 75], "color": "rgba(245,158,11,0.15)"},
                    {"range": [75, 100], "color": "rgba(34,197,94,0.15)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=paper,
        font_color=font_color,
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def _compute_heuristic_metrics(runs: list) -> dict:
    """
    Compute proxy metrics from DB data when DeepEval isn't configured.
    These are heuristic approximations, not LLM-graded.
    """
    if not runs:
        return {"faithfulness": 0.0, "relevancy": 0.0, "task_completion": 0.0, "overall": 0.0}

    # Faithfulness proxy: % of runs with non-empty explanation (not hallucinated)
    faithful = sum(1 for r in runs if len(r.get("explanation", "")) > 30) / len(runs)

    # Relevancy proxy: % of runs with >0 skills extracted (analyst did its job)
    relevant = sum(1 for r in runs if len(r.get("skills", [])) > 0) / len(runs)

    # Task completion proxy: % of runs with valid recommendation
    valid_recs = {"Interview", "Hold", "Reject", "Need Human Review"}
    completed = sum(1 for r in runs if r.get("recommendation") in valid_recs) / len(runs)

    overall = (faithful + relevant + completed) / 3
    return {
        "faithfulness": faithful,
        "relevancy": relevant,
        "task_completion": completed,
        "overall": overall,
    }


def show():
    dark = st.session_state.get("dark_mode", True)

    st.markdown(
        "<h1 class='gradient-header' style='font-size:2rem; margin-bottom:0.2rem;'>📈 Evaluation</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:var(--text-secondary);'>Agent quality metrics and DeepEval integration.</p>",
        unsafe_allow_html=True,
    )

    runs = get_all_runs()

    if not runs:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:3rem;">
                <div style="font-size:3rem;">📊</div>
                <h3 style="color:var(--text-secondary);">No evaluation data yet</h3>
                <p style="color:var(--text-secondary);">Complete at least one recruitment run first.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    metrics = _compute_heuristic_metrics(runs)

    # ── DeepEval integration attempt ──────────────────────────────────────
    deepeval_available = False
    deepeval_results = {}
    try:
        from deepeval import evaluate  # type: ignore
        from deepeval.metrics import (  # type: ignore
            FaithfulnessMetric,
            AnswerRelevancyMetric,
        )
        from deepeval.test_case import LLMTestCase  # type: ignore
        deepeval_available = True
    except ImportError:
        pass

    if deepeval_available:
        st.success("✅ DeepEval is installed. Click below to run LLM-graded evaluation.")
        if st.button("🧪 Run DeepEval Evaluation", key="run_deepeval"):
            with st.spinner("Running DeepEval metrics..."):
                try:
                    # Build test cases from recent runs
                    test_cases = []
                    for run in runs[:5]:
                        tc = LLMTestCase(
                            input=run.get("explanation", "")[:200],
                            actual_output=run.get("recommendation", ""),
                            expected_output="Interview" if run.get("overall_score", 0) >= 75 else "Reject",
                            retrieval_context=[run.get("explanation", "")],
                        )
                        test_cases.append(tc)

                    faithfulness_metric = FaithfulnessMetric(threshold=0.5, async_mode=False)
                    relevancy_metric = AnswerRelevancyMetric(threshold=0.5, async_mode=False)

                    results_f = []
                    results_r = []
                    for tc in test_cases:
                        faithfulness_metric.measure(tc)
                        relevancy_metric.measure(tc)
                        results_f.append(faithfulness_metric.score)
                        results_r.append(relevancy_metric.score)

                    deepeval_results = {
                        "faithfulness": sum(results_f) / len(results_f) if results_f else 0,
                        "relevancy": sum(results_r) / len(results_r) if results_r else 0,
                    }
                    metrics["faithfulness"] = deepeval_results["faithfulness"]
                    metrics["relevancy"] = deepeval_results["relevancy"]
                    metrics["overall"] = (metrics["faithfulness"] + metrics["relevancy"] + metrics["task_completion"]) / 3
                    st.success("DeepEval evaluation complete!")
                except Exception as exc:
                    st.warning(f"DeepEval evaluation failed: {exc}. Showing heuristic metrics.")
    else:
        st.info("💡 DeepEval not configured. Showing heuristic proxy metrics. Install deepeval and set DEEPEVAL_API_KEY to enable LLM-graded evaluation.")

    # ── Metric gauges ─────────────────────────────────────────────────────
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Agent Quality Metrics")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.plotly_chart(_gauge(metrics["faithfulness"], "Faithfulness", dark), use_container_width=True)
    with g2:
        st.plotly_chart(_gauge(metrics["relevancy"], "Answer Relevancy", dark), use_container_width=True)
    with g3:
        st.plotly_chart(_gauge(metrics["task_completion"], "Task Completion", dark), use_container_width=True)
    with g4:
        st.plotly_chart(_gauge(metrics["overall"], "Overall Agent Score", dark), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Progress bars ─────────────────────────────────────────────────────
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📈 Metric Progress")
    metric_defs = [
        ("🔍 Faithfulness", metrics["faithfulness"],
         "Measures if agent explanations are grounded in actual resume content"),
        ("🎯 Answer Relevancy", metrics["relevancy"],
         "Measures if recommendations are relevant to the job description"),
        ("✅ Task Completion", metrics["task_completion"],
         "Measures if all candidates received a valid final decision"),
        ("⭐ Overall Score", metrics["overall"],
         "Weighted average of all metrics"),
    ]
    for label, val, desc in metric_defs:
        col_l, col_p = st.columns([1, 3])
        with col_l:
            color = "#22c55e" if val >= 0.75 else "#f59e0b" if val >= 0.5 else "#ef4444"
            st.markdown(
                f"<div style='padding-top:0.5rem;'><strong>{label}</strong><br>"
                f"<span style='font-size:0.75rem; color:var(--text-secondary);'>{desc}</span></div>",
                unsafe_allow_html=True,
            )
        with col_p:
            st.markdown(
                f"""
                <div style="margin-top:0.8rem;">
                    <div style="background:var(--glass-border); border-radius:999px; height:10px; overflow:hidden;">
                        <div style="width:{val*100:.1f}%; height:100%; background:{color};
                                    border-radius:999px; transition:width 0.6s;"></div>
                    </div>
                    <div style="text-align:right; font-size:0.8rem; color:{color}; margin-top:2px;">
                        {val*100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Run summary table ─────────────────────────────────────────────────
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📋 Evaluation Summary")
    import pandas as pd
    df = pd.DataFrame(runs)[["candidate_name", "overall_score", "recommendation", "timestamp"]].head(20)
    df.columns = ["Candidate", "Score", "Recommendation", "Timestamp"]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.strftime("%b %d %Y %H:%M")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
