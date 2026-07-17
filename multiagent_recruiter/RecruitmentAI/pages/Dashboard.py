"""
Dashboard Page
--------------
KPI cards, charts, and recent activity.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database.db import get_all_runs, get_stats


def show():
    dark = st.session_state.get("dark_mode", True)
    bg = "#0f0f1a" if dark else "#f8fafc"
    paper = "#1a1a2e" if dark else "#ffffff"
    font_color = "#e2e8f0" if dark else "#0f172a"
    plotly_template = "plotly_dark" if dark else "plotly_white"

    st.markdown(
        "<h1 class='gradient-header' style='font-size:2rem; margin-bottom:0.2rem;'>📊 Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:var(--text-secondary); margin-top:0;'>Overview of all recruitment runs and metrics</p>",
        unsafe_allow_html=True,
    )

    stats = get_stats()
    runs = get_all_runs()

    # ── KPI Metrics ────────────────────────────────────────────────────────
    cols = st.columns(4)
    kpis = [
        ("👥", "Total Candidates", stats["total"], "#6366f1"),
        ("✅", "Interviews", stats["interview"], "#22c55e"),
        ("❌", "Rejected", stats["rejected"], "#ef4444"),
        ("⏳", "Pending / Hold", stats["pending"], "#f59e0b"),
    ]
    for col, (icon, label, value, color) in zip(cols, kpis):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:1.8rem;">{icon}</div>
                    <div class="metric-value" style="color:{color};">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    if not runs:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:3rem;">
                <div style="font-size:3rem;">🚀</div>
                <h3 style="color:var(--text-secondary);">No runs yet</h3>
                <p style="color:var(--text-secondary);">Head to the <strong>Recruitment</strong> page to evaluate your first candidate.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    df = pd.DataFrame(runs)

    # ── Row 1: Score distribution + Recommendation breakdown ──────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("**📈 Score Distribution**")
        fig = px.histogram(
            df,
            x="overall_score",
            nbins=20,
            color_discrete_sequence=["#6366f1"],
            template=plotly_template,
            labels={"overall_score": "Overall Score", "count": "Candidates"},
        )
        fig.update_layout(
            paper_bgcolor=paper,
            plot_bgcolor=paper,
            font_color=font_color,
            margin=dict(l=0, r=0, t=20, b=0),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("**🎯 Recommendation Breakdown**")
        rec_counts = df["recommendation"].value_counts().reset_index()
        rec_counts.columns = ["Recommendation", "Count"]
        color_map = {
            "Interview": "#22c55e",
            "Hold": "#f59e0b",
            "Reject": "#ef4444",
            "Need Human Review": "#3b82f6",
        }
        fig2 = px.pie(
            rec_counts,
            names="Recommendation",
            values="Count",
            color="Recommendation",
            color_discrete_map=color_map,
            template=plotly_template,
        )
        fig2.update_layout(
            paper_bgcolor=paper,
            font_color=font_color,
            margin=dict(l=0, r=0, t=20, b=0),
            height=280,
            legend=dict(orientation="v", x=1.0),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 2: Candidate Ranking ───────────────────────────────────────────
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("**🏆 Candidate Ranking**")
    top = df.sort_values("overall_score", ascending=False).head(15)
    fig3 = px.bar(
        top,
        x="candidate_name",
        y="overall_score",
        color="recommendation",
        color_discrete_map=color_map,
        template=plotly_template,
        labels={"overall_score": "Score", "candidate_name": "Candidate"},
    )
    fig3.update_layout(
        paper_bgcolor=paper,
        plot_bgcolor=paper,
        font_color=font_color,
        margin=dict(l=0, r=0, t=20, b=60),
        height=300,
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 3: Skills frequency + Timeline ────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("**🛠️ Top Skills**")
        all_skills = []
        for row in runs:
            all_skills.extend(row.get("skills", []))
        if all_skills:
            skill_series = pd.Series(all_skills).value_counts().head(15)
            fig4 = px.bar(
                x=skill_series.values,
                y=skill_series.index,
                orientation="h",
                color=skill_series.values,
                color_continuous_scale="Viridis",
                template=plotly_template,
                labels={"x": "Count", "y": "Skill"},
            )
            fig4.update_layout(
                paper_bgcolor=paper,
                plot_bgcolor=paper,
                font_color=font_color,
                margin=dict(l=0, r=0, t=20, b=0),
                height=300,
                showlegend=False,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No skills data yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("**🕐 Recent Activity**")
        recent = df.head(10)[["candidate_name", "recommendation", "overall_score", "timestamp"]].copy()
        recent["timestamp"] = pd.to_datetime(recent["timestamp"]).dt.strftime("%b %d, %H:%M")
        for _, row in recent.iterrows():
            rec = row["recommendation"]
            icon = {"Interview": "✅", "Hold": "⏸️", "Reject": "❌"}.get(rec, "🔍")
            color = {"Interview": "#22c55e", "Hold": "#f59e0b", "Reject": "#ef4444"}.get(rec, "#3b82f6")
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                            padding:0.4rem 0; border-bottom:1px solid var(--glass-border);">
                    <span>{icon} <strong>{row['candidate_name']}</strong></span>
                    <span style="color:{color}; font-weight:600;">{row['overall_score']:.0f}</span>
                    <span style="font-size:0.75rem; color:var(--text-secondary);">{row['timestamp']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
