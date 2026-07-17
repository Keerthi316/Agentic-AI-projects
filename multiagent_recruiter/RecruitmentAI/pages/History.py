"""
History Page
------------
Search, view, and delete historical recruitment runs.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st

from database.db import get_all_runs, get_run_by_id, delete_run
from utils.helpers import recommendation_icon, recommendation_color, score_color, format_list


def show():
    dark = st.session_state.get("dark_mode", True)

    st.markdown(
        "<h1 class='gradient-header' style='font-size:2rem; margin-bottom:0.2rem;'>📜 History</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:var(--text-secondary);'>Browse, search, and manage previous recruitment runs.</p>",
        unsafe_allow_html=True,
    )

    runs = get_all_runs()

    if not runs:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:3rem;">
                <div style="font-size:3rem;">📭</div>
                <h3 style="color:var(--text-secondary);">No history yet</h3>
                <p style="color:var(--text-secondary);">Complete recruitment runs will appear here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    df = pd.DataFrame(runs)

    # ── Search / filter ────────────────────────────────────────────────────
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    fcol1, fcol2, fcol3 = st.columns([2, 1, 1])
    with fcol1:
        search = st.text_input("🔍 Search by candidate name or filename", placeholder="e.g. John Doe")
    with fcol2:
        rec_filter = st.selectbox(
            "Filter by recommendation",
            ["All", "Interview", "Hold", "Reject", "Need Human Review"],
        )
    with fcol3:
        min_score = st.slider("Min score", 0, 100, 0)
    st.markdown("</div>", unsafe_allow_html=True)

    # Apply filters
    filtered = df.copy()
    if search.strip():
        mask = (
            filtered["candidate_name"].str.contains(search, case=False, na=False)
            | filtered["filename"].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]
    if rec_filter != "All":
        filtered = filtered[filtered["recommendation"] == rec_filter]
    filtered = filtered[filtered["overall_score"] >= min_score]

    st.markdown(
        f"<p style='color:var(--text-secondary); margin:0.5rem 0;'>Showing {len(filtered)} of {len(df)} records</p>",
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.info("No records match your filters.")
        return

    # ── Run cards ──────────────────────────────────────────────────────────
    for _, row in filtered.iterrows():
        run_id = row["run_id"]
        name = row["candidate_name"]
        rec = row["recommendation"]
        score = row["overall_score"]
        filename = row["filename"]
        timestamp = row["timestamp"]
        explanation = row.get("explanation", "")
        approved = row.get("approved")
        final_status = row.get("final_status")
        finalized_at = row.get("finalized_at")

        rec_color = recommendation_color(rec)
        rec_icon = recommendation_icon(rec)
        sc = score_color(score)
        ts_fmt = pd.to_datetime(timestamp).strftime("%b %d %Y, %H:%M") if timestamp else "—"

        approved_badge = ""
        if final_status == "Interview Finalized":
            approved_badge = "<span style='background:#22c55e22; color:#22c55e; padding:0.2rem 0.6rem; border-radius:999px; font-size:0.75rem;'>✅ Interview Finalized</span>"
        elif final_status == "Rejected by Human Reviewer":
            approved_badge = "<span style='background:#ef444422; color:#ef4444; padding:0.2rem 0.6rem; border-radius:999px; font-size:0.75rem;'>❌ Rejected by Reviewer</span>"
        elif approved is not None:
            if approved == 1 or approved is True:
                approved_badge = "<span style='background:#22c55e22; color:#22c55e; padding:0.2rem 0.6rem; border-radius:999px; font-size:0.75rem;'>✅ Approved</span>"
            else:
                approved_badge = "<span style='background:#ef444422; color:#ef4444; padding:0.2rem 0.6rem; border-radius:999px; font-size:0.75rem;'>❌ Overridden</span>"

        with st.expander(
            f"{rec_icon} {name}  —  {score:.0f}/100  —  {ts_fmt}", expanded=False
        ):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(
                    f"""
                    <div style="margin-bottom:0.5rem;">
                        <strong>File:</strong> {filename} &nbsp;|&nbsp;
                        <strong>Score:</strong> <span style="color:{sc};">{score:.1f}</span> &nbsp;|&nbsp;
                        <strong>Decision:</strong> <span style="color:{rec_color};">{rec_icon} {rec}</span>
                        &nbsp; {approved_badge}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("**💬 Explanation**")
                st.write(explanation[:400] + "…" if len(explanation) > 400 else explanation)

                skills = row.get("skills", [])
                if skills:
                    st.markdown(f"**🛠️ Skills:** {format_list(skills, 10)}")

                exp = row.get("experience", [])
                if exp:
                    st.markdown("**💼 Experience:**")
                    for e in exp[:2]:
                        st.markdown(f"- {e}")

                # Score breakdown
                sb = row.get("score_breakdown", {})
                if sb:
                    st.markdown("**📊 Score Breakdown:**")
                    sb_cols = st.columns(5)
                    dims = [
                        ("Technical", sb.get("technical_score", "—")),
                        ("Experience", sb.get("experience_score", "—")),
                        ("Education", sb.get("education_score", "—")),
                        ("Projects", sb.get("projects_score", "—")),
                        ("Comm.", sb.get("communication_score", "—")),
                    ]
                    for sb_col, (dim, val) in zip(sb_cols, dims):
                        sb_col.metric(dim, f"{val:.0f}" if isinstance(val, float) else val)

            with c2:
                st.markdown(f"**Run ID:**")
                st.code(run_id[:8], language=None)
                st.markdown(f"**Date:** {ts_fmt}")
                if finalized_at:
                    fin_fmt = pd.to_datetime(finalized_at).strftime("%b %d %Y, %H:%M")
                    st.markdown(f"**Finalized:** {fin_fmt}")
                if st.button("🗑️ Delete", key=f"del_{run_id}"):
                    delete_run(run_id)
                    st.success("Deleted.")
                    st.rerun()

    # ── Export ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("**📥 Export History**")
    export_df = filtered[["candidate_name", "filename", "overall_score", "recommendation", "timestamp", "explanation"]].copy()
    csv = export_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name="recruitment_history.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)
