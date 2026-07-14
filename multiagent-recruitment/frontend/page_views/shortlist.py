"""
Shortlist page — ranked candidates with recommendation status and CSV download.
"""

import streamlit as st
import pandas as pd
from io import BytesIO


def show():
    """Render the Shortlist page."""
    st.markdown('<p class="main-header">📋 Shortlist</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ranked candidate shortlist with recommendations</p>', unsafe_allow_html=True)

    state = st.session_state.workflow_state
    shortlist = state.get("shortlist", []) or []
    profiles = state.get("parsed_profiles", []) or []

    # Build name lookup from profiles
    name_map = {}
    for p in profiles:
        cid = p.candidate_id if hasattr(p, "candidate_id") else p.get("candidate_id", "")
        name = p.name if hasattr(p, "name") else p.get("name", "Unknown")
        name_map[cid] = name

    # Convert shortlist entries to dicts
    entries = []
    for sl in shortlist:
        if hasattr(sl, "model_dump"):
            d = sl.model_dump()
        elif isinstance(sl, dict):
            d = sl
        else:
            continue
        # Fill name if missing
        if not d.get("name") or d.get("name") == "Unknown":
            d["name"] = name_map.get(d.get("candidate_id", ""), "Unknown")
        entries.append(d)

    if not entries:
        if state.get("scorecards"):
            st.warning("⚠️ Candidates have been scored but no shortlist generated yet.")
            st.info("Run the workflow to generate the shortlist.")
            if st.button("▶️ Run Workflow Now", use_container_width=True):
                from utils.backend import run_full_workflow
                with st.spinner("🔄 Generating shortlist..."):
                    st.session_state.workflow_state = run_full_workflow(st.session_state.workflow_state)
                    st.rerun()
        else:
            st.info("📂 No shortlist available yet. Upload resumes and run the workflow first.")
        return

    # Summary metrics
    shortlisted = [e for e in entries if e.get("status") == "shortlisted"]
    hold = [e for e in entries if e.get("status") == "hold"]
    rejected = [e for e in entries if e.get("status") == "rejected"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Total", len(entries))
    with col2:
        st.metric("✅ Shortlisted", len(shortlisted))
    with col3:
        st.metric("⏳ Hold", len(hold))
    with col4:
        st.metric("❌ Rejected", len(rejected))

    st.divider()

    # Sort by rank
    entries.sort(key=lambda x: x.get("rank", 999))

    # Display ranked table
    st.markdown("### 🏆 Ranked Candidate List")

    rows = []
    for e in entries:
        status = e.get("status", "pending")
        status_emoji = {"shortlisted": "✅", "hold": "⏳", "rejected": "❌"}.get(status, "❓")
        rows.append({
            "Rank": e.get("rank", 0),
            "Name": e.get("name", "Unknown"),
            "Score": f"{e.get('final_score', 0):.1f}",
            "Status": f"{status_emoji} {status.capitalize()}",
            "Candidate ID": e.get("candidate_id", ""),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df.set_index("Rank"), use_container_width=True)

    # Download CSV
    st.divider()
    col_d1, col_d2 = st.columns([1, 3])

    with col_d1:
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        st.download_button(
            label="📥 Download CSV",
            data=csv_buffer,
            file_name="recruitment_shortlist.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_d2:
        st.info("Download the shortlist as CSV for external processing or record-keeping.")

    # Detailed view
    st.divider()
    st.markdown("### 🔍 Candidate Details by Rank")

    for e in entries:
        rank = e.get("rank", 0)
        name = e.get("name", "Unknown")
        status = e.get("status", "pending")
        score = e.get("final_score", 0)

        status_colors = {"shortlisted": "#00cc66", "hold": "#ffaa00", "rejected": "#ff4444"}
        color = status_colors.get(status, "#888888")

        with st.container(border=True):
            col_r1, col_r2, col_r3 = st.columns([1, 3, 1])
            with col_r1:
                st.markdown(f"<h2 style='color:{color};'>#{rank}</h2>", unsafe_allow_html=True)
            with col_r2:
                st.markdown(f"**{name}**")
                st.caption(f"Status: {status.capitalize()}")
            with col_r3:
                st.markdown(f"<h3 style='color:{color};'>{score:.1f}</h3>", unsafe_allow_html=True)