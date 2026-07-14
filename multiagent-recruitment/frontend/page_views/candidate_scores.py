"""
Candidate Scores page — sortable scorecards with overall and category scores,
highlighting borderline candidates.
"""

import streamlit as st
import pandas as pd
from components.status_badge import score_badge


def show():
    """Render the Candidate Scores page."""
    st.markdown('<p class="main-header">📊 Candidate Scores</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Scoring results against the job description</p>', unsafe_allow_html=True)

    state = st.session_state.workflow_state
    scorecards = state.get("scorecards", []) or []
    profiles = state.get("parsed_profiles", []) or []

    # Build name lookup
    name_map = {}
    for p in profiles:
        cid = p.candidate_id if hasattr(p, "candidate_id") else p.get("candidate_id", "")
        name = p.name if hasattr(p, "name") else p.get("name", "Unknown")
        name_map[cid] = name

    # Convert scorecards to dicts
    scorecard_dicts = []
    for sc in scorecards:
        if hasattr(sc, "model_dump"):
            d = sc.model_dump()
        elif isinstance(sc, dict):
            d = sc
        else:
            continue
        d["name"] = name_map.get(d.get("candidate_id", ""), "Unknown")
        scorecard_dicts.append(d)

    if not scorecard_dicts:
        if state.get("parsed_profiles"):
            st.warning("⚠️ Candidates parsed but not yet scored. Run the workflow to score them.")
            if st.button("▶️ Run Workflow Now", use_container_width=True):
                from utils.backend import run_full_workflow
                with st.spinner("🔄 Running workflow..."):
                    st.session_state.workflow_state = run_full_workflow(st.session_state.workflow_state)
                    st.rerun()
        else:
            st.info("📂 No candidates scored yet. Upload resumes and run the workflow first.")
        return

    # Summary metrics
    borderline_count = sum(1 for d in scorecard_dicts if d.get("is_borderline", False))
    avg_score = sum(d.get("total_score", 0) for d in scorecard_dicts) / len(scorecard_dicts) if scorecard_dicts else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Scored", len(scorecard_dicts))
    with col2:
        st.metric("📈 Avg Score", f"{avg_score:.1f}")
    with col3:
        st.metric("⚠️ Borderline", borderline_count)
    with col4:
        high_count = sum(1 for d in scorecard_dicts if d.get("total_score", 0) >= 75)
        st.metric("🏆 High Confidence", high_count)

    st.divider()

    # Sort controls
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        sort_by = st.selectbox(
            "Sort by",
            ["total_score", "skill_score", "experience_score", "education_score", "name"],
            format_func=lambda x: {
                "total_score": "Overall Score",
                "skill_score": "Skill Score",
                "experience_score": "Experience Score",
                "education_score": "Education Score",
                "name": "Name",
            }.get(x, x),
        )
    with col_s2:
        sort_asc = st.checkbox("Ascending", value=False)

    # Sort
    reverse = not sort_asc
    if sort_by == "name":
        scorecard_dicts.sort(key=lambda x: x.get("name", ""), reverse=reverse)
    else:
        scorecard_dicts.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)

    # Display as table
    st.markdown("### 📋 Scorecard Table")

    # Build DataFrame
    rows = []
    for d in scorecard_dicts:
        rows.append({
            "Name": d.get("name", "Unknown"),
            "Overall": d.get("total_score", 0),
            "Skills": d.get("skill_score", 0),
            "Experience": d.get("experience_score", 0),
            "Education": d.get("education_score", 0),
            "Borderline": "⚠️ Yes" if d.get("is_borderline", False) else "✅ No",
        })

    df = pd.DataFrame(rows)

    # Color-code the dataframe
    def highlight_scores(val):
        if isinstance(val, (int, float)):
            if val >= 80:
                return "background-color: #d4edda; color: #155724"
            elif val >= 60:
                return "background-color: #fff3cd; color: #856404"
            elif val >= 40:
                return "background-color: #ffeeba; color: #856404"
            else:
                return "background-color: #f8d7da; color: #721c24"
        return ""

    styled_df = df.style.applymap(highlight_scores, subset=["Overall", "Skills", "Experience", "Education"])
    st.dataframe(styled_df, use_container_width=True, height=min(400, 50 * len(rows) + 50))

    # Detailed view per candidate
    st.divider()
    st.markdown("### 🔍 Detailed Score Breakdown")

    for d in scorecard_dicts:
        name = d.get("name", "Unknown")
        cid = d.get("candidate_id", "N/A")
        is_borderline = d.get("is_borderline", False)
        reasoning = d.get("reasoning", "")

        with st.expander(f"{'⚠️' if is_borderline else '✅'} {name} — {d.get('total_score', 0):.1f}/100", expanded=is_borderline):
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Overall", f"{d.get('total_score', 0):.1f}")
            with col_b:
                st.metric("Skills", f"{d.get('skill_score', 0):.1f}")
            with col_c:
                st.metric("Experience", f"{d.get('experience_score', 0):.1f}")
            with col_d:
                st.metric("Education", f"{d.get('education_score', 0):.1f}")

            if is_borderline:
                st.warning("⚠️ This candidate is **borderline** and will be sent for verification.")

            if reasoning:
                st.markdown("**Reasoning:**")
                st.caption(reasoning)