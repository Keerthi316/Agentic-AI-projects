"""
Human Approval page — approve or reject candidates before scheduling.
"""

import streamlit as st
import pandas as pd


def show():
    """Render the Human Approval page."""
    st.markdown('<p class="main-header">👤 Human Approval</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Review and approve candidates before scheduling interviews</p>', unsafe_allow_html=True)

    state = st.session_state.workflow_state
    shortlist = state.get("shortlist", []) or []

    entries = []
    for sl in shortlist:
        if hasattr(sl, "model_dump"):
            d = sl.model_dump()
        elif isinstance(sl, dict):
            d = sl
        else:
            continue
        entries.append(d)

    if not entries:
        if state.get("scorecards"):
            st.warning("⚠️ Shortlist not yet generated. Run the workflow first.")
            if st.button("▶️ Run Workflow Now", use_container_width=True):
                from utils.backend import run_full_workflow
                with st.spinner("🔄 Running workflow..."):
                    st.session_state.workflow_state = run_full_workflow(st.session_state.workflow_state)
                    st.rerun()
        else:
            st.info("📂 No shortlist available yet. Upload resumes and run the workflow first.")
        return

    # Sort by rank
    entries.sort(key=lambda x: x.get("rank", 999))

    # Show approval status
    if state.get("human_approved", False):
        st.success("✅ **Human approval has already been granted.** Scheduling will proceed.")
        st.markdown("---")
        st.info("To modify the approval, you can reset it below.")

        col_r1, col_r2 = st.columns([1, 3])
        with col_r1:
            if st.button("🔄 Reset Approval", use_container_width=True, type="secondary"):
                st.session_state.workflow_state["human_approved"] = False
                st.session_state.human_approved = False
                st.rerun()
        with col_r2:
            if st.button("📅 Go to Interview Scheduler", use_container_width=True):
                st.session_state.page = "Interview Scheduler"
                st.rerun()
        return

    # Approval summary
    shortlisted_candidates = [e for e in entries if e.get("status") == "shortlisted"]
    hold_candidates = [e for e in entries if e.get("status") == "hold"]
    rejected_candidates = [e for e in entries if e.get("status") == "rejected"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ To Approve", len(shortlisted_candidates))
    with col2:
        st.metric("⏳ On Hold", len(hold_candidates))
    with col3:
        st.metric("❌ Rejected", len(rejected_candidates))

    st.divider()

    # Display candidates for approval
    st.markdown("### 📋 Candidates Awaiting Approval")

    for e in entries:
        name = e.get("name", "Unknown")
        rank = e.get("rank", 0)
        score = e.get("final_score", 0)
        status = e.get("status", "pending")

        with st.container(border=True):
            st.markdown(f"**#{rank} — {name}** — Score: {score:.1f}")
            st.caption(f"Recommendation: {status.upper()}")

            if status == "shortlisted":
                st.markdown("✅ **Recommended for interview**")
            elif status == "hold":
                st.markdown("⏳ **On hold — may be reconsidered**")
            else:
                st.markdown("❌ **Not recommended**")

    st.divider()

    # Approval action
    st.markdown("### 🚀 Approval Decision")

    st.info(
        f"**{len(shortlisted_candidates)} candidate(s)** are recommended for interview scheduling. "
        "Review their profiles and scores before approving."
    )

    col_a1, col_a2 = st.columns([1, 3])

    with col_a1:
        if st.button(
            "✅ Approve All Shortlisted",
            use_container_width=True,
            type="primary",
            disabled=len(shortlisted_candidates) == 0,
        ):
            st.session_state.workflow_state["human_approved"] = True
            st.session_state.human_approved = True
            st.success("✅ Human approval granted! Proceeding to scheduling.")
            st.rerun()

    with col_a2:
        st.caption(
            "Approval triggers the Scheduler agent to generate interview invitations "
            "for all shortlisted candidates."
        )

    # Alternative: approve specific candidates
    st.divider()
    st.markdown("#### 🎯 Select Specific Candidates to Approve")

    selected_ids = []
    for e in entries:
        if e.get("status") == "shortlisted":
            name = e.get("name", "Unknown")
            cid = e.get("candidate_id", "")
            selected = st.checkbox(f"✅ {name} ({cid})", value=True, key=f"approve_{cid}")
            if selected:
                selected_ids.append(cid)

    if selected_ids:
        if st.button("✅ Approve Selected", use_container_width=True, type="primary"):
            st.session_state.workflow_state["human_approved"] = True
            st.session_state.human_approved = True
            st.session_state.approved_ids = selected_ids
            st.success(f"✅ {len(selected_ids)} candidate(s) approved! Proceeding to scheduling.")
            st.rerun()

    # Escalation notice
    if state.get("needs_human_escalation", False):
        st.divider()
        st.error("🚨 **Human escalation required.**")
        st.warning(
            "The system has exhausted all retry attempts. Manual review is needed "
            "to resolve outstanding issues before continuing."
        )