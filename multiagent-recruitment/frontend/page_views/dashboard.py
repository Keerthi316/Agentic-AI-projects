"""
Dashboard page — project overview, workflow visualization, and system status.
"""

import streamlit as st
from utils.backend import state_to_summary
from components.workflow_viz import render_workflow_diagram


def show():
    """Render the Dashboard page."""
    st.markdown('<p class="main-header">🏠 Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Agent Recruitment System Overview</p>', unsafe_allow_html=True)

    state = st.session_state.workflow_state
    summary = state_to_summary(state)

    # Quick stats row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📄 Job Description", "✅ Set" if summary["has_jd"] else "❌ Missing")
    with col2:
        st.metric("📤 Resumes", summary["candidate_count"])
    with col3:
        st.metric("🔍 Parsed", summary["parsed_count"])
    with col4:
        st.metric("📊 Scored", summary["scored_count"])
    with col5:
        st.metric("📋 Shortlisted", summary["shortlisted_count"])

    st.divider()

    # Two columns: Workflow + System Info
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Workflow visualization
        events = state.get("workflow_events", [])
        step_count = summary["step_count"]
        render_workflow_diagram(events, step_count)

    with col_right:
        # System information
        st.markdown("### ⚙️ System Information")

        with st.container(border=True):
            st.markdown("**Workflow Status**")
            if summary["workflow_complete"]:
                st.success("✅ Workflow completed successfully")
            elif summary["needs_escalation"]:
                st.error("🚨 Human escalation needed")
            elif summary["shortlist_count"] > 0 and not summary["human_approved"]:
                st.warning("⏳ Awaiting human approval")
            elif summary["scored_count"] > 0:
                st.info("🔄 Workflow in progress")
            else:
                st.info("💤 Waiting for input")

            st.markdown(f"**Total Steps:** {summary['step_count']}")
            st.markdown(f"**Execution Time:** {state.get('execution_time_ms', 0)} ms")
            st.markdown(f"**Errors:** {summary['error_count']}")

        # Agent legend
        st.markdown("### 🤖 Agents")
        agents = [
            ("📄 Resume Analyst", "Parses resumes, detects injections"),
            ("📊 Scorer", "Scores candidates vs job description"),
            ("🔍 Verifier", "Blind re-score for fairness"),
            ("⚖️ Decider", "Generates ranked shortlist"),
            ("👤 Human Approval", "Approval gate before scheduling"),
            ("📅 Scheduler", "Generates interview invites"),
        ]
        for icon_name, desc in agents:
            st.markdown(f"- **{icon_name}**: {desc}")

        # Quick actions
        st.markdown("### 🚀 Quick Actions")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📄 Go to Job Description", use_container_width=True):
                st.session_state.page = "Job Description"
                st.rerun()
        with col_b:
            if st.button("📤 Go to Resume Upload", use_container_width=True):
                st.session_state.page = "Resume Upload"
                st.rerun()

    # Error display
    if summary["error_count"] > 0:
        st.divider()
        st.markdown("### ⚠️ Errors")
        errors = state.get("errors", [])
        for err in errors:
            st.error(err)