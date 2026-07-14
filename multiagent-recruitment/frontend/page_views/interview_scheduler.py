"""
Interview Scheduler page — schedule interviews for approved candidates.
"""

import streamlit as st
import pandas as pd
from io import BytesIO


def show():
    """Render the Interview Scheduler page."""
    st.markdown('<p class="main-header">📅 Interview Scheduler</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Schedule and manage interviews for approved candidates</p>', unsafe_allow_html=True)

    state = st.session_state.workflow_state
    human_approved = state.get("human_approved", False)
    shortlist = state.get("shortlist", []) or []

    # Check if human approval was given
    if not human_approved:
        st.warning("⚠️ Human approval has not been granted yet.")
        st.info("Go to **Human Approval** page to approve candidates before scheduling.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("👤 Go to Human Approval", use_container_width=True):
                st.session_state.page = "Human Approval"
                st.rerun()
        with col2:
            # Allow direct trigger with warning
            if st.button("▶️ Approve & Schedule Now", use_container_width=True, type="primary"):
                st.session_state.workflow_state["human_approved"] = True
                st.session_state.human_approved = True
                st.success("✅ Auto-approved. Running scheduler...")
                st.rerun()
        return

    # Get shortlisted candidates
    shortlisted_ids = set()
    shortlisted_names = {}
    for sl in shortlist:
        if hasattr(sl, "model_dump"):
            d = sl.model_dump()
        elif isinstance(sl, dict):
            d = sl
        else:
            continue
        if d.get("status") == "shortlisted":
            cid = d.get("candidate_id", "")
            shortlisted_ids.add(cid)
            shortlisted_names[cid] = d.get("name", "Unknown")

    if not shortlisted_ids:
        st.warning("⚠️ No shortlisted candidates to schedule.")
        st.info("Go to the **Shortlist** page to review candidates.")
        return

    st.success(f"✅ Human approval granted. Ready to schedule interviews for {len(shortlisted_ids)} candidate(s).")

    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Approved Candidates", len(shortlisted_ids))
    with col2:
        st.metric("📅 Schedules Generated", 0)  # Will update after scheduling
    with col3:
        st.metric("⏳ Pending Scheduling", len(shortlisted_ids))

    st.divider()

    # Scheduling trigger
    st.markdown("### 🚀 Generate Interview Schedules")
    st.info(
        "The Scheduler agent will generate personalized interview invitation emails "
        "for all approved candidates."
    )

    if st.button("📅 Generate Interview Schedules", use_container_width=True, type="primary"):
        from utils.backend import schedule_interviews_backend
        with st.spinner("🔄 Generating interview schedules..."):
            result = schedule_interviews_backend(st.session_state.workflow_state)
            st.session_state.workflow_state.update(result)
            st.success(f"✅ Interview schedules generated for {len(shortlisted_ids)} candidate(s)!")
            st.rerun()

    st.divider()

    # Show generated schedules
    schedules = state.get("schedules", []) or []

    if schedules:
        st.markdown("### 📋 Scheduled Interviews")

        # Convert to dicts
        schedule_dicts = []
        for s in schedules:
            if hasattr(s, "model_dump"):
                d = s.model_dump()
            elif isinstance(s, dict):
                d = s
            else:
                continue
            schedule_dicts.append(d)

        # Display each schedule
        for s in schedule_dicts:
            name = s.get("name", "Unknown")
            cid = s.get("candidate_id", "")
            email_subject = s.get("email_subject", "")
            email_body = s.get("email_body", "")
            interview_format = s.get("interview_format", "N/A")
            duration = s.get("duration_minutes", 0)

            with st.container(border=True):
                st.markdown(f"**👤 {name}** ({cid})")
                st.markdown(f"**Format:** {interview_format} | **Duration:** {duration} min")
                with st.expander("📧 View Email Template"):
                    st.markdown(f"**Subject:** {email_subject}")
                    st.divider()
                    st.markdown(email_body)

        # Download all schedules as CSV
        st.divider()
        col_d1, col_d2 = st.columns([1, 3])

        with col_d1:
            rows = []
            for s in schedule_dicts:
                rows.append({
                    "Name": s.get("name", ""),
                    "Candidate ID": s.get("candidate_id", ""),
                    "Format": s.get("interview_format", ""),
                    "Duration (min)": s.get("duration_minutes", 0),
                    "Email Subject": s.get("email_subject", ""),
                })
            df = pd.DataFrame(rows)
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            st.download_button(
                label="📥 Download Schedules CSV",
                data=csv_buffer,
                file_name="interview_schedules.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_d2:
            st.info("Download interview schedules as CSV for integration with calendar tools.")

    else:
        # Show candidates ready for scheduling
        st.markdown("### 📝 Candidates Ready for Scheduling")

        for cid, name in shortlisted_names.items():
            with st.container(border=True):
                st.markdown(f"**👤 {name}** ({cid})")
                st.caption("Awaiting schedule generation")