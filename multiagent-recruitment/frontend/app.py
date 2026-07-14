"""
Multi-Agent Recruitment System — Streamlit Frontend

This is the entry point for the frontend application. It:
1. Configures the Streamlit page (title, layout, sidebar)
2. Initializes session state
3. Renders the sidebar navigation
4. Routes to the selected page

Design decisions:
- Session state persists workflow progress across pages and re-runs.
- The sidebar shows a workflow progress indicator with checkmarks.
- Each page is a separate module in pages/ for clean separation.
- The backend bridge (utils.backend) handles all backend interaction.
"""

import os
import sys
import streamlit as st

# Add project root AND frontend directory to path so imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
frontend_dir = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if frontend_dir not in sys.path:
    sys.path.insert(0, frontend_dir)

# Set demo mode if no API key
if not os.getenv("OPENAI_API_KEY", ""):
    os.environ["RECRUITMENT_DEMO_MODE"] = "true"

from utils.backend import get_initial_state, state_to_summary

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent Recruitment System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    .stApp { max-width: 100%; }
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 1rem;
        color: #424242;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 4px solid #1E88E5;
    }
    .metric-card {
        background-color: #f0f4f8;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .sidebar-workflow-step {
        padding: 4px 8px;
        border-radius: 4px;
        margin: 2px 0;
    }
    .stButton button {
        width: 100%;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "workflow_state" not in st.session_state:
    st.session_state.workflow_state = get_initial_state()

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "workflow_running" not in st.session_state:
    st.session_state.workflow_running = False

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "human_approved" not in st.session_state:
    st.session_state.human_approved = False


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def render_sidebar():
    """Render the sidebar with navigation and workflow progress."""
    summary = state_to_summary(st.session_state.workflow_state)

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
        st.markdown("### 🤖 Recruitment System")
        st.divider()

        # Navigation
        st.markdown("#### 📋 Pages")

        pages = [
            ("🏠", "Dashboard"),
            ("📄", "Job Description"),
            ("📤", "Resume Upload"),
            ("🔍", "Candidate Analysis"),
            ("📊", "Candidate Scores"),
            ("✅", "Verification"),
            ("📋", "Shortlist"),
            ("👤", "Human Approval"),
            ("📅", "Interview Scheduler"),
            ("📜", "Execution Logs"),
        ]

        for icon, page_name in pages:
            is_active = st.session_state.page == page_name
            btn_label = f"{icon} {page_name}"
            if st.button(
                btn_label,
                key=f"nav_{page_name}",
                use_container_width=True,
                type="secondary" if not is_active else "primary",
            ):
                st.session_state.page = page_name
                st.rerun()

        st.divider()

        # Workflow progress
        st.markdown("#### 📊 Workflow Progress")

        steps = [
            ("📄", "Job Description", summary["has_jd"]),
            ("📤", "Resumes Uploaded", summary["candidate_count"] > 0),
            ("🔍", "Resumes Parsed", summary["parsed_count"] > 0),
            ("📊", "Candidates Scored", summary["scored_count"] > 0),
            ("✅", "Verification Done", summary["verified_count"] > 0 or summary["borderline_count"] == 0),
            ("📋", "Shortlist Ready", summary["shortlist_count"] > 0),
            ("👤", "Human Approval", summary["human_approved"]),
            ("📅", "Scheduling Done", st.session_state.workflow_state.get("step_count", 0) >= 5),
        ]

        for icon, label, done in steps:
            if done:
                st.markdown(f"✅ **{icon} {label}**  ")
            else:
                st.markdown(f"⬜ {icon} {label}  ")

        st.divider()

        # Summary metrics
        st.markdown("#### 📈 Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Candidates", summary["candidate_count"])
        with col2:
            st.metric("Shortlisted", summary["shortlisted_count"])

        col3, col4 = st.columns(2)
        with col3:
            st.metric("Errors", summary["error_count"])
        with col4:
            st.metric("Steps", summary["step_count"])

        # Reset button
        st.divider()
        if st.button("🔄 Reset Workflow", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ---------------------------------------------------------------------------
# Main app router
# ---------------------------------------------------------------------------

def main():
    """Main entry point — renders sidebar and routes to the selected page."""
    render_sidebar()

    # Page routing
    page = st.session_state.page

    if page == "Dashboard":
        from page_views.dashboard import show
        show()
    elif page == "Job Description":
        from page_views.job_description import show
        show()
    elif page == "Resume Upload":
        from page_views.resume_upload import show
        show()
    elif page == "Candidate Analysis":
        from page_views.candidate_analysis import show
        show()
    elif page == "Candidate Scores":
        from page_views.candidate_scores import show
        show()
    elif page == "Verification":
        from page_views.verification import show
        show()
    elif page == "Shortlist":
        from page_views.shortlist import show
        show()
    elif page == "Human Approval":
        from page_views.human_approval import show
        show()
    elif page == "Interview Scheduler":
        from page_views.interview_scheduler import show
        show()
    elif page == "Execution Logs":
        from page_views.execution_logs import show
        show()


if __name__ == "__main__":
    main()