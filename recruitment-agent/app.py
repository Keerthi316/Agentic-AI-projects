"""
Recruitment Agent Dashboard - Streamlit Frontend with LLM-based Agent
Uses LangGraph + OpenRouter API for autonomous candidate evaluation.
"""

import streamlit as st
import json
import sys
import os
import time
import io
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.graph import run_agent, run_sequential
from graph.state import AgentState


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n".join(text_parts)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")


def extract_text_from_file(file) -> str:
    """Extract text from an uploaded file (TXT, MD, or PDF)."""
    file_bytes = file.getvalue()
    file_name = file.name.lower()
    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    else:
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1")


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="TechVest Recruit AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    /* ===== BASE ===== */
    .stApp { background: #ffffff; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .main > div { padding: 0rem 1rem; }
    
    /* ===== TEXT COLORS (dark on light, light on dark) ===== */
    .stApp, .stMarkdown, p, li, span, div, .stText, .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
        color: #1f2937 !important;
    }
    .st-cb, .st-cr, .st-bm, .st-bn, .st-bo, .st-bp, .st-bq, .st-br { color: #1f2937 !important; }
    .stTextInput > div > div > input { color: #1f2937 !important; background: #ffffff !important; }
    .stSelectbox > div > div > div { color: #1f2937 !important; }
    .stFileUploader > div > div { color: #1f2937 !important; }
    .stCaption { color: #4b5563 !important; font-size: 0.85rem; }
    .st-expander, .st-expander-header { color: #1f2937 !important; }
    h1, h2, h3, h4, h5, h6 { color: #111827 !important; }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e5e7eb; height: 100%;
    }
    .metric-card .label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #4b5563; }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #111827; }
    .metric-card.interview { border-left: 4px solid #10b981; }
    .metric-card.hold { border-left: 4px solid #f59e0b; }
    .metric-card.reject { border-left: 4px solid #ef4444; }
    .metric-card.total { border-left: 4px solid #3b82f6; }
    
    /* ===== BADGES ===== */
    .badge {
        display: inline-flex; align-items: center; padding: 0.2rem 0.7rem;
        border-radius: 20px; font-size: 0.7rem; font-weight: 600;
    }
    .badge-interview { background: #d1fae5; color: #065f46; }
    .badge-hold { background: #fef3c7; color: #92400e; }
    .badge-reject { background: #fee2e2; color: #991b1b; }
    .badge-pending { background: #e0e7ff; color: #3730a3; }
    
    /* ===== SECTION HEADERS ===== */
    .section-header {
        font-size: 0.85rem; font-weight: 700; text-transform: uppercase;
        color: #4b5563; margin: 1.5rem 0 0.75rem 0; border-bottom: 1px solid #d1d5db;
    }
    
    /* ===== STATUS WIDGET ===== */
    .status-widget {
        background: #f3f4f6; border-radius: 8px; padding: 0.75rem 1rem;
        border: 1px solid #d1d5db; margin-bottom: 0.5rem;
    }
    .status-widget .status-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; color: #4b5563; }
    .status-widget .status-value { font-size: 0.9rem; font-weight: 500; color: #111827; }
    
    /* ===== DATA TABLE ===== */
    .data-table { width: 100%; border-collapse: collapse; }
    .data-table th { text-align: left; padding: 0.75rem 1rem; font-size: 0.7rem; font-weight: 600; color: #4b5563; border-bottom: 2px solid #d1d5db; }
    .data-table td { padding: 0.75rem 1rem; font-size: 0.85rem; color: #1f2937; border-bottom: 1px solid #e5e7eb; }
    
    /* ===== SCORE BAR ===== */
    .score-bar-bg { background: #e5e7eb; border-radius: 4px; height: 6px; width: 100%; overflow: hidden; }
    .score-bar-fill { height: 100%; border-radius: 4px; }
    
    /* ===== TRAJECTORY ===== */
    .trajectory-step { background: #f3f4f6; border-radius: 8px; padding: 0.75rem 1rem; border-left: 3px solid #3b82f6; margin-bottom: 0.5rem; }
    .trajectory-step .step-header { font-size: 0.75rem; font-weight: 600; color: #1d4ed8; }
    .trajectory-step .step-thought { font-size: 0.85rem; color: #1f2937; font-style: italic; }
    
    /* ===== DARK MODE SUPPORT ===== */
    @media (prefers-color-scheme: dark) {
        .stApp { background: #0f172a; }
        .stApp, .stMarkdown, p, li, span, div, .stText, .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
            color: #e2e8f0 !important;
        }
        .st-cb, .st-cr, .st-bm, .st-bn, .st-bo, .st-bp, .st-bq, .st-br { color: #e2e8f0 !important; }
        .stTextInput > div > div > input { color: #e2e8f0 !important; background: #1e293b !important; }
        .stSelectbox > div > div > div { color: #e2e8f0 !important; }
        .stFileUploader > div > div { color: #e2e8f0 !important; }
        .stCaption { color: #94a3b8 !important; }
        .st-expander, .st-expander-header { color: #e2e8f0 !important; }
        h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }
        .metric-card { background: #1e293b; border-color: #334155; }
        .metric-card .label { color: #94a3b8; }
        .metric-card .value { color: #f1f5f9; }
        .section-header { color: #94a3b8; border-bottom-color: #334155; }
        .status-widget { background: #1e293b; border-color: #334155; }
        .status-widget .status-label { color: #94a3b8; }
        .status-widget .status-value { color: #f1f5f9; }
        .data-table th { color: #94a3b8; border-bottom-color: #334155; }
        .data-table td { color: #e2e8f0; border-bottom-color: #1e293b; }
        .score-bar-bg { background: #334155; }
        .trajectory-step { background: #1e293b; }
        .trajectory-step .step-thought { color: #e2e8f0; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if "initialized" not in st.session_state:
    st.session_state.agent_result = None
    st.session_state.jd_text = ""
    st.session_state.candidates_data = {}
    st.session_state.agent_running = False
    st.session_state.agent_status = "IDLE"
    st.session_state.current_step = 0
    st.session_state.current_candidate = ""
    st.session_state.initialized = True

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;padding:1rem 0;border-bottom:1px solid #e5e7eb;margin-bottom:1rem;">
        <div style="background:linear-gradient(135deg,#3b82f6,#8b5cf6);width:32px;height:32px;border-radius:8px;display:flex;align-items:center;justify-content:center;color:white;font-size:16px;">🤖</div>
        <div>
            <div style="font-size:1.1rem;font-weight:700;color:#111827;">TechVest Recruit</div>
            <div style="font-size:0.7rem;color:#6b7280;">AI-Powered Hiring Agent</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key - loaded from .env only (not exposed in UI)
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        st.error("⚠️ OPENROUTER_API_KEY not found in .env file. Add it to run the agent.")
    else:
        st.success("✅ API key loaded from .env")
    
    model_choice = st.selectbox("Model", [
        "openai/gpt-4o-mini", "openai/gpt-4o", "anthropic/claude-3.5-sonnet",
        "google/gemini-2.5-flash"
    ], index=0)
    os.environ["MODEL"] = model_choice
    
    # JD Upload
    st.markdown('<div class="section-header">Job Description</div>', unsafe_allow_html=True)
    jd_file = st.file_uploader("Upload JD (TXT, MD, PDF)", type=["txt", "md", "pdf"], label_visibility="collapsed")
    if jd_file:
        try:
            st.session_state.jd_text = extract_text_from_file(jd_file)
            st.success(f"✅ JD loaded ({jd_file.name})")
        except Exception as e:
            st.error(str(e))
    elif not st.session_state.jd_text:
        st.caption("Upload a job description file to begin")
    
    # Resume Upload
    st.markdown('<div class="section-header">Resumes</div>', unsafe_allow_html=True)
    resume_files = st.file_uploader("Upload Resumes (TXT, MD, PDF)", type=["txt", "md", "pdf"],
                                    accept_multiple_files=True, label_visibility="collapsed")
    if resume_files:
        for f in resume_files:
            name = f.name.rsplit(".", 1)[0]
            try:
                content = extract_text_from_file(f)
                st.session_state.candidates_data[name] = content
            except Exception as e:
                st.error(f"{f.name}: {str(e)}")
                continue
        st.success(f"✅ {len(resume_files)} resumes loaded")
    elif not st.session_state.candidates_data:
        st.caption("Upload resume files to begin")
    
    # Actions
    st.markdown('<div class="section-header">Actions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("▶ Run Agent", type="primary", use_container_width=True,
                           disabled=st.session_state.agent_running or not api_key or not st.session_state.jd_text or not st.session_state.candidates_data)
    with col2:
        reset_btn = st.button("⟳ Reset", use_container_width=True)
    
    if run_btn:
        st.session_state.agent_running = True
        st.session_state.agent_status = "RUNNING"
        st.session_state.agent_result = None
        st.rerun()
    
    if reset_btn:
        st.session_state.agent_result = None
        st.session_state.agent_running = False
        st.session_state.agent_status = "IDLE"
        st.session_state.current_step = 0
        st.session_state.current_candidate = ""
        st.rerun()
    
    # Status
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Agent Status</div>', unsafe_allow_html=True)
    status_color = {"IDLE": "#6b7280", "RUNNING": "#3b82f6", "WAITING_APPROVAL": "#f59e0b", "COMPLETED": "#10b981", "ERROR": "#ef4444"}.get(st.session_state.agent_status, "#6b7280")
    st.markdown(f'<div class="status-widget"><div class="status-label">Status</div><div class="status-value" style="color:{status_color};">● {st.session_state.agent_status}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="status-widget"><div class="status-label">Current Candidate</div><div class="status-value">{st.session_state.current_candidate or "—"}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="status-widget"><div class="status-label">Current Step</div><div class="status-value">{st.session_state.current_step or "—"}</div></div>', unsafe_allow_html=True)

# ============================================================
# MAIN PAGE
# ============================================================
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;padding:0.5rem 0 0 0;">
    <h1 style="font-size:1.5rem;font-weight:700;color:#111827;margin:0;">Recruitment Dashboard</h1>
    <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-size:0.75rem;color:#6b7280;">TechVest Solutions</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# RUN AGENT
# ============================================================
if st.session_state.agent_running and st.session_state.agent_result is None:
    progress_bar = st.progress(0, text="Initializing agent...")
    status_placeholder = st.empty()
    
    try:
        status_placeholder.info("🔧 Analyzing JD and building rubric...")
        progress_bar.progress(10, text="Analyzing JD...")
        time.sleep(0.2)
        
        candidates_list = list(st.session_state.candidates_data.items())
        for i, (name, _) in enumerate(candidates_list):
            st.session_state.current_candidate = name
            progress = 15 + (i / len(candidates_list)) * 40
            progress_bar.progress(int(progress), text=f"Processing: {name}...")
            status_placeholder.info(f"📄 Processing {name}...")
            time.sleep(0.2)
        
        st.session_state.current_candidate = "All candidates"
        status_placeholder.info("🤖 Running autonomous agent with LLM...")
        progress_bar.progress(60, text="Running agent...")
        
        result = run_agent(st.session_state.jd_text, st.session_state.candidates_data)
        st.session_state.agent_result = result
        st.session_state.current_step = result.step_count
        
        if result.status == "WAITING_APPROVAL":
            st.session_state.agent_status = "WAITING_APPROVAL"
        else:
            st.session_state.agent_status = result.status
        
        progress_bar.progress(100, text="Complete!")
        status_placeholder.success("✅ Agent execution complete!")
        time.sleep(0.5)
        
    except Exception as e:
        st.session_state.agent_status = "ERROR"
        status_placeholder.error(f"❌ Error: {str(e)}")
        progress_bar.empty()
    
    st.session_state.agent_running = False
    st.rerun()

# ============================================================
# RESULTS
# ============================================================
result = st.session_state.agent_result

if result and result.shortlist:
    shortlist = result.shortlist
    interview_count = sum(1 for e in shortlist if e.get("decision") == "Interview")
    hold_count = sum(1 for e in shortlist if e.get("decision") == "Hold")
    reject_count = sum(1 for e in shortlist if e.get("decision") == "Reject")
    total = len(shortlist)
else:
    interview_count = hold_count = reject_count = 0
    total = len(st.session_state.candidates_data)

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1: st.markdown(f'<div class="metric-card total"><div class="label">Total Candidates</div><div class="value">{total}</div></div>', unsafe_allow_html=True)
with col2: st.markdown(f'<div class="metric-card interview"><div class="label">Interview</div><div class="value">{interview_count}</div></div>', unsafe_allow_html=True)
with col3: st.markdown(f'<div class="metric-card hold"><div class="label">Hold</div><div class="value">{hold_count}</div></div>', unsafe_allow_html=True)
with col4: st.markdown(f'<div class="metric-card reject"><div class="label">Reject</div><div class="value">{reject_count}</div></div>', unsafe_allow_html=True)

# Ranking Table
st.markdown('<div class="section-header">Candidate Ranking</div>', unsafe_allow_html=True)
if result and result.shortlist:
    st.markdown("""<table class="data-table"><thead><tr><th>Rank</th><th>Candidate</th><th>Score</th><th>Progress</th><th>Decision</th><th>Summary</th></tr></thead><tbody>""", unsafe_allow_html=True)
    for i, entry in enumerate(result.shortlist):
        rank = i + 1
        name = entry.get("candidate", "Unknown")
        score = entry.get("score", 0)
        decision = entry.get("decision", "Hold")
        summary = entry.get("summary", "")[:100]
        score_color = "#10b981" if score >= 70 else ("#f59e0b" if score >= 40 else "#ef4444")
        badge_class = f"badge-{decision.lower()}"
        st.markdown(f"<tr><td><strong>#{rank}</strong></td><td><strong>{name}</strong></td><td><strong>{score}</strong></td><td style='width:200px;'><div class='score-bar-bg'><div class='score-bar-fill' style='width:{score}%;background:{score_color};height:6px;border-radius:4px;'></div></div></td><td><span class='badge {badge_class}'>{decision}</span></td><td style='color:#6b7280;font-size:0.8rem;'>{summary}</td></tr>", unsafe_allow_html=True)
    st.markdown("</tbody></table>", unsafe_allow_html=True)
else:
    st.info("👆 Run the agent to view candidate rankings")

# Candidate Details
st.markdown('<div class="section-header">Candidate Details</div>', unsafe_allow_html=True)
if result and result.shortlist:
    for i, entry in enumerate(result.shortlist):
        name = entry.get("candidate", "Unknown")
        score = entry.get("score", 0)
        decision = entry.get("decision", "Hold")
        evidence = entry.get("evidence", [])
        focus = entry.get("interview_focus", [])
        
        with st.expander(f"#{i+1}  {name}  —  {score}  {decision}", expanded=(i == 0)):
            col_left, col_right = st.columns([2, 3])
            with col_left:
                st.markdown("**📋 Evidence**")
                for ev in evidence[:3]:
                    st.markdown(f"- {ev}")
                if focus:
                    st.markdown("**🎯 Interview Focus**")
                    for f in focus[:3]:
                        st.markdown(f"- {f}")
            with col_right:
                st.markdown("**📊 Score**")
                rec_colors = {"Interview": "#10b981", "Hold": "#f59e0b", "Reject": "#ef4444"}
                rec_color = rec_colors.get(decision, "#6b7280")
                st.markdown(f"""
                <div style="border:2px solid {rec_color};border-radius:12px;padding:1rem;text-align:center;">
                    <div style="font-size:1.2rem;font-weight:700;color:{rec_color};">{decision}</div>
                    <div style="font-size:0.8rem;color:#6b7280;margin-top:0.3rem;">{entry.get('summary', '')[:200]}</div>
                </div>
                """, unsafe_allow_html=True)
else:
    st.info("👆 Run the agent to view candidate details")

# Bottom Tabs
st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
if result:
    tab1, tab2, tab3 = st.tabs(["📜 Trajectory", "📋 Audit Log", "📅 Interview Scheduling"])
    
    with tab1:
        st.markdown("### Reasoning Trajectory")
        st.caption("Every step the agent took with LLM-powered decisions")
        trajectory = result.trajectory
        if trajectory:
            selected = st.select_slider("Step Timeline", options=range(len(trajectory)), value=len(trajectory)-1,
                                        format_func=lambda x: f"Step {trajectory[x]['step_number']}: {trajectory[x]['tool']}")
            step = trajectory[selected]
            st.markdown(f"""
            <div class="trajectory-step">
                <div class="step-header">Step {step['step_number']} — {step['tool'].upper()}</div>
                <div class="step-thought">💭 {step['thought']}</div>
                <div style="margin-top:0.5rem;font-size:0.8rem;color:#6b7280;">
                    <strong>Observation:</strong> {str(step.get('observation', ''))[:300]}
                </div>
                <div style="margin-top:0.5rem;"><strong>Decision:</strong> {step['decision']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Audit Log")
        st.caption("Complete record for compliance")
        for s in result.trajectory:
            st.markdown(f"<div style='background:#f9fafb;border-radius:6px;padding:0.5rem;margin-bottom:0.3rem;border-left:3px solid #3b82f6;font-size:0.8rem;'><strong>#{s['step_number']}</strong> <span style='color:#3b82f6;'>{s['tool']}</span> — <em>{s['thought'][:80]}</em></div>", unsafe_allow_html=True)
        st.download_button("📥 Download Audit Log", 
                          data=json.dumps(result.trajectory, indent=2, default=str),
                          file_name=f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with tab3:
        st.markdown("### Interview Scheduling")
        st.caption("Requires human approval")
        if result.actions:
            for action in result.actions:
                is_pending = action.get("status") == "Pending Human Approval"
                st.markdown(f"""
                <div style="background:white;border-radius:12px;padding:1rem;box-shadow:0 1px 3px rgba(0,0,0,0.08);margin-bottom:0.5rem;">
                    <div style="display:flex;align-items:center;justify-content:space-between;">
                        <div><strong>{action.get('candidate', '')}</strong><br><span style="color:#6b7280;font-size:0.85rem;">📅 {action.get('slot', '')}</span></div>
                        <span class="badge badge-{'pending' if is_pending else 'success'}">{action.get('status', '')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No interview proposals yet.")
else:
    st.info("👆 Run the agent to view results")

st.markdown("""
<div style="text-align:center;padding:2rem 0 0.5rem 0;font-size:0.7rem;color:#9ca3af;">
    TechVest Recruit AI · Autonomous Hiring Agent · Built with LangGraph + OpenRouter + Streamlit
</div>
""", unsafe_allow_html=True)