"""
Recruitment Page
----------------
Full multi-agent workflow with complete human approval flow:
- Approve  → "Interview Finalized", workflow node turns green, DB updated
- Decline  → "Rejected by Human Reviewer", DB updated
Both paths disable buttons and mark workflow as completed.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from typing import Any, Dict, List

import streamlit as st

from database.db import save_run, finalize_approval, get_run_by_id
from utils.logger import get_run_log
from utils.helpers import score_color, recommendation_icon, recommendation_color, format_list


# ── Workflow node definitions (includes final scheduler node) ─────────────
WORKFLOW_NODES = [
    ("coordinator", "🗂️",  "Coordinator"),
    ("analyst",     "🔍",  "Resume Analyst"),
    ("scorer",      "📊",  "Scorer"),
    ("verifier",    "🛡️",  "Verifier"),
    ("decider",     "⚖️",  "Decider"),
    ("human",       "👤",  "Human Approval"),
    ("finalized",   "🗓️",  "Interview Scheduled"),
]

# ── Session-state keys ────────────────────────────────────────────────────
_KEY_RESULTS   = "recruitment_results"      # list of saved run dicts
_KEY_NODES     = "recruitment_node_statuses"
_KEY_LOGS      = "recruitment_logs_snapshot"


# ── Workflow renderer ─────────────────────────────────────────────────────
def _node_html(node_id: str, icon: str, label: str, status: str) -> str:
    if status == "done":
        border = "var(--success)"
        bg     = "rgba(34,197,94,0.12)"
        badge  = '<span style="margin-left:0.5rem; color:var(--success); font-weight:700;">✓</span>'
        anim   = ""
    elif status == "active":
        border = "var(--accent)"
        bg     = "var(--accent-glow)"
        badge  = '<span style="margin-left:0.5rem;">⟳</span>'
        anim   = "animation:glow-pulse 1.5s ease-in-out infinite;"
    elif status == "skipped":
        border = "var(--glass-border)"
        bg     = "transparent"
        badge  = ""
        anim   = "opacity:0.35;"
    else:  # pending
        border = "var(--glass-border)"
        bg     = "transparent"
        badge  = ""
        anim   = "opacity:0.5;"

    return (
        f'<div class="workflow-node" style="border-color:{border}; background:{bg}; {anim}">'
        f"<span>{icon}</span><span>{label}</span>{badge}"
        f"</div>"
    )


def render_workflow(node_statuses: Dict[str, str]) -> None:
    arrow = '<div style="color:var(--text-secondary);font-size:1.1rem;margin:2px 0 2px 1.2rem;">↓</div>'
    html  = '<div style="display:flex;flex-direction:column;padding:0.75rem;">'
    nodes = list(enumerate(WORKFLOW_NODES))
    for i, (nid, icon, label) in nodes:
        html += _node_html(nid, icon, label, node_statuses.get(nid, "pending"))
        if i < len(nodes) - 1:
            html += arrow
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_logs(logs: List[dict]) -> None:
    if not logs:
        st.markdown(
            "<div class='log-panel'><span style='color:var(--text-secondary);'>Waiting for logs…</span></div>",
            unsafe_allow_html=True,
        )
        return
    entries = ""
    for e in logs[-60:]:
        cls = {"info": "log-info", "warning": "log-warning", "error": "log-error"}.get(
            e.get("level", "info"), "log-info"
        )
        ts = e.get("timestamp", "")[-8:-3] if e.get("timestamp") else ""
        entries += (
            f'<div class="log-entry">'
            f'<span class="{cls}">[{ts}] [{e["agent"]}]</span> {e["message"]}'
            f"</div>"
        )
    st.markdown(f'<div class="log-panel">{entries}</div>', unsafe_allow_html=True)


# ── Human approval widget ─────────────────────────────────────────────────
def _render_approval_widget(run_id: str, name: str, original_rec: str) -> None:
    """
    Renders the approval widget for a single candidate.
    State is persisted in st.session_state keyed by run_id so it survives reruns.
    """
    state_key = f"approval_state_{run_id}"

    # Read current DB state (source of truth)
    record = get_run_by_id(run_id)
    db_approved    = record.get("approved")        if record else None
    db_final_status = record.get("final_status")   if record else None
    db_completed   = record.get("workflow_completed", 0) if record else 0

    # If already finalized (from DB), show final state
    if db_completed and db_final_status:
        if db_final_status == "Interview Finalized":
            st.markdown(
                """
                <div style="background:rgba(34,197,94,0.12); border:1.5px solid #22c55e;
                            border-radius:12px; padding:1rem 1.2rem; margin-top:0.5rem;">
                    <div style="font-size:1.1rem; font-weight:700; color:#22c55e;">
                        ✅ Candidate Finalized for the Interview Round
                    </div>
                    <div style="color:var(--text-secondary); font-size:0.85rem; margin-top:0.3rem;">
                        Recruitment workflow completed. Candidate is scheduled for interview.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:  # Rejected by Human Reviewer
            st.markdown(
                """
                <div style="background:rgba(239,68,68,0.10); border:1.5px solid #ef4444;
                            border-radius:12px; padding:1rem 1.2rem; margin-top:0.5rem;">
                    <div style="font-size:1.1rem; font-weight:700; color:#ef4444;">
                        ❌ Rejected by Human Reviewer
                    </div>
                    <div style="color:var(--text-secondary); font-size:0.85rem; margin-top:0.3rem;">
                        Recruitment workflow completed. Candidate will not proceed.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return  # buttons disabled — nothing more to render

    # Still pending — show approval prompt
    st.markdown(
        f"""
        <div style="background:rgba(245,158,11,0.10); border:1.5px solid #f59e0b;
                    border-radius:12px; padding:1rem 1.2rem; margin-top:0.5rem;">
            <div style="font-weight:700; color:#f59e0b; font-size:1rem;">
                ⏳ Human Approval Required
            </div>
            <div style="color:var(--text-secondary); font-size:0.85rem; margin-top:0.3rem;">
                AI decision: <strong>{original_rec}</strong> for <strong>{name}</strong>.
                Review and confirm before this candidate progresses.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button(
            "✅  Approve — Finalize for Interview",
            key=f"approve_{run_id}",
            use_container_width=True,
        ):
            # 1. Write to DB
            finalize_approval(run_id, approved=True)
            # 2. Update workflow node statuses in session
            if _KEY_NODES in st.session_state:
                ns = st.session_state[_KEY_NODES]
                ns["human"]     = "done"
                ns["finalized"] = "done"
                st.session_state[_KEY_NODES] = ns
            # 3. Log it
            from utils.logger import log_event
            log_event("Human", f"APPROVED — {name} finalized for Interview Round")
            st.rerun()

    with btn_col2:
        if st.button(
            "❌  Decline — Reject Candidate",
            key=f"decline_{run_id}",
            use_container_width=True,
        ):
            finalize_approval(run_id, approved=False)
            if _KEY_NODES in st.session_state:
                ns = st.session_state[_KEY_NODES]
                ns["human"]     = "done"
                ns["finalized"] = "skipped"
                st.session_state[_KEY_NODES] = ns
            from utils.logger import log_event
            log_event("Human", f"DECLINED — {name} rejected by human reviewer", "warning")
            st.rerun()


# ── Candidate result card ─────────────────────────────────────────────────
def candidate_card(decision: Dict[str, Any], run_id: str) -> None:
    profile   = decision.get("_profile", {})
    scorecard = decision.get("_scorecard", {})
    name      = decision.get("_candidate_name", "Unknown")
    rec       = decision.get("recommendation", "Hold")
    score     = decision.get("_overall_score", 0)
    explanation = decision.get("explanation", "")
    filename  = decision.get("_filename", "")

    rc  = recommendation_color(rec)
    ri  = recommendation_icon(rec)
    sc  = score_color(score)

    # Read final status from DB for live badge
    record       = get_run_by_id(run_id)
    final_status = record.get("final_status") if record else None
    completed    = record.get("workflow_completed", 0) if record else 0

    if final_status == "Interview Finalized":
        status_badge = (
            '<span style="background:rgba(34,197,94,0.15); color:#22c55e; '
            'padding:0.2rem 0.8rem; border-radius:999px; font-size:0.8rem; font-weight:700;">'
            '✅ Interview Finalized</span>'
        )
    elif final_status == "Rejected by Human Reviewer":
        status_badge = (
            '<span style="background:rgba(239,68,68,0.12); color:#ef4444; '
            'padding:0.2rem 0.8rem; border-radius:999px; font-size:0.8rem; font-weight:700;">'
            '❌ Rejected by Reviewer</span>'
        )
    else:
        status_badge = ""

    st.markdown(
        f"""
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;
                        flex-wrap:wrap; gap:0.75rem;">
                <div>
                    <h3 style="margin:0; font-size:1.2rem;">{name}</h3>
                    <div style="font-size:0.78rem; color:var(--text-secondary); margin-top:0.1rem;">
                        {filename}
                    </div>
                    <div style="margin-top:0.5rem;">{status_badge}</div>
                </div>
                <div style="text-align:right;">
                    <div class="score-badge"
                         style="background:{sc}22; color:{sc}; font-size:1.3rem; padding:0.4rem 1rem;">
                        {score:.0f} / 100
                    </div>
                    <div style="margin-top:0.4rem; color:{rc}; font-weight:700; font-size:0.95rem;">
                        {ri} {rec}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔍 View Full Details", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🛠️ Skills**")
            st.write(format_list(profile.get("skills", []), 20))
            st.markdown("**🎓 Education**")
            for e in profile.get("education", [])[:3]:
                st.markdown(f"- {e}")
        with col2:
            st.markdown("**💼 Experience**")
            for e in profile.get("experience", [])[:3]:
                st.markdown(f"- {e}")
            st.markdown("**🏆 Projects**")
            for p in profile.get("projects", [])[:3]:
                st.markdown(f"- {p}")

        st.markdown("**📊 Score Breakdown**")
        sc_cols = st.columns(5)
        for sc_col, (dim, val) in zip(sc_cols, [
            ("Technical",    scorecard.get("technical_score", 0)),
            ("Experience",   scorecard.get("experience_score", 0)),
            ("Education",    scorecard.get("education_score", 0)),
            ("Projects",     scorecard.get("projects_score", 0)),
            ("Comm.",        scorecard.get("communication_score", 0)),
        ]):
            sc_col.metric(dim, f"{val:.0f}")

        st.markdown("**💬 Reasoning**")
        st.info(explanation)

        if scorecard.get("strengths"):
            st.markdown("**✅ Strengths**")
            for s in scorecard["strengths"]:
                st.markdown(f"- {s}")
        if scorecard.get("weaknesses"):
            st.markdown("**⚠️ Areas to Improve**")
            for w in scorecard["weaknesses"]:
                st.markdown(f"- {w}")
        if scorecard.get("missing_skills"):
            st.markdown("**❌ Missing Skills**")
            st.write(", ".join(scorecard["missing_skills"]))

    # Show approval widget only for Interview / Reject decisions
    if rec in ("Interview", "Reject"):
        _render_approval_widget(run_id, name, rec)

    st.markdown("<hr style='border-color:var(--glass-border); margin:1rem 0;'>", unsafe_allow_html=True)


# ── Main page ─────────────────────────────────────────────────────────────
def show():
    st.markdown(
        "<h1 class='gradient-header' style='font-size:2rem; margin-bottom:0.2rem;'>🎯 Recruitment</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:var(--text-secondary);'>Upload resumes, define the JD, and watch agents collaborate live.</p>",
        unsafe_allow_html=True,
    )

    # ── Input form ────────────────────────────────────────────────────────
    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        jd = st.text_area(
            "📋 Job Description",
            height=180,
            placeholder="Paste the full job description including required skills, experience, responsibilities…",
            key="jd_input",
        )
        uploaded = st.file_uploader(
            "📎 Upload Resumes (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="resume_upload",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    run_btn = st.button("🚀 Run Recruitment Pipeline", use_container_width=True)

    # ── If we have stored results, show them (persists across reruns) ─────
    if _KEY_RESULTS in st.session_state and st.session_state[_KEY_RESULTS]:
        _render_results_section()

    if not run_btn:
        return

    # ── Validation ────────────────────────────────────────────────────────
    if not jd.strip():
        st.error("Please enter a Job Description.")
        return
    if not uploaded:
        st.error("Please upload at least one resume.")
        return

    # ── Clear previous run state ─────────────────────────────────────────
    st.session_state.pop(_KEY_RESULTS, None)
    st.session_state.pop(_KEY_NODES, None)
    st.session_state.pop(_KEY_LOGS, None)

    # ── Live layout ───────────────────────────────────────────────────────
    left_col, right_col = st.columns([1, 2])
    with left_col:
        st.markdown("### 🔄 Workflow")
        wf_placeholder = st.empty()
    with right_col:
        st.markdown("### 📋 Execution Logs")
        log_placeholder = st.empty()

    node_statuses = {n: "pending" for n, _, __ in WORKFLOW_NODES}

    files = [{"filename": f.name, "bytes": f.read(), "text": ""} for f in uploaded]

    # Show coordinator as active immediately
    node_statuses["coordinator"] = "active"
    with wf_placeholder.container():
        render_workflow(node_statuses)

    # ── Run workflow ──────────────────────────────────────────────────────
    import importlib
    workflow_mod = importlib.import_module("graph.workflow")

    with st.spinner("Running multi-agent pipeline…"):
        try:
            from utils.logger import clear_run_log, get_run_log
            clear_run_log()

            graph = workflow_mod.get_workflow()
            initial_state = {
                "job_description": jd,
                "uploaded_files": files,
                "profiles": [], "scorecards": [], "verified_scores": [],
                "shortlist": [], "revision_count": 0,
                "current_candidate_index": 0,
                "needs_verification": False, "needs_human_approval": False,
                "human_approved": None, "run_id": None,
                "logs": [], "errors": [],
            }

            node_order = ["coordinator", "analyst", "scorer", "verifier", "decider"]
            seen_nodes: set = set()

            # Stream node-by-node
            for event in graph.stream(initial_state, stream_mode="updates"):
                for node_name in event:
                    if node_name not in seen_nodes:
                        seen_nodes.add(node_name)
                        for prev in node_order:
                            if prev != node_name and node_statuses.get(prev) == "active":
                                node_statuses[prev] = "done"
                        if node_name in node_statuses:
                            node_statuses[node_name] = "active"
                        with wf_placeholder.container():
                            render_workflow(node_statuses)
                        with log_placeholder.container():
                            render_logs(get_run_log())
                        time.sleep(0.25)

                with log_placeholder.container():
                    render_logs(get_run_log())

            # Full invoke for final state
            final_state = graph.invoke(initial_state)

        except Exception as exc:
            st.error(f"Workflow error: {exc}")
            import traceback
            st.code(traceback.format_exc())
            return

    # Mark agent nodes done; set human node
    for n in node_order:
        if n in seen_nodes:
            node_statuses[n] = "done"

    shortlist     = final_state.get("shortlist", [])
    needs_human   = final_state.get("needs_human_approval", False)
    node_statuses["human"]     = "active" if needs_human else "skipped"
    node_statuses["finalized"] = "pending" if needs_human else "skipped"

    with wf_placeholder.container():
        render_workflow(node_statuses)
    with log_placeholder.container():
        render_logs(get_run_log())

    # ── Errors panel ──────────────────────────────────────────────────────
    errors = final_state.get("errors", [])
    if errors:
        with st.expander("⚠️ Warnings / Security Alerts", expanded=True):
            for e in errors:
                if "SECURITY" in e or "injection" in e.lower():
                    st.error(e)
                else:
                    st.warning(e)

    if not shortlist:
        st.error("No decisions produced. Check API key and resume content.")
        return

    # ── Save all results to DB and session ────────────────────────────────
    saved_results = []
    for decision in shortlist:
        run_id = save_run(
            filename      = decision.get("_filename", ""),
            candidate_name= decision.get("_candidate_name", "Unknown"),
            overall_score = decision.get("_overall_score", 0),
            recommendation= decision.get("recommendation", "Hold"),
            explanation   = decision.get("explanation", ""),
            skills        = decision.get("_profile", {}).get("skills", []),
            experience    = decision.get("_profile", {}).get("experience", []),
            education     = decision.get("_profile", {}).get("education", []),
            score_breakdown= decision.get("_scorecard", {}),
        )
        saved_results.append({"decision": decision, "run_id": run_id})

    # Persist to session so approval clicks don't lose context
    st.session_state[_KEY_RESULTS]  = saved_results
    st.session_state[_KEY_NODES]    = node_statuses
    st.session_state[_KEY_LOGS]     = get_run_log()

    st.rerun()   # rerun so _render_results_section picks up saved state cleanly


def _render_results_section() -> None:
    """Render saved results + live approval widgets. Called on every rerun."""
    saved_results = st.session_state.get(_KEY_RESULTS, [])
    node_statuses = st.session_state.get(_KEY_NODES, {})
    logs          = st.session_state.get(_KEY_LOGS, [])

    # ── Workflow + logs summary ───────────────────────────────────────────
    left_col, right_col = st.columns([1, 2])
    with left_col:
        st.markdown("### 🔄 Workflow")
        render_workflow(node_statuses)
    with right_col:
        st.markdown("### 📋 Execution Logs")
        render_logs(logs)

    st.markdown("---")
    st.markdown(
        f"<h2 class='gradient-header'>Results — {len(saved_results)} candidate(s)</h2>",
        unsafe_allow_html=True,
    )

    # Check if ALL approval-required candidates are resolved
    all_resolved = all(
        get_run_by_id(r["run_id"]).get("workflow_completed", 0)
        for r in saved_results
        if r["decision"].get("recommendation") in ("Interview", "Reject")
    ) if saved_results else True

    if all_resolved and any(
        r["decision"].get("recommendation") in ("Interview", "Reject")
        for r in saved_results
    ):
        # Update node statuses to show workflow fully completed
        node_statuses["human"] = "done"
        # finalized node: green if any interview, skipped if all rejected
        any_interview = any(
            get_run_by_id(r["run_id"]).get("final_status") == "Interview Finalized"
            for r in saved_results
        )
        node_statuses["finalized"] = "done" if any_interview else "skipped"
        st.session_state[_KEY_NODES] = node_statuses

        st.markdown(
            """
            <div style="background:rgba(34,197,94,0.10); border:1.5px solid #22c55e;
                        border-radius:12px; padding:0.85rem 1.2rem; margin-bottom:1.2rem;">
                <strong style="color:#22c55e;">🎉 Recruitment workflow fully completed.</strong>
                <span style="color:var(--text-secondary); font-size:0.9rem;">
                  Dashboard and History have been updated.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Candidate cards ───────────────────────────────────────────────────
    for item in saved_results:
        candidate_card(item["decision"], item["run_id"])
