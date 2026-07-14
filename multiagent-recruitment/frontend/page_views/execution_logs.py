"""
Execution Logs page — LangGraph execution flow, routing decisions, retries,
execution time, and errors.
"""

import streamlit as st
from components.workflow_viz import render_workflow_diagram


def show():
    """Render the Execution Logs page."""
    st.markdown('<p class="main-header">📜 Execution Logs</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">LangGraph execution flow, routing decisions, and errors</p>', unsafe_allow_html=True)

    state = st.session_state.workflow_state
    events = state.get("workflow_events", []) or []
    step_count = state.get("step_count", 0)
    errors = state.get("errors", []) or []

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔄 Events", len(events))
    with col2:
        st.metric("📊 Steps", step_count)
    with col3:
        st.metric("⏱️ Time", f"{state.get('execution_time_ms', 0)} ms")
    with col4:
        st.metric("❌ Errors", len(errors))

    st.divider()

    # Workflow visualization
    st.markdown("### 🔄 Workflow Execution Flow")
    render_workflow_diagram(events, step_count)

    st.divider()

    # Event timeline
    st.markdown("### 📋 Event Timeline")

    if not events:
        st.info("💤 No workflow events recorded yet. Run the workflow to see execution logs.")
        return

    for event_idx, event in enumerate(events):
        for node_name, node_output in event.items():
            with st.container(border=True):
                # Node header
                col_e1, col_e2 = st.columns([3, 1])
                with col_e1:
                    node_label = node_name.replace("_", " ").title()
                    st.markdown(f"**▶ {node_label}** (Event {event_idx + 1})")
                with col_e2:
                    st.caption(f"Step: {event_idx + 1}")

                # Node output details
                if isinstance(node_output, dict):
                    # Show errors
                    node_errors = node_output.get("errors", [])
                    if node_errors:
                        for err in node_errors:
                            st.error(f"❌ {err}")

                    # Show profiles parsed
                    profiles = node_output.get("parsed_profiles", [])
                    if profiles:
                        st.info(f"📄 Parsed {len(profiles)} candidate profile(s)")
                        for p in profiles:
                            name = p.name if hasattr(p, "name") else p.get("name", "Unknown")
                            cid = p.candidate_id if hasattr(p, "candidate_id") else p.get("candidate_id", "")
                            injected = p.is_injection_detected if hasattr(p, "is_injection_detected") else p.get("is_injection_detected", False)
                            if injected:
                                st.markdown(f"  - 🚨 {name} ({cid}) — **INJECTION DETECTED**")
                            else:
                                st.markdown(f"  - ✅ {name} ({cid})")

                    # Show scorecards
                    scorecards = node_output.get("scorecards", [])
                    if scorecards:
                        st.info(f"📊 Generated {len(scorecards)} scorecard(s)")
                        for sc in scorecards:
                            cid = sc.candidate_id if hasattr(sc, "candidate_id") else sc.get("candidate_id", "")
                            score = sc.total_score if hasattr(sc, "total_score") else sc.get("total_score", 0)
                            borderline = sc.is_borderline if hasattr(sc, "is_borderline") else sc.get("is_borderline", False)
                            flag = "⚠️ BORDERLINE" if borderline else ""
                            st.markdown(f"  - {cid}: {score:.1f} {flag}")

                    # Show verified scores
                    verified = node_output.get("verified_scores", [])
                    if verified:
                        st.info(f"✅ Verified {len(verified)} candidate(s)")
                        for vs in verified:
                            cid = vs.candidate_id if hasattr(vs, "candidate_id") else vs.get("candidate_id", "")
                            orig = vs.original_score if hasattr(vs, "original_score") else vs.get("original_score", 0)
                            blind = vs.blind_score if hasattr(vs, "blind_score") else vs.get("blind_score", 0)
                            fair = vs.is_fair if hasattr(vs, "is_fair") else vs.get("is_fair", False)
                            fair_str = "✅ Fair" if fair else "❌ Unfair"
                            st.markdown(f"  - {cid}: Original={orig:.1f} → Blind={blind:.1f} ({fair_str})")

                    # Show shortlist
                    shortlist = node_output.get("shortlist", [])
                    if shortlist:
                        st.info(f"📋 Generated shortlist with {len(shortlist)} candidate(s)")
                        for sl in shortlist:
                            name = sl.name if hasattr(sl, "name") else sl.get("name", "Unknown")
                            score = sl.final_score if hasattr(sl, "final_score") else sl.get("final_score", 0)
                            status = sl.status if hasattr(sl, "status") else sl.get("status", "pending")
                            rank = sl.rank if hasattr(sl, "rank") else sl.get("rank", 0)
                            st.markdown(f"  - #{rank} {name}: {score:.1f} ({status})")

                    # Show step count
                    sc = node_output.get("step_count", 0)
                    st.caption(f"State step count: {sc}")

    st.divider()

    # Routing decisions
    st.markdown("### 🔀 Routing Decisions")

    # Infer routing from events
    routing_events = []
    for event in events:
        for node_name in event.keys():
            routing_events.append(node_name)

    routing_descriptions = {
        "resume_analyst": "📄 Resume Analyst → Scorer",
        "scorer": "📊 Scorer → Conditional Route",
        "verifier": "🔍 Verifier → Decider",
        "decider": "⚖️ Decider → Human Approval",
        "human_approval_gate": "👤 Human Approval Gate → Scheduler",
        "scheduler": "📅 Scheduler → END",
    }

    for node_name in routing_events:
        description = routing_descriptions.get(node_name, f"▶ {node_name}")
        st.success(f"✅ {description}")

    # Full JSON for debugging
    st.divider()
    st.markdown("### 🔧 Raw Event Data (Debug)")
    with st.expander("View Raw JSON"):
        import json
        st.json(events, expanded=False)