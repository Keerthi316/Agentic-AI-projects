"""
Verification page — blind re-score results, fairness check, injection detection.
"""

import streamlit as st
import pandas as pd


def show():
    """Render the Verification page."""
    st.markdown('<p class="main-header">✅ Verification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Blind re-score verification for borderline candidates</p>', unsafe_allow_html=True)

    state = st.session_state.workflow_state
    scorecards = state.get("scorecards", []) or []
    verified = state.get("verified_scores", []) or []

    # Check if verification is needed
    borderline_ids = set()
    for sc in scorecards:
        s = sc.model_dump() if hasattr(sc, "model_dump") else sc
        if s.get("is_borderline", False):
            borderline_ids.add(s.get("candidate_id", ""))

    # Convert verified scores to dicts
    verified_dicts = []
    for vs in verified:
        if hasattr(vs, "model_dump"):
            d = vs.model_dump()
        elif isinstance(vs, dict):
            d = vs
        else:
            continue
        verified_dicts.append(d)

    if not borderline_ids and not verified_dicts:
        st.info("✅ No borderline candidates. Verification was not required in this run.")
        # Show scorecards info
        if scorecards:
            st.info("All candidates were scored with high confidence (no borderline cases).")
        else:
            st.info("📂 No candidates scored yet. Run the workflow first.")
        return

    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔍 Borderline Candidates", len(borderline_ids))
    with col2:
        st.metric("✅ Verified", len(verified_dicts))
    with col3:
        unfair_count = sum(1 for d in verified_dicts if not d.get("is_fair", True))
        st.metric("⚠️ Unfair Scores", unfair_count)

    st.divider()

    # Show borderline candidates needing verification
    if borderline_ids and not verified_dicts:
        st.warning(f"⚠️ {len(borderline_ids)} borderline candidate(s) detected but not yet verified.")
        st.info("Run the workflow to trigger verification via blind re-scoring.")
        if st.button("▶️ Run Verification Now", use_container_width=True):
            from utils.backend import run_full_workflow
            with st.spinner("🔄 Running verification..."):
                st.session_state.workflow_state = run_full_workflow(st.session_state.workflow_state)
                st.rerun()

    # Show verification results
    if verified_dicts:
        st.markdown("### 📋 Verification Results")

        # Build name lookup
        profiles = state.get("parsed_profiles", []) or []
        name_map = {}
        for p in profiles:
            cid = p.candidate_id if hasattr(p, "candidate_id") else p.get("candidate_id", "")
            name = p.name if hasattr(p, "name") else p.get("name", "Unknown")
            name_map[cid] = name

        # Table view
        rows = []
        for d in verified_dicts:
            cid = d.get("candidate_id", "")
            rows.append({
                "Name": name_map.get(cid, "Unknown"),
                "Original Score": d.get("original_score", 0),
                "Blind Score": d.get("blind_score", 0),
                "Difference": f"{d.get('score_difference', 0):.1f}",
                "Fair": "✅ Yes" if d.get("is_fair", True) else "❌ No",
                "Injection Affected": "⚠️ Yes" if d.get("injection_affected", False) else "✅ No",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        st.divider()

        # Detailed view
        st.markdown("### 🔍 Detailed Verification")

        for d in verified_dicts:
            cid = d.get("candidate_id", "")
            name = name_map.get(cid, "Unknown")
            is_fair = d.get("is_fair", True)
            diff = d.get("score_difference", 0)
            fairness_notes = d.get("fairness_notes", "")

            with st.expander(
                f"{'✅' if is_fair else '❌'} {name} — Original: {d.get('original_score', 0):.1f} → Blind: {d.get('blind_score', 0):.1f}",
                expanded=not is_fair,
            ):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Original Score", f"{d.get('original_score', 0):.1f}")
                with col_b:
                    st.metric("Blind Score", f"{d.get('blind_score', 0):.1f}")
                with col_c:
                    st.metric("Difference", f"{diff:.1f}")

                if not is_fair:
                    st.error("⚠️ **Unfair scoring detected!** This score will be flagged for review.")
                elif diff > 10:
                    st.warning(f"⚠️ Score difference ({diff:.1f}) exceeds threshold. Using blind score.")
                else:
                    st.success(f"✅ Scores are consistent (difference: {diff:.1f}). Using average.")

                if d.get("injection_affected", False):
                    st.error("🚨 **Prompt injection may have affected scoring!**")

                if fairness_notes:
                    st.markdown("**Fairness Notes:**")
                    st.caption(fairness_notes)