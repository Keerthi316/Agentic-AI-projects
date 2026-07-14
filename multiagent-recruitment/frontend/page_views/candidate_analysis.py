"""
Candidate Analysis page — display parsed profiles with skills, education, experience,
projects, certifications, and prompt injection warnings.
"""

import streamlit as st
from components.status_badge import injection_badge
from components.candidate_card import candidate_card, candidate_expander


def show():
    """Render the Candidate Analysis page."""
    st.markdown('<p class="main-header">🔍 Candidate Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Parsed candidate profiles from resumes</p>', unsafe_allow_html=True)

    state = st.session_state.workflow_state
    profiles = state.get("parsed_profiles", []) or []

    # Handle both dict and Pydantic model formats
    profiles_dicts = []
    for p in profiles:
        if hasattr(p, "model_dump"):
            profiles_dicts.append(p.model_dump())
        elif isinstance(p, dict):
            profiles_dicts.append(p)
        else:
            profiles_dicts.append({})

    if not profiles_dicts:
        # Check if raw candidates exist but haven't been parsed
        candidates = state.get("candidates", []) or []
        if candidates:
            st.warning(f"⚠️ {len(candidates)} resume(s) uploaded but not yet parsed.")
            st.info("Run the workflow from the **Resume Upload** page to parse resumes.")
            if st.button("📤 Go to Resume Upload", use_container_width=True):
                st.session_state.page = "Resume Upload"
                st.rerun()
        else:
            st.info("📂 No resumes uploaded yet. Start by uploading resumes.")
            if st.button("📤 Go to Resume Upload", use_container_width=True):
                st.session_state.page = "Resume Upload"
                st.rerun()
        return

    # Summary stats
    injection_count = sum(1 for p in profiles_dicts if p.get("is_injection_detected", False))
    total_skills = sum(len(p.get("skills", [])) for p in profiles_dicts)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👤 Candidates", len(profiles_dicts))
    with col2:
        st.metric("🧠 Total Skills Listed", total_skills)
    with col3:
        st.metric("🚨 Injections Detected", injection_count)
    with col4:
        injection_risk = "⚠️ High" if injection_count > 0 else "✅ None"
        st.metric("Injection Risk", injection_risk)

    st.divider()

    # Search and filter
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        search_term = st.text_input("🔍 Search candidates by name or skills", placeholder="e.g., Python, John...")
    with col_s2:
        filter_injected = st.checkbox("Show only injection warnings", value=False)

    # Filter profiles
    filtered_profiles = profiles_dicts
    if search_term:
        search_lower = search_term.lower()
        filtered_profiles = [
            p for p in filtered_profiles
            if search_lower in p.get("name", "").lower()
            or any(search_lower in skill.lower() for skill in p.get("skills", []))
        ]
    if filter_injected:
        filtered_profiles = [p for p in filtered_profiles if p.get("is_injection_detected", False)]

    st.markdown(f"### 📋 Parsed Profiles ({len(filtered_profiles)} displayed)")

    # Display each candidate
    for i, profile in enumerate(filtered_profiles):
        with st.container(border=True):
            candidate_card(profile, idx=i)
            candidate_expander(profile, key=f"candidate_{i}_details")