"""
Reusable candidate card and expander components for displaying profile details.
"""

import streamlit as st
from components.status_badge import injection_badge


def candidate_card(profile: dict, idx: int = 0) -> None:
    """Display a compact summary card for a candidate.

    Args:
        profile: Dict representation of a CandidateProfile.
        idx: Index for unique key generation.
    """
    name = profile.get("name", "Unknown")
    cid = profile.get("candidate_id", "N/A")
    skills = profile.get("skills", [])
    inj_conf = profile.get("injection_confidence", 0.0)
    is_injected = profile.get("is_injection_detected", False)

    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        st.markdown(f"**{name}**  ")
        st.caption(f"ID: {cid}")
    with col2:
        st.markdown(f"Skills: {len(skills)}")
    with col3:
        if is_injected:
            badge = injection_badge(inj_conf)
            st.markdown(f"<span style='color:red'>{badge}</span>", unsafe_allow_html=True)
        else:
            st.markdown("✅ Clean")


def candidate_expander(profile: dict, key: str = "expander") -> None:
    """Display an expandable detailed view of a candidate profile.

    Args:
        profile: Dict representation of a CandidateProfile.
        key: Unique key for the expander widget.
    """
    name = profile.get("name", "Unknown")
    skills = profile.get("skills", [])
    education = profile.get("education", [])
    experience = profile.get("experience", [])
    projects = profile.get("projects", [])
    certifications = profile.get("certifications", [])
    is_injected = profile.get("is_injection_detected", False)
    inj_conf = profile.get("injection_confidence", 0.0)
    raw_text = profile.get("raw_text", "")

    with st.expander(f"📋 {name} — Details", key=key):
        # Skills
        st.markdown("**🧠 Skills**")
        if skills:
            st.write(", ".join(skills))
        else:
            st.caption("No skills listed")

        # Education
        st.markdown("**🎓 Education**")
        if education:
            for edu in education:
                st.write(f"- {edu}")
        else:
            st.caption("No education listed")

        # Experience
        st.markdown("**💼 Experience**")
        if experience:
            for exp in experience:
                role = exp.get("role", "N/A")
                company = exp.get("company", "N/A")
                years = exp.get("years", 0)
                desc = exp.get("description", "")
                st.markdown(f"**{role}** @ {company} ({years} yrs)")
                if desc:
                    st.caption(desc)
        else:
            st.caption("No experience listed")

        # Projects
        st.markdown("**📁 Projects**")
        if projects:
            for proj in projects:
                pname = proj.get("name", "Untitled")
                pdesc = proj.get("description", "")
                techs = proj.get("technologies", [])
                st.markdown(f"**{pname}**")
                if pdesc:
                    st.caption(pdesc)
                if techs:
                    st.caption(f"Tech: {', '.join(techs)}")
        else:
            st.caption("No projects listed")

        # Certifications
        st.markdown("**📜 Certifications**")
        if certifications:
            for cert in certifications:
                st.write(f"- {cert}")
        else:
            st.caption("No certifications listed")

        # Injection warning
        if is_injected:
            st.warning(f"🚨 Prompt Injection Detected (confidence: {inj_conf:.0%})")

        # Raw text (collapsible)
        if raw_text:
            with st.expander("View Raw Resume Text"):
                st.text(raw_text[:2000] + ("..." if len(raw_text) > 2000 else ""))