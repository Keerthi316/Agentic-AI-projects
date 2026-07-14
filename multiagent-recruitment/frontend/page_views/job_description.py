"""
Job Description page — paste, preview, edit, and submit job descriptions.
"""

import streamlit as st
from utils.backend import validate_jd


# Sample job descriptions for quick filling
SAMPLE_JDS = {
    "Senior Python Backend Engineer": {
        "title": "Senior Python Backend Engineer",
        "description": "We are looking for a Senior Python Backend Engineer to join our team. "
                       "The ideal candidate has strong experience with Python, FastAPI/Django, PostgreSQL, "
                       "and cloud services (AWS/GCP). You will design and build scalable microservices, "
                       "work with event-driven architectures, and mentor junior engineers.",
        "required_skills": ["Python", "FastAPI", "PostgreSQL", "AWS", "REST APIs"],
        "preferred_skills": ["Kubernetes", "Docker", "Redis", "Kafka", "GraphQL"],
        "min_experience": 4,
        "education": "Bachelor's in Computer Science or related field",
    },
    "Data Scientist": {
        "title": "Data Scientist",
        "description": "We need a Data Scientist with expertise in machine learning, "
                       "statistical analysis, and Python. Experience with deep learning "
                       "frameworks and cloud ML services is a plus.",
        "required_skills": ["Python", "Machine Learning", "Statistics", "SQL", "PyTorch"],
        "preferred_skills": ["TensorFlow", "AWS SageMaker", "Spark", "Kubernetes"],
        "min_experience": 3,
        "education": "Master's in Data Science, CS, or related field",
    },
}


def show():
    """Render the Job Description page."""
    st.markdown('<p class="main-header">📄 Job Description</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Define the position requirements</p>', unsafe_allow_html=True)

    state = st.session_state.workflow_state
    existing_jd = state.get("jd")

    # Show existing JD if already set
    if existing_jd:
        st.success("✅ Job description is already configured.")
        with st.expander("📄 View Current Job Description", expanded=True):
            st.markdown(f"**Title:** {existing_jd.title}")
            st.markdown(f"**Description:** {existing_jd.description}")
            st.markdown(f"**Required Skills:** {', '.join(existing_jd.required_skills)}")
            st.markdown(f"**Preferred Skills:** {', '.join(existing_jd.preferred_skills)}")
            st.markdown(f"**Min Experience:** {existing_jd.min_experience_years} years")
            st.markdown(f"**Education:** {existing_jd.education_requirement or 'Not specified'}")

        if st.button("✏️ Edit Job Description", use_container_width=True):
            st.session_state.workflow_state["jd"] = None
            st.rerun()

        st.divider()

    # Load sample or edit form
    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### 📋 Sample Templates")
        for name, data in SAMPLE_JDS.items():
            if st.button(f"📄 {name}", use_container_width=True):
                # Fill the form fields via session state
                st.session_state.jd_form = data.copy()
                st.rerun()

        st.markdown("### ℹ️ Tips")
        st.info(
            "• Be specific about required skills\n"
            "• Include experience years\n"
            "• List preferred qualifications\n"
            "• Add education requirements\n"
            "• Use bullet points for clarity"
        )

    with col1:
        st.markdown("### ✏️ Job Description Form")

        # Initialize form with existing or sample data
        form_defaults = getattr(st.session_state, "jd_form", {}) or {}

        with st.form("jd_form"):
            title = st.text_input(
                "Job Title *",
                value=form_defaults.get("title", existing_jd.title if existing_jd else ""),
                placeholder="e.g., Senior Python Backend Engineer",
            )

            description = st.text_area(
                "Job Description *",
                value=form_defaults.get("description", existing_jd.description if existing_jd else ""),
                placeholder="Describe the role, responsibilities, and ideal candidate...",
                height=200,
            )

            col_a, col_b = st.columns(2)
            with col_a:
                required_skills_str = st.text_area(
                    "Required Skills (one per line)",
                    value="\n".join(form_defaults.get("required_skills", existing_jd.required_skills if existing_jd else [])),
                    placeholder="Python\nFastAPI\nPostgreSQL",
                    height=100,
                )
            with col_b:
                preferred_skills_str = st.text_area(
                    "Preferred Skills (one per line)",
                    value="\n".join(form_defaults.get("preferred_skills", existing_jd.preferred_skills if existing_jd else [])),
                    placeholder="Kubernetes\nDocker\nRedis",
                    height=100,
                )

            min_experience = st.number_input(
                "Minimum Experience (years)",
                min_value=0,
                max_value=20,
                value=form_defaults.get("min_experience", existing_jd.min_experience_years if existing_jd else 0),
            )

            education = st.text_input(
                "Education Requirement",
                value=form_defaults.get("education", existing_jd.education_requirement if existing_jd else ""),
                placeholder="e.g., Bachelor's in Computer Science",
            )

            submitted = st.form_submit_button("✅ Save Job Description", use_container_width=True, type="primary")

            if submitted:
                required_skills = [s.strip() for s in required_skills_str.split("\n") if s.strip()]
                preferred_skills = [s.strip() for s in preferred_skills_str.split("\n") if s.strip()]

                is_valid, jd_obj, error = validate_jd(
                    title=title,
                    description=description,
                    required_skills=required_skills,
                    preferred_skills=preferred_skills,
                    min_experience=min_experience,
                    education=education,
                )

                if is_valid:
                    st.session_state.workflow_state["jd"] = jd_obj
                    if hasattr(st.session_state, "jd_form"):
                        del st.session_state.jd_form
                    st.success("✅ Job description saved successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Validation failed: {error}")

    # Visual separator
    st.divider()

    # Flow guidance
    if state.get("jd"):
        st.info("✅ Job description is ready. Proceed to **Resume Upload** to add candidates.")
        if st.button("📤 Go to Resume Upload", use_container_width=True):
            st.session_state.page = "Resume Upload"
            st.rerun()
    else:
        st.warning("⚠️ No job description configured yet. Fill out the form above to start.")