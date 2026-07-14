"""
Resume Upload page — upload multiple PDF/DOCX/TXT resumes, preview, and manage.
"""

import streamlit as st
from utils.backend import extract_text_from_file


def show():
    """Render the Resume Upload page."""
    st.markdown('<p class="main-header">📤 Resume Upload</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload candidate resumes (PDF, DOCX, TXT)</p>', unsafe_allow_html=True)

    state = st.session_state.workflow_state

    # Check if JD is configured first
    if not state.get("jd"):
        st.warning("⚠️ Please configure a Job Description first before uploading resumes.")
        if st.button("📄 Go to Job Description", use_container_width=True):
            st.session_state.page = "Job Description"
            st.rerun()
        return

    # File upload area
    st.markdown("### 📎 Upload Resumes")

    uploaded_files = st.file_uploader(
        "Choose resume files (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload one or more resume files. Supported formats: PDF, DOCX, TXT",
    )

    if uploaded_files:
        with st.spinner("📄 Processing uploaded files..."):
            for uploaded_file in uploaded_files:
                # Check if already uploaded
                already_uploaded = any(
                    uf.name == uploaded_file.name for uf in st.session_state.uploaded_files
                )
                if not already_uploaded:
                    st.session_state.uploaded_files.append(uploaded_file)

        st.success(f"✅ {len(uploaded_files)} file(s) processed")

    # Display uploaded files
    if st.session_state.uploaded_files:
        st.markdown("### 📋 Uploaded Files")
        st.markdown(f"**Total:** {len(st.session_state.uploaded_files)} files")

        for i, uf in enumerate(st.session_state.uploaded_files):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.markdown(f"📄 **{uf.name}**")
            with col2:
                size_kb = len(uf.getvalue()) / 1024
                st.caption(f"{size_kb:.1f} KB")
            with col3:
                file_type = uf.name.split(".")[-1].upper()
                st.caption(file_type)
            with col4:
                if st.button("❌ Remove", key=f"remove_{i}"):
                    st.session_state.uploaded_files.pop(i)
                    st.rerun()

        # Extract text and add to state
        st.divider()
        st.markdown("### 🔄 Process Resumes")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("📄 Extract Text from All Files", use_container_width=True, type="primary"):
                with st.spinner("Extracting text from resumes..."):
                    new_candidates = []
                    progress_bar = st.progress(0)
                    for i, uf in enumerate(st.session_state.uploaded_files):
                        try:
                            text = extract_text_from_file(uf)
                            if text.strip():
                                new_candidates.append(text)
                        except Exception as e:
                            st.error(f"Failed to extract {uf.name}: {e}")
                        progress_bar.progress((i + 1) / len(st.session_state.uploaded_files))

                    if new_candidates:
                        st.session_state.workflow_state["candidates"] = new_candidates
                        st.success(f"✅ Extracted text from {len(new_candidates)} resume(s)")
                        st.rerun()
                    else:
                        st.error("❌ No text could be extracted from the uploaded files")

        with col_b:
            if st.button("📝 Paste Resume Text Instead", use_container_width=True):
                st.session_state.show_text_input = True
                st.rerun()

        # Manual text input option
        if st.session_state.get("show_text_input", False):
            st.markdown("### ✏️ Paste Resume Text")
            resume_text = st.text_area(
                "Paste the full resume text below:",
                height=300,
                placeholder="Paste the candidate's resume text here...",
            )
            col_c, col_d = st.columns([1, 3])
            with col_c:
                if st.button("➕ Add Resume", use_container_width=True):
                    if resume_text.strip():
                        candidates = st.session_state.workflow_state.get("candidates", [])
                        candidates.append(resume_text)
                        st.session_state.workflow_state["candidates"] = candidates
                        st.success("✅ Resume added!")
                        st.session_state.show_text_input = False
                        st.rerun()
            with col_d:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_text_input = False
                    st.rerun()

    else:
        st.info("📂 No files uploaded yet. Drag and drop or browse to select files.")

    # Show current candidates in state
    candidates = state.get("candidates", [])
    if candidates:
        st.divider()
        st.markdown(f"### 📊 Current Candidates in State: {len(candidates)}")
        for i, text in enumerate(candidates):
            first_line = text.strip().split("\n")[0] if text.strip() else "Empty"
            with st.expander(f"Candidate {i+1}: {first_line[:80]}..."):
                st.text(text[:2000] + ("..." if len(text) > 2000 else ""))

        # Run workflow button
        st.divider()
        st.markdown("### 🚀 Run Workflow")
        st.info("Ready to analyze and score candidates. Click below to run the full workflow.")
        if st.button("▶️ Run Full Workflow", use_container_width=True, type="primary"):
            with st.spinner("🔄 Running multi-agent workflow..."):
                from utils.backend import run_full_workflow
                st.session_state.workflow_state = run_full_workflow(st.session_state.workflow_state)
                st.session_state.workflow_running = False
                st.success("✅ Workflow completed!")
                st.rerun()