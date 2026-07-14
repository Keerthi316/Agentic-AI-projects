"""
Backend integration bridge — connects the Streamlit frontend to the existing
LangGraph recruitment workflow.

Design decisions:
1. Lazy imports: The backend modules are only imported when needed, so the
   frontend can start up quickly and doesn't crash if backend deps are missing.

2. Session state management: All workflow state is stored in Streamlit's
   session_state so it persists across page navigations and re-runs.

3. Demo mode detection: The bridge respects the existing demo mode in
   tools/llm.py, so the frontend works without an API key.

4. No business logic duplication: The bridge only calls existing backend
   functions (agents, graph, models). It does NOT re-implement any logic.

5. File parsing: Uses the existing io module and textract/pdfminer for
   PDF/DOCX parsing. Falls back to plain text extraction.
"""

import io
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

# Backend imports (lazy — imported when first used)
_BACKEND_LOADED = False

logger = logging.getLogger(__name__)


def _load_backend():
    """Lazy-load backend modules.

    This is called once when the first backend operation is triggered.
    It ensures the frontend can start even if backend dependencies are
    not yet installed.
    """
    global _BACKEND_LOADED
    if _BACKEND_LOADED:
        return

    # Add the project root to sys.path so imports work
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Set demo mode if no API key
    if not os.getenv("OPENAI_API_KEY", ""):
        os.environ["RECRUITMENT_DEMO_MODE"] = "true"

    _BACKEND_LOADED = True


def get_initial_state() -> Dict[str, Any]:
    """Return the default initial state with all fields.

    This mirrors RecruitmentState but with empty/default values,
    ready to be populated by the user through the UI.
    """
    return {
        "jd": None,
        "candidates": [],
        "parsed_profiles": [],
        "scorecards": [],
        "verified_scores": [],
        "revision_count": 0,
        "shortlist": [],
        "step_count": 0,
        "errors": [],
        "needs_human_escalation": False,
        "human_approved": False,
        "workflow_events": [],
        "workflow_complete": False,
        "execution_time_ms": 0,
        "schedules": [],
    }


# Keys that exist only in the frontend state dict and must be stripped
# before passing to RecruitmentState or graph.stream().
_FRONTEND_ONLY_KEYS = {"workflow_events", "workflow_complete", "execution_time_ms"}


def _to_recruitment_state(state: Dict[str, Any]):
    """Build a RecruitmentState from the frontend state dict.

    Strips frontend-only keys that are not part of RecruitmentState so
    LangGraph does not raise a validation error on unknown fields.

    Args:
        state: The frontend workflow state dictionary.

    Returns:
        A RecruitmentState TypedDict ready for graph.stream().
    """
    from models.state import RecruitmentState
    clean = {k: v for k, v in state.items() if k not in _FRONTEND_ONLY_KEYS}
    return RecruitmentState(**clean)


def validate_jd(title: str, description: str, required_skills: List[str],
                preferred_skills: List[str], min_experience: int,
                education: str) -> Tuple[bool, Optional[Any], str]:
    """Validate a job description using the backend JDInput model.

    Args:
        title, description, required_skills, preferred_skills, min_experience, education

    Returns:
        (is_valid, JDInput_instance, error_message)
    """
    _load_backend()
    from models.state import JDInput
    from pydantic import ValidationError

    try:
        jd = JDInput(
            title=title,
            description=description,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            min_experience_years=min_experience,
            education_requirement=education,
        )
        return True, jd, ""
    except ValidationError as e:
        return False, None, str(e)


def parse_resume_text(text: str, candidate_id: str) -> Optional[Dict[str, Any]]:
    """Parse resume text using the backend Resume Analyst.

    Args:
        text: Raw resume text.
        candidate_id: Unique identifier for the candidate.

    Returns:
        Dict representation of CandidateProfile, or None on failure.
    """
    _load_backend()
    from agents.resume_analyst import _detect_injection, _extract_profile, _validate_profile
    from models.state import CandidateProfile

    try:
        injection_check = _detect_injection(text)
        profile = _extract_profile(text, candidate_id, injection_check)

        if profile and _validate_profile(profile):
            return profile.model_dump()
        return None
    except Exception as e:
        logger.error(f"Resume parsing failed: {e}")
        return None


def score_candidates_backend(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the Scorer agent on parsed profiles.

    Args:
        state: Current workflow state with parsed_profiles and jd.

    Returns:
        Updated state with scorecards populated.
    """
    _load_backend()
    from agents.scorer import score_candidates

    ts = _to_recruitment_state(state)
    result = score_candidates(ts)
    state.update(result)
    return state


def verify_scores_backend(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the Verifier agent on borderline candidates.

    Args:
        state: Current workflow state with scorecards.

    Returns:
        Updated state with verified_scores populated.
    """
    _load_backend()
    from agents.verifier import verify_scores

    ts = _to_recruitment_state(state)
    result = verify_scores(ts)
    state.update(result)
    return state


def generate_shortlist_backend(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the Decider agent to generate the ranked shortlist.

    Args:
        state: Current workflow state with scorecards and verified_scores.

    Returns:
        Updated state with shortlist populated.
    """
    _load_backend()
    from agents.decider import generate_shortlist

    ts = _to_recruitment_state(state)
    result = generate_shortlist(ts)
    state.update(result)
    return state


def schedule_interviews_backend(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the Scheduler agent to generate interview templates.

    Args:
        state: Current workflow state (must have shortlist and human_approved=True).

    Returns:
        Updated state with scheduling results.
    """
    _load_backend()
    from agents.scheduler import schedule_interviews

    ts = _to_recruitment_state(state)
    result = schedule_interviews(ts)
    state.update(result)
    return state


def run_full_workflow(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete LangGraph workflow from start to finish.

    This executes the full pipeline: Analyst → Scorer → (Verifier) → Decider
    → Human Approval → Scheduler, using the existing graph.

    Args:
        state: Initial workflow state with jd and candidates.

    Returns:
        Final state with all results and execution events.
    """
    _load_backend()
    from graph.workflow import build_recruitment_graph

    start_time = time.time()
    events = []
    errors = []

    # Strip frontend-only keys before handing state to LangGraph
    ts = _to_recruitment_state(state)

    # Build the graph
    graph = build_recruitment_graph()

    # Stream execution with event capture
    try:
        for event in graph.stream(ts):
            events.append(event)
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    state.update(node_output)
                    if node_output.get("errors"):
                        errors.extend(node_output["errors"])
    except Exception as e:
        errors.append(f"Workflow execution error: {str(e)}")
        logger.error(f"Workflow failed: {e}")

    execution_time = int((time.time() - start_time) * 1000)

    state["workflow_events"] = events
    state["workflow_complete"] = len(errors) == 0 or state.get("shortlist") is not None
    state["execution_time_ms"] = execution_time
    state["errors"] = state.get("errors", []) + errors

    return state


def extract_text_from_file(uploaded_file) -> str:
    """Extract text from an uploaded file (PDF, DOCX, TXT).

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        Extracted text content.
    """
    import tempfile

    _load_backend()

    file_name = uploaded_file.name.lower()
    content = uploaded_file.read()

    if file_name.endswith(".txt"):
        return content.decode("utf-8", errors="replace")

    elif file_name.endswith(".pdf"):
        # Try PyPDF2, then pdfminer, then fallback to raw text
        try:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(io.BytesIO(content))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                pass

            try:
                from pdfminer.high_level import extract_text
                return extract_text(io.BytesIO(content))
            except ImportError:
                pass

            # Fallback: write to temp and use pdftotext
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                import subprocess
                result = subprocess.run(["pdftotext", tmp_path, "-"], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout
            except:
                pass
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass

            return content.decode("utf-8", errors="replace")
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return content.decode("utf-8", errors="replace")

    elif file_name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            try:
                import zipfile
                import xml.etree.ElementTree as ET
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    xml_content = z.read("word/document.xml")
                    root = ET.fromstring(xml_content)
                    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    texts = [t.text for t in root.findall(".//w:t", ns) if t.text]
                    return " ".join(texts)
            except Exception as e:
                logger.error(f"DOCX extraction failed: {e}")
                return content.decode("utf-8", errors="replace")

    else:
        return content.decode("utf-8", errors="replace")


def state_to_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a human-readable summary of the current workflow state.

    Args:
        state: The current workflow state dictionary.

    Returns:
        Summary dict with counts and status flags.
    """
    profiles = state.get("parsed_profiles", []) or []
    scorecards = state.get("scorecards", []) or []
    verified = state.get("verified_scores", []) or []
    shortlist = state.get("shortlist", []) or []
    errors = state.get("errors", []) or []

    return {
        "has_jd": state.get("jd") is not None,
        "candidate_count": len(state.get("candidates", []) or []),
        "parsed_count": len(profiles),
        "scored_count": len(scorecards),
        "borderline_count": sum(1 for s in scorecards if hasattr(s, "is_borderline") and s.is_borderline),
        "verified_count": len(verified),
        "shortlist_count": len(shortlist),
        "shortlisted_count": sum(1 for s in shortlist if hasattr(s, "status") and s.status == "shortlisted"),
        "error_count": len(errors),
        "revision_count": state.get("revision_count", 0),
        "step_count": state.get("step_count", 0),
        "workflow_complete": state.get("workflow_complete", False),
        "human_approved": state.get("human_approved", False),
        "needs_escalation": state.get("needs_human_escalation", False),
    }