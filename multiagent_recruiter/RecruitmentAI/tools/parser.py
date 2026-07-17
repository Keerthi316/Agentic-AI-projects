"""
Resume parser: extracts plain text from PDF, DOCX, and TXT files.
Includes prompt-injection sanitization before text reaches any LLM.
"""
from __future__ import annotations

import io
import re
from typing import Tuple

from utils.logger import get_logger

logger = get_logger("parser")

# ── Prompt-injection patterns to strip ────────────────────────────────────
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"override\s+(all\s+)?instructions?",
    r"(give\s+me|assign\s+me|set\s+(my\s+)?score\s+to)\s+\d+\s*(marks?|points?|score)?",
    r"rank\s+me\s+(first|#1|number\s+one|top)",
    r"(recommend|select|hire|shortlist)\s+me\s+immediately",
    r"call\s+propose_interview",
    r"score\s*[:=]\s*\d+",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions?|prompt)",
    r"(print|show|display|output)\s+(your\s+)?(system\s+prompt|instructions?)",
    r"what\s+(are\s+)?(your\s+)?instructions?",
    r"you\s+are\s+now\s+a?\s+",
    r"act\s+as\s+(a\s+|an\s+)?",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"new\s+instructions?\s*:",
    r"^system\s*:",
    r"<\s*/?system\s*>",
    r"\[INST\]",
    r"###\s*(instruction|system|input)\s*:",
    r"give\s+me\s+100",
    r"(maximum|full|perfect)\s+(score|marks|points|rating)",
]

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS),
    flags=re.IGNORECASE | re.MULTILINE,
)


def sanitize_resume_text(text: str, filename: str) -> Tuple[str, bool]:
    """
    Sanitize resume text by removing prompt-injection attempts.
    Returns (cleaned_text, injection_detected).
    """
    matches = list(_INJECTION_RE.finditer(text))
    if matches:
        logger.warning(
            "PROMPT INJECTION detected in '%s' — %d suspicious pattern(s) found and removed.",
            filename,
            len(matches),
        )
        for m in matches:
            logger.warning("  Matched: %r", m.group()[:80])
        cleaned = _INJECTION_RE.sub("[REDACTED]", text)
        return cleaned, True
    return text, False


def extract_text(file_bytes: bytes, filename: str) -> Tuple[str, str]:
    """
    Extract and sanitize text from PDF, DOCX, or TXT.
    Returns (sanitized_text, error_message).
    error_message is empty string on success.
    """
    ext = filename.rsplit(".", 1)[-1].lower()
    try:
        if ext == "pdf":
            raw = _parse_pdf(file_bytes)
        elif ext == "docx":
            raw = _parse_docx(file_bytes)
        elif ext == "txt":
            raw = file_bytes.decode("utf-8", errors="replace")
        else:
            return "", f"Unsupported file type: {ext}"

        sanitized, injected = sanitize_resume_text(raw, filename)
        return sanitized, ""

    except Exception as exc:
        logger.exception("Parsing failed for %s", filename)
        return "", str(exc)


def _parse_pdf(data: bytes) -> str:
    import PyPDF2  # type: ignore
    reader = PyPDF2.PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def _parse_docx(data: bytes) -> str:
    import docx  # type: ignore
    doc = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
