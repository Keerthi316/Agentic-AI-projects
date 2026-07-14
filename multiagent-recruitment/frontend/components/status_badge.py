"""
Reusable status badge components for displaying candidate states.

Each badge returns an emoji + colored label for visual clarity.
"""

import streamlit as st


def status_badge(status: str) -> str:
    """Return an emoji badge for a candidate's status.

    Args:
        status: One of 'shortlisted', 'hold', 'rejected', 'pending'.

    Returns:
        Emoji string for the status.
    """
    badges = {
        "shortlisted": "✅",
        "hold": "⏳",
        "rejected": "❌",
        "pending": "🔄",
    }
    return badges.get(status.lower(), "❓")


def score_badge(score: float) -> None:
    """Render a color-coded score badge in the sidebar.

    Args:
        score: Numeric score (0-100).
    """
    if score >= 80:
        st.markdown(f"<span style='color:#00cc66; font-weight:bold; font-size:1.2em;'>{score:.1f}/100</span>",
                    unsafe_allow_html=True)
    elif score >= 60:
        st.markdown(f"<span style='color:#ffaa00; font-weight:bold; font-size:1.2em;'>{score:.1f}/100</span>",
                    unsafe_allow_html=True)
    elif score >= 40:
        st.markdown(f"<span style='color:#ff6600; font-weight:bold; font-size:1.2em;'>{score:.1f}/100</span>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:#ff3333; font-weight:bold; font-size:1.2em;'>{score:.1f}/100</span>",
                    unsafe_allow_html=True)


def injection_badge(confidence: float) -> str:
    """Return an emoji + text for injection detection confidence.

    Args:
        confidence: Confidence score (0.0 to 1.0).

    Returns:
        Emoji + label string.
    """
    if confidence >= 0.7:
        return "🚨 HIGH"
    elif confidence >= 0.4:
        return "⚠️ MEDIUM"
    elif confidence > 0.0:
        return "🔍 LOW"
    return "✅ CLEAN"