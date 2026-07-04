"""
Bug Report Triage – Streamlit UI
=================================
Run with:  streamlit run app.py
"""

import json
import time

import streamlit as st
from main import stage1_understand, stage2_reason, stage3_produce, stage4_selfcheck

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bug Report Triage",
    page_icon="🐛",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Preset bug reports for quick testing
# ---------------------------------------------------------------------------
PRESETS = {
    "— choose a preset —": "",
    "Case 1 · Normal (Mobile Safari Login)": """\
Title: Login button unresponsive on mobile Safari

Description:
When a user tries to log in using mobile Safari on iOS 17, tapping the
"Login" button does nothing. The button appears to register the tap
(briefly highlights) but the form is never submitted and the page does
not navigate. This works fine on Chrome for Android and desktop browsers.

Steps to reproduce:
1. Open https://app.example.com/login on an iPhone running iOS 17 with Safari.
2. Enter valid credentials.
3. Tap the "Login" button.

Expected: The user is authenticated and redirected to /dashboard.
Actual: Nothing happens. No network request is made (confirmed via proxy).

Environment: iOS 17.4, Safari 17, iPhone 14 Pro. App version 3.2.1.""",

    "Case 2 · Complex (Java NullPointerException)": """\
Title: NullPointerException in PaymentService during checkout

Description:
Production is throwing a NullPointerException roughly 30% of the time
during the final step of the checkout flow. The cart service appears to
return a payment intent object that sometimes has a null `currency` field.
This happens exclusively for guest checkout users; logged-in users are
unaffected. Started occurring after the v4.1.0 deploy on 2026-06-28.

Stack Trace:
java.lang.NullPointerException: Cannot invoke "String.toLowerCase()" because "currency" is null
    at com.example.payment.PaymentService.normalizeCurrency(PaymentService.java:142)
    at com.example.payment.PaymentService.processPayment(PaymentService.java:87)
    at com.example.checkout.CheckoutController.completeOrder(CheckoutController.java:214)

Frequency: ~30% of guest checkout attempts.
Deploy that introduced the issue: v4.1.0 (2026-06-28).
Related PR: #1847 – "Unify currency handling across cart and payment services".""",

    "Case 3 · Broken / Gibberish Input": """\
asdfghjkl thing broken pls fix urgent!!!
the widget thingy does not do the stuff it should when you click the
blue doohickey after logging in sometimes maybe on Tuesdays??
error says something like "bad" idk""",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def severity_color(severity: str) -> str:
    """Return a CSS hex color for a severity label."""
    return {
        "critical": "#d62728",
        "high":     "#ff7f0e",
        "medium":   "#f5c518",
        "low":      "#2ca02c",
    }.get(severity.lower(), "#888888")


def score_color(score) -> str:
    """Return a CSS hex color for a quality score (1-10)."""
    try:
        s = int(score)
    except (TypeError, ValueError):
        return "#888888"
    if s >= 8:
        return "#2ca02c"
    if s >= 5:
        return "#f5c518"
    return "#d62728"


def render_json_card(data: dict, title: str, expanded: bool = True) -> None:
    """Render a dict as a formatted JSON block inside an expander."""
    with st.expander(title, expanded=expanded):
        st.json(data)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

st.title("🐛 Bug Report Triage Pipeline")
st.caption("4-Stage LLM Pipeline  ·  Powered by OpenRouter")

st.markdown("---")

# -- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.info(
        "Make sure your `OPENROUTER_API_KEY` is set in the `.env` file "
        "in the project directory before running.",
        icon="🔑",
    )
    st.markdown("---")
    st.markdown("**Pipeline stages**")
    st.markdown(
        "1. 🔍 **Understand** – QA extraction\n"
        "2. 🧠 **Reason** – Root-cause CoT\n"
        "3. 🛠️ **Produce** – Fix recommendation\n"
        "4. ✅ **Self-Check** – Quality audit"
    )
    st.markdown("---")
    st.caption("Results are also saved to `pipeline_results.json`.")

# -- Input section -----------------------------------------------------------
st.subheader("📝 Enter a Bug Report")

col_preset, _ = st.columns([2, 3])
with col_preset:
    preset_choice = st.selectbox("Load a preset", options=list(PRESETS.keys()))

# Pre-fill textarea when a preset is selected
initial_text = PRESETS[preset_choice]
bug_report = st.text_area(
    "Bug report text",
    value=initial_text,
    height=260,
    placeholder="Paste your bug report here, or pick a preset above…",
    label_visibility="collapsed",
)

run_btn = st.button("🚀 Run Triage Pipeline", type="primary", use_container_width=True)

st.markdown("---")

# -- Pipeline execution ------------------------------------------------------
if run_btn:
    if not bug_report.strip():
        st.warning("Please enter a bug report before running the pipeline.")
        st.stop()

    # Progress bar + status placeholders
    progress = st.progress(0, text="Starting pipeline…")
    status   = st.empty()

    results = {}
    start   = time.time()

    # ---- Stage 1 -----------------------------------------------------------
    status.info("🔍 Stage 1 – Understanding the bug report…")
    try:
        results["stage1"] = stage1_understand(bug_report)
        progress.progress(25, text="Stage 1 complete")
    except Exception as exc:
        results["stage1"] = {"error": str(exc)}
        st.error(f"Stage 1 failed: {exc}")

    # ---- Stage 2 -----------------------------------------------------------
    status.info("🧠 Stage 2 – Reasoning about root cause…")
    try:
        results["stage2"] = stage2_reason(results["stage1"])
        progress.progress(50, text="Stage 2 complete")
    except Exception as exc:
        results["stage2"] = {"error": str(exc)}
        st.error(f"Stage 2 failed: {exc}")

    # ---- Stage 3 -----------------------------------------------------------
    status.info("🛠️ Stage 3 – Producing fix recommendation…")
    try:
        results["stage3"] = stage3_produce(results["stage1"], results["stage2"])
        progress.progress(75, text="Stage 3 complete")
    except Exception as exc:
        results["stage3"] = {"error": str(exc)}
        st.error(f"Stage 3 failed: {exc}")

    # ---- Stage 4 -----------------------------------------------------------
    status.info("✅ Stage 4 – Running quality self-check…")
    try:
        results["stage4"] = stage4_selfcheck(
            bug_report,
            results["stage1"],
            results["stage2"],
            results["stage3"],
        )
        progress.progress(100, text="Pipeline complete!")
    except Exception as exc:
        results["stage4"] = {"error": str(exc)}
        st.error(f"Stage 4 failed: {exc}")

    elapsed = time.time() - start
    status.success(f"✅ Pipeline finished in {elapsed:.1f}s")

    # Save to JSON
    with open("pipeline_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    st.markdown("---")

    # ---- Summary metrics row -----------------------------------------------
    st.subheader("📊 Summary")

    s1 = results.get("stage1", {})
    s2 = results.get("stage2", {})
    s3 = results.get("stage3", {})
    s4 = results.get("stage4", {})

    severity   = s2.get("severity", "N/A")
    confidence = s2.get("confidence", "N/A")
    score      = s4.get("quality_score", "N/A")
    needs_rev  = s4.get("needs_revision", "N/A")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🔥 Severity",    severity)
    m2.metric("🎯 Confidence",  confidence)
    m3.metric("⭐ Quality Score", f"{score}/10" if score != "N/A" else "N/A")
    m4.metric("🔄 Needs Revision", "Yes" if needs_rev is True else ("No" if needs_rev is False else str(needs_rev)))

    st.markdown("---")

    # ---- Stage outputs side by side ----------------------------------------
    st.subheader("🔍 Stage 1 – Understand")
    col1a, col1b = st.columns(2)
    with col1a:
        st.markdown("**Bug Summary**")
        st.info(s1.get("bug_summary", "N/A"))
        st.markdown("**Affected Component**")
        st.code(s1.get("affected_component", "N/A"), language=None)
    with col1b:
        st.markdown("**Steps to Reproduce**")
        steps = s1.get("steps_to_reproduce", [])
        if steps:
            for i, step in enumerate(steps, 1):
                st.markdown(f"{i}. {step}")
        else:
            st.caption("None extracted")
        st.markdown("**Missing Information**")
        missing = s1.get("missing_information", [])
        if missing:
            for m in missing:
                st.markdown(f"- {m}")
        else:
            st.caption("None")
    render_json_card(s1, "Full Stage 1 JSON", expanded=False)

    st.markdown("---")

    # ---- Stage 2 -----------------------------------------------------------
    st.subheader("🧠 Stage 2 – Reason")
    col2a, col2b = st.columns(2)
    with col2a:
        st.markdown("**Likely Root Cause**")
        st.warning(s2.get("likely_root_cause", "N/A"))
        sev_val = s2.get("severity", "N/A")
        sev_col = severity_color(sev_val)
        st.markdown(
            f"**Severity:** <span style='color:{sev_col}; font-weight:bold'>{sev_val}</span> "
            f"&nbsp;|&nbsp; **Confidence:** {s2.get('confidence', 'N/A')}",
            unsafe_allow_html=True,
        )
    with col2b:
        st.markdown("**Chain-of-Thought Reasoning**")
        reasoning = s2.get("reasoning", [])
        if reasoning:
            for i, step in enumerate(reasoning, 1):
                st.markdown(f"{i}. {step}")
        else:
            st.caption("No reasoning steps returned")
    render_json_card(s2, "Full Stage 2 JSON", expanded=False)

    st.markdown("---")

    # ---- Stage 3 -----------------------------------------------------------
    st.subheader("🛠️ Stage 3 – Produce")
    st.markdown("**Developer Summary**")
    st.success(s3.get("developer_summary", "N/A"))
    st.markdown("**Recommended Fix**")
    st.markdown(s3.get("recommended_fix", "N/A"))
    st.markdown("**Next Debugging Steps**")
    steps3 = s3.get("next_debugging_steps", [])
    if steps3:
        for i, step in enumerate(steps3, 1):
            st.markdown(f"{i}. {step}")
    else:
        st.caption("No steps returned")
    render_json_card(s3, "Full Stage 3 JSON", expanded=False)

    st.markdown("---")

    # ---- Stage 4 -----------------------------------------------------------
    st.subheader("✅ Stage 4 – Self-Check")
    col4a, col4b = st.columns(2)
    with col4a:
        sc  = s4.get("quality_score", "N/A")
        col = score_color(sc)
        st.markdown(
            f"**Quality Score:** <span style='font-size:2rem; color:{col}; font-weight:bold'>{sc}/10</span>",
            unsafe_allow_html=True,
        )
        st.markdown("**Issues Found**")
        issues = s4.get("issues_found", [])
        if issues:
            for issue in issues:
                st.markdown(f"- {issue}")
        else:
            st.caption("No issues found")
    with col4b:
        st.markdown("**Revised Summary**")
        st.info(s4.get("revised_summary", "N/A"))
    render_json_card(s4, "Full Stage 4 JSON", expanded=False)

    st.markdown("---")

    # ---- Download button ---------------------------------------------------
    st.download_button(
        label="⬇️ Download Full Results (JSON)",
        data=json.dumps(results, indent=2, ensure_ascii=False),
        file_name="pipeline_results.json",
        mime="application/json",
        use_container_width=True,
    )
