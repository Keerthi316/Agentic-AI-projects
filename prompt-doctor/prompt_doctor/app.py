"""
app.py — Prompt Doctor Streamlit Application
==============================================

Design decisions:
1. **Session state for progress** — Streamlit reruns on every interaction.
   Session state persists `current_level`, `unlocked_levels`, and evaluation
   results between reruns.
2. **Domains sidebar** — Lets users choose a topic area. Currently domains are
   a single selector, but the architecture supports per-domain levels later.
3. **Two-step evaluation flow** — First "Run Prompt" executes the user's prompt
   and shows the AI's output. Then "Evaluate Prompt" runs the examiner and
   shows grading. This lets the user see what their prompt produced before
   getting feedback.
4. **Visual grading** — Principles are shown as expandable cards with
   pass/fail badges, the weak phrase, and the guiding question.
5. **Level unlock** — When all principles pass, a celebratory message appears
   and "Next Level" button enables.
6. **Reset option** — A sidebar button to reset progress for testing.
"""

import streamlit as st

from examiner import evaluate
from levels import LEVELS, get_level, get_max_level
from runner import run_prompt

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Prompt Doctor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "current_level" not in st.session_state:
    st.session_state.current_level = 1

if "unlocked_levels" not in st.session_state:
    st.session_state.unlocked_levels = {1}  # Level 1 is always unlocked

if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None

if "runner_result" not in st.session_state:
    st.session_state.runner_result = None

if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

if "custom_sample_input" not in st.session_state:
    st.session_state.custom_sample_input = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/doctor-male.png",
        width=64,
    )
    st.title("Prompt Doctor 🏥")
    st.caption("Learn prompt engineering by doing.")

    st.divider()

    # Domain selection
    domain = st.selectbox(
        "Select Domain",
        ["General", "Writing", "Programming", "Data Analysis", "Customer Support"],
        index=0,
    )

    st.divider()

    # Level navigation
    st.subheader("Levels")

    for level in LEVELS:
        is_unlocked = level.id in st.session_state.unlocked_levels
        is_current = level.id == st.session_state.current_level

        if is_unlocked:
            label = f"✅ Level {level.id}: {level.title}"
        else:
            label = f"🔒 Level {level.id}: {level.title}"

        st.button(
            label,
            key=f"nav_{level.id}",
            disabled=not is_unlocked,
            type="primary" if is_current else "secondary",
            on_click=lambda lid=level.id: _navigate_to_level(lid),
        )

    st.divider()

    # Reset button
    if st.button("🔄 Reset Progress", type="secondary", use_container_width=True):
        for key in ["current_level", "unlocked_levels", "evaluation_result", "runner_result", "last_prompt", "custom_sample_input"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.divider()
    st.caption("Built with ❤️ using Streamlit + OpenRouter")


def _navigate_to_level(level_id: int) -> None:
    """Navigate to the given level."""
    st.session_state.current_level = level_id
    st.session_state.evaluation_result = None
    st.session_state.runner_result = None
    st.session_state.last_prompt = ""
    st.session_state.custom_sample_input = None


# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------

current_level_id = st.session_state.current_level
level = get_level(current_level_id)

if level is None:
    st.error(f"Level {current_level_id} not found.")
    st.stop()

# --- Level header ---
st.title(f"Level {level.id}: {level.title}")
st.markdown(level.description)

# --- Progress indicator ---
unlocked_count = len(st.session_state.unlocked_levels)
total_count = get_max_level()
st.progress(unlocked_count / total_count, text=f"{unlocked_count} of {total_count} levels unlocked")

st.divider()

# --- Level details in columns ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Your Task")
    st.markdown(level.task)

    st.subheader("📝 Sample Input")
    sample_input_key = f"sample_input_{level.id}"
    default_sample = st.session_state.custom_sample_input or level.sample_input
    custom_sample_input = st.text_area(
        "Edit this sample input or write your own — it will be fed to your prompt:",
        value=default_sample,
        height=120,
        key=sample_input_key,
    )
    st.session_state.custom_sample_input = custom_sample_input

with col2:
    st.subheader("🎯 Principles to Master")
    for p in level.principles:
        st.markdown(f"- **{p}**")

    st.subheader("💡 Expected Output")
    st.info(level.expected_output_hint)

st.divider()

# --- Prompt input ---
st.subheader("✍️ Write Your Prompt")
user_prompt = st.text_area(
    "Enter your prompt below. It will be combined with the sample input above.",
    value=st.session_state.last_prompt,
    placeholder="e.g., You are a travel advisor. Recommend 3 budget-friendly beach destinations in Southeast Asia...",
    height=150,
    key="prompt_input",
)

# Save prompt to session state
if user_prompt:
    st.session_state.last_prompt = user_prompt

# --- Action buttons ---
action_col1, action_col2, _ = st.columns([1, 1, 2])

with action_col1:
    run_clicked = st.button(
        "🚀 Run Prompt",
        type="primary",
        use_container_width=True,
        disabled=not user_prompt.strip(),
    )

with action_col2:
    evaluate_clicked = st.button(
        "📊 Evaluate Prompt",
        type="secondary",
        use_container_width=True,
        disabled=not user_prompt.strip(),
    )

st.divider()

# ---------------------------------------------------------------------------
# Handle "Run Prompt" action
# ---------------------------------------------------------------------------

if run_clicked and user_prompt.strip():
    sample_input_to_use = st.session_state.custom_sample_input or level.sample_input
    with st.spinner("Running your prompt against the sample input..."):
        st.session_state.runner_result = run_prompt(
            user_prompt, sample_input_to_use
        )
    st.rerun()

# ---------------------------------------------------------------------------
# Handle "Evaluate Prompt" action
# ---------------------------------------------------------------------------

if evaluate_clicked and user_prompt.strip():
    with st.spinner("🧐 The examiner is reviewing your prompt..."):
        st.session_state.evaluation_result = evaluate(
            level.id, user_prompt
        )
    st.rerun()

# ---------------------------------------------------------------------------
# Display runner result
# ---------------------------------------------------------------------------

if st.session_state.runner_result is not None:
    result = st.session_state.runner_result

    with st.expander("🤖 Prompt Output", expanded=True):
        if result["ran_ok"]:
            st.markdown(result["output"])
        else:
            st.error(f"⚠️ {result['error']}")

# ---------------------------------------------------------------------------
# Display evaluation result
# ---------------------------------------------------------------------------

if st.session_state.evaluation_result is not None:
    eval_result = st.session_state.evaluation_result

    if not eval_result["ran_ok"]:
        # Examiner failed — show error
        st.error("### Examiner Error")
        for p in eval_result["principles"]:
            st.warning(p["weakness"])
    else:
        # Examiner succeeded — show grading
        verdict = eval_result["verdict"]
        principles = eval_result["principles"]

        if verdict == "pass":
            st.success("### ✅ All Principles Passed! 🎉")
        else:
            st.warning("### 🔄 Needs Revision — Keep Improving!")

        st.divider()

        # Show each principle as a card
        for i, p in enumerate(principles, 1):
            passed = p["pass"]
            badge = "✅ Pass" if passed else "❌ Needs Work"

            with st.container():
                cols = st.columns([1, 10])
                with cols[0]:
                    st.markdown(f"**{i}.**")
                with cols[1]:
                    if passed:
                        st.success(f"**{p['name']}** — {badge}")
                    else:
                        st.error(f"**{p['name']}** — {badge}")

                    # Show weakness and question only on failure
                    if not passed and p.get("weakness"):
                        st.markdown(f"**Weakness:** _{p['weakness']}_")
                    if not passed and p.get("question"):
                        st.markdown(f"💡 **Guiding Question:** {p['question']}")

                st.divider()

        # --- Level unlock logic ---
        if verdict == "pass":
            all_passed = all(p["pass"] for p in principles)

            if all_passed:
                next_level = current_level_id + 1
                if next_level <= get_max_level():
                    if next_level not in st.session_state.unlocked_levels:
                        st.session_state.unlocked_levels.add(next_level)
                        st.balloons()

                    st.success(
                        f"### 🎉 Level {current_level_id} Complete! "
                        f"Level {next_level} is now unlocked."
                    )

                    if st.button(
                        f"➡️ Go to Level {next_level}",
                        type="primary",
                        use_container_width=True,
                    ):
                        _navigate_to_level(next_level)
                        st.rerun()
                else:
                    st.success(
                        "### 🏆 Congratulations! You've completed all levels!"
                    )
                    st.balloons()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "💡 **Tip:** Start with a clear role and instruction. "
    "Click 'Run Prompt' to see what your prompt produces, then 'Evaluate Prompt' "
    "to get feedback from the AI examiner."
)