import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from main import ask, MODELS, PRICES  # noqa: F401  (PRICES imported for completeness)

st.title("Multi-Model Comparison Tool")
st.caption("Ask one question and compare answers, speed, and cost across multiple LLMs side by side.")

if "results" not in st.session_state:
    st.session_state.results = None

question = st.text_area("Your question", placeholder="e.g. What is the capital of France?")

selected_models = st.multiselect(
    "Models to compare",
    options=MODELS,
    default=MODELS,
)

btn = st.empty()
clicked = btn.button("Compare models")

if clicked:
    if not question.strip():
        st.warning("Please type a question first.")
    elif not selected_models:
        st.warning("Please select at least one model.")
    else:
        # Swap the button for a disabled version while work is in progress
        btn.button("Compare models", disabled=True, key="btn_disabled")
        st.session_state.results = None

        bucket = {}
        with st.spinner("Asking all models in parallel…"):
            with ThreadPoolExecutor(max_workers=len(selected_models)) as pool:
                futures = {pool.submit(ask, question, m): m for m in selected_models}
                for future in as_completed(futures):
                    m = futures[future]
                    try:
                        answer, latency, in_tok, out_tok, cost, error = future.result()
                        bucket[m] = (answer, latency, cost, error)
                    except Exception as e:
                        bucket[m] = (None, 0.0, 0.0, str(e))

        # Restore original column order (as_completed arrives in completion order)
        st.session_state.results = [(m, *bucket[m]) for m in selected_models]

if st.session_state.results:
    cols = st.columns(len(st.session_state.results))
    for col, (model, answer, latency, cost, error) in zip(cols, st.session_state.results):
        with col:
            st.header(model, divider="gray")
            if error:
                st.error(error)
            else:
                st.write(answer)
            st.metric("Latency", f"{latency:.2f}s")
            st.metric("Cost", f"${cost:.6f}")

    st.caption("Prices shown are illustrative. All current models are free-tier; swap in paid model IDs to see real cost estimates.")
