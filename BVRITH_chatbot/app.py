"""
Main Streamlit application for the College FAQ Chatbot.

Provides a modern chat interface for querying the college knowledge base
using RAG (Retrieval-Augmented Generation) with LangChain and ChromaDB.
"""

import os
import sys
import uuid
import logging
from typing import Optional
from pathlib import Path

import streamlit as st
import pandas as pd

from utils import get_config
from vector_store import get_vector_store, get_chunk_count, get_embeddings
from chatbot import CollegeChatbot
from prompts import SUGGESTED_QUESTIONS
from memory import MemoryManager

# ── Observability imports (graceful fallback if not installed yet) ──────────
try:
    from observability.session_stats import SessionStats
    from observability.alerts import alert_engine
    from observability.ab_testing import ab_test_manager
    from observability.log_analyzer import LogAnalyzer
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("observability/ module not found — stats sidebar disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="BVRIT College FAQ Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for dark mode friendly UI
# ──────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    /* Main content area */
    .main > div {
        padding-top: 1rem;
    }

    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }

    .user-message {
        background-color: rgba(33, 150, 243, 0.1);
        border-left: 3px solid #2196F3;
    }

    .bot-message {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 3px solid #4CAF50;
    }

    /* Citation styling */
    .citation {
        display: inline-block;
        background-color: rgba(255, 193, 7, 0.2);
        color: #FF9800;
        padding: 0.1rem 0.4rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.1rem;
    }

    /* Metrics styling */
    .metric-card {
        background-color: rgba(128, 128, 128, 0.05);
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }

    /* Source viewer styling */
    .source-box {
        background-color: rgba(33, 33, 33, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 0.5rem;
        padding: 0.8rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        max-height: 200px;
        overflow-y: auto;
    }

    /* Typing animation */
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        gap: 3px;
    }
    .typing-indicator span {
        width: 6px;
        height: 6px;
        background-color: #888;
        border-radius: 50%;
        animation: typing-bounce 1.4s ease-in-out infinite both;
    }
    .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
    .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
    .typing-indicator span:nth-child(3) { animation-delay: 0s; }
    @keyframes typing-bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }

    /* Copy button */
    .copy-btn {
        background: none;
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 0.25rem;
        padding: 0.2rem 0.5rem;
        cursor: pointer;
        font-size: 0.75rem;
        color: #888;
        transition: all 0.2s;
    }
    .copy-btn:hover {
        border-color: #2196F3;
        color: #2196F3;
    }

    /* Fix chat input at bottom */
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: calc(100% - 400px);
        max-width: 900px;
        background: var(--background-color);
        padding: 1rem 1rem 1.5rem;
        z-index: 100;
    }
    @media (max-width: 768px) {
        .stChatInputContainer {
            width: 100%;
        }
    }
</style>
"""


# ──────────────────────────────────────────────
# Initialization Functions
# ──────────────────────────────────────────────

def initialize_session_state() -> None:
    """Initialize all session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.chatbot = None
        st.session_state.vector_store = None
        st.session_state.db_status = None
        st.session_state.messages = []
        st.session_state.chunk_count = 0
        st.session_state.show_source = False
        st.session_state.current_chunks = []
        st.session_state.show_tool_panel = False
        st.session_state.last_tool_debug = {}
        # Memory system state — restore user_id from URL query param so it
        # survives page refreshes and the user's stored memories are not lost.
        saved_uid = st.query_params.get("uid", "")
        st.session_state.user_id = saved_uid
        st.session_state.session_id = str(uuid.uuid4())   # unique per browser session
        st.session_state.show_memory_panel = False
        st.session_state.memory_count = 0
        st.session_state.total_memory_count = 0
        st.session_state.pending_memory_refresh = False
        # ── Observability state ──
        if OBSERVABILITY_AVAILABLE:
            st.session_state.session_stats = SessionStats()
            st.session_state.obs_alerts = []      # active alert list for UI
            st.session_state.show_stats = True


def initialize_vector_store() -> None:
    """Initialize or load the vector store and chatbot."""
    if st.session_state.vector_store is not None:
        return

    with st.spinner("🔄 Initializing knowledge base..."):
        try:
            vector_store, status = get_vector_store()
            if vector_store is None:
                st.error(status or "Failed to initialize vector store.")
                return

            st.session_state.vector_store = vector_store
            st.session_state.db_status = status
            st.session_state.chunk_count = get_chunk_count()

            # Initialize chatbot with memory system
            memory_manager = MemoryManager()
            st.session_state.chatbot = CollegeChatbot(vector_store, memory_manager=memory_manager)
            logger.info("Chatbot initialized successfully with memory system.")

            # If the user_id was restored from the URL query param, register it
            # on the chatbot immediately and load the correct memory count so the
            # sidebar shows the right number from the very first render.
            if st.session_state.user_id:
                st.session_state.chatbot.set_user(
                    user_id=st.session_state.user_id,
                    session_id=st.session_state.session_id,
                )
                st.session_state.memory_count = st.session_state.chatbot.get_user_memory_count(
                    st.session_state.user_id
                )
                logger.info(
                    f"Restored user '{st.session_state.user_id}' with "
                    f"{st.session_state.memory_count} memories."
                )

        except ValueError as e:
            st.error(f"⚠️ {str(e)}")
            st.info("Please set your API key in the .env file and restart the app.")
        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")
            logger.exception("Initialization failed.")


# ──────────────────────────────────────────────
# UI Components
# ──────────────────────────────────────────────

def render_sidebar() -> None:
    """Render the sidebar with configuration info and controls."""
    with st.sidebar:
        # College logo placeholder
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0;">
                <div style="font-size: 3rem;">🎓</div>
                <h2 style="margin: 0;">BVRIT</h2>
                <p style="color: #888; font-size: 0.85rem;">College FAQ Assistant</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Knowledge Base Status
        st.subheader("📚 Knowledge Base")
        if st.session_state.db_status:
            st.success(f"✓ {st.session_state.db_status}")
        else:
            st.warning("⚠️ Not initialized")

        # Configuration Details
        config = get_config()

        st.subheader("⚙️ Configuration")

        # Number of chunks
        if st.session_state.chunk_count > 0:
            st.metric("Indexed Chunks", st.session_state.chunk_count)
        else:
            st.metric("Indexed Chunks", "—")

        # Embedding model
        st.metric(
            "Embedding Model",
            config.get("EMBEDDING_MODEL", "—"),
        )

        # LLM
        st.metric(
            "LLM",
            config.get("OPENROUTER_MODEL", "—"),
        )

        # Retrieval Top K
        st.metric("Retrieval Top K", "6")

        # Document name
        doc_name = "college_kb.docx"
        st.metric("Document", doc_name)

        st.divider()

        # Section Filter (placeholder - for future enhancement)
        st.subheader("🏷️ Section Filter")
        section_options = [
            "All Sections",
            "Admissions",
            "Placements",
            "Fees",
            "Courses",
            "Facilities",
            "Examinations",
        ]
        selected_section = st.selectbox(
            "Filter by section",
            section_options,
            label_visibility="collapsed",
        )

        st.divider()

        # Source Viewer Toggle
        st.subheader("🔍 Source Viewer")
        st.session_state.show_source = st.toggle(
            "Show retrieved chunks",
            value=st.session_state.show_source,
            help="Toggle to view the actual document chunks used for each answer.",
        )

        st.divider()

        # Tool Debug Panel Toggle
        st.subheader("🔧 Tool Debug")
        st.session_state.show_tool_panel = st.toggle(
            "Show Tool Debug Panel",
            value=st.session_state.show_tool_panel,
            help="Toggle to view tool execution details (routing, arguments, outputs).",
        )
        if st.session_state.show_tool_panel and st.session_state.last_tool_debug:
            debug = st.session_state.last_tool_debug
            st.markdown(
                f'<div style="background:rgba(128,128,128,0.05);padding:0.5rem;border-radius:0.5rem;font-size:0.8rem;">'
                f'<b>Routing:</b> {debug.get("routing", "—")}<br>',
                unsafe_allow_html=True,
            )
            if debug.get("tool_name"):
                st.markdown(f'<b>Tool:</b> {debug["tool_name"]}', unsafe_allow_html=True)
                st.markdown(f'<b>Args:</b> <pre style="font-size:0.7rem;">{debug.get("tool_args", {})}</pre>', unsafe_allow_html=True)
                st.markdown(f'<b>Output:</b> <pre style="font-size:0.7rem;">{debug.get("tool_output", "")[:300]}</pre>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # ── Memory System Section ──
        st.subheader("🧠 User Memory")
        
        # User ID input
        user_id = st.text_input(
            "User ID",
            value=st.session_state.user_id,
            placeholder="Enter your user ID...",
            help="Your unique identifier. Memories are stored per user.",
            key="user_id_input",
        )
        if user_id != st.session_state.user_id:
            st.session_state.user_id = user_id
            # Persist to URL so memories survive page refresh
            if user_id:
                st.query_params["uid"] = user_id
            else:
                st.query_params.pop("uid", None)
            if st.session_state.chatbot:
                st.session_state.chatbot.set_user(
                    user_id=user_id,
                    session_id=st.session_state.session_id,
                )
                # Load the real count for this user immediately
                if user_id:
                    st.session_state.memory_count = st.session_state.chatbot.get_user_memory_count(user_id)
                else:
                    st.session_state.memory_count = 0
            st.rerun()
        
        # Show memory count if user is set.
        # We display st.session_state.memory_count (the authoritative cached value)
        # rather than querying ChromaDB on every render cycle, which caused the
        # count to fluctuate while the background extraction thread was mid-write.
        if st.session_state.user_id and st.session_state.chatbot:
            st.metric("Stored Memories", st.session_state.memory_count)
        
        # Memory Panel Toggle
        st.session_state.show_memory_panel = st.toggle(
            "Show Memory Details",
            value=st.session_state.show_memory_panel,
            help="Toggle to view memory system details.",
        )
        if st.session_state.show_memory_panel:
            total = st.session_state.chatbot.memory_manager.get_total_memory_count() if st.session_state.chatbot else 0
            st.session_state.total_memory_count = total
            st.markdown(
                f'<div style="background:rgba(128,128,128,0.05);padding:0.5rem;'
                f'border-radius:0.5rem;font-size:0.8rem;">'
                f'<b>Total Memories (all users):</b> {total}<br>'
                f'<b>Your Memories:</b> {st.session_state.memory_count}<br>'
                f'<b>Session ID:</b> {st.session_state.session_id[:8]}...'
                f'</div>',
                unsafe_allow_html=True,
            )
        
        # Clear My Data Button
        if st.session_state.user_id:
            if st.button(
                "🗑️ Clear My Data",
                use_container_width=True,
                type="secondary",
                help="Delete ALL your stored memories from the memory database.",
            ):
                if st.session_state.chatbot:
                    deleted = st.session_state.chatbot.clear_user_memories(
                        st.session_state.user_id
                    )
                    if deleted > 0:
                        st.success(f"✅ Cleared {deleted} memories!")
                    else:
                        st.info("No memories to clear.")
                    st.rerun()

        st.divider()

        # ── 📊 Session Stats (Observability) ──────────────────────────────
        if OBSERVABILITY_AVAILABLE and hasattr(st.session_state, "session_stats"):
            st.subheader("📊 Session Stats")
            stats = st.session_state.session_stats

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Queries", stats.total_queries)
                st.metric("Avg Latency", f"{stats.avg_latency:.2f}s")
                st.metric("P95 Latency", f"{stats.p95_latency:.2f}s")
            with col_b:
                st.metric("Total Cost", f"${stats.total_cost_usd:.5f}")
                st.metric("Total Tokens", stats.total_tokens)
                st.metric("Errors", stats.error_count)

            # Active alerts (latency / cost / error-rate breaches)
            if st.session_state.get("obs_alerts"):
                st.markdown("**⚠️ Active Alerts:**")
                for alert in st.session_state.obs_alerts[-5:]:  # show last 5
                    severity_icon = "🔴" if alert.severity == "critical" else "🟡"
                    st.markdown(
                        f'<div style="background:rgba(255,100,100,0.08);padding:0.3rem 0.5rem;'
                        f'border-radius:0.4rem;font-size:0.78rem;margin-bottom:0.2rem;">'
                        f'{severity_icon} {alert.message}</div>',
                        unsafe_allow_html=True,
                    )

            # A/B version counts
            if hasattr(ab_test_manager, "get_version_counts"):
                ab_counts = ab_test_manager.get_version_counts()
                st.caption(
                    f"A/B calls — v1: {ab_counts.get('v1', 0)} | v2: {ab_counts.get('v2', 0)}"
                )

            col_rst, col_log = st.columns(2)
            with col_rst:
                if st.button("↺ Reset Stats", use_container_width=True, key="reset_stats"):
                    stats.reset()
                    st.session_state.obs_alerts = []
                    st.rerun()
            with col_log:
                if st.button("🔍 Analyze Logs", use_container_width=True, key="analyze_logs"):
                    with st.spinner("Analyzing…"):
                        report = LogAnalyzer().analyze()
                    st.session_state.log_analysis = report
                    st.rerun()

            # Show log analysis results if available
            if st.session_state.get("log_analysis"):
                with st.expander("📋 Log Analysis", expanded=False):
                    rep = st.session_state.log_analysis
                    st.write(f"**Records analysed:** {rep['total_records']}")
                    if rep.get("anomalies"):
                        st.write(f"**Anomalies:** {len(rep['anomalies'])}")
                        for a in rep["anomalies"][:5]:
                            st.write(f"• {a['detail']}")
                    st.write("**Suggestions:**")
                    for s in rep.get("suggestions", []):
                        st.write(s)

        # ── Clear Chat Button ──────────────────────────────────────────────
        st.divider()

        # Clear Chat Button
        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state.messages = []
            if st.session_state.chatbot:
                st.session_state.chatbot.clear_history()
            st.rerun()

        st.divider()

        # Footer
        st.markdown(
            """
            <div style="text-align: center; color: #888; font-size: 0.75rem; padding-top: 1rem;">
                Built with LangChain + ChromaDB + OpenAI Function Calling
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_suggested_questions() -> None:
    """Display suggested questions as clickable buttons."""
    if st.session_state.messages:
        return  # Only show on initial load

    st.markdown("### 👋 Welcome to BVRIT College FAQ Assistant!")
    st.markdown("Ask me anything about BVRIT College. Here are some suggested questions:")

    cols = st.columns(2)
    for i, question in enumerate(SUGGESTED_QUESTIONS):
        col = cols[i % 2]
        if col.button(
            question,
            use_container_width=True,
            key=f"suggested_{i}",
        ):
            st.session_state.pending_question = question
            st.rerun()


def render_chat_messages() -> None:
    """Render all chat messages with citations and metadata."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                # Bot message with styling
                st.markdown(
                    f'<div class="chat-message bot-message">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

                # Display citations if available
                if msg.get("citations"):
                    st.markdown("**📌 Citations:**")
                    for citation in msg["citations"]:
                        st.markdown(
                            f'<span class="citation">📄 {citation}</span>',
                            unsafe_allow_html=True,
                        )

                # Display metadata row
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.markdown(
                        f'<div class="metric-card">⏱️ {msg.get("response_time", "—")}</div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        f'<div class="metric-card">📄 {msg.get("chunk_count", 0)} chunks</div>',
                        unsafe_allow_html=True,
                    )
                with col3:
                    confidence = msg.get("confidence", 0)
                    conf_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                    st.markdown(
                        f'<div class="metric-card" style="color: {conf_color};">🎯 {confidence:.2%}</div>',
                        unsafe_allow_html=True,
                    )
                with col4:
                    # Memory info
                    memories_used = msg.get("memories_used", 0)
                    memories_stored = msg.get("memories_stored", 0)
                    memory_text = f"🧠 {memories_used} used"
                    if memories_stored > 0:
                        memory_text += f" / {memories_stored} stored"
                    st.markdown(
                        f'<div class="metric-card">{memory_text}</div>',
                        unsafe_allow_html=True,
                    )
                with col5:
                    # Copy button - preprocess content for JS safety
                    safe_content = msg['content'].replace('`', '').replace("'", "").replace('"', '')
                    copy_js = f"""
                    <button class="copy-btn" onclick="
                        navigator.clipboard.writeText('{safe_content.replace(chr(10), ' ')}');
                        this.textContent='✓ Copied!';
                        setTimeout(() => this.textContent='📋 Copy', 2000);
                    ">📋 Copy</button>
                    """
                    st.markdown(f'<div style="text-align: right;">{copy_js}</div>', unsafe_allow_html=True)

                # Source viewer for retrieved chunks (if enabled)
                if st.session_state.show_source and msg.get("retrieved_chunks"):
                    with st.expander("📖 View Retrieved Chunks", expanded=False):
                        for i, chunk in enumerate(msg["retrieved_chunks"], 1):
                            source = chunk.metadata.get("source", "Unknown")
                            section = chunk.metadata.get("section", "General")
                            score = chunk.metadata.get("relevance_score", 0)
                            content = chunk.page_content[:500]  # Truncated for display

                            st.markdown(
                                f'<div class="source-box">'
                                f'<strong>Chunk {i}</strong> | '
                                f'<em>Source:</em> {source} | '
                                f'<em>Section:</em> {section} | '
                                f'<em>Score:</em> {score:.4f}'
                                f'<br/><br/>{content}'
                                f'{"..." if len(chunk.page_content) > 500 else ""}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
            else:
                # User message
                st.markdown(
                    f'<div class="chat-message user-message">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )


def handle_user_input(user_question: str) -> None:
    """Process a user question and generate a response, storing it in session state.

    Does NOT render anything — rendering is handled by render_chat_messages()
    on the next Streamlit render cycle.

    Args:
        user_question (str): The user's question text.
    """
    # ── 1. Input validation (observability alert check) ───────────────────
    if OBSERVABILITY_AVAILABLE:
        ok, input_alert = alert_engine.validate_input(user_question)
        if not ok:
            st.session_state.obs_alerts.append(input_alert)
            st.session_state.messages.append({
                "role": "user",
                "content": user_question[:200] + "…",
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"⚠️ Your message is too long ({len(user_question)} characters). "
                    f"Please shorten it to under 2,000 characters and try again."
                ),
                "citations": [], "retrieved_chunks": [],
                "response_time": "0.00s", "chunk_count": 0, "confidence": 0,
            })
            return

    # ── 2. A/B prompt variant assignment ─────────────────────────────────
    prompt_version = "v1"
    if OBSERVABILITY_AVAILABLE:
        prompt_version, _ = ab_test_manager.assign_variant()
        # Inject prompt version into chatbot so it can pass to logger
        if st.session_state.chatbot:
            st.session_state.chatbot._ab_prompt_version = prompt_version

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_question,
    })

    # ── 3. Start observability call tracking ──────────────────────────────
    import time as _time
    call_start = _time.time()

    try:
        # Generate answer
        response = st.session_state.chatbot.answer_question(user_question)
        answer_text = response["answer"]
        call_success = True
        call_error = ""

        # Extract citations
        import re
        citation_matches = re.findall(r'\[(.*?)\]', answer_text)
        exclude_words = {"Document", "Source", "Section", "BVRIT College Information Assistant"}
        citations = list(set(
            m for m in citation_matches
            if m not in exclude_words and len(m) < 100
        ))

        # Store response in session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "citations": citations,
            "retrieved_chunks": response.get("retrieved_chunks", []),
            "response_time": response["response_time"],
            "chunk_count": response["chunk_count"],
            "confidence": response["confidence"],
            "routing": response.get("routing", "RAG"),
            "tool_debug": response.get("tool_debug", {}),
            "memories_used": response.get("memories_used", 0),
            "memories_stored": response.get("memories_stored", 0),
        })

        # Store last tool debug for sidebar panel
        st.session_state.last_tool_debug = response.get("tool_debug", {})

        # Schedule a deferred memory count refresh.
        # We cannot query the count here because the background extraction thread
        # may still be writing to ChromaDB — reading now gives a stale/lower value
        # and makes the counter appear to shrink.  Instead we set a flag and let
        # the next render cycle (triggered by st.rerun()) do the refresh after a
        # small sleep so the write is guaranteed to have landed.
        st.session_state.pending_memory_refresh = True

    except ValueError as e:
        call_success = False
        call_error = str(e)
        answer_text = f"⚠️ {str(e)}"
        citations = []
        response = {"response_time": "0.01s", "chunk_count": 0, "confidence": 0,
                    "routing": "Error", "tool_debug": {}, "memories_used": 0, "memories_stored": 0}
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "citations": [], "retrieved_chunks": [],
            "response_time": "0.01s", "chunk_count": 0, "confidence": 0,
        })
    except Exception as e:
        call_success = False
        call_error = str(e)
        answer_text = f"⚠️ An error occurred: {str(e)}"
        citations = []
        response = {"response_time": "0.01s", "chunk_count": 0, "confidence": 0,
                    "routing": "Error", "tool_debug": {}, "memories_used": 0, "memories_stored": 0}
        logger.exception("Error generating response.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "citations": [], "retrieved_chunks": [],
            "response_time": "0.01s", "chunk_count": 0, "confidence": 0,
        })

    # ── 4. Record observability metrics ───────────────────────────────────
    if OBSERVABILITY_AVAILABLE:
        call_latency = _time.time() - call_start
        # Parse latency number from response_time string (e.g. "1.23s" → 1.23)
        try:
            rt_str = response.get("response_time", "0s")
            latency_val = float(rt_str.rstrip("s")) if isinstance(rt_str, str) else call_latency
        except (ValueError, AttributeError):
            latency_val = call_latency

        # Estimate token count & cost (rough: use session_stats + llm_logger)
        est_tokens = max(50, len(user_question) // 4 + len(answer_text) // 4)
        est_cost = est_tokens / 1_000_000 * 0.6   # GPT-4o-mini output rate approx

        # Update session stats
        stats = st.session_state.session_stats
        stats.record(
            latency=latency_val,
            tokens=est_tokens,
            cost=est_cost,
            success=call_success,
        )

        # Check per-call thresholds
        new_alerts = alert_engine.check_call(latency=latency_val, cost=est_cost)
        new_alerts += alert_engine.check_session(error_rate=stats.error_rate)
        st.session_state.obs_alerts.extend(new_alerts)

        # Record A/B result
        is_refusal = any(
            p in answer_text.lower()
            for p in ["i can only answer", "i don't have information", "outside my scope"]
        )
        ab_test_manager.record_result(
            version=prompt_version,
            latency=latency_val,
            cost=est_cost,
            refusal=is_refusal,
            citations=citations,
            success=call_success,
        )


# ──────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────

def main() -> None:
    """Main entry point for the Streamlit application."""
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()

    # Initialize vector store and chatbot
    initialize_vector_store()

    # Render sidebar
    render_sidebar()

    # ── Step 1: Render all messages from session state ──
    render_suggested_questions()
    render_chat_messages()

    # ── Step 2: Get user input (chat_input at the bottom) ──
    pending = st.session_state.get("pending_question")
    if pending:
        user_question = pending
        st.session_state.pending_question = None
    else:
        user_question = st.chat_input("Ask a question about BVRIT College...")

    # ── Step 3: Ensure user is set on chatbot ──
    if st.session_state.chatbot:
        # Use the typed user_id if provided, otherwise use a stable session-scoped fallback
        effective_user_id = st.session_state.user_id or f"anon_{st.session_state.session_id[:8]}"
        st.session_state.chatbot.set_user(
            user_id=effective_user_id,
            session_id=st.session_state.session_id,
        )

    # ── Step 4: Process input if provided ──
    if user_question and st.session_state.chatbot is not None:
        handle_user_input(user_question)
        st.rerun()

    # ── Step 5: Deferred memory count refresh ──────────────────────────────
    # If a background extraction thread was launched last cycle, wait briefly
    # to let it finish, then query the true count and trigger one more rerun
    # to update the sidebar metric.  This gives a stable, monotonically
    # increasing counter rather than the flicker caused by reading mid-write.
    if st.session_state.get("pending_memory_refresh") and st.session_state.chatbot:
        import time as _refresh_time
        _refresh_time.sleep(0.4)   # give the daemon thread time to commit
        effective_uid = st.session_state.user_id or f"anon_{st.session_state.session_id[:8]}"
        fresh_count = st.session_state.chatbot.get_user_memory_count(effective_uid)
        # Only update (and rerun) if the count actually changed — avoids infinite rerun loops
        if fresh_count != st.session_state.memory_count:
            st.session_state.memory_count = fresh_count
            st.session_state.pending_memory_refresh = False
            st.rerun()
        else:
            st.session_state.pending_memory_refresh = False


if __name__ == "__main__":
    main()