"""
Main Streamlit application for the College FAQ Chatbot.

Provides a modern chat interface for querying the college knowledge base
using RAG (Retrieval-Augmented Generation) with LangChain and ChromaDB.
"""

import os
import sys
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
        # Memory system state
        st.session_state.user_id = ""
        st.session_state.session_id = ""
        st.session_state.show_memory_panel = False
        st.session_state.memory_count = 0
        st.session_state.total_memory_count = 0


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
            if st.session_state.chatbot:
                st.session_state.chatbot.set_user(
                    user_id=user_id,
                    session_id=st.session_state.session_id,
                )
            st.rerun()
        
        # Show memory count if user is set
        if st.session_state.user_id and st.session_state.chatbot:
            memory_count = st.session_state.chatbot.get_user_memory_count(
                st.session_state.user_id
            )
            st.session_state.memory_count = memory_count
            st.metric("Stored Memories", memory_count)
        
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
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_question,
    })

    try:
        # Generate answer
        response = st.session_state.chatbot.answer_question(user_question)
        answer_text = response["answer"]

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

        # Update memory count in sidebar
        if st.session_state.user_id and st.session_state.chatbot:
            st.session_state.memory_count = st.session_state.chatbot.get_user_memory_count(
                st.session_state.user_id
            )

    except ValueError as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ {str(e)}",
            "citations": [],
            "retrieved_chunks": [],
            "response_time": "0.01s",
            "chunk_count": 0,
            "confidence": 0,
        })
    except Exception as e:
        logger.exception("Error generating response.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ An error occurred: {str(e)}",
            "citations": [],
            "retrieved_chunks": [],
            "response_time": "0.01s",
            "chunk_count": 0,
            "confidence": 0,
        })


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
    if st.session_state.chatbot and st.session_state.user_id:
        if st.session_state.chatbot.current_user_id != st.session_state.user_id:
            st.session_state.chatbot.set_user(
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
            )

    # ── Step 4: Process input if provided ──
    if user_question and st.session_state.chatbot is not None:
        handle_user_input(user_question)
        st.rerun()


if __name__ == "__main__":
    main()