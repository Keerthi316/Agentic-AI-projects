"""
Memory Manager — Orchestrator for the Memory System.

This module ties together the MemoryStore, MemoryExtractor, MemoryRetriever,
and MemoryCleanup into a single cohesive interface.

It provides the main integration points used by the chatbot:
1. `retrieve_and_inject_memories()` — Called before LLM answer generation
2. `extract_and_store_memories()` — Called after LLM answer generation
3. `clear_user_data()` — Called when user says "clear my data"
4. `run_cleanup()` — Called periodically to remove old memories

Pipeline:
    User Query → Search Knowledge ChromaDB → Search Memory ChromaDB (this)
    → Combine → Build Prompt → LLM → Return Answer → Extract New Memories (this)
    → Store into Memory ChromaDB
"""

import logging
from typing import List, Optional
from datetime import datetime, timezone

from langchain_core.documents import Document

from .memory_store import MemoryStore
from .memory_extractor import MemoryExtractor
from .memory_retriever import MemoryRetriever
from .memory_cleanup import MemoryCleanup

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Orchestrator for the memory system.

    Provides a single interface for the chatbot to interact with
    all memory-related functionality, keeping the chatbot code clean.
    """

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        memory_extractor: Optional[MemoryExtractor] = None,
        memory_retriever: Optional[MemoryRetriever] = None,
        memory_cleanup: Optional[MemoryCleanup] = None,
    ):
        """
        Initialize the MemoryManager with its sub-components.

        Args:
            memory_store: Optional MemoryStore instance.
            memory_extractor: Optional MemoryExtractor instance.
            memory_retriever: Optional MemoryRetriever instance.
            memory_cleanup: Optional MemoryCleanup instance.
        """
        self.memory_store = memory_store or MemoryStore()
        self.memory_extractor = memory_extractor or MemoryExtractor()
        self.memory_retriever = memory_retriever or MemoryRetriever(self.memory_store)
        self.memory_cleanup = memory_cleanup or MemoryCleanup(self.memory_store)

        # Track whether cleanup has been run this session
        self._cleanup_done = False

    # ──────────────────────────────────────────
    # Pre-Answer: Retrieve and Inject Memories
    # ──────────────────────────────────────────

    def retrieve_and_inject_memories(
        self,
        user_id: str,
        query: str = "",
        k: int = 5,
    ) -> str:
        """
        Retrieve user memories and format them for prompt injection.

        This is called BEFORE the LLM generates an answer.
        The returned string should be injected into the system prompt.

        Args:
            user_id: The user to retrieve memories for.
            query: The current user query (used for semantic search).
            k: Maximum number of memories to retrieve.

        Returns:
            str: Formatted memory string for prompt injection.
                 Empty string if no memories found or no user_id.
        """
        if not user_id:
            return ""

        # Run cleanup once per session (on first memory retrieval)
        if not self._cleanup_done:
            self._run_initial_cleanup()

        # Retrieve memories with the user's query as semantic context
        memories = self.memory_retriever.retrieve_memories(
            user_id=user_id,
            query=query,
            k=k,
        )

        # Format for prompt injection
        formatted = self.memory_retriever.format_memories_for_prompt(memories)

        if formatted:
            logger.info(
                f"Injected {len(memories)} memories for user '{user_id}'"
            )
        else:
            logger.info(f"No memories to inject for user '{user_id}'")

        return formatted

    # ──────────────────────────────────────────
    # Post-Answer: Extract and Store Memories
    # ──────────────────────────────────────────

    def extract_and_store_memories(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str = "",
        session_id: str = "",
    ) -> int:
        """
        Extract memories from the conversation and store them.

        This is called AFTER the LLM generates an answer.
        Handles deduplication: updates existing memories instead of
        creating duplicates.

        Args:
            user_id: The user to store memories for.
            user_message: The user's message text.
            assistant_response: The assistant's response text.
            session_id: Current session identifier.

        Returns:
            int: Number of new memories stored or updated.
        """
        if not user_id:
            return 0

        # Extract memories from the conversation
        extracted = self.memory_extractor.extract_memories(
            user_message=user_message,
            assistant_response=assistant_response,
        )

        if not extracted:
            return 0

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        stored_count = 0
        for memory in extracted:
            memory_type = memory["memory_type"]
            content = memory["content"]
            importance = memory["importance"]

            # Check if this memory already exists
            existing_id = self.memory_store.find_existing_memory(
                user_id=user_id,
                memory_type=memory_type,
                content=content,
            )

            if existing_id:
                # Update existing memory with new timestamp and session
                success = self.memory_store.update_memory(
                    doc_id=existing_id,
                    content=content,
                    timestamp=timestamp,
                    importance=importance,
                    session_id=session_id,
                )
                if success:
                    logger.info(
                        f"Updated memory: user={user_id}, "
                        f"type={memory_type}, content={content}"
                    )
                    stored_count += 1
            else:
                # Add new memory
                self.memory_store.add_memory(
                    user_id=user_id,
                    memory_type=memory_type,
                    content=content,
                    importance=importance,
                    session_id=session_id,
                    timestamp=timestamp,
                )
                logger.info(
                    f"Stored new memory: user={user_id}, "
                    f"type={memory_type}, content={content}"
                )
                stored_count += 1

        return stored_count

    # ──────────────────────────────────────────
    # Privacy: Clear User Data
    # ──────────────────────────────────────────

    def clear_user_data(self, user_id: str) -> int:
        """
        Delete ALL memories for a user from the memory ChromaDB.

        This implements the "Clear my data" privacy feature.
        Only affects the memory ChromaDB — the knowledge base is untouched.

        Args:
            user_id: The user whose data should be deleted.

        Returns:
            int: Number of deleted memories.
        """
        if not user_id:
            logger.warning("clear_user_data called with empty user_id")
            return 0

        logger.info(f"Clearing all memories for user '{user_id}'")
        deleted_count = self.memory_store.delete_user_memories(user_id)
        logger.info(
            f"Cleared {deleted_count} memories for user '{user_id}'"
        )
        return deleted_count

    # ──────────────────────────────────────────
    # Cleanup: Remove Old Memories
    # ──────────────────────────────────────────

    def _run_initial_cleanup(self) -> None:
        """
        Run cleanup once when the chatbot first starts.

        Deletes memories older than 30 days.
        """
        try:
            deleted = self.memory_store.delete_old_memories()
            if deleted > 0:
                logger.info(f"Initial cleanup: deleted {deleted} old memories")
            self._cleanup_done = True
        except Exception as e:
            logger.warning(f"Initial cleanup failed: {e}")
            self._cleanup_done = True  # Don't retry on failure

    def run_cleanup(self) -> int:
        """
        Manually trigger memory cleanup.

        This can be called from the application on a schedule.

        Returns:
            int: Number of deleted memories.
        """
        return self.memory_store.delete_old_memories()

    # ──────────────────────────────────────────
    # Utility Methods
    # ──────────────────────────────────────────

    def get_user_memory_count(self, user_id: str) -> int:
        """
        Get the number of memories stored for a user.

        Args:
            user_id: The user to check.

        Returns:
            int: Number of memory entries.
        """
        return self.memory_store.count_user_memories(user_id)

    def get_total_memory_count(self) -> int:
        """
        Get the total number of memories across all users.

        Returns:
            int: Total memory entries.
        """
        return self.memory_store.get_total_memory_count()

    def detect_clear_data_command(self, user_message: str) -> bool:
        """
        Check if the user is requesting to clear their data.

        Args:
            user_message: The user's message.

        Returns:
            bool: True if this is a clear-data request.
        """
        return self.memory_extractor.detect_clear_data_command(user_message)