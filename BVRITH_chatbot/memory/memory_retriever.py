"""
Memory Retriever — Retrieve Relevant User Memories Before Answer Generation.

This module handles the retrieval of user memories from the memory ChromaDB
and formats them into a prompt-friendly string that can be injected into
the system prompt for personalized responses.

Execution order (per the pipeline):
    User Query → Search Knowledge ChromaDB → Search Memory ChromaDB
    → Combine → Build Prompt → LLM → Return Answer → Extract New Memories
"""

import logging
from typing import List, Optional
from langchain_core.documents import Document

from .memory_store import MemoryStore

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """
    Retrieves user memories and formats them for prompt injection.

    This sits between the knowledge base retrieval and the prompt builder
    in the RAG pipeline, ensuring user context is available to the LLM.
    """

    def __init__(self, memory_store: MemoryStore):
        """
        Initialize the MemoryRetriever.

        Args:
            memory_store: The MemoryStore instance to query.
        """
        self.memory_store = memory_store

    def retrieve_memories(
        self,
        user_id: str,
        query: Optional[str] = None,
        k: int = 5,
    ) -> List[Document]:
        """
        Retrieve the top-K most relevant memories for a user.

        Args:
            user_id: The user to retrieve memories for.
            query: Optional query for semantic search against memories.
            k: Maximum number of memories to return (default: 5).

        Returns:
            List[Document]: Retrieved memory documents.
        """
        if not user_id:
            logger.debug("No user_id provided; skipping memory retrieval.")
            return []

        try:
            memories = self.memory_store.get_user_memories(
                user_id=user_id,
                query=query,
                k=k,
            )
            logger.info(
                f"Retrieved {len(memories)} memories for user '{user_id}'"
            )
            return memories
        except Exception as e:
            logger.warning(f"Failed to retrieve memories for user '{user_id}': {e}")
            return []

    def format_memories_for_prompt(self, memories: List[Document]) -> str:
        """
        Format retrieved memories into a string for system prompt injection.

        Example output:
            User Memory:
            - Name: Priya
            - Interest: CSE
            - Language: English
            - Skill: Python

        Args:
            memories: List of memory Documents from the memory store.

        Returns:
            str: Formatted memory string, or empty string if no memories.
        """
        if not memories:
            return ""

        memory_lines = ["## User Memory"]
        for i, doc in enumerate(memories, 1):
            meta = doc.metadata
            memory_type = meta.get("memory_type", "unknown")
            content = meta.get("content", doc.page_content)
            importance = meta.get("importance", 0.5)

            # Format the memory type nicely
            type_display = memory_type.replace("_", " ").title()

            # Add to the formatted output
            memory_lines.append(f"- {type_display}: {content}")

            # Add importance indicator for high-value memories
            if importance >= 0.9:
                memory_lines[-1] += " (important)"

        return "\n".join(memory_lines)

    def get_memory_summary(self, user_id: str, k: int = 5) -> str:
        """
        Convenience method: retrieve and format memories in one call.

        Args:
            user_id: The user to get memories for.
            k: Maximum number of memories.

        Returns:
            str: Formatted memory string for prompt injection.
        """
        memories = self.retrieve_memories(user_id=user_id, k=k)
        if not memories:
            logger.info(f"No memories found for user '{user_id}'")
            return ""
        return self.format_memories_for_prompt(memories)

    def count_user_memories(self, user_id: str) -> int:
        """
        Count how many memories are stored for a given user.

        Args:
            user_id: The user to count memories for.

        Returns:
            int: Number of memory entries.
        """
        return self.memory_store.count_user_memories(user_id)