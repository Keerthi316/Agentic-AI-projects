"""
Memory Cleanup — Automatic Deletion of Old Memories.

This module handles the automatic cleanup of memory entries that are
older than a configurable threshold (default: 30 days).

Cleanup runs:
1. On first memory retrieval of a session (from MemoryManager)
2. Can be triggered manually via the application

Only affects the memory ChromaDB — the knowledge base is never touched.
"""

import logging
from typing import Optional

from .memory_store import MemoryStore, MEMORY_MAX_AGE_DAYS

logger = logging.getLogger(__name__)


class MemoryCleanup:
    """
    Handles cleanup of expired memories from the memory ChromaDB.

    This ensures user privacy by automatically removing old data
    while keeping the knowledge base completely untouched.
    """

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        """
        Initialize the MemoryCleanup.

        Args:
            memory_store: The MemoryStore instance to clean up.
                         If None, creates a new one.
        """
        self.memory_store = memory_store or MemoryStore()

    def clean_old_memories(self, max_age_days: int = MEMORY_MAX_AGE_DAYS) -> int:
        """
        Delete memories older than the specified number of days.

        Args:
            max_age_days: Maximum age of memories in days.
                         Memories older than this are deleted.
                         Default: 30 (from MEMORY_MAX_AGE_DAYS).

        Returns:
            int: Number of memories deleted.
        """
        logger.info(
            f"Starting cleanup of memories older than {max_age_days} days..."
        )
        deleted_count = self.memory_store.delete_old_memories(
            max_age_days=max_age_days
        )
        if deleted_count > 0:
            logger.info(f"Cleanup completed: deleted {deleted_count} old memories")
        else:
            logger.info("Cleanup completed: no old memories to delete")
        return deleted_count

    def get_memory_age_stats(self) -> dict:
        """
        Get statistics about memory ages for monitoring.

        Returns:
            dict: Statistics including total count and age distribution.
        """
        total = self.memory_store.get_total_memory_count()

        return {
            "total_memories": total,
            "max_age_days": MEMORY_MAX_AGE_DAYS,
        }