"""
Memory Module for BVRIT College Chatbot.

This module provides a completely separate ChromaDB-based memory system
that runs alongside the existing knowledge base ChromaDB.

It handles:
- Storing user memories (preferences, facts, skills) in a dedicated ChromaDB
- Retrieving relevant memories before each answer generation
- Extracting new memories from conversations
- Updating existing memories (no duplicates)
- Automatic cleanup of memories older than 30 days
- User data deletion ("Clear my data" command)

Architecture:
    memory_db/          ← Separate ChromaDB for memory (NOT the knowledge base)
    memory/
        __init__.py
        memory_store.py      ← Low-level ChromaDB CRUD operations
        memory_extractor.py  ← Extract facts from conversation text
        memory_retriever.py  ← Retrieve top-K memories for a user
        memory_manager.py    ← Orchestrator that ties everything together
        memory_cleanup.py    ← Scheduled cleanup of expired memories
"""

from .memory_manager import MemoryManager
from .memory_store import MemoryStore
from .memory_extractor import MemoryExtractor
from .memory_retriever import MemoryRetriever
from .memory_cleanup import MemoryCleanup

__all__ = [
    "MemoryManager",
    "MemoryStore",
    "MemoryExtractor",
    "MemoryRetriever",
    "MemoryCleanup",
]