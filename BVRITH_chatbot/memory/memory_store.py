"""
Memory Store — Low-level ChromaDB CRUD Operations for User Memory.

This module provides a dedicated ChromaDB instance (completely separate from
the knowledge base ChromaDB) for storing and managing user memories.

Key design decisions:
- Uses a separate persistence directory: `memory_db/` (NOT `chroma_db/`)
- Collection name: `user_memory` (separate from the KB collection `college_kb`)
- Each memory entry is a ChromaDB Document with metadata fields:
    user_id, memory_type, content, timestamp, importance, session_id
- Supports: add, retrieve, update, delete by user, delete old memories
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from vector_store import get_embeddings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

# Completely separate directory from the knowledge base chroma_db/
MEMORY_DB_DIR = "memory_db"
MEMORY_COLLECTION_NAME = "user_memory"

# When retrieving memories, return at most this many
MEMORY_TOP_K = 5

# Memories older than this many days are eligible for automatic cleanup
MEMORY_MAX_AGE_DAYS = 30


class MemoryStore:
    """
    Low-level CRUD operations for the memory ChromaDB.

    This class manages a completely independent ChromaDB instance
    that only stores user memory vectors — NOT knowledge base documents.
    """

    def __init__(self, embeddings: Optional[Embeddings] = None):
        """
        Initialize the MemoryStore with a dedicated ChromaDB instance.

        Args:
            embeddings: Embedding model. If None, uses the same one
                       from vector_store.get_embeddings().
        """
        self.embeddings = embeddings or get_embeddings()
        self._vector_store: Optional[Chroma] = None

    def _get_vector_store(self) -> Chroma:
        """
        Lazy-load the memory ChromaDB instance.

        Returns:
            Chroma: The memory vector store (separate from knowledge base).
        """
        if self._vector_store is None:
            # Ensure the memory_db directory exists
            os.makedirs(MEMORY_DB_DIR, exist_ok=True)

            logger.info(
                f"Loading/creating memory vector store at '{MEMORY_DB_DIR}' "
                f"(collection: '{MEMORY_COLLECTION_NAME}')"
            )

            self._vector_store = Chroma(
                collection_name=MEMORY_COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=MEMORY_DB_DIR,
            )
        return self._vector_store

    # ──────────────────────────────────────────
    # Memory CRUD Operations
    # ──────────────────────────────────────────

    def add_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        importance: float = 0.5,
        session_id: str = "",
        timestamp: Optional[str] = None,
    ) -> str:
        """
        Add a new memory entry to the memory ChromaDB.

        Args:
            user_id: Unique identifier for the user.
            memory_type: Type of memory (e.g., "name", "preference",
                        "skill", "language", "interest").
            content: The actual memory content (e.g., "Priya", "CSE", "English").
            importance: Relevance score 0.0–1.0. Higher = more important.
            session_id: Session identifier for tracking.
            timestamp: ISO timestamp string. Defaults to current UTC time.

        Returns:
            str: The ChromaDB document ID of the stored memory.
        """
        vector_store = self._get_vector_store()

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        metadata = {
            "user_id": user_id,
            "memory_type": memory_type,
            "content": content,
            "timestamp": timestamp,
            "importance": importance,
            "session_id": session_id,
        }

        # Generate a deterministic ID: user_id + memory_type + content_lower
        # This helps identify duplicates later
        doc_id = f"{user_id}_{memory_type}_{content.lower().replace(' ', '_')}"

        # Create a Document with the content being the memory description
        # This is what gets embedded
        doc_text = f"{memory_type}: {content}"
        doc = Document(page_content=doc_text, metadata=metadata)

        logger.info(
            f"Adding memory: user={user_id}, type={memory_type}, "
            f"content={content}, importance={importance}"
        )

        # Add to ChromaDB with a specific ID
        vector_store.add_documents(
            documents=[doc],
            ids=[doc_id],
        )

        return doc_id

    def get_user_memories(
        self,
        user_id: str,
        query: Optional[str] = None,
        k: int = MEMORY_TOP_K,
    ) -> List[Document]:
        """
        Retrieve top-K memories for a given user.

        If a query string is provided, performs semantic search over the
        user's memories. Otherwise, retrieves by metadata filter only.

        Args:
            user_id: The user to retrieve memories for.
            query: Optional query to semantically search against memories.
            k: Maximum number of memories to return (default: 5).

        Returns:
            List[Document]: Retrieved memory documents, sorted by relevance.
        """
        vector_store = self._get_vector_store()

        # Filter by user_id
        filter_dict = {"user_id": {"$eq": user_id}}

        try:
            if query:
                # Semantic search with user filter
                logger.info(
                    f"Searching memories for user={user_id} with query='{query}'"
                )
                results = vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict,
                )
            else:
                # Just get recent memories by metadata filter
                logger.info(f"Fetching recent memories for user={user_id}")
                # Chroma doesn't support sort by metadata directly,
                # so we use a generic query
                results = vector_store.similarity_search(
                    query="memory",
                    k=k,
                    filter=filter_dict,
                )

            logger.info(f"Retrieved {len(results)} memories for user={user_id}")
            return results

        except Exception as e:
            logger.warning(f"Failed to retrieve memories for user={user_id}: {e}")
            return []

    def get_all_user_memories(self, user_id: str) -> List[Document]:
        """
        Retrieve ALL memories for a given user (no relevance limit).

        Uses ChromaDB's native collection.get() with metadata filter,
        which is more reliable than similarity_search for bulk retrieval.

        Args:
            user_id: The user whose memories should be retrieved.

        Returns:
            List[Document]: All memory documents for the user.
        """
        vector_store = self._get_vector_store()
        filter_dict = {"user_id": {"$eq": user_id}}

        try:
            # Use collection.get() for direct metadata-based retrieval
            # This avoids embedding calls and works reliably with filters
            results = vector_store._collection.get(
                where=filter_dict,
                include=["metadatas", "documents"],
            )

            docs = []
            for i, doc_id in enumerate(results.get("ids", [])):
                metadata = results["metadatas"][i] if results.get("metadatas") else {}
                page_content = results["documents"][i] if results.get("documents") else ""
                doc = Document(page_content=page_content, metadata=metadata)
                docs.append(doc)

            logger.info(f"Retrieved {len(docs)} total memories for user={user_id}")
            return docs

        except Exception as e:
            logger.warning(f"Failed to get all memories for user={user_id}: {e}")
            return []

    def find_existing_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
    ) -> Optional[str]:
        """
        Check if a memory already exists for this user+type combination.

        Used to update existing memories instead of creating duplicates.

        ChromaDB has limited compound filter support, so we use a two-step
        approach: first filter by user_id, then check memory_type in Python.

        Args:
            user_id: User identifier.
            memory_type: Type of memory (e.g., "preference", "language").
            content: The content to check (used to search).

        Returns:
            Optional[str]: The ChromaDB document ID if found, else None.
        """
        vector_store = self._get_vector_store()

        # Build a deterministic ID to check
        expected_id = f"{user_id}_{memory_type}_{content.lower().replace(' ', '_')}"

        # Step 1: Filter by user_id only (single filter works well)
        filter_dict = {"user_id": {"$eq": user_id}}

        try:
            # Use collection.get() for direct metadata-based retrieval
            results = vector_store._collection.get(
                where=filter_dict,
                include=["metadatas", "documents"],
            )

            # Step 2: Check memory_type and content in Python
            for i, doc_id in enumerate(results.get("ids", [])):
                metadata = results["metadatas"][i] if results.get("metadatas") else {}
                meta_type = metadata.get("memory_type", "")
                meta_content = metadata.get("content", "")

                if meta_type == memory_type:
                    existing_content = meta_content.lower().strip()
                    new_content = content.lower().strip()
                    if existing_content == new_content:
                        # Exact duplicate content found
                        logger.info(
                            f"Found exact duplicate memory: user={user_id}, "
                            f"type={memory_type}, content={content}"
                        )
                        return doc_id

            return None

        except Exception as e:
            logger.warning(f"Failed to check existing memory: {e}")
            return None

    def update_memory(
        self,
        doc_id: str,
        content: str,
        timestamp: Optional[str] = None,
        importance: float = 0.5,
        session_id: str = "",
    ) -> bool:
        """
        Update an existing memory entry with new content and timestamp.

        ChromaDB does not support in-place updates, so we delete and re-add.

        Args:
            doc_id: The ChromaDB document ID to update.
            content: New content for the memory.
            timestamp: New timestamp (default: current UTC time).
            importance: New importance score.
            session_id: New session identifier.

        Returns:
            bool: True if update succeeded, False otherwise.
        """
        vector_store = self._get_vector_store()

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            # Delete the old document
            vector_store._collection.delete(ids=[doc_id])
            logger.info(f"Deleted old memory: id={doc_id}")

            # Reconstruct metadata from the doc_id
            parts = doc_id.split("_", 2)
            if len(parts) >= 3:
                user_id = parts[0]
                memory_type = parts[1]
            else:
                logger.warning(f"Cannot parse doc_id: {doc_id}")
                return False

            metadata = {
                "user_id": user_id,
                "memory_type": memory_type,
                "content": content,
                "timestamp": timestamp,
                "importance": importance,
                "session_id": session_id,
                "doc_id": doc_id,
            }

            doc_text = f"{memory_type}: {content}"
            doc = Document(page_content=doc_text, metadata=metadata)

            vector_store.add_documents(documents=[doc], ids=[doc_id])
            logger.info(f"Updated memory: id={doc_id}, content={content}")
            return True

        except Exception as e:
            logger.error(f"Failed to update memory id={doc_id}: {e}")
            return False

    def delete_user_memories(self, user_id: str) -> int:
        """
        Delete ALL memories for a given user from the memory ChromaDB.

        This is used for the "Clear my data" privacy feature.

        Args:
            user_id: The user whose memories should be deleted.

        Returns:
            int: Number of deleted memory entries.
        """
        vector_store = self._get_vector_store()

        try:
            filter_dict = {"user_id": {"$eq": user_id}}

            # Get IDs from the collection using the filter
            try:
                results = vector_store._collection.get(where=filter_dict)
                ids_to_delete = results.get("ids", [])
            except Exception as e:
                logger.warning(f"Could not get IDs via filter: {e}")
                ids_to_delete = []

            if ids_to_delete:
                vector_store._collection.delete(ids=ids_to_delete)
                logger.info(
                    f"Deleted {len(ids_to_delete)} memories for user={user_id}"
                )
                return len(ids_to_delete)
            else:
                logger.info(f"No memories found for user={user_id}")
                return 0

        except Exception as e:
            logger.error(f"Failed to delete memories for user={user_id}: {e}")
            return 0

    def delete_old_memories(self, max_age_days: int = MEMORY_MAX_AGE_DAYS) -> int:
        """
        Delete memories older than a specified number of days.

        This implements the automatic cleanup feature.

        Args:
            max_age_days: Memories older than this many days will be deleted.

        Returns:
            int: Number of deleted memories.
        """
        vector_store = self._get_vector_store()

        try:
            # Calculate the cutoff date
            cutoff = datetime.now(timezone.utc)
            # We need to get all documents and check timestamps
            # ChromaDB doesn't support date comparison in metadata filters natively,
            # so we iterate through all entries.

            all_results = vector_store._collection.get(include=["metadatas"])
            ids_to_delete = []
            ids_to_keep = []

            for i, doc_id in enumerate(all_results.get("ids", [])):
                metadata = all_results["metadatas"][i] if all_results.get("metadatas") else {}
                timestamp_str = metadata.get("timestamp", "")

                try:
                    # Parse the timestamp
                    if "T" in timestamp_str:
                        doc_time = datetime.strptime(
                            timestamp_str.split(".")[0], "%Y-%m-%dT%H:%M:%S"
                        ).replace(tzinfo=timezone.utc)
                    else:
                        doc_time = datetime.strptime(
                            timestamp_str, "%Y-%m-%d"
                        ).replace(tzinfo=timezone.utc)

                    age_days = (cutoff - doc_time).days
                    if age_days > max_age_days:
                        ids_to_delete.append(doc_id)
                    else:
                        ids_to_keep.append(doc_id)

                except (ValueError, TypeError):
                    # If we can't parse the timestamp, keep the memory
                    ids_to_keep.append(doc_id)

            if ids_to_delete:
                vector_store._collection.delete(ids=ids_to_delete)
                logger.info(
                    f"Deleted {len(ids_to_delete)} old memories "
                    f"(older than {max_age_days} days)"
                )
            else:
                logger.info("No old memories to delete.")

            return len(ids_to_delete)

        except Exception as e:
            logger.error(f"Failed to delete old memories: {e}")
            return 0

    def count_user_memories(self, user_id: str) -> int:
        """
        Count the number of memories stored for a given user.

        Args:
            user_id: The user to count memories for.

        Returns:
            int: Number of memories for this user.
        """
        try:
            memories = self.get_all_user_memories(user_id)
            return len(memories)
        except Exception:
            return 0

    def get_total_memory_count(self) -> int:
        """
        Get the total number of memories across all users.

        Returns:
            int: Total memory entries in the database.
        """
        vector_store = self._get_vector_store()
        try:
            return vector_store._collection.count()
        except Exception as e:
            logger.warning(f"Failed to get total memory count: {e}")
            return 0