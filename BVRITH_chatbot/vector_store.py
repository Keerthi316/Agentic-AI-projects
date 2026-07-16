"""
Vector store management for the College FAQ Chatbot.

Handles ChromaDB initialization, embedding generation,
document indexing, similarity search, and database persistence.
"""

import os
import json
import hashlib
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from utils import get_config

logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
TOP_K = 5
CHROMA_DB_DIR = "chroma_db"
DATA_DIR = "data"
COLLECTION_NAME = "college_kb"
HASH_FILE = os.path.join(CHROMA_DB_DIR, "doc_hash.json")

# ── Embeddings singleton — created once and reused across all callers ──────
_embeddings_instance: Optional[OpenAIEmbeddings] = None


def get_embeddings() -> OpenAIEmbeddings:
    """
    Get the embedding model using OpenRouter/OpenAI compatible API.

    The instance is cached as a module-level singleton so that repeated
    calls (e.g. from MemoryStore.__init__) don't create a new object
    on every invocation — saving overhead and keeping HTTP connection pools.

    Returns:
        OpenAIEmbeddings: Configured embedding model instance.

    Raises:
        ValueError: If the API key is missing.
    """
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    config = get_config()
    api_key = config.get("OPENROUTER_API_KEY")

    if not api_key or api_key == "your-api-key-here":
        raise ValueError(
            "OpenRouter API key is missing. "
            "Please set OPENROUTER_API_KEY in your .env file."
        )

    _embeddings_instance = OpenAIEmbeddings(
        model=config.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_api_key=api_key,
        openai_api_base=config.get("OPENROUTER_BASE_URL"),
    )
    return _embeddings_instance


def _compute_doc_hash(file_path: str) -> str:
    """
    Compute a SHA-256 hash of the document file to detect changes.

    Args:
        file_path (str): Path to the document file.

    Returns:
        str: Hex digest of the file hash.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _get_stored_hash() -> Optional[str]:
    """
    Read the previously stored document hash from disk.

    Returns:
        Optional[str]: The stored hash, or None if not available.
    """
    try:
        if os.path.exists(HASH_FILE):
            with open(HASH_FILE, "r") as f:
                data = json.load(f)
                return data.get("doc_hash")
    except Exception as e:
        logger.warning(f"Could not read stored hash: {e}")
    return None


def _save_doc_hash(doc_hash: str) -> None:
    """
    Persist the document hash to disk for future change detection.

    Args:
        doc_hash (str): The hash to store.
    """
    try:
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        with open(HASH_FILE, "w") as f:
            json.dump({"doc_hash": doc_hash}, f)
    except Exception as e:
        logger.warning(f"Could not save document hash: {e}")


def _document_changed(doc_path: str) -> bool:
    """
    Check if the document has changed since the last vector store build.

    Args:
        doc_path (str): Path to the current document.

    Returns:
        bool: True if the document has changed or no hash is stored.
    """
    current_hash = _compute_doc_hash(doc_path)
    stored_hash = _get_stored_hash()
    return current_hash != stored_hash


def get_vector_store() -> Tuple[Optional[Chroma], Optional[str]]:
    """
    Get or create the ChromaDB vector store.

    Automatically detects if the knowledge base document has changed
    and rebuilds the vector store if needed.

    Returns:
        Tuple[Optional[Chroma], Optional[str]]:
            - Chroma vector store instance (or None on failure)
            - Status/error message string
    """
    embeddings = get_embeddings()
    persist_directory = CHROMA_DB_DIR
    doc_path = _find_document()

    # Check if document has changed — if so, delete old DB and rebuild
    if doc_path and _document_changed(doc_path):
        logger.info("Document has changed. Rebuilding vector store...")
        _delete_vector_store(persist_directory)

    # Check if persisted database already exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        try:
            logger.info("Loading existing vector store from disk...")
            vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=persist_directory,
            )
            # Verify the collection has documents
            collection_size = vector_store._collection.count()
            if collection_size > 0:
                logger.info(
                    f"Loaded existing vector store with {collection_size} chunks."
                )
                return vector_store, f"Loaded existing database ({collection_size} chunks)"
            else:
                logger.warning("Existing vector store is empty. Recreating...")
        except Exception as e:
            logger.warning(f"Could not load existing vector store: {e}. Recreating...")

    # Create vector store from document
    logger.info("Creating new vector store from document...")
    return _create_vector_store_from_document(embeddings, persist_directory)


def _delete_vector_store(persist_directory: str) -> None:
    """
    Delete the persisted vector store and hash file so it gets rebuilt.

    Args:
        persist_directory (str): Path to the ChromaDB directory.
    """
    import shutil
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)
    os.makedirs(persist_directory, exist_ok=True)
    logger.info("Deleted old vector store for rebuild.")


def _create_vector_store_from_document(
    embeddings: OpenAIEmbeddings,
    persist_directory: str,
) -> Tuple[Optional[Chroma], Optional[str]]:
    """
    Create a new vector store by loading, splitting, and indexing the document.

    Args:
        embeddings: The embedding model to use.
        persist_directory: Directory to persist the ChromaDB database.

    Returns:
        Tuple[Optional[Chroma], Optional[str]]: Vector store and status message.
    """
    from utils import load_document, split_documents

    # Find the document file
    doc_path = _find_document()

    if not doc_path:
        error_msg = (
            "No knowledge base document found. "
            f"Please place a .docx file in the '{DATA_DIR}' folder."
        )
        logger.error(error_msg)
        return None, error_msg

    try:
        # Load and split the document
        documents = load_document(doc_path)
        chunks = split_documents(
            documents,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        if not chunks:
            error_msg = "Document loaded but no chunks were created."
            logger.error(error_msg)
            return None, error_msg

        # Create and persist the vector store
        logger.info(f"Creating vector store with {len(chunks)} chunks...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=COLLECTION_NAME,
        )

        # Save document hash for change detection on next startup
        _save_doc_hash(_compute_doc_hash(doc_path))

        logger.info(
            f"Created and persisted vector store with {len(chunks)} chunks."
        )
        return vector_store, f"Created new database ({len(chunks)} chunks)"

    except Exception as e:
        error_msg = f"Failed to create vector store: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def _find_document() -> Optional[str]:
    """
    Find a .docx file in the data directory.

    Returns:
        Optional[str]: Path to the first .docx file found, or None.
    """
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        return None

    docx_files = list(data_path.glob("*.docx"))
    if not docx_files:
        return None

    return str(docx_files[0])


def retrieve_documents(
    vector_store: Chroma,
    query: str,
    k: int = TOP_K,
) -> Tuple[List[Document], float]:
    """
    Retrieve relevant documents for a given query.

    Uses similarity_search_with_score (L2 distance) as the primary method.
    L2 distance is always positive and converts cleanly to 0-1 relevance.
    Falls back to MMR for diversity if needed.

    Args:
        vector_store (Chroma): The vector store to search.
        query (str): The user's query string.
        k (int): Number of top results to return. Defaults to TOP_K.

    Returns:
        Tuple[List[Document], float]:
            - List of retrieved documents with metadata.
            - Maximum relevance score (confidence) among retrieved documents.
    """
    documents = []
    max_score = 0.0
    seen_ids = set()

    # Primary strategy: similarity_search_with_score (L2 distance)
    # L2 distance is always positive → relevance = 1/(1+distance) is always 0-1
    try:
        results = vector_store.similarity_search_with_score(query, k=k)
        for doc, distance in results:
            doc_id = doc.metadata.get("chunk_id", id(doc))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                # L2 distance → relevance using exponential decay
                # This gives better score separation:
                # dist=0.5 → 0.82, dist=0.7 → 0.76, dist=1.0 → 0.63, dist=1.5 → 0.47
                relevance = 1.0 / (1.0 + (abs(distance) ** 1.5))
                doc.metadata["relevance_score"] = round(relevance, 4)
                doc.metadata["distance"] = round(abs(distance), 4)
                documents.append(doc)
                max_score = max(max_score, relevance)
    except Exception as e:
        logger.warning(f"similarity_search_with_score failed: {e}")

    # Fallback: MMR for diverse results if needed
    if len(documents) < k:
        try:
            query_embedding = vector_store._embedding_function.embed_query(query)
            mmr_results = vector_store.max_marginal_relevance_search_by_vector(
                query_embedding,
                k=k,
                fetch_k=k * 2,
                lambda_mult=0.5,
            )
            for doc in mmr_results:
                doc_id = doc.metadata.get("chunk_id", id(doc))
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    doc.metadata["relevance_score"] = 0.3
                    documents.append(doc)
        except Exception as e:
            logger.warning(f"MMR search failed: {e}")

    logger.info(
        f"Retrieved {len(documents)} documents for query. "
        f"Max relevance score: {max_score:.4f}"
    )

    return documents[:k], max_score


def get_chunk_count() -> int:
    """
    Get the number of chunks in the vector store.

    Returns:
        int: Number of indexed chunks, or 0 if unavailable.
    """
    try:
        embeddings = get_embeddings()
        persist_directory = CHROMA_DB_DIR

        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=persist_directory,
            )
            return vector_store._collection.count()
    except Exception as e:
        logger.warning(f"Could not get chunk count: {e}")

    return 0