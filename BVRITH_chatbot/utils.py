"""
Utility functions for the College FAQ Chatbot.

Handles document loading, text splitting, configuration management,
and helper utilities used across the application.
"""

import os
import time
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def get_config() -> Dict[str, str]:
    """
    Load configuration from environment variables.

    Returns:
        Dict[str, str]: Dictionary containing configuration values.
    """
    config = {
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
        "OPENROUTER_BASE_URL": os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        "OPENROUTER_MODEL": os.getenv("OPENROUTER_MODEL", "gpt-4o-mini"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    }
    return config


def load_document(file_path: str) -> List[Document]:
    """
    Load a Word (.docx) document and return LangChain Document objects.

    Args:
        file_path (str): Path to the .docx file.

    Returns:
        List[Document]: List of LangChain Document objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a .docx file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if path.suffix.lower() not in [".docx"]:
        raise ValueError(f"Unsupported file format: {path.suffix}. Only .docx is supported.")

    logger.info(f"Loading document: {file_path}")
    loader = Docx2txtLoader(str(path))
    documents = loader.load()

    logger.info(f"Loaded {len(documents)} document(s) from {file_path}")
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding and retrieval.

    Each chunk is enriched with metadata including the source filename
    and section heading (extracted from the first line of each chunk).

    Args:
        documents (List[Document]): List of documents to split.
        chunk_size (int): Maximum size of each chunk in characters.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        List[Document]: List of split document chunks with metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    # Enrich metadata and prepend section heading to content for better embeddings
    for i, chunk in enumerate(chunks):
        # Extract a section heading from the first line of the chunk
        first_line = chunk.page_content.strip().split("\n")[0][:80]
        section_heading = first_line if first_line else "General"
        chunk.metadata["chunk_id"] = i
        chunk.metadata["section"] = section_heading

        # CRITICAL: Prepend section heading to content so embeddings capture context
        # This makes "who is the principal" match chunks whose section says "Principal"
        chunk.page_content = f"[Section: {section_heading}] {chunk.page_content}"

    logger.info(f"Split {len(documents)} document(s) into {len(chunks)} chunks")
    return chunks


def format_time(seconds: float) -> str:
    """
    Format a time duration in seconds to a human-readable string.

    Args:
        seconds (float): Time duration in seconds.

    Returns:
        str: Formatted time string (e.g., '1.23s').
    """
    return f"{seconds:.2f}s"


class Timer:
    """Simple context manager for measuring execution time."""

    def __enter__(self) -> "Timer":
        """Start the timer."""
        self.start = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop the timer on exit."""
        self.end = time.time()
        self.duration = self.end - self.start
