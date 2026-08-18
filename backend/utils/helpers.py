"""
utils/helpers.py

Common helper functions used throughout the project.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List

from langchain_core.documents import Document


# ==========================================================
# Text Helpers
# ==========================================================

def clean_text(text: str) -> str:
    """
    Clean extracted text.
    """

    if not text:
        return ""

    text = text.replace("\x00", "")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# Chunk Hash
# ==========================================================

def calculate_chunk_hash(
    text: str,
    source: str = "",
    page: object = "",
) -> str:
    """
    Generate a unique hash for a chunk.

    Salted with source + page so that identical boilerplate text
    appearing on different pages or in different documents (common
    in annual reports — repeated headers, footers, disclaimers)
    does not collide into a single chunk_hash. A collision here
    would cause real, distinct chunks to be wrongly treated as
    duplicates during RRF fusion and deduplication, silently
    dropping one of them from the retrieved context.
    """

    payload = f"{source}::{page}::{text}"

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


# ==========================================================
# Document Formatting
# ==========================================================

def format_documents(
    documents: List[Document],
) -> str:
    """
    Convert retrieved documents into a prompt context.
    """

    formatted = []

    for i, doc in enumerate(documents, start=1):

        page = doc.metadata.get("page", "Unknown")

        source = doc.metadata.get("source", "Unknown")

        formatted.append(

            f"""
==============================
Document {i}
Source : {source}
Page   : {page}
------------------------------
{doc.page_content}
""".strip()

        )

    return "\n\n".join(formatted)


# ==========================================================
# Duplicate Removal
# ==========================================================

def deduplicate_documents(
    documents: List[Document],
) -> List[Document]:
    """
    Remove duplicate chunks.
    """

    unique = []

    seen = set()

    for doc in documents:

        chunk_hash = doc.metadata.get(
            "chunk_hash"
        )

        if chunk_hash is None:

            chunk_hash = calculate_chunk_hash(
                doc.page_content,
                source=doc.metadata.get("source", ""),
                page=doc.metadata.get("page", ""),
            )

            doc.metadata["chunk_hash"] = chunk_hash

        if chunk_hash in seen:
            continue

        seen.add(chunk_hash)

        unique.append(doc)

    return unique


# ==========================================================
# JSON
# ==========================================================

def safe_json_load(
    text: str,
) -> Dict[str, Any]:
    """
    Safely parse JSON.

    Returns empty dictionary if parsing fails.
    """

    try:

        return json.loads(text)

    except Exception:

        return {}


# ==========================================================
# Metadata
# ==========================================================

def merge_metadata(
    *metadata_objects,
) -> Dict[str, Any]:
    """
    Merge multiple metadata dictionaries.
    """

    merged = {}

    for metadata in metadata_objects:

        if metadata:

            merged.update(metadata)

    return merged


# ==========================================================
# Token Estimation
# ==========================================================

def estimate_tokens(
    text: str,
) -> int:
    """
    Rough token estimation.

    1 token ≈ 4 characters (English).
    """

    return max(1, len(text) // 4)


# ==========================================================
# Context Length
# ==========================================================

def total_context_length(
    documents: List[Document],
) -> int:
    """
    Total number of characters in retrieved context.
    """

    return sum(

        len(doc.page_content)

        for doc in documents

    )


# ==========================================================
# Pretty Print Sources
# ==========================================================

def format_sources(
    documents: List[Document],
) -> str:
    """
    Create source list for answer footer.
    """

    lines = []

    seen = set()

    for doc in documents:

        source = doc.metadata.get(
            "source",
            "Unknown",
        )

        page = doc.metadata.get(
            "page",
            "Unknown",
        )

        key = (source, page)

        if key in seen:
            continue

        seen.add(key)

        lines.append(

            f"{source} (Page {page})"

        )

    return "\n".join(lines)