"""
core/rag_engine.py

Enterprise RAG Engine
"""

from __future__ import annotations

from pathlib import Path

from backend.extractor.loader import get_pdf_loader
from backend.processing.chunker import get_chunker
from backend.vectorstore.chroma_store import get_chroma_store

from backend.retrieval.hybrid import HybridRetriever

from backend.chain.rag_chain import (
    get_rag_chain,
    reset_rag_chain,
)

from backend.utils.logger import get_logger

logger = get_logger(__name__)

_rag = None


# ==========================================================
# Build RAG
# ==========================================================

def build_rag(
    pdf_paths: list[str],
    progress_callback=None,
):
    """
    Build the RAG pipeline.

    progress_callback: optional callable(percent: int, message: str) -> None
    Used by the UI layer (e.g. Streamlit) to report live progress.
    Safe to leave as None for CLI usage.
    """

    global _rag

    def _report(percent: int, message: str):
        logger.info("[%3d%%] %s", percent, message)
        if progress_callback is not None:
            progress_callback(percent, message)

    logger.info("=" * 70)
    logger.info("BUILDING RAG PIPELINE")
    logger.info("=" * 70)

    reset_rag_chain()

    _report(2, "Starting knowledge base build...")

    loader = get_pdf_loader()

    documents = []

    total_pdfs = max(len(pdf_paths), 1)

    for idx, pdf in enumerate(pdf_paths):

        logger.info("Loading PDF : %s", pdf)

        pdf_name = Path(pdf).name

        _report(
            int(5 + (idx / total_pdfs) * 40),
            f"Reading {pdf_name} ({idx + 1}/{total_pdfs})...",
        )

        documents.extend(

            loader.load_pdf(pdf)

        )

    _report(45, "Splitting documents into chunks...")

    chunker = get_chunker()

    chunks = chunker.split_documents(

        documents

    )

    _report(55, f"Generating embeddings for {len(chunks)} chunks...")

    store = get_chroma_store()

    store.initialize(

        chunks

    )

    _report(85, "Building hybrid retriever...")

    hybrid = HybridRetriever(

        vector_retriever=store.get_vector_retriever(),

        bm25_manager=store.get_bm25(),

    )

    _report(95, "Finalizing RAG chain...")

    _rag = get_rag_chain(

        hybrid

    )

    _report(100, "Knowledge Base Ready")

    logger.info("RAG READY")

    return _rag


# ==========================================================
# Get
# ==========================================================

def get_rag():

    if _rag is None:

        raise RuntimeError(

            "Knowledge Base has not been built."

        )

    return _rag


# ==========================================================
# Ask
# ==========================================================

def ask_question(

    rag,

    question: str,

    history: str = "",

):

    return rag.invoke(

        question=question,

        chat_history=history,

    )


# ==========================================================
# Clear
# ==========================================================

def clear_rag():

    global _rag

    reset_rag_chain()

    _rag = None