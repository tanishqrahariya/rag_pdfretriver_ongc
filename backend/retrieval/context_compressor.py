"""
retrieval/context_compressor.py

Context Compressor

Removes irrelevant information from retrieved documents
before sending them to the LLM.

Pipeline

Retrieved Documents
        │
        ▼
CrossEncoder
        │
        ▼
Sentence Selection
        │
        ▼
Compressed Context
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document

from config import RERANK_TOP_K
from backend.retrieval.reranker import get_reranker
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_context_compressor = None


class ContextCompressor:

    """
    Compress retrieved context.

    Removes noisy paragraphs while preserving
    important evidence.
    """

    def __init__(self):

        self.reranker = get_reranker()

    # =====================================================
    # Sentence Splitter
    # =====================================================

    def _split_sentences(
        self,
        text: str,
    ) -> List[str]:

        sentences = re.split(

            r"(?<=[.!?])\s+",

            text,

        )

        return [

            sentence.strip()

            for sentence in sentences

            if sentence.strip()

        ]

    # =====================================================
    # Compress
    # =====================================================

    def compress(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Document]:

        logger.info("=" * 70)
        logger.info("CONTEXT COMPRESSION")
        logger.info("=" * 70)

        compressed = []

        for document in documents:

            sentences = self._split_sentences(
                document.page_content
            )

            if len(sentences) <= 4:

                compressed.append(document)

                continue

            sentence_docs = [

                Document(

                    page_content=s,

                    metadata=document.metadata,

                )

                for s in sentences

            ]

            top_sentences = self.reranker.rerank(

                query=query,

                documents=sentence_docs,

                top_k=min(5, len(sentence_docs)),

            )

            # The reranker returns sentences ordered by relevance
            # score, not by where they appeared in the source text.
            # Re-sort the *kept* sentences back into their original
            # document order so the compressed passage still reads
            # sequentially (important for things like "FY24 was X,
            # FY25 was Y" where order carries meaning).
            kept_texts = {
                sentence.page_content
                for sentence in top_sentences
            }

            ordered_kept = [
                original
                for original in sentences
                if original in kept_texts
            ]

            text = " ".join(ordered_kept)

            compressed.append(

                Document(

                    page_content=text,

                    metadata=document.metadata,

                )

            )

        logger.info(

            "Compressed %d documents.",

            len(compressed),

        )

        logger.info("=" * 70)

        return compressed


# ==========================================================
# Factory
# ==========================================================

def get_context_compressor():

    global _context_compressor

    if _context_compressor is None:

        _context_compressor = ContextCompressor()

    return _context_compressor