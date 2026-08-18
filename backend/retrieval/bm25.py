"""
retrieval/bm25.py

BM25 Sparse Retriever

Responsibilities
----------------
1. Build BM25 Index
2. Retrieve Documents
3. Singleton Manager
"""

from __future__ import annotations

import re
from typing import List

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config import BM25_TOP_K
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_bm25_manager = None


class BM25Manager:

    def __init__(self):

        self.bm25 = None

        self.documents: List[Document] = []

        self.corpus = []

    # ======================================================
    # Tokenizer
    # ======================================================

    def _tokenize(
        self,
        text: str,
    ) -> List[str]:

        text = text.lower()

        return re.findall(
            r"\b\w+\b",
            text,
        )

    # ======================================================
    # Build Index
    # ======================================================

    def build(
        self,
        documents: List[Document],
    ):

        logger.info("=" * 70)
        logger.info("BUILDING BM25 INDEX")
        logger.info("=" * 70)

        self.documents = documents

        if not documents:

            logger.warning(
                "No documents supplied for BM25."
            )

            return

        self.corpus = [

            self._tokenize(
                doc.page_content
            )

            for doc in documents

        ]

        self.bm25 = BM25Okapi(
            self.corpus
        )

        logger.info(
            "Indexed %d documents.",
            len(documents),
        )

        logger.info("=" * 70)

    # ======================================================
    # Retrieve
    # ======================================================

    def retrieve(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
    ) -> List[Document]:

        if self.bm25 is None:

            logger.warning(
                "BM25 index has not been built."
            )

            return []

        tokens = self._tokenize(query)

        scores = self.bm25.get_scores(
            tokens
        )

        ranked = sorted(

            zip(scores, self.documents),

            key=lambda x: x[0],

            reverse=True,

        )

        results = []

        for score, doc in ranked[:top_k]:

            metadata = dict(doc.metadata)

            metadata["bm25_score"] = float(score)

            results.append(

                Document(

                    page_content=doc.page_content,

                    metadata=metadata,

                )

            )

        logger.info(

            "BM25 Retrieved : %d",

            len(results),

        )

        return results

    # ======================================================
    # Statistics
    # ======================================================

    def document_count(
        self,
    ) -> int:

        return len(self.documents)

    # ======================================================
    # Reset
    # ======================================================

    def reset(self):

        self.bm25 = None

        self.documents = []

        self.corpus = []

        logger.info(
            "BM25 index cleared."
        )


# ==========================================================
# Factory
# ==========================================================

def get_bm25_manager():

    global _bm25_manager

    if _bm25_manager is None:

        _bm25_manager = BM25Manager()

    return _bm25_manager