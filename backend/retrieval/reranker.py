"""
retrieval/reranker.py

CrossEncoder Reranker

Pipeline

Query
    │
    ▼
Retrieved Documents
    │
    ▼
CrossEncoder
    │
    ▼
Score Every Chunk
    │
    ▼
Sort
    │
    ▼
Top Documents
"""

from __future__ import annotations

import time
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import (
    RERANKER_MODEL,
    RERANK_TOP_K,
)

from backend.utils.logger import get_logger

logger = get_logger(__name__)

_reranker = None


class CrossEncoderReranker:

    def __init__(self):

        logger.info("=" * 70)
        logger.info("LOADING CROSSENCODER")
        logger.info("=" * 70)

        logger.info(
            "Model : %s",
            RERANKER_MODEL,
        )

        self.model = CrossEncoder(
            RERANKER_MODEL,
        )

        logger.info(
            "CrossEncoder loaded successfully."
        )

    # ======================================================
    # Rerank
    # ======================================================

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = RERANK_TOP_K,
    ) -> List[Document]:

        if not documents:
            return []

        start = time.perf_counter()

        pairs = [

            (

                query,

                doc.page_content,

            )

            for doc in documents

        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        ranked = sorted(

            zip(scores, documents),

            key=lambda x: float(x[0]),

            reverse=True,

        )

        results = []

        for score, doc in ranked[:top_k]:

            metadata = dict(doc.metadata)

            metadata["rerank_score"] = float(score)

            results.append(

                Document(

                    page_content=doc.page_content,

                    metadata=metadata,

                )

            )

        elapsed = time.perf_counter() - start

        logger.info(

            "Reranked %d → %d documents (%.2f sec)",

            len(documents),

            len(results),

            elapsed,

        )

        return results

    # ======================================================
    # Best Score
    # ======================================================

    def best_score(
        self,
        query: str,
        documents: List[Document],
    ) -> float:

        reranked = self.rerank(

            query,

            documents,

            top_k=1,

        )

        if not reranked:

            return 0.0

        return reranked[0].metadata.get(

            "rerank_score",

            0.0,

        )


# ==========================================================
# Factory
# ==========================================================

def get_reranker():

    global _reranker

    if _reranker is None:

        _reranker = CrossEncoderReranker()

    return _reranker