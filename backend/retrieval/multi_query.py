"""
retrieval/multi_query.py

Multi Query Retriever

Pipeline

Question
    │
    ▼
Query Planner
    │
    ▼
Multiple Search Queries
    │
    ▼
Hybrid Retrieval
    │
    ▼
Merge
    │
    ▼
Duplicate Removal
    │
    ▼
Frequency Ranking
    │
    ▼
Final Documents
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

from langchain_core.documents import Document

from backend.utils.logger import get_logger

logger = get_logger(__name__)

_multi_query = None


class MultiQueryRetriever:

    """
    Execute retrieval for multiple search queries and
    merge the results.
    """

    def __init__(self, hybrid_retriever):

        self.hybrid = hybrid_retriever

    # =====================================================
    # Retrieve
    # =====================================================

    def retrieve(
        self,
        queries: list[str],
    ) -> List[Document]:

        logger.info("=" * 70)
        logger.info("MULTI QUERY RETRIEVAL")
        logger.info("=" * 70)

        score_map = defaultdict(int)

        document_map = {}

        # ---------------------------------------------
        # Run retrieval for every query
        # ---------------------------------------------

        for index, query in enumerate(queries, start=1):

            logger.info(
                "Query %d : %s",
                index,
                query,
            )

            docs = self.hybrid.retrieve(query)

            logger.info(
                "Retrieved %d documents.",
                len(docs),
            )

            for doc in docs:

                chunk_hash = doc.metadata.get(
                    "chunk_hash"
                )

                if chunk_hash is None:

                    chunk_hash = hash(
                        doc.page_content
                    )

                score_map[chunk_hash] += 1

                document_map[chunk_hash] = doc

        # ---------------------------------------------
        # Rank by frequency
        # ---------------------------------------------

        ranked = sorted(

            score_map.items(),

            key=lambda x: x[1],

            reverse=True,

        )

        results = []

        for chunk_hash, frequency in ranked:

            doc = document_map[chunk_hash]

            metadata = dict(doc.metadata)

            metadata["query_frequency"] = frequency

            results.append(

                Document(

                    page_content=doc.page_content,

                    metadata=metadata,

                )

            )

        logger.info(
            "Merged Documents : %d",
            len(results),
        )

        logger.info("=" * 70)

        return results


# ==========================================================
# Factory
# ==========================================================

def get_multi_query_retriever(
    hybrid_retriever,
):

    global _multi_query

    if _multi_query is None:

        _multi_query = MultiQueryRetriever(
            hybrid_retriever
        )

    return _multi_query