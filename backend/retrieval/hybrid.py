"""
retrieval/hybrid.py

Production Hybrid Retriever
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

from langchain_core.documents import Document

from config import (
    HYBRID_TOP_K,
    VECTOR_TOP_K,
    BM25_TOP_K,
    RRF_K,
    MAX_CHUNKS_PER_PAGE,
)

from backend.retrieval.reranker import get_reranker
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:

    def __init__(
        self,
        vector_retriever,
        bm25_manager,
    ):

        self.vector = vector_retriever

        self.bm25 = bm25_manager

        self.reranker = get_reranker()

    # =====================================================
    # Reciprocal Rank Fusion
    # =====================================================

    def _rrf(
        self,
        dense_docs: List[Document],
        sparse_docs: List[Document],
    ) -> List[Document]:

        scores = defaultdict(float)

        document_map = {}

        for docs in [dense_docs, sparse_docs]:

            for rank, doc in enumerate(docs):

                chunk_hash = doc.metadata.get(
                    "chunk_hash"
                )

                if chunk_hash is None:
                    continue

                scores[chunk_hash] += 1.0 / (
                    RRF_K + rank + 1
                )

                document_map[chunk_hash] = doc

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        merged = []

        for chunk_hash, score in ranked:

            doc = document_map[chunk_hash]

            metadata = dict(doc.metadata)

            metadata["rrf_score"] = score

            merged.append(

                Document(

                    page_content=doc.page_content,

                    metadata=metadata,

                )

            )

        return merged

    # =====================================================
    # Metadata Boost
    # =====================================================

    def _metadata_boost(
        self,
        docs: List[Document],
    ):

        if not docs:
            return

        base_scores = [
            doc.metadata.get(
                "rerank_score",
                doc.metadata.get("rrf_score", 0.0),
            )
            for doc in docs
        ]

        lo, hi = min(base_scores), max(base_scores)
        spread = (hi - lo) or 1.0

        for doc, base_score in zip(docs, base_scores):

            boost = 0.0

            meta = doc.metadata

            if meta.get("financial_content"):

                boost += 0.20

            if meta.get("contains_table"):

                boost += 0.10

            if meta.get("contains_currency"):

                boost += 0.10

            meta["metadata_score"] = boost

            # Normalize the reranker's raw score to 0-1 first, then
            # apply the boost. Adding a small flat boost straight to
            # a raw cross-encoder logit would either be swamped
            # (score scale much bigger than the boost) or dominate
            # (score scale much smaller) depending on the model —
            # normalizing puts them on the same footing so relevance
            # still leads and the boost only breaks close calls.
            normalized = (base_score - lo) / spread

            meta["combined_score"] = normalized + boost

        docs.sort(

            key=lambda d:

            d.metadata.get(
                "combined_score",
                0.0,
            ),

            reverse=True,

        )

    # =====================================================
    # Diversity
    # =====================================================

    def _diversity_filter(
        self,
        docs: List[Document],
    ) -> List[Document]:

        final = []

        page_counter = {}

        for doc in docs:

            page = doc.metadata.get(
                "page",
                -1,
            )

            count = page_counter.get(page, 0)

            if count >= MAX_CHUNKS_PER_PAGE:
                continue

            page_counter[page] = count + 1

            final.append(doc)

            if len(final) >= HYBRID_TOP_K:
                break

        return final

    # =====================================================
    # Retrieve
    # =====================================================

    def retrieve(
        self,
        query: str,
    ) -> List[Document]:

        logger.info("=" * 70)
        logger.info("HYBRID RETRIEVAL")
        logger.info("=" * 70)

        dense = self.vector.invoke(query)

        logger.info(
            "Dense : %d",
            len(dense),
        )

        sparse = self.bm25.retrieve(
            query,
            top_k=BM25_TOP_K,
        )

        logger.info(
            "Sparse : %d",
            len(sparse),
        )

        merged = self._rrf(
            dense,
            sparse,
        )

        logger.info(
            "Merged : %d",
            len(merged),
        )

        reranked = self.reranker.rerank(

            query=query,

            documents=merged,

            top_k=min(
                len(merged),
                VECTOR_TOP_K * 2,
            ),

        )

        logger.info(
            "Reranked : %d",
            len(reranked),
        )

        unique = []

        seen = set()

        for doc in reranked:

            chunk_hash = doc.metadata.get(
                "chunk_hash"
            )

            if chunk_hash in seen:
                continue

            seen.add(chunk_hash)

            unique.append(doc)

        self._metadata_boost(unique)

        final = self._diversity_filter(
            unique
        )

        logger.info(
            "Final : %d",
            len(final),
        )

        logger.info("=" * 70)

        return final