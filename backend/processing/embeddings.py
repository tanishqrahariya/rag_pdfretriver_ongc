"""
processing/embeddings.py

Embedding Manager

Loads embedding model only once and provides
helper methods for document and query embeddings.
"""

from __future__ import annotations

from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    EMBEDDING_MODEL,
    NORMALIZE_EMBEDDINGS,
)

from backend.utils.logger import get_logger

logger = get_logger(__name__)

_embedding_model = None


class EmbeddingManager:

    """
    Singleton embedding manager.
    """

    def __init__(self):

        logger.info("=" * 70)
        logger.info("LOADING EMBEDDING MODEL")
        logger.info("=" * 70)

        logger.info(
            "Embedding Model : %s",
            EMBEDDING_MODEL,
        )

        self.model = HuggingFaceEmbeddings(

            model_name=EMBEDDING_MODEL,

            encode_kwargs={

                "normalize_embeddings": NORMALIZE_EMBEDDINGS,

            },

        )

        logger.info(
            "Embedding model loaded successfully."
        )

    # ======================================================
    # Query Embedding
    # ======================================================

    def embed_query(
        self,
        query: str,
    ) -> List[float]:
        """
        Generate embedding for a search query.
        """

        return self.model.embed_query(query)

    # ======================================================
    # Document Embedding
    # ======================================================

    def embed_documents(
        self,
        documents: List[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        """

        return self.model.embed_documents(
            documents
        )

    # ======================================================
    # Accessor
    # ======================================================

    def get_model(self):
        """
        Return LangChain embedding model.
        """

        return self.model


# ==========================================================
# Factory
# ==========================================================

def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:

        _embedding_model = EmbeddingManager()

    return _embedding_model