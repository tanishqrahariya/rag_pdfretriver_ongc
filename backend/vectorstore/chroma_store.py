"""
vectorstore/chroma_store.py

Chroma Vector Store Manager
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    COLLECTION_NAME,
    PERSIST_DIRECTORY,
    VECTOR_TOP_K,
)

from backend.processing.embeddings import get_embedding_model
from backend.retrieval.bm25 import get_bm25_manager
from backend.utils.logger import get_logger

logger = get_logger(__name__)

METADATA_FILE = Path(PERSIST_DIRECTORY) / "metadata.json"

_vector_store = None


class ChromaStore:

    def __init__(self):

        self.embedding_manager = get_embedding_model()

        self.embeddings = self.embedding_manager.get_model()

        self.vectorstore = None

        self.retriever = None

        self.bm25 = get_bm25_manager()

    # ======================================================
    # Content Fingerprint
    # ======================================================

    def _compute_signature(
        self,
        documents: list[Document],
    ) -> str:
        """
        Fingerprint the actual document set (which files, which
        chunks) rather than just counting chunks. Two different
        sets of PDFs can coincidentally produce the same chunk
        count, which would fool a count-only staleness check into
        reusing the wrong embeddings.
        """

        parts = sorted(
            f"{doc.metadata.get('source', '')}::{doc.metadata.get('chunk_hash', '')}"
            for doc in documents
        )

        return hashlib.sha256(
            "|".join(parts).encode("utf-8")
        ).hexdigest()

    # ======================================================
    # Metadata
    # ======================================================

    def _save_metadata(
        self,
        chunk_count: int,
        signature: str = "",
    ):

        metadata = {

            "chunk_count": chunk_count,

            "signature": signature,

            "embedding_model":

                self.embedding_manager.model.model_name,

        }

        Path(PERSIST_DIRECTORY).mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(

            METADATA_FILE,

            "w",

            encoding="utf-8",

        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
            )

    # ------------------------------------------------------

    def _load_metadata(self):

        if not METADATA_FILE.exists():

            return None

        with open(

            METADATA_FILE,

            "r",

            encoding="utf-8",

        ) as f:

            return json.load(f)

    # ======================================================
    # Create
    # ======================================================

    def create(
        self,
        documents: list[Document],
    ):

        logger.info("=" * 70)
        logger.info("CREATING CHROMA DATABASE")
        logger.info("=" * 70)

        if Path(PERSIST_DIRECTORY).exists():

            shutil.rmtree(
                PERSIST_DIRECTORY,
                ignore_errors=True,
            )

        self.vectorstore = Chroma.from_documents(

            documents=documents,

            embedding=self.embeddings,

            collection_name=COLLECTION_NAME,

            persist_directory=PERSIST_DIRECTORY,

        )

        logger.info(

            "Stored vectors : %d",

            self.vectorstore._collection.count(),

        )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": VECTOR_TOP_K},
        )

        self.bm25.build(documents)

        self._save_metadata(
            chunk_count=len(documents),
            signature=self._compute_signature(documents),
        )

    # ======================================================
    # Load
    # ======================================================

    def load(self):

        logger.info("=" * 70)
        logger.info("LOADING CHROMA DATABASE")
        logger.info("=" * 70)

        self.vectorstore = Chroma(

            collection_name=COLLECTION_NAME,

            embedding_function=self.embeddings,

            persist_directory=PERSIST_DIRECTORY,

        )

        logger.info(

            "Existing vectors : %d",

            self.vectorstore._collection.count(),

        )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": VECTOR_TOP_K},
        )

        data = self.vectorstore.get(

            include=[

                "documents",

                "metadatas",

            ]

        )

        documents = []

        for text, meta in zip(

            data["documents"],

            data["metadatas"],

        ):

            documents.append(

                Document(

                    page_content=text,

                    metadata=meta,

                )

            )

        self.bm25.build(documents)

        return documents

    # ======================================================
    # Initialize
    # ======================================================

    def initialize(
        self,
        documents: list[Document],
    ):

        if not Path(PERSIST_DIRECTORY).exists():

            self.create(documents)

            return

        loaded = self.load()

        metadata = self._load_metadata()

        if metadata is None:

            logger.info(
                "Metadata missing. Rebuilding..."
            )

            self.create(documents)

            return

        new_signature = self._compute_signature(documents)

        old_signature = metadata.get("signature")

        if old_signature is None:

            # Older metadata.json from before signatures existed —
            # fall back to the count check once, then a signature
            # will be saved and used from now on.
            if len(loaded) != len(documents):

                logger.info(
                    "Chunk count changed. Rebuilding..."
                )

                self.create(documents)

                return

        elif old_signature != new_signature:

            logger.info(
                "Document set changed. Rebuilding..."
            )

            self.create(documents)

            return

        logger.info(
            "Using existing vector database."
        )

    # ======================================================
    # Accessors
    # ======================================================

    def get_vectorstore(self):

        return self.vectorstore

    def get_vector_retriever(self):

        return self.retriever

    def get_bm25(self):

        return self.bm25


# ==========================================================
# Factory
# ==========================================================

def get_chroma_store():

    global _vector_store

    if _vector_store is None:

        _vector_store = ChromaStore()

    return _vector_store