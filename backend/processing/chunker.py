"""
processing/chunker.py

Document Chunking Module

Responsibilities
----------------
1. Split large documents
2. Preserve metadata
3. Generate chunk ids
4. Generate chunk hashes
5. Estimate token count
"""

from __future__ import annotations

import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from backend.utils.helpers import (
    calculate_chunk_hash,
    estimate_tokens,
)

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentChunker:

    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(

            chunk_size=CHUNK_SIZE,

            chunk_overlap=CHUNK_OVERLAP,

            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],

        )

    # ======================================================

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:

        logger.info("=" * 70)
        logger.info("DOCUMENT CHUNKING")
        logger.info("=" * 70)

        chunks = []

        total_input = len(documents)

        logger.info(
            "Input Documents : %d",
            total_input,
        )

        for parent_id, document in enumerate(documents):

            split_docs = self.splitter.create_documents(

                [document.page_content]

            )

            logger.info(

                "Page %s -> %d chunks",

                document.metadata.get(
                    "page",
                    "?",
                ),

                len(split_docs),

            )

            for index, chunk in enumerate(split_docs):

                metadata = dict(document.metadata)

                metadata.update(

                    {

                        "chunk_id": str(uuid.uuid4()),

                        "parent_document": parent_id,

                        "chunk_index": index,

                        "chunk_hash":

                            calculate_chunk_hash(

                                chunk.page_content,

                                source=document.metadata.get("source", ""),

                                page=document.metadata.get("page", ""),

                            ),

                        "token_count":

                            estimate_tokens(

                                chunk.page_content

                            ),

                    }

                )

                chunks.append(

                    Document(

                        page_content=chunk.page_content,

                        metadata=metadata,

                    )

                )

        logger.info(

            "Generated Chunks : %d",

            len(chunks),

        )

        logger.info("=" * 70)

        return chunks


# ==========================================================
# Factory
# ==========================================================

_chunker = None


def get_chunker():

    global _chunker

    if _chunker is None:

        _chunker = DocumentChunker()

    return _chunker