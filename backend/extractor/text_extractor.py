"""
extractor/text_extractor.py

Extract text from PDF using PyMuPDF.

Features
--------
✔ Fast extraction
✔ Metadata preservation
✔ Page filtering
✔ Text cleaning
✔ Chunk hash generation
"""

from __future__ import annotations

import fitz

from langchain_core.documents import Document

from backend.utils.logger import get_logger
from backend.utils.helpers import (
    clean_text,
    calculate_chunk_hash,
)
from backend.utils.pdf_utils import iter_pages

logger = get_logger(__name__)


class TextExtractor:
    """
    Extract text from PDF.
    """

    def extract(
        self,
        pdf_path: str,
    ) -> list[Document]:

        logger.info("=" * 70)
        logger.info("TEXT EXTRACTION")
        logger.info("=" * 70)

        pdf = fitz.open(pdf_path)

        documents = []

        total_pages = len(pdf)

        logger.info(
            "Total Pages : %d",
            total_pages,
        )

        extracted_pages = 0

        for page_number, page in iter_pages(pdf):

            text = page.get_text("text")

            text = clean_text(text)

            if not text:
                continue

            metadata = {

                "source": pdf_path,

                "page": page_number + 1,

                "content_type": "text",

                "chunk_hash": calculate_chunk_hash(text),

            }

            documents.append(

                Document(

                    page_content=text,

                    metadata=metadata,

                )

            )

            extracted_pages += 1

        pdf.close()

        logger.info(
            "Extracted Pages : %d",
            extracted_pages,
        )

        logger.info(
            "Documents : %d",
            len(documents),
        )

        logger.info("=" * 70)

        return documents


_text_extractor = None


def get_text_extractor():

    global _text_extractor

    if _text_extractor is None:

        _text_extractor = TextExtractor()

    return _text_extractor