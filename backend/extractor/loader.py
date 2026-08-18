"""
extractor/loader.py

Main PDF Loader.

Responsibilities
----------------
1. Load PDF
2. Extract Text
3. Extract Tables
4. Extract OCR Images
5. Merge Documents
6. Remove Duplicate Documents
"""

from __future__ import annotations

from pathlib import Path

import fitz

from langchain_core.documents import Document

from backend.extractor.text_extractor import get_text_extractor
from backend.extractor.table_extractor import get_table_extractor
from backend.extractor.image_extractor import get_image_extractor

from backend.utils.helpers import deduplicate_documents
from backend.utils.logger import get_logger

from config import MIN_TEXT_CHARS_PER_PAGE

logger = get_logger(__name__)


class PDFLoader:

    """
    Main extraction pipeline.
    """

    def __init__(self):

        self.text_extractor = get_text_extractor()

        self.table_extractor = get_table_extractor()

        self.image_extractor = get_image_extractor()

    # =====================================================

    def load_pdf(
        self,
        pdf_path: str,
    ) -> list[Document]:

        logger.info("=" * 70)
        logger.info("PDF LOADING")
        logger.info("=" * 70)

        pdf_path = str(Path(pdf_path))

        logger.info(
            "PDF : %s",
            pdf_path,
        )

        # -------------------------------------------------
        # Text
        # -------------------------------------------------

        text_docs = self.text_extractor.extract(
            pdf_path
        )

        logger.info(
            "Text Documents : %d",
            len(text_docs),
        )

        # -------------------------------------------------
        # Sparse-page detection
        # -------------------------------------------------
        # Pages whose real text layer is very short are often
        # templated/certificate-style pages (scorecards, certificates,
        # forms) where the label text lives in a background image and
        # only a handful of dynamic values are real text. Flag these
        # so the image extractor can run a full-page high-DPI OCR pass
        # on them, recovering the label context that plain text
        # extraction misses.

        chars_per_page: dict[int, int] = {}

        for doc in text_docs:

            page_num = doc.metadata.get("page")

            chars_per_page[page_num] = (
                chars_per_page.get(page_num, 0)
                + len(doc.page_content)
            )

        with fitz.open(pdf_path) as _pdf:
            total_pages = len(_pdf)

        sparse_pages = {
            page_num
            for page_num in range(1, total_pages + 1)
            if chars_per_page.get(page_num, 0) < MIN_TEXT_CHARS_PER_PAGE
        }

        if sparse_pages:

            logger.info(
                "Sparse pages flagged for full-page OCR : %s",
                sorted(sparse_pages),
            )

        # -------------------------------------------------
        # Tables
        # -------------------------------------------------

        table_docs = self.table_extractor.extract(
            pdf_path
        )

        logger.info(
            "Table Documents : %d",
            len(table_docs),
        )

        # -------------------------------------------------
        # Images / OCR
        # -------------------------------------------------

        image_docs = self.image_extractor.extract(
            pdf_path,
            ocr_full_pages=sparse_pages,
        )

        logger.info(
            "Image Documents : %d",
            len(image_docs),
        )

        # -------------------------------------------------
        # Merge
        # -------------------------------------------------

        documents = (

            text_docs

            + table_docs

            + image_docs

        )

        logger.info(
            "Merged Documents : %d",
            len(documents),
        )

        # -------------------------------------------------
        # Remove Duplicates
        # -------------------------------------------------

        documents = deduplicate_documents(
            documents
        )

        logger.info(
            "Unique Documents : %d",
            len(documents),
        )

        logger.info("=" * 70)

        return documents


# ==========================================================
# Factory
# ==========================================================

_loader = None


def get_pdf_loader():

    global _loader

    if _loader is None:

        _loader = PDFLoader()

    return _loader