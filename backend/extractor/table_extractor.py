"""
extractor/table_extractor.py

Extract tables from PDF using pdfplumber.

Features
--------
✓ Multi-table extraction
✓ Markdown conversion
✓ Financial table detection
✓ Metadata preservation
✓ Compatible with LangChain Documents
"""

from __future__ import annotations

import pdfplumber
import pandas as pd

from langchain_core.documents import Document

from backend.utils.logger import get_logger
from backend.utils.helpers import (
    calculate_chunk_hash,
)
from backend.utils.pdf_utils import iter_pages

logger = get_logger(__name__)


class TableExtractor:

    def __init__(self):

        self.financial_keywords = {

            "revenue",

            "profit",

            "loss",

            "assets",

            "liabilities",

            "cash",

            "ebitda",

            "pat",

            "income",

            "expense",

            "equity",

            "share",

            "capital",

            "dividend",

        }

    # ======================================================

    def _is_financial_table(
        self,
        markdown: str,
    ) -> bool:

        text = markdown.lower()

        return any(

            word in text

            for word in self.financial_keywords

        )

    # ======================================================

    def _table_to_markdown(
        self,
        table,
    ) -> str:

        try:

            df = pd.DataFrame(table)

            if df.empty:

                return ""

            df.columns = df.iloc[0]

            df = df[1:]

            df = df.fillna("")

            return df.to_markdown(index=False)

        except Exception:

            return ""

    # ======================================================

    def extract(
        self,
        pdf_path: str,
    ) -> list[Document]:

        logger.info("=" * 70)
        logger.info("TABLE EXTRACTION")
        logger.info("=" * 70)

        documents = []

        with pdfplumber.open(pdf_path) as pdf:

            for page_number, page in iter_pages(pdf.pages):

                try:

                    tables = page.extract_tables()

                except Exception:

                    logger.exception(

                        "Failed page %d",

                        page_number + 1,

                    )

                    continue

                if not tables:

                    continue

                logger.info(

                    "Page %d : %d tables",

                    page_number + 1,

                    len(tables),

                )

                for table_id, table in enumerate(tables):

                    markdown = self._table_to_markdown(table)

                    if not markdown:

                        continue

                    metadata = {

                        "source": pdf_path,

                        "page": page_number + 1,

                        "table_id": table_id,

                        "content_type": "table",

                        "financial_table":

                            self._is_financial_table(

                                markdown

                            ),

                        "chunk_hash":

                            calculate_chunk_hash(

                                markdown

                            ),

                    }

                    documents.append(

                        Document(

                            page_content=markdown,

                            metadata=metadata,

                        )

                    )

        logger.info(

            "Extracted Tables : %d",

            len(documents),

        )

        logger.info("=" * 70)

        return documents


_table_extractor = None


def get_table_extractor():

    global _table_extractor

    if _table_extractor is None:

        _table_extractor = TableExtractor()

    return _table_extractor