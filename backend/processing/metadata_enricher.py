"""
processing/metadata_enricher.py

Metadata enrichment for extracted documents.

This module analyzes every chunk and enriches it
with useful metadata for retrieval, routing,
reasoning and answer formatting.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class MetadataEnricher:

    """
    Enrich metadata for every chunk.
    """

    FINANCIAL_KEYWORDS = {

        "revenue",
        "profit",
        "loss",
        "income",
        "expense",
        "assets",
        "liabilities",
        "equity",
        "cash",
        "ebitda",
        "pat",
        "ebit",
        "dividend",
        "share",
        "capital",
        "reserve",
        "production",
        "oil",
        "gas",
        "barrel",

    }

    SECTION_PATTERNS = [

        "financial highlights",
        "management discussion",
        "board of directors",
        "chairman's message",
        "corporate governance",
        "balance sheet",
        "cash flow",
        "statement of profit",
        "auditors report",

    ]

    # =====================================================

    def enrich(
        self,
        documents: list[Document],
    ) -> list[Document]:

        logger.info("=" * 70)
        logger.info("METADATA ENRICHMENT")
        logger.info("=" * 70)

        enriched = []

        for doc in documents:

            metadata = dict(doc.metadata)

            text = doc.page_content

            lower = text.lower()

            # ------------------------------------------
            # Statistics
            # ------------------------------------------

            metadata["character_count"] = len(text)

            metadata["word_count"] = len(text.split())

            metadata["line_count"] = len(text.splitlines())

            # ------------------------------------------
            # Numeric Features
            # ------------------------------------------

            metadata["contains_numbers"] = bool(

                re.search(r"\d", text)

            )

            metadata["contains_currency"] = bool(

                re.search(

                    r"(₹|\$|rs\.?|crore|million|billion)",

                    lower,

                )

            )

            metadata["contains_percentage"] = "%" in text

            metadata["contains_year"] = bool(

                re.search(

                    r"\b(19|20)\d{2}\b",

                    text,

                )

            )

            # ------------------------------------------
            # Financial Detection
            # ------------------------------------------

            metadata["financial_content"] = any(

                keyword in lower

                for keyword in self.FINANCIAL_KEYWORDS

            )

            # ------------------------------------------
            # Section Detection
            # ------------------------------------------

            section = "general"

            for pattern in self.SECTION_PATTERNS:

                if pattern in lower:

                    section = pattern.title()

                    break

            metadata["section"] = section

            # ------------------------------------------
            # Heading Detection
            # ------------------------------------------

            first_line = ""

            lines = text.splitlines()

            if lines:

                first_line = lines[0].strip()

            metadata["heading"] = first_line

            # ------------------------------------------
            # Content Flags
            # ------------------------------------------

            metadata["contains_table"] = (

                metadata.get("content_type") == "table"

            )

            metadata["contains_image"] = (

                metadata.get("content_type") in ("image", "page_ocr")

            )

            metadata["contains_text"] = (

                metadata.get("content_type") == "text"

            )

            enriched.append(

                Document(

                    page_content=text,

                    metadata=metadata,

                )

            )

        logger.info(

            "Enriched %d chunks.",

            len(enriched),

        )

        logger.info("=" * 70)

        return enriched


# ==========================================================
# Factory
# ==========================================================

_enricher = None


def get_metadata_enricher():

    global _enricher

    if _enricher is None:

        _enricher = MetadataEnricher()

    return _enricher