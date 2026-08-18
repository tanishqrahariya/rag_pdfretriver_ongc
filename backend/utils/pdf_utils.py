"""
utils/pdf_utils.py

PDF utility functions.

Centralizes page selection, validation and
common PDF-related helper methods.
"""

from __future__ import annotations

from typing import Iterable

from config import (
    START_PAGE,
    MAX_PAGES,
)


# ==========================================================
# Page Range
# ==========================================================

def get_page_range(
    total_pages: int,
) -> range:
    """
    Return the pages that should be processed.

    Examples
    --------

    START_PAGE = 0
    MAX_PAGES = None

    -> range(0,total_pages)

    START_PAGE = 0
    MAX_PAGES = 50

    -> range(0,50)

    START_PAGE = 200
    MAX_PAGES = 100

    -> range(200,300)
    """

    if total_pages <= 0:

        return range(0)

    start = max(0, START_PAGE)

    if start >= total_pages:

        return range(0)

    if MAX_PAGES is None:

        end = total_pages

    else:

        end = min(

            total_pages,

            start + MAX_PAGES,

        )

    return range(start, end)


# ==========================================================
# Total Pages
# ==========================================================

def total_selected_pages(
    total_pages: int,
) -> int:
    """
    Number of pages that will actually be processed.
    """

    return len(get_page_range(total_pages))


# ==========================================================
# Validate Page
# ==========================================================

def valid_page(
    page_number: int,
    total_pages: int,
) -> bool:
    """
    Check whether a page exists.
    """

    return 0 <= page_number < total_pages


# ==========================================================
# Is Selected
# ==========================================================

def page_selected(
    page_number: int,
    total_pages: int,
) -> bool:
    """
    Returns True if page belongs to processing range.
    """

    return page_number in get_page_range(
        total_pages
    )


# ==========================================================
# Human Readable
# ==========================================================

def page_description() -> str:
    """
    Human-readable processing description.
    """

    if MAX_PAGES is None:

        return (
            f"Pages {START_PAGE + 1} → End"
        )

    return (

        f"Pages "

        f"{START_PAGE + 1}"

        f" → "

        f"{START_PAGE + MAX_PAGES}"

    )


# ==========================================================
# Iterate Pages
# ==========================================================

def iter_pages(
    pages: Iterable,
):
    """
    Generic page iterator.

    Works for

    - fitz.Document
    - pdfplumber
    - list
    """

    page_list = list(pages)

    for page_number in get_page_range(
        len(page_list)
    ):

        yield page_number, page_list[page_number]