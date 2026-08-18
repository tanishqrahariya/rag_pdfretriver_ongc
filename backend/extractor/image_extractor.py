"""
extractor/image_extractor.py

Extract images from PDF and perform OCR.

Features
--------
✓ Extract embedded images
✓ Ignore tiny images
✓ OCR using EasyOCR
✓ Chart/Image detection
✓ LangChain Document output
"""

from __future__ import annotations

import io

import fitz
import easyocr
import numpy as np

from PIL import Image

from langchain_core.documents import Document

from config import (
    MIN_IMAGE_WIDTH,
    MIN_IMAGE_HEIGHT,
    OCR_CONFIDENCE,
    OCR_RENDER_DPI,
)

from backend.utils.helpers import (
    calculate_chunk_hash,
    clean_text,
)

from backend.utils.logger import get_logger
from backend.utils.pdf_utils import iter_pages

logger = get_logger(__name__)


class ImageExtractor:

    def __init__(self):

        self.reader = easyocr.Reader(
            ["en"],
            gpu=False,
        )

    # ======================================================

    def _ocr_image(
        self,
        image: Image.Image,
    ) -> str:

        # EasyOCR's readtext() only accepts a file path, raw bytes,
        # or a numpy array — never a PIL.Image object. Passing a
        # PIL.Image straight through raises an internal "Invalid
        # input type" error, which the try/except around every call
        # site was silently swallowing — so OCR quietly returned
        # nothing for every image and every full-page fallback,
        # including exactly the kind of label text (e.g. "GATE
        # Score:", "Marks out of 100:") that certificate/scorecard
        # style PDFs bake into a background template rather than the
        # real text layer.
        image_array = np.array(image.convert("RGB"))

        results = self.reader.readtext(image_array)

        lines = []

        for item in results:

            confidence = item[2]

            if confidence < OCR_CONFIDENCE:
                continue

            lines.append(item[1])

        return clean_text(
            "\n".join(lines)
        )

    # ======================================================

    def _ocr_full_page(
        self,
        page,
    ) -> str:
        """
        Render the whole page at a high DPI and OCR that render.

        Some PDFs (certificates, scorecards, templated forms) draw
        their label text as part of a low-resolution background image
        or as vector artwork. OCR-ing the raw embedded image XObject
        directly can miss this text entirely or read it with very low
        confidence. Rendering the page itself at a controlled high DPI
        gives OCR a much better chance of picking up that label text,
        regardless of how it was originally authored.
        """

        pixmap = page.get_pixmap(dpi=OCR_RENDER_DPI)

        image = Image.open(
            io.BytesIO(pixmap.tobytes("png"))
        )

        return self._ocr_image(image)

    # ======================================================

    def extract(
        self,
        pdf_path: str,
        ocr_full_pages: set[int] | None = None,
    ) -> list[Document]:
        """
        ocr_full_pages: a set of 1-indexed page numbers that should
        additionally get a full-page high-DPI OCR pass, on top of the
        normal embedded-image OCR below. Passed in by the loader for
        pages whose real text layer came back sparse.
        """

        ocr_full_pages = ocr_full_pages or set()

        logger.info("=" * 70)
        logger.info("IMAGE EXTRACTION")
        logger.info("=" * 70)

        pdf = fitz.open(pdf_path)

        documents = []

        total_images = 0

        for page_number, page in iter_pages(pdf):

            page_num_1_indexed = page_number + 1

            # -------------------------------------------------
            # Full-page OCR fallback for sparse/templated pages
            # -------------------------------------------------

            if page_num_1_indexed in ocr_full_pages:

                try:

                    full_page_text = self._ocr_full_page(page)

                    if full_page_text:

                        documents.append(

                            Document(

                                page_content=full_page_text,

                                metadata={

                                    "source": pdf_path,

                                    "page": page_num_1_indexed,

                                    "content_type": "page_ocr",

                                    "contains_ocr": True,

                                    "chunk_hash":
                                        calculate_chunk_hash(
                                            full_page_text
                                        ),

                                },

                            )

                        )

                        logger.info(
                            "Full-page OCR recovered %d chars "
                            "on sparse page %d",
                            len(full_page_text),
                            page_num_1_indexed,
                        )

                except Exception:

                    logger.exception(
                        "Failed full-page OCR on page %d",
                        page_num_1_indexed,
                    )

            image_list = page.get_images(full=True)

            if not image_list:
                continue

            logger.info(
                "Page %d : %d images",
                page_number + 1,
                len(image_list),
            )

            for image_id, image_info in enumerate(image_list):

                try:

                    xref = image_info[0]

                    image_dict = pdf.extract_image(xref)

                    image_bytes = image_dict["image"]

                    image = Image.open(
                        io.BytesIO(image_bytes)
                    )

                    width, height = image.size

                    if (
                        width < MIN_IMAGE_WIDTH
                        or height < MIN_IMAGE_HEIGHT
                    ):
                        continue

                    total_images += 1

                    ocr_text = self._ocr_image(image)

                    if not ocr_text:
                        continue

                    metadata = {

                        "source": pdf_path,

                        "page": page_number + 1,

                        "image_id": image_id,

                        "width": width,

                        "height": height,

                        "content_type": "image",

                        "contains_ocr": True,

                        "chunk_hash":
                            calculate_chunk_hash(
                                ocr_text
                            ),

                    }

                    documents.append(

                        Document(

                            page_content=ocr_text,

                            metadata=metadata,

                        )

                    )

                except Exception:

                    logger.exception(
                        "Failed image extraction "
                        "Page %d Image %d",
                        page_number + 1,
                        image_id,
                    )

        pdf.close()

        logger.info(
            "Processed Images : %d",
            total_images,
        )

        logger.info(
            "OCR Documents : %d",
            len(documents),
        )

        logger.info("=" * 70)

        return documents


_image_extractor = None


def get_image_extractor():

    global _image_extractor

    if _image_extractor is None:

        _image_extractor = ImageExtractor()

    return _image_extractor