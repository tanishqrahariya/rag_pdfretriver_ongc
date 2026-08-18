"""
utils/logger.py

Centralized logging for the ONGC Enterprise RAG Assistant.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import (
    LOG_FILE,
    LOG_LEVEL,
)

# ==========================================================
# Logger Configuration
# ==========================================================

LOGGER_NAME = "ONGC_RAG"

_configured = False


def setup_logger():
    """
    Configure application logger.

    This function is executed only once.
    """

    global _configured

    if _configured:
        return

    Path(LOG_FILE).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(LOGGER_NAME)

    logger.setLevel(LOG_LEVEL)

    logger.handlers.clear()

    formatter = logging.Formatter(

        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",

        datefmt="%Y-%m-%d %H:%M:%S",

    )

    # ------------------------------------------------------
    # Console
    # ------------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # ------------------------------------------------------
    # File
    # ------------------------------------------------------

    file_handler = logging.FileHandler(

        LOG_FILE,

        encoding="utf-8",

    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.propagate = False

    _configured = True


# ==========================================================
# Public API
# ==========================================================

def get_logger(name: str | None = None):

    setup_logger()

    if name:

        return logging.getLogger(
            f"{LOGGER_NAME}.{name}"
        )

    return logging.getLogger(LOGGER_NAME)