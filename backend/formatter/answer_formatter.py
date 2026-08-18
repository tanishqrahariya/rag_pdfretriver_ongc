"""
formatter/answer_formatter.py

Enterprise Answer Formatter

Responsibilities
----------------
1. Clean LLM output
2. Remove source references
3. Remove RAG-specific phrases
4. Remove duplicate paragraphs
5. Remove duplicate headings
6. Normalize whitespace
7. Normalize empty responses
8. Produce clean ChatGPT-like answers
"""

from __future__ import annotations

import re

from backend.utils.logger import get_logger

logger = get_logger(__name__)

_formatter = None


# ==========================================================
# Answer Formatter
# ==========================================================

class AnswerFormatter:

    def __init__(self):

        logger.info("Answer Formatter Initialized.")

    # =====================================================
    # Remove Source References
    # =====================================================

    def _remove_sources(self, text: str) -> str:

        patterns = [

            r"(?is)#+\s*Sources.*$",
            r"(?is)#+\s*References.*$",
            r"(?is)Sources\s*:.*$",
            r"(?is)References\s*:.*$",
            r"\(Source.*?\)",
            r"\(Page\s*\d+\)",
            r"Page\s*:\s*\d+",
            r"Source\s*:\s*.*",
            r"\[\d+\]",

        ]

        for pattern in patterns:

            text = re.sub(pattern, "", text)

        return text

    # =====================================================
    # Remove RAG Phrases
    # =====================================================

    def _remove_rag_phrases(self, text: str) -> str:

        patterns = [

            r"(?i)based on the (provided|given|retrieved) (context|information|documents),?\s*",

            r"(?i)according to the (provided|retrieved) (documents|context),?\s*",

            r"(?i)from the structured facts,?\s*",

            r"(?i)the retrieved documents (show|indicate|suggest) that\s*",

            r"(?i)the provided documents (show|indicate|suggest) that\s*",

            r"(?i)the context indicates that\s*",

            r"(?i)the context shows that\s*",

            r"(?i)based on the available information,?\s*",

            r"(?i)according to the available information,?\s*",

        ]

        for pattern in patterns:

            text = re.sub(pattern, "", text)

        return text

    # =====================================================
    # Remove Duplicate Paragraphs
    # =====================================================

    def _remove_duplicates(self, text: str) -> str:

        paragraphs = []

        seen = set()

        for paragraph in text.split("\n\n"):

            paragraph = paragraph.strip()

            if not paragraph:

                continue

            key = re.sub(r"\s+", " ", paragraph.lower())

            if key in seen:

                continue

            seen.add(key)

            paragraphs.append(paragraph)

        return "\n\n".join(paragraphs)

    # =====================================================
    # Remove Duplicate Headings
    # =====================================================

    def _remove_duplicate_headings(self, text: str) -> str:

        lines = text.splitlines()

        cleaned = []

        previous = ""

        for line in lines:

            current = line.strip()

            if current and current.lower() == previous.lower():

                continue

            cleaned.append(line)

            previous = current

        return "\n".join(cleaned)

    # =====================================================
    # Normalize Whitespace
    # =====================================================

    def _normalize(self, text: str) -> str:

        text = text.replace("\r", "")

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = "\n".join(

            line.rstrip()

            for line in text.splitlines()

        )

        return text.strip()

    # =====================================================
    # Normalize Empty Responses
    # =====================================================

    def _normalize_empty_answer(self, text: str) -> str:

        phrases = [

            "no relevant information",

            "insufficient information",

            "not enough information",

            "couldn't find",

            "cannot find",

            "unable to find",

            "information is unavailable",

        ]

        lower = text.lower()

        if any(p in lower for p in phrases):

            if len(text.split()) < 25:

                return (

                    "I couldn't find sufficient information "

                    "in the available documents."

                )

        return text

    # =====================================================
    # Remove Repeated Blank Bullet Points
    # =====================================================

    def _cleanup_lists(self, text: str) -> str:

        text = re.sub(

            r"\n[-•]\s*\n",

            "\n",

            text,

        )

        return text

    # =====================================================
    # Final Formatting
    # =====================================================

    def format(

        self,

        answer: str,

        facts: dict | None = None,

        documents=None,

    ) -> str:

        logger.info("=" * 70)
        logger.info("FORMATTING ANSWER")
        logger.info("=" * 70)

        if not answer or not answer.strip():

            return (

                "I couldn't find sufficient information "

                "in the available documents."

            )

        answer = self._remove_sources(answer)

        answer = self._remove_rag_phrases(answer)

        answer = self._remove_duplicate_headings(answer)

        answer = self._remove_duplicates(answer)

        answer = self._cleanup_lists(answer)

        answer = self._normalize(answer)

        answer = self._normalize_empty_answer(answer)

        logger.info(

            "Final Answer Length : %d",

            len(answer),

        )

        logger.info("Formatting Completed.")

        logger.info("=" * 70)

        return answer


# ==========================================================
# Factory
# ==========================================================

def get_answer_formatter():

    global _formatter

    if _formatter is None:

        _formatter = AnswerFormatter()

    return _formatter