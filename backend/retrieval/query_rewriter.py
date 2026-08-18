"""
retrieval/query_rewriter.py

Conversation-aware Query Rewriter.

Responsibilities
----------------
1. Rewrite follow-up questions
2. Resolve pronouns
3. Preserve user intent
4. Return standalone search query
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from backend.llm.model import get_llm
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_rewriter = None


class QueryRewriter:

    def __init__(self):

        self.llm = get_llm()

        self.chain = (
            self._prompt()
            | self.llm
            | StrOutputParser()
        )

    # =====================================================
    # Prompt
    # =====================================================

    def _prompt(self):

        from langchain_core.prompts import ChatPromptTemplate

        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an expert query rewriting assistant.

Rewrite the user's latest question into a
fully self-contained search query.

Rules

• Use conversation history.

• Resolve all pronouns.

• Preserve meaning.

• Do NOT answer.

• Return only the rewritten query.
                    """,
                ),
                (
                    "human",
                    """
Conversation

{history}

Question

{question}
                    """,
                ),
            ]
        )

    # =====================================================
    # Rewrite
    # =====================================================

    def rewrite(
        self,
        question: str,
        history: str,
    ) -> str:

        logger.info("=" * 70)
        logger.info("QUERY REWRITING")
        logger.info("=" * 70)

        if not history.strip():

            logger.info(
                "No history. Original query used."
            )

            return question

        try:

            rewritten = self.chain.invoke(
                {
                    "history": history,
                    "question": question,
                }
            ).strip()

            if not rewritten:

                return question

            logger.info(
                "Original : %s",
                question,
            )

            logger.info(
                "Rewritten : %s",
                rewritten,
            )

            return rewritten

        except Exception:

            logger.exception(
                "Query rewriting failed."
            )

            return question


# ==========================================================
# Factory
# ==========================================================

def get_query_rewriter():

    global _rewriter

    if _rewriter is None:

        _rewriter = QueryRewriter()

    return _rewriter