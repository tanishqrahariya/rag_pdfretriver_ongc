"""
evaluation/answer_verifier.py

Answer Verifier

Verifies whether the generated answer is
supported by the retrieved evidence.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from backend.llm.model import get_llm
from backend.llm.prompt import PromptManager
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_verifier = None


class AnswerVerifier:

    """
    Verify generated answers against evidence.
    """

    def __init__(self):

        self.llm = get_llm()

        self.chain = (

            PromptManager.verification_prompt()

            | self.llm

            | StrOutputParser()

        )

    # ======================================================
    # Verify
    # ======================================================

    def verify(
        self,
        evidence: str,
        answer: str,
    ) -> bool:

        logger.info("=" * 70)
        logger.info("ANSWER VERIFICATION")
        logger.info("=" * 70)

        if not evidence.strip():

            logger.warning(
                "No evidence available."
            )

            return False

        try:

            result = self.chain.invoke(

                {

                    "evidence": evidence,

                    "answer": answer,

                }

            ).strip().upper()

            logger.info(
                "Verification Result : %s",
                result,
            )

            return result.startswith("SUPPORTED")

        except Exception:

            logger.exception(
                "Verification failed."
            )

            return False

    # ======================================================
    # Verify or Replace
    # ======================================================

    def verify_answer(
        self,
        evidence: str,
        answer: str,
    ) -> str:

        if self.verify(
            evidence,
            answer,
        ):
            return answer

        logger.warning(
            "Answer not fully supported."
        )

        return (
            "I couldn't verify this answer using "
            "the retrieved evidence. Please try "
            "rephrasing your question or provide "
            "additional context."
        )


# ==========================================================
# Factory
# ==========================================================

def get_answer_verifier():

    global _verifier

    if _verifier is None:

        _verifier = AnswerVerifier()

    return _verifier