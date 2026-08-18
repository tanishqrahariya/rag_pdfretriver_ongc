"""
llm/prompt_builder.py

Dynamic Prompt Builder

Central place for building prompts used throughout the
Enterprise RAG pipeline.
"""

from __future__ import annotations

from backend.llm.prompt import PromptManager
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_builder = None


# ==========================================================
# Prompt Builder
# ==========================================================

class PromptBuilder:
    """
    Build prompts for different stages of the Enterprise RAG pipeline.
    """

    def __init__(self):
        pass

    # =====================================================
    # Final Answer Prompt
    # =====================================================

    def build(
        self,
        question: str,
        facts: str,
        answer_style: str = "normal",
        history: str = "",
    ):

        logger.info("=" * 70)
        logger.info("BUILDING ANSWER PROMPT")
        logger.info("=" * 70)

        prompt = PromptManager.answer_prompt()

        payload = {
            "question": question,
            "facts": facts,
            "history": history,
            "answer_style": answer_style.lower(),
        }

        logger.info(
            "Answer Style : %s",
            answer_style,
        )

        return prompt, payload

    # =====================================================
    # Reasoning Prompt
    # =====================================================

    def build_reasoning(
        self,
        question: str,
        facts: str,
        analysis_mode: bool = False,
    ):

        logger.info("=" * 70)
        logger.info("BUILDING REASONING PROMPT")
        logger.info("=" * 70)

        prompt = PromptManager.reasoning_prompt()

        payload = {
            "question": question,
            "facts": facts,
            "analysis_mode": analysis_mode,
        }

        logger.info(
            "Analysis Mode : %s",
            analysis_mode,
        )

        return prompt, payload

    # =====================================================
    # Verification Prompt
    # =====================================================

    def build_verification(
        self,
        facts: str,
        answer: str,
    ):

        logger.info("=" * 70)
        logger.info("BUILDING VERIFICATION PROMPT")
        logger.info("=" * 70)

        prompt = PromptManager.verification_prompt()

        payload = {
            "facts": facts,
            "answer": answer,
        }

        return prompt, payload

    # =====================================================
    # Citation Prompt
    # =====================================================

    def build_citation(
        self,
        answer: str,
        sources: str,
    ):

        logger.info("=" * 70)
        logger.info("BUILDING CITATION PROMPT")
        logger.info("=" * 70)

        prompt = PromptManager.citation_prompt()

        payload = {
            "answer": answer,
            "sources": sources,
        }

        return prompt, payload


# ==========================================================
# Factory
# ==========================================================

def get_prompt_builder():

    global _builder

    if _builder is None:
        _builder = PromptBuilder()

    return _builder