"""
chain/reasoning_chain.py

Enterprise Reasoning Chain

Pipeline

Question
    │
    ▼
Structured Facts
    │
    ▼
Reasoning Prompt
    │
    ▼
LLM
    │
    ▼
Professional Answer
"""

from __future__ import annotations

import json

from langchain_core.output_parsers import StrOutputParser

from backend.llm.model import get_llm
from backend.llm.prompt_builder import get_prompt_builder

from backend.utils.logger import get_logger

logger = get_logger(__name__)

_reasoning_chain = None


class ReasoningChain:
    """
    Performs reasoning using structured facts extracted from
    retrieved documents.
    """

    def __init__(self):

        self.llm = get_llm()
        self.prompt_builder = get_prompt_builder()

    # =====================================================
    # Convert facts to readable text
    # =====================================================

    @staticmethod
    def _facts_to_text(facts: dict) -> str:

        return json.dumps(
            facts,
            indent=2,
            ensure_ascii=False,
        )

    # =====================================================
    # Detect analytical questions
    # =====================================================

    @staticmethod
    def _is_analysis_question(question: str) -> bool:

        keywords = [
            "analysis",
            "analyze",
            "analytical",
            "review",
            "performance",
            "report",
            "insight",
            "compare",
            "comparison",
            "growth",
            "trend",
            "strength",
            "weakness",
            "future",
            "outlook",
            "financial",
            "swot",
        ]

        q = question.lower()

        return any(word in q for word in keywords)

    # =====================================================
    # Validate facts
    # =====================================================

    @staticmethod
    def _has_facts(facts: dict) -> bool:

        if not facts:
            return False

        for value in facts.values():

            if isinstance(value, list) and value:
                return True

            if isinstance(value, dict) and value:
                return True

        return False

    # =====================================================
    # Run
    # =====================================================

    def run(
        self,
        question: str,
        facts: dict,
    ) -> str:

        logger.info("=" * 70)
        logger.info("REASONING CHAIN")
        logger.info("=" * 70)

        if not self._has_facts(facts):

            return (
                "I couldn't find enough relevant information "
                "to answer your question."
            )

        logger.info(
            "Analysis Mode : %s",
            self._is_analysis_question(question),
        )

        logger.info(
            "Financial Facts : %d",
            len(facts.get("financial", [])),
        )

        logger.info(
            "Production Facts : %d",
            len(facts.get("production", [])),
        )

        logger.info(
            "Risk Facts : %d",
            len(facts.get("risk", [])),
        )

        logger.info(
            "People Facts : %d",
            len(facts.get("people", [])),
        )

        prompt, payload = self.prompt_builder.build_reasoning(

            question=question,

            facts=self._facts_to_text(facts),

            analysis_mode=self._is_analysis_question(question),

        )

        chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke(payload).strip()

        logger.info("Reasoning completed successfully.")
        logger.info("=" * 70)

        return answer

    # =====================================================
    # Batch
    # =====================================================

    def batch(
        self,
        questions: list[str],
        facts_list: list[dict],
    ) -> list[str]:

        return [
            self.run(question, facts)
            for question, facts in zip(
                questions,
                facts_list,
            )
        ]


# ==========================================================
# Factory
# ==========================================================

def get_reasoning_chain():

    global _reasoning_chain

    if _reasoning_chain is None:

        _reasoning_chain = ReasoningChain()

    return _reasoning_chain