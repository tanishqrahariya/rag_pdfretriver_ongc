"""
retrieval/query_planner.py

Enterprise Query Planner

Only uses LLM for complex questions.
Simple questions bypass the planner.

Pipeline

Question
    │
    ▼
Simple?
 │         │
Yes        No
 │          │
 ▼          ▼
Question   LLM Planner
 │          │
 └──────┬───┘
        ▼
   QueryPlan
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser

from backend.llm.model import get_llm
from backend.llm.prompt import PromptManager

from backend.utils.logger import get_logger

logger = get_logger(__name__)

_planner = None


# ==========================================================
# Query Plan
# ==========================================================

@dataclass
class QueryPlan:

    intent: str = "general"

    answer_style: str = "normal"

    sub_queries: list[str] = field(default_factory=list)


# ==========================================================
# Planner
# ==========================================================

class QueryPlanner:

    def __init__(self):

        self.llm = get_llm()

        self.chain = (

            PromptManager.query_planner_prompt()

            | self.llm

            | StrOutputParser()

        )

    # ======================================================
    # Simple Question Detection
    # ======================================================

    def _is_simple(self, question: str) -> bool:

        q = question.lower()

        # Comparison / multi-part questions benefit from being
        # decomposed into sub-queries (e.g. one search per year)
        # even when short — "Compare FY24 and FY25 performance" is
        # only 5 words but is exactly the kind of question the
        # planner exists for. Force these through the LLM planner
        # regardless of length.
        force_planning_keywords = [

            "compare",

            "comparison",

            "versus",

            " vs ",

            "difference between",

            "trend",

        ]

        if any(word in q for word in force_planning_keywords):

            return False

        keywords = [

            "who",

            "what",

            "when",

            "where",

            "which",

            "whose",

            "ceo",

            "chairman",

            "director",

            "md",

            "head",

            "revenue",

            "profit",

            "production",

            "employee",

            "headquarters",

        ]

        if len(q.split()) <= 8:

            return True

        if any(word in q for word in keywords):

            return True

        return False

    # ======================================================
    # Plan
    # ======================================================

    def plan(
        self,
        question: str,
    ) -> QueryPlan:

        logger.info("=" * 70)
        logger.info("QUERY PLANNER")
        logger.info("=" * 70)

        # ----------------------------------------------
        # Skip planner for simple questions
        # ----------------------------------------------

        if self._is_simple(question):

            logger.info("Simple question detected.")

            return QueryPlan(

                intent="general",

                answer_style="normal",

                sub_queries=[question],

            )

        # ----------------------------------------------
        # Complex Questions
        # ----------------------------------------------

        try:

            response = self.chain.invoke(

                {

                    "question": question,

                }

            )

            logger.info(response)

            response = response.strip()

            if response.startswith("```"):

                response = (

                    response

                    .replace("```json", "")

                    .replace("```", "")

                    .strip()

                )

            data = json.loads(response)

            queries = data.get(

                "sub_queries",

                [question],

            )

            if not queries:

                queries = [question]

            return QueryPlan(

                intent=data.get(

                    "intent",

                    "general",

                ),

                answer_style=data.get(

                    "answer_style",

                    "normal",

                ),

                sub_queries=queries,

            )

        except Exception:

            logger.exception(

                "Planner failed."

            )

            return QueryPlan(

                intent="general",

                answer_style="normal",

                sub_queries=[question],

            )


# ==========================================================
# Factory
# ==========================================================

def get_query_planner():

    global _planner

    if _planner is None:

        _planner = QueryPlanner()

    return _planner