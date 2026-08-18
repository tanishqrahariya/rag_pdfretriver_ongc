"""
retrieval/fact_extractor.py

Enterprise Question-Aware Fact Extractor

Pipeline

Retrieved Documents
        │
        ▼
Sentence Extraction
        │
        ▼
Intent Detection
        │
        ▼
Relevance Filtering
        │
        ▼
Fact Categorization
        │
        ▼
Ranking
        │
        ▼
Structured Facts
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import List

from langchain_core.documents import Document

from backend.utils.logger import get_logger

logger = get_logger(__name__)

_fact_extractor = None


class FactExtractor:
    """
    Enterprise Fact Extractor.

    Unlike a simple regex extractor, this extractor first
    understands what the user is asking before selecting
    relevant facts.

    Goal

    Question
        ↓
    Relevant Facts
        ↓
    Reasoning LLM
    """

    # =====================================================
    # Regular Expressions
    # =====================================================

    MONEY_PATTERN = re.compile(
        r"(?:₹|Rs\.?|INR)?\s?[\d,]+(?:\.\d+)?\s*(?:crore|cr|million|billion)?",
        re.IGNORECASE,
    )

    PERCENT_PATTERN = re.compile(
        r"\d+(?:\.\d+)?%",
        re.IGNORECASE,
    )

    YEAR_PATTERN = re.compile(
        r"(?:FY'? ?\d{2}|FY ?\d{4}|20\d{2})",
        re.IGNORECASE,
    )

    PRODUCTION_PATTERN = re.compile(
        r"\d+(?:\.\d+)?\s*(?:MMT|MMSCMD|BCM|MW|KM)",
        re.IGNORECASE,
    )

    # =====================================================

    def __init__(self):

        logger.info("Fact Extractor Initialized.")

    # =====================================================
    # Detect User Intent
    # =====================================================

    def _detect_intent(
        self,
        question: str,
    ) -> str:

        question = question.lower()

        intents = {

            "financial": [

                "revenue",
                "profit",
                "pat",
                "income",
                "financial",
                "ebitda",
                "cash",
                "earnings",
                "dividend",
                "expense",
                "tax",

            ],

            "production": [

                "production",
                "crude",
                "oil",
                "gas",
                "refinery",
                "exploration",
                "mmt",
                "mmscmd",

            ],

            "leadership": [

                "chairman",
                "ceo",
                "director",
                "board",
                "leadership",

            ],

            "project": [

                "project",
                "asset",
                "plant",
                "pipeline",
                "field",
                "offshore",

            ],

            "risk": [

                "risk",
                "challenge",
                "problem",
                "threat",
                "issue",

            ],

            "esg": [

                "esg",
                "green",
                "carbon",
                "environment",
                "sustainability",
                "csr",

            ],

            "comparison": [

                "compare",
                "comparison",
                "difference",
                "versus",
                "vs",

            ],

            "trend": [

                "trend",
                "increase",
                "decrease",
                "growth",
                "decline",

            ],

        }

        for intent, keywords in intents.items():

            if any(

                keyword in question

                for keyword in keywords

            ):

                return intent

        return "general"
      
      # =====================================================
    # Split Document into Sentences
    # =====================================================

    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:

        text = text.replace("\n", " ")

        sentences = re.split(

            r"(?<=[.!?])\s+",

            text,

        )

        cleaned = []

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence) < 12:

                continue

            cleaned.append(sentence)

        return cleaned

    # =====================================================
    # Calculate Sentence Score
    # =====================================================

    def _sentence_score(

        self,

        sentence: str,

        question: str,

        intent: str,

    ) -> int:

        score = 0

        sentence_lower = sentence.lower()

        question_lower = question.lower()

        # -------------------------------------------------
        # Common keywords
        # -------------------------------------------------

        for word in question_lower.split():

            if len(word) <= 2:
                continue

            if word in sentence_lower:

                score += 3

        # -------------------------------------------------
        # Numbers
        # -------------------------------------------------

        if self.MONEY_PATTERN.search(sentence):

            score += 5

        if self.PERCENT_PATTERN.search(sentence):

            score += 2

        if self.YEAR_PATTERN.search(sentence):

            score += 2

        if self.PRODUCTION_PATTERN.search(sentence):

            score += 3

        # -------------------------------------------------
        # Intent Boost
        # -------------------------------------------------

        if intent == "financial":

            for word in [

                "revenue",

                "profit",

                "pat",

                "income",

                "earnings",

                "cash",

                "tax",

                "crore",

            ]:

                if word in sentence_lower:

                    score += 5

        elif intent == "production":

            for word in [

                "production",

                "crude",

                "gas",

                "oil",

                "mmt",

                "mmscmd",

            ]:

                if word in sentence_lower:

                    score += 5

        elif intent == "leadership":

            for word in [

                "chairman",

                "director",

                "ceo",

                "board",

            ]:

                if word in sentence_lower:

                    score += 5

        elif intent == "project":

            for word in [

                "project",

                "field",

                "asset",

                "pipeline",

                "plant",

            ]:

                if word in sentence_lower:

                    score += 5

        elif intent == "risk":

            for word in [

                "risk",

                "challenge",

                "issue",

                "uncertainty",

            ]:

                if word in sentence_lower:

                    score += 5

        elif intent == "esg":

            for word in [

                "esg",

                "environment",

                "green",

                "carbon",

                "sustainability",

            ]:

                if word in sentence_lower:

                    score += 5

        elif intent == "comparison":

            for word in [

                "compare",

                "comparison",

                "difference",

                "versus",

                "higher",

                "lower",

                "increase",

                "decrease",

            ]:

                if word in sentence_lower:

                    score += 5

        elif intent == "trend":

            for word in [

                "trend",

                "increase",

                "decrease",

                "growth",

                "decline",

                "higher",

                "lower",

            ]:

                if word in sentence_lower:

                    score += 5

        # -------------------------------------------------
        # Trend / Comparison Questions
        # -------------------------------------------------

        if any(

            word in question_lower

            for word in [

                "trend",

                "increase",

                "decrease",

                "growth",

                "compare",

                "comparison",

                "difference",

                "why",

                "how",

            ]

        ):

            if self.YEAR_PATTERN.search(sentence):

                score += 4

            if "from" in sentence_lower and "to" in sentence_lower:

                score += 3

            if "increase" in sentence_lower:

                score += 4

            if "decrease" in sentence_lower:

                score += 4

            if "higher" in sentence_lower:

                score += 3

            if "lower" in sentence_lower:

                score += 3

        return score

    # =====================================================
    # Keep Only Relevant Sentences
    # =====================================================

    def _filter_sentences(

        self,

        sentences: list[str],

        question: str,

        intent: str,

    ) -> list[str]:

        ranked = []

        for sentence in sentences:

            score = self._sentence_score(

                sentence,

                question,

                intent,

            )

            ranked.append((score, sentence))

        ranked.sort(

            key=lambda x: x[0],

            reverse=True,

        )

        strong = [

            sentence

            for score, sentence in ranked

            if score >= 6

        ]

        if strong:

            return strong

        # Fallback: nothing crossed the "strong" bar, but these
        # sentences came from documents that already survived
        # hybrid retrieval + reranking — they are still the most
        # relevant material we have. Keep the best few rather than
        # discarding everything and forcing a hard "no information
        # found" answer.
        fallback = [
            sentence
            for score, sentence in ranked[:3]
            if score > 0
        ]

        return fallback
        
      # =====================================================
    # Extract Structured Facts
    # =====================================================

    def extract(

        self,

        question: str,

        documents: List[Document],

    ) -> dict:

        logger.info("=" * 70)
        logger.info("FACT EXTRACTION")
        logger.info("=" * 70)

        intent = self._detect_intent(question)

        logger.info(
            "Detected Intent : %s",
            intent,
        )

        facts = {

            "summary": [],

            "financial": [],

            "production": [],

            "people": [],

            "risk": [],

            "esg": [],

            "projects": [],

            "pages": set(),

        }

        seen = set()

        # --------------------------------------------------
        # Process every retrieved document
        # --------------------------------------------------

        for doc in documents:

            page = doc.metadata.get("page")

            if page is not None:

                facts["pages"].add(page)

            sentences = self._split_sentences(

                doc.page_content

            )

            relevant = self._filter_sentences(

                sentences,

                question,

                intent,

            )

            for sentence in relevant:

                key = sentence.lower()

                if key in seen:

                    continue

                seen.add(key)

                facts["summary"].append(sentence)

                # ------------------------------------------
                # Financial
                # ------------------------------------------

                if any(

                    word in key

                    for word in [

                        "revenue",

                        "profit",

                        "pat",

                        "income",

                        "earnings",

                        "crore",

                        "dividend",

                        "tax",

                    ]

                ):

                    facts["financial"].append(sentence)

                # ------------------------------------------
                # Production
                # ------------------------------------------

                if any(

                    word in key

                    for word in [

                        "production",

                        "crude",

                        "gas",

                        "oil",

                        "mmt",

                        "mmscmd",

                    ]

                ):

                    facts["production"].append(sentence)

                # ------------------------------------------
                # Leadership
                # ------------------------------------------

                if any(

                    word in key

                    for word in [

                        "chairman",

                        "director",

                        "ceo",

                        "board",

                    ]

                ):

                    facts["people"].append(sentence)

                # ------------------------------------------
                # ESG
                # ------------------------------------------

                if any(

                    word in key

                    for word in [

                        "environment",

                        "green",

                        "carbon",

                        "esg",

                        "sustainability",

                    ]

                ):

                    facts["esg"].append(sentence)

                # ------------------------------------------
                # Risks
                # ------------------------------------------

                if any(

                    word in key

                    for word in [

                        "risk",

                        "challenge",

                        "issue",

                        "threat",

                        "uncertainty",

                    ]

                ):

                    facts["risk"].append(sentence)

                # ------------------------------------------
                # Projects
                # ------------------------------------------

                if any(

                    word in key

                    for word in [

                        "project",

                        "asset",

                        "field",

                        "plant",

                        "pipeline",

                        "offshore",

                    ]

                ):

                    facts["projects"].append(sentence)

        # --------------------------------------------------
        # Remove duplicates
        # --------------------------------------------------

        for category in [

            "summary",

            "financial",

            "production",

            "people",

            "risk",

            "esg",

            "projects",

        ]:

            facts[category] = list(

                dict.fromkeys(

                    facts[category]

                )

            )

        # --------------------------------------------------
        # Keep only the best facts
        # --------------------------------------------------

        MAX_FACTS = {

            "summary": 10,

            "financial": 8,

            "production": 6,

            "people": 5,

            "risk": 5,

            "esg": 5,

            "projects": 5,

        }

        for category, limit in MAX_FACTS.items():

            facts[category] = facts[category][:limit]

        facts["pages"] = sorted(

            list(facts["pages"])

        )
        
          # --------------------------------------------------
        # Prioritize Explanation Sentences
        # --------------------------------------------------

        explanation_words = [

            "because",

            "due to",

            "driven by",

            "result of",

            "led to",

            "caused by",

            "increase",

            "decrease",

            "growth",

            "decline",

            "higher",

            "lower",

            "improved",

            "reduced",

        ]

        if intent in [

            "financial",

            "comparison",

            "trend",

        ]:

            explanation = []

            normal = []

            for sentence in facts["summary"]:

                lower = sentence.lower()

                if any(

                    word in lower

                    for word in explanation_words

                ):

                    explanation.append(sentence)

                else:

                    normal.append(sentence)

            facts["summary"] = explanation + normal

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        logger.info(
            "Summary Facts : %d",
            len(facts["summary"]),
        )

        logger.info(
            "Financial Facts : %d",
            len(facts["financial"]),
        )

        logger.info(
            "Production Facts : %d",
            len(facts["production"]),
        )

        logger.info(
            "People Facts : %d",
            len(facts["people"]),
        )

        logger.info(
            "Risk Facts : %d",
            len(facts["risk"]),
        )

        logger.info(
            "ESG Facts : %d",
            len(facts["esg"]),
        )

        logger.info(
            "Project Facts : %d",
            len(facts["projects"]),
        )

        logger.info(
            "Pages : %d",
            len(facts["pages"]),
        )

        logger.info("=" * 70)

        return facts


# ==========================================================
# Factory
# ==========================================================

def get_fact_extractor():

    global _fact_extractor

    if _fact_extractor is None:

        _fact_extractor = FactExtractor()

    return _fact_extractor