"""
retrieval/entity_memory.py

Entity Memory

Stores important entities discovered during the conversation.

Examples

CEO
↓

Arun Kumar Singh

PAT
↓

Profit After Tax

ONGC

↓

Oil and Natural Gas Corporation

This helps rewrite follow-up questions like

"Tell me more about him"

into

"Tell me more about Arun Kumar Singh"
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict


from backend.utils.logger import get_logger

logger = get_logger(__name__)

_memory = None


class EntityMemory:

    """
    Simple conversation entity memory.
    """

    def __init__(self):

        self.entities: Dict[str, str] = OrderedDict()

        self.max_entities = 50

    # ======================================================
    # Add Entity
    # ======================================================

    def add(
        self,
        entity: str,
        value: str,
    ):

        entity = entity.strip().lower()

        if not entity:
            return

        self.entities[entity] = value

        while len(self.entities) > self.max_entities:

            self.entities.popitem(last=False)

    # ======================================================
    # Update From Answer
    # ======================================================

    def update_from_text(
        self,
        text: str,
    ):

        """
        Very lightweight entity extraction.

        Can later be replaced with SpaCy or an LLM.
        """

        patterns = [

            r"Chairman and CEO\s*[:\-]?\s*([A-Z][A-Za-z .]+)",

            r"CEO\s*[:\-]?\s*([A-Z][A-Za-z .]+)",

            r"Managing Director\s*[:\-]?\s*([A-Z][A-Za-z .]+)",

        ]

        for pattern in patterns:

            match = re.search(pattern, text)

            if match:

                self.add(

                    "ceo",

                    match.group(1).strip(),

                )

    # ======================================================
    # Resolve Pronouns
    # ======================================================

    def resolve(
        self,
        query: str,
    ) -> str:

        q = query

        if " him " in f" {q.lower()} ":

            if "ceo" in self.entities:

                q = re.sub(

                    r"\bhim\b",

                    self.entities["ceo"],

                    q,

                    flags=re.IGNORECASE,

                )

        return q

    # ======================================================
    # Lookup
    # ======================================================

    def get(
        self,
        key: str,
    ):

        return self.entities.get(

            key.lower()

        )

    # ======================================================
    # Clear
    # ======================================================

    def clear(self):

        self.entities.clear()

        logger.info(

            "Entity memory cleared."

        )

    # ======================================================
    # Export
    # ======================================================

    def export(self):

        return dict(self.entities)


# ==========================================================
# Factory
# ==========================================================

def get_entity_memory():

    global _memory

    if _memory is None:

        _memory = EntityMemory()

    return _memory