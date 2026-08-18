"""
router/intent_router.py

Intent Router

Purpose
-------
Detect whether a query requires RAG retrieval or can
be answered directly.

Pipeline

User Question
      │
      ▼
Intent Router
      │
 ┌────┴────────────┐
 │                 │
Greeting       Knowledge Question
 │                 │
 ▼                 ▼
Direct Reply    RAG Pipeline
"""

from __future__ import annotations

import re


class IntentRouter:

    def __init__(self):

        self.greetings = {

            "hi",

            "hello",

            "hey",

            "good morning",

            "good evening",

            "good afternoon",

            "hii",

        }

        self.thanks = {

            "thanks",

            "thank you",

            "thankyou",

            "thx",

        }

        self.goodbye = {

            "bye",

            "goodbye",

            "see you",

            "exit",

            "quit",

        }

    # ----------------------------------------------------
    # Greeting
    # ----------------------------------------------------

    def is_greeting(self, question):

        q = question.lower().strip()

        return any(g in q for g in self.greetings)

    # ----------------------------------------------------
    # Thanks
    # ----------------------------------------------------

    def is_thanks(self, question):

        q = question.lower().strip()

        return any(t in q for t in self.thanks)

    # ----------------------------------------------------
    # Goodbye
    # ----------------------------------------------------

    def is_goodbye(self, question):

        q = question.lower().strip()

        return any(g in q for g in self.goodbye)

    # ----------------------------------------------------
    # Route
    # ----------------------------------------------------

    def route(self, question):

        if self.is_greeting(question):

            return {

                "intent": "greeting",

                "answer": (
                    "Hello! I'm your ONGC Annual Report Assistant. "
                    "Ask me anything about ONGC's annual reports, "
                    "financials, projects, production, or management."
                ),

            }

        if self.is_thanks(question):

            return {

                "intent": "thanks",

                "answer": (
                    "You're welcome! Let me know if you have any "
                    "questions about ONGC or its annual reports."
                ),

            }

        if self.is_goodbye(question):

            return {

                "intent": "goodbye",

                "answer": (
                    "Goodbye! Have a great day."
                ),

            }

        return {

            "intent": "rag",

            "answer": None,

        }


_router = IntentRouter()


def get_router():

    return _router