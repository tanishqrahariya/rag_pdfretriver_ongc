"""
llm/model.py

Enterprise LLM Manager

Responsibilities
----------------
1. Load LLM once
2. Support HuggingFace Inference API
3. Singleton
4. Central configuration
"""

from __future__ import annotations

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from dotenv import load_dotenv
load_dotenv()

from config import (
    HF_TOKEN,
    LLM_MODEL,
    TEMPERATURE,
    MAX_NEW_TOKENS,
)

from backend.utils.logger import get_logger

logger = get_logger(__name__)

_llm = None


class LLMManager:

    def __init__(self):

        logger.info("=" * 70)
        logger.info("LOADING LLM")
        logger.info("=" * 70)

        self.chat_model = ChatHuggingFace(

          llm=HuggingFaceEndpoint(

          repo_id=LLM_MODEL,

          huggingfacehub_api_token=HF_TOKEN,

          task="text-generation",

          max_new_tokens=MAX_NEW_TOKENS,

          temperature=TEMPERATURE,

    )

)

        logger.info(

            "LLM Loaded : %s",

            LLM_MODEL,

        )

    # ------------------------------------------------------

    def get_llm(self):

        return self.chat_model

    # ------------------------------------------------------

    def invoke(self, prompt):

        return self.chat_model.invoke(prompt)


# ==========================================================
# Factory
# ==========================================================

def get_llm_manager():

    global _llm

    if _llm is None:

        _llm = LLMManager()

    return _llm


def get_llm():

    return get_llm_manager().get_llm()