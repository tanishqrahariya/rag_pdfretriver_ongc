"""
chain/rag_chain.py

Enterprise RAG Pipeline
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from backend.retrieval.query_rewriter import get_query_rewriter
from backend.retrieval.query_planner import get_query_planner
from backend.retrieval.context_compressor import get_context_compressor
from backend.retrieval.fact_extractor import get_fact_extractor
from backend.retrieval.entity_memory import get_entity_memory
from backend.retrieval.hybrid import HybridRetriever

from backend.chain.reasoning_chain import get_reasoning_chain

from backend.formatter.answer_formatter import get_answer_formatter

from backend.llm.model import get_llm
from backend.llm.prompt_builder import get_prompt_builder

from backend.utils.logger import get_logger

logger = get_logger(__name__)

_rag = None


# ==========================================================
# Enterprise RAG
# ==========================================================

class RAGChain:

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
    ):

        self.hybrid = hybrid_retriever

        self.rewriter = get_query_rewriter()

        self.planner = get_query_planner()

        self.memory = get_entity_memory()

        self.compressor = get_context_compressor()

        self.fact_extractor = get_fact_extractor()

        self.reasoning = get_reasoning_chain()

        self.prompt_builder = get_prompt_builder()

        self.llm = get_llm()

        self.formatter = get_answer_formatter()

    # =====================================================
    # Entity Resolution
    # =====================================================

    def _resolve_entities(
        self,
        question: str,
    ) -> str:

        logger.info("Resolving entities...")

        return self.memory.resolve(question)

    # =====================================================
    # Query Rewrite
    # =====================================================

    def _rewrite_query(
        self,
        question: str,
        history: str,
    ) -> str:

        logger.info("Rewriting query...")

        rewritten = self.rewriter.rewrite(

            question,

            history,

        )

        logger.info(
            "Final Query : %s",
            rewritten,
        )

        return rewritten

    # =====================================================
    # Query Planning
    # =====================================================

    def _plan_query(
        self,
        question: str,
    ):

        logger.info("Planning query...")

        plan = self.planner.plan(question)

        logger.info(
            "Intent : %s",
            plan.intent,
        )

        logger.info(
            "Answer Style : %s",
            plan.answer_style,
        )

        logger.info(
            "Sub Queries : %d",
            len(plan.sub_queries),
        )

        return plan
    
        # =====================================================
    # Hybrid Retrieval
    # =====================================================

    def _retrieve_documents(
        self,
        plan,
    ):

        logger.info("=" * 70)
        logger.info("HYBRID RETRIEVAL")
        logger.info("=" * 70)

        retrieved = []

        for query in plan.sub_queries:

            logger.info(
                "Searching : %s",
                query,
            )

            docs = self.hybrid.retrieve(query)

            retrieved.extend(docs)

        logger.info(
            "Retrieved Chunks : %d",
            len(retrieved),
        )

        return retrieved

    # =====================================================
    # Remove Duplicate Chunks
    # =====================================================

    def _remove_duplicates(
        self,
        documents,
    ):

        logger.info("Removing duplicate chunks...")

        unique = []

        seen = set()

        for document in documents:

            chunk_hash = document.metadata.get(
                "chunk_hash"
            )

            if chunk_hash in seen:

                continue

            seen.add(chunk_hash)

            unique.append(document)

        logger.info(
            "Unique Chunks : %d",
            len(unique),
        )

        return unique

    # =====================================================
    # Context Compression
    # =====================================================

    def _compress_context(
        self,
        question: str,
        documents,
    ):

        logger.info("=" * 70)
        logger.info("CONTEXT COMPRESSION")
        logger.info("=" * 70)

        compressed = self.compressor.compress(

            question,

            documents,

        )

        logger.info(
            "Compressed Chunks : %d",
            len(compressed),
        )

        return compressed
    
        # =====================================================
    # Structured Fact Extraction
    # =====================================================

    def _extract_facts(
        self,
        question: str,
        documents,
    ):

        logger.info("=" * 70)
        logger.info("FACT EXTRACTION")
        logger.info("=" * 70)

        facts = self.fact_extractor.extract(

            question=question,

            documents=documents,

        )

        logger.info(
            "Summary Facts : %d",
            len(facts.get("summary", [])),
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
            "People Facts : %d",
            len(facts.get("people", [])),
        )

        logger.info(
            "Risk Facts : %d",
            len(facts.get("risk", [])),
        )

        logger.info(
            "Project Facts : %d",
            len(facts.get("projects", [])),
        )

        logger.info(
            "ESG Facts : %d",
            len(facts.get("esg", [])),
        )

        return facts

    # =====================================================
    # Reasoning
    # =====================================================

    def _reason(
        self,
        question: str,
        facts,
    ):

        logger.info("=" * 70)
        logger.info("REASONING")
        logger.info("=" * 70)

        reasoning = self.reasoning.run(

            question=question,

            facts=facts,

        )

        logger.info("Reasoning completed.")

        return reasoning

    # =====================================================
    # Generate Final Answer
    # =====================================================

    def _generate_answer(
        self,
        question: str,
        reasoning,
        answer_style: str,
        history: str,
    ) -> str:

        logger.info("=" * 70)
        logger.info("ANSWER GENERATION")
        logger.info("=" * 70)

        prompt, payload = self.prompt_builder.build(

            question=question,

            facts=reasoning,

            answer_style=answer_style,

            history=history,

        )

        chain = (

            prompt

            | self.llm

            | StrOutputParser()

        )

        answer = chain.invoke(payload).strip()

        logger.info("Answer generated.")

        return answer

    # =====================================================
    # Update Entity Memory
    # =====================================================

    def _update_memory(
        self,
        answer: str,
    ):

        logger.info("Updating entity memory...")

        self.memory.update_from_text(answer)

    # =====================================================
    # Final Formatting
    # =====================================================

    def _format_answer(
        self,
        answer: str,
        facts,
        documents,
    ) -> str:

        logger.info("=" * 70)
        logger.info("FORMATTING ANSWER")
        logger.info("=" * 70)

        final = self.formatter.format(

            answer=answer,

            facts=facts,

            documents=documents,

        )

        return final
    
        # =====================================================
    # Main Pipeline
    # =====================================================

    def invoke(
        self,
        question: str,
        chat_history: str = "",
    ) -> str:

        logger.info("=" * 70)
        logger.info("STARTING RAG PIPELINE")
        logger.info("=" * 70)

        # --------------------------------------------------
        # Entity Resolution
        # --------------------------------------------------

        resolved_question = self._resolve_entities(
            question,
        )

        # --------------------------------------------------
        # Query Rewrite
        # --------------------------------------------------

        rewritten_question = self._rewrite_query(

            resolved_question,

            chat_history,

        )

        # --------------------------------------------------
        # Query Planning
        # --------------------------------------------------

        plan = self._plan_query(

            rewritten_question,

        )

        # --------------------------------------------------
        # Retrieval
        # --------------------------------------------------

        retrieved_documents = self._retrieve_documents(
            plan,
        )

        # --------------------------------------------------
        # Remove Duplicates
        # --------------------------------------------------

        unique_documents = self._remove_duplicates(

            retrieved_documents,

        )

        # --------------------------------------------------
        # Context Compression
        # --------------------------------------------------

        compressed_documents = self._compress_context(

            rewritten_question,

            unique_documents,

        )

        # --------------------------------------------------
        # Fact Extraction
        # --------------------------------------------------

        facts = self._extract_facts(

            rewritten_question,

            compressed_documents,

        )

        # --------------------------------------------------
        # Reasoning
        # --------------------------------------------------

        reasoning = self._reason(

            rewritten_question,

            facts,

        )

        # --------------------------------------------------
        # Final Answer
        # --------------------------------------------------

        answer = self._generate_answer(

            question=rewritten_question,

            reasoning=reasoning,

            answer_style=plan.answer_style,

            history=chat_history,

        )

        # --------------------------------------------------
        # Update Memory
        # --------------------------------------------------

        self._update_memory(

            answer,

        )

        # --------------------------------------------------
        # Format Answer
        # --------------------------------------------------

        final_answer = self._format_answer(

            answer=answer,

            facts=facts,

            documents=compressed_documents,

        )

        logger.info("=" * 70)
        logger.info("RAG PIPELINE COMPLETED")
        logger.info("=" * 70)

        return final_answer


# ==========================================================
# Factory
# ==========================================================

def get_rag_chain(
    hybrid_retriever: HybridRetriever,
) -> RAGChain:

    global _rag

    if _rag is None:

        logger.info("Creating Enterprise RAG Chain...")

        _rag = RAGChain(

            hybrid_retriever,

        )

    return _rag


# ==========================================================
# Reset
# ==========================================================

def reset_rag_chain():

    global _rag

    _rag = None