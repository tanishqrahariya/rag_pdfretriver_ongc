"""
app.py

ONGC Enterprise RAG Assistant
Terminal Interface
"""

from __future__ import annotations

from backend.utils.logger import get_logger

from core.rag_engine import (
    build_rag,
    ask_question,
)

logger = get_logger(__name__)


# ==========================================================
# Banner
# ==========================================================

def print_banner():

    print("\n" + "=" * 70)
    print("              ONGC Enterprise RAG Assistant")
    print("=" * 70)
    print("Commands:")
    print("  exit  - Quit")
    print("  clear - Clear conversation history")
    print("=" * 70)


# ==========================================================
# Chat
# ==========================================================

def chat():

    logger.info("=" * 70)
    logger.info("INITIALIZING RAG ENGINE")
    logger.info("=" * 70)

    # Build the complete RAG pipeline
    PDFS = [
    r"pdfs\AI_and_GenAI_Notes.pdf",
    ]
    rag = build_rag(
      pdf_paths=PDFS,
    )

    history = ""

    print_banner()

    while True:

        try:
            question = input("\nYou : ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not question:
            continue

        cmd = question.lower()

        if cmd in ("exit", "quit"):
            print("\nGoodbye!")
            break

        if cmd == "clear":
            history = ""
            print("Conversation history cleared.")
            continue

        try:

            answer = ask_question(

                rag=rag,

                question=question,

                history=history,

            )

            print("\nAssistant:\n")
            print(answer)

            history += (
                f"\nUser: {question}"
                f"\nAssistant: {answer}"
            )

        except Exception:
            logger.exception("Failed to answer question.")
            print("\nAn error occurred while generating the answer.")


# ==========================================================
# Main
# ==========================================================

def main():

    logger.info("=" * 70)
    logger.info("STARTING CLI")
    logger.info("=" * 70)

    chat()


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":

    main()