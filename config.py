"""
config.py

Central configuration file for the ONGC Enterprise RAG Assistant.

Change values here instead of modifying source code.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).parent

DATA_DIR = PROJECT_ROOT / "pdfs"

CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

LOG_DIR = PROJECT_ROOT / "logs"

PROMPT_DIR = PROJECT_ROOT / "prompts"

# ==========================================================
# PDF Processing
# ==========================================================

# First page = 0
START_PAGE = 0

# None = Process Complete PDF
MAX_PAGES = None

# ==========================================================
# Chunking
# ==========================================================

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

# ==========================================================
# Embedding Model
# ==========================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

NORMALIZE_EMBEDDINGS = True

# ==========================================================
# CrossEncoder Reranker
# ==========================================================

RERANKER_MODEL = "BAAI/bge-reranker-base"

RERANK_TOP_K = 20

RERANK_SCORE_THRESHOLD = 0.25

# ==========================================================
# Language Model
# ==========================================================

LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

TEMPERATURE = 0.2

MAX_NEW_TOKENS = 1024

# ==========================================================
# Chroma
# ==========================================================

COLLECTION_NAME = "ongc_annual_reports"

PERSIST_DIRECTORY = str(CHROMA_DB_DIR)

# ==========================================================
# Hybrid Retrieval
# ==========================================================

VECTOR_TOP_K = 20

BM25_TOP_K = 20

HYBRID_TOP_K = 10

FETCH_K = 60

RRF_K = 60

MAX_CHUNKS_PER_PAGE = 3

# ==========================================================
# Query Planning
# ==========================================================

MAX_SUB_QUERIES = 5

# ==========================================================
# OCR
# ==========================================================

OCR_CONFIDENCE = 0.60

MIN_IMAGE_WIDTH = 100

MIN_IMAGE_HEIGHT = 100

# A page whose real text layer is shorter than this is treated as
# "sparse" — likely a templated/certificate-style page where labels
# live in a background image and only a few dynamic values are real
# text (e.g. scorecards, certificates). Sparse pages get a full-page
# high-DPI OCR render as a fallback so label context isn't lost.
MIN_TEXT_CHARS_PER_PAGE = 300

# DPI used when rendering a sparse page to an image for OCR. Embedded
# image XObjects are sometimes authored at very low resolution (~70
# DPI), which is too low for reliable OCR of small label text — so for
# sparse pages we render the page itself at this DPI instead of OCR-ing
# the raw embedded image.
OCR_RENDER_DPI = 300

# ==========================================================
# Logging
# ==========================================================

LOG_LEVEL = "INFO"

LOG_FILE = LOG_DIR / "rag.log"

# ==========================================================
# Supported File Types
# ==========================================================

SUPPORTED_FILES = [
    ".pdf",
]

# ==========================================================
# Streamlit
# ==========================================================

APP_TITLE = "ONGC Enterprise AI"

APP_ICON = "🤖"

# ==========================================================
# Create Required Directories
# ==========================================================

LOG_DIR.mkdir(exist_ok=True)

CHROMA_DB_DIR.mkdir(exist_ok=True)

DATA_DIR.mkdir(exist_ok=True)