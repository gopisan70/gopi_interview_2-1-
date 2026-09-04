"""Central configuration. Every value can be overridden via environment variables or a .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Knowledge source. If the real PDF is missing, ingest.py falls back to the sample document.
PDF_PATH = DATA_DIR / "hotel_rag_document_v2.pdf"
SAMPLE_DOC_PATH = DATA_DIR / "sample_hotel_document.pdf"
INDEX_DIR = DATA_DIR / "index"          # faiss.index + chunks.json live here
DB_PATH = DATA_DIR / "reservations.db"

# RAG settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_TARGET_TOKENS = 300               # aim for ~200-400 tokens per chunk
CHUNK_MAX_TOKENS = 400
TOP_K = 4
MIN_SIMILARITY = 0.1                    # sanity floor only; the LLM decides whether the context answers the question

# LLM settings. Anthropic is used when an API key is present, otherwise a local Ollama model.
LLM_PROVIDER = os.getenv("LLM_PROVIDER") or ("anthropic" if os.getenv("ANTHROPIC_API_KEY") else "ollama")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
ANTHROPIC_EFFORT = os.getenv("ANTHROPIC_EFFORT", "medium")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

HOTEL_NAME = "Grand Azure Bay Hotel"
FALLBACK_MESSAGE = (
    f"I'm sorry, I can only help with questions about the {HOTEL_NAME} "
    "and with creating, viewing or cancelling your own reservation."
)
PRIVACY_MESSAGE = (
    "I can't share reservation data in bulk or information about other guests. "
    "To look up your own booking, please give me your reservation ID and the email you booked with."
)
INJECTION_MESSAGE = (
    "I can't follow that request. I can answer questions about the hotel or help with your reservation."
)
UNAVAILABLE_MESSAGE = "The assistant is temporarily unavailable. Please try again in a moment."
