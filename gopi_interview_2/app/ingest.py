"""Ingest: document -> section-aware chunks -> embeddings -> FAISS index.

Usage:
    python -m app.ingest                     # uses data/hotel_rag_document_v2.pdf (or the sample if missing)
    python -m app.ingest --source my.pdf     # any .pdf / .txt / .md
"""
import argparse
import json
import logging
import re
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app import config
from app.guardrails import contains_injection

log = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\.?\s+\S")


# --------------------------------------------------------------------------- loading
def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- chunking
def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~1.3 tokens per whitespace-delimited word for English prose)."""
    return max(1, int(len(text.split()) * 1.3))


def is_heading(line: str) -> bool:
    """Heuristic: short line, no sentence punctuation, numbered / markdown / Title Case / ALL CAPS."""
    words = line.split()
    if not words or len(words) > 10 or line[-1] in ".,;":
        return False
    if line.startswith("#") or _NUMBERED_HEADING_RE.match(line) or line.isupper():
        return True
    capitalised = sum(1 for w in words if w[0].isupper())
    return capitalised / len(words) >= 0.6 and len(line) <= 70


def split_sections(text: str) -> list[tuple[str, str]]:
    """Group lines under the most recent heading. Returns [(section_title, section_body)]."""
    sections: list[tuple[str, str]] = []
    title, paragraphs, current = "Introduction", [], []

    def flush_paragraph():
        if current:
            paragraphs.append(" ".join(current))
            current.clear()

    def flush_section():
        flush_paragraph()
        if paragraphs:
            sections.append((title, "\n\n".join(paragraphs)))
            paragraphs.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush_paragraph()
        elif is_heading(line):
            flush_section()
            title = line.lstrip("#").strip()
        else:
            current.append(line)
    flush_section()
    return sections


def chunk_section(title: str, body: str, target: int, maximum: int) -> list[str]:
    """Pack sentences into chunks of roughly `target` tokens, never exceeding `maximum`."""
    sentences = [s for para in body.split("\n\n") for s in _SENTENCE_RE.split(para) if s.strip()]
    chunks, current, current_tokens = [], [], 0
    for sentence in sentences:
        tokens = estimate_tokens(sentence)
        if current and current_tokens + tokens > maximum:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += tokens
        if current_tokens >= target:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
    if current:
        chunks.append(" ".join(current))
    # The section title is kept inside the chunk text so it is embedded together with the content.
    return [f"{title}\n{chunk}" for chunk in chunks]


def chunk_document(text: str, source: str) -> list[dict]:
    chunks = []
    for title, body in split_sections(text):
        for chunk_text in chunk_section(title, body, config.CHUNK_TARGET_TOKENS, config.CHUNK_MAX_TOKENS):
            if contains_injection(chunk_text):
                log.warning("Skipping chunk in section %r: looks like a prompt-injection attempt", title)
                continue
            chunks.append({"id": len(chunks), "section": title, "text": chunk_text, "source": source})
    return chunks


# --------------------------------------------------------------------------- embedding / storage
def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    # normalize_embeddings=True -> inner product == cosine similarity in the FAISS index.
    return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype="float32")


def save_index(index: faiss.Index, chunks: list[dict], index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    (index_dir / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")


def ingest(source: Path, index_dir: Path = config.INDEX_DIR) -> list[dict]:
    log.info("Loading %s", source.name)
    text = load_text(source)
    chunks = chunk_document(text, source.name)
    if not chunks:
        raise ValueError(f"No text could be extracted from {source}")
    log.info("Built %d chunks; embedding with %s", len(chunks), config.EMBEDDING_MODEL)
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    vectors = embed_texts([c["text"] for c in chunks], model)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    save_index(index, chunks, index_dir)
    log.info("Saved index with %d vectors to %s", index.ntotal, index_dir)
    return chunks


def resolve_source(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if config.PDF_PATH.exists():
        return config.PDF_PATH
    log.warning("%s not found - falling back to the SAMPLE document %s. "
                "Drop the real PDF into data/ and re-run ingest.", config.PDF_PATH.name, config.SAMPLE_DOC_PATH.name)
    return config.SAMPLE_DOC_PATH


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build the vector index for the hotel document.")
    parser.add_argument("--source", help="Path to a .pdf/.txt/.md file (default: data/hotel_rag_document_v2.pdf)")
    parser.add_argument("--show", action="store_true", help="Print the chunks after building the index")
    args = parser.parse_args()
    source = resolve_source(args.source)
    if not source.exists():
        raise SystemExit(f"Source document not found: {source}")
    chunks = ingest(source)
    if args.show:
        for c in chunks:
            print(f"\n--- chunk {c['id']} [{c['section']}] (~{estimate_tokens(c['text'])} tokens) ---\n{c['text']}")


if __name__ == "__main__":
    main()
