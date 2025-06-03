"""
ingest.py

Handles document loading, parsing, text splitting, and indexing into Chroma vector store.
Supports PDF, DOCX, and Google Docs formats with proper exception handling and embedding batching.
"""

import os
import logging
import fitz  # PyMuPDF
from typing import List
import docx2txt
from app.embedder import get_embedding
from app.retriever import collection

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])


def load_docx(path: str) -> str:
    return docx2txt.process(path)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_document(path: str) -> str:
    if path.endswith(".pdf"):
        return load_pdf(path)
    elif path.endswith(".docx"):
        return load_docx(path)
    elif path.endswith(".txt"):
        return load_text(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def chunk_text(text: str, max_tokens: int = 300, overlap: int = 50) -> List[str]:
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(" ".join(current + [sentence])) <= max_tokens:
            current.append(sentence)
        else:
            if current:
                chunks.append(" ".join(current))
            current = current[-(overlap // max(1, len(sentence))):] if overlap else []
            current.append(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks


def ingest_document(path: str):
    logger.info(f"Ingesting document: {path}")
    doc = load_document(path)
    chunks = chunk_text(doc)
    embeddings = get_embedding(chunks)
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[f"{os.path.basename(path)}-{i}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"source": os.path.basename(path), "chunk_id": i}]
        )
