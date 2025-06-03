"""
embedder.py

This module handles local embedding generation for input texts.
It supports Ollama embedding models if available, and falls back to
SentenceTransformer for environments without Ollama.

Embeddings are never generated via remote APIs like OpenAI, as per design constraints.
Now includes batch support for faster embedding performance.
"""

import logging
from typing import List

from app.rag.ollama import get_ollama_embedding, is_ollama_available
from app.config import settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Fallback local embedding model
_sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")


def _batch_ollama_embeddings(
    texts: List[str], batch_size: int = 32
) -> List[List[float]]:
    """
    Send embedding requests to Ollama in batches.

    Args:
        texts (List[str]): List of texts to embed.
        batch_size (int): Number of texts per batch.

    Returns:
        List[List[float]]: Embedding vectors.
    """
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vectors.extend(get_ollama_embedding(batch))
    return vectors


def get_embedding(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for the given list of texts using a local embedding model.

    Priority:
    1. Use Ollama if available and configured with a compatible embedding model.
    2. Otherwise, fall back to local SentenceTransformer embedding.

    Args:
        texts (List[str]): List of text strings to embed.

    Returns:
        List[List[float]]: Embedding vectors.
    """
    try:
        if is_ollama_available() and settings.embedding_model.startswith("ollama/"):
            logger.info(f"Using Ollama to embed {len(texts)} texts in batches")
            return _batch_ollama_embeddings(texts)
        else:
            logger.info("Using SentenceTransformer fallback embedding model")
            return _sentence_transformer.encode(texts, convert_to_numpy=True).tolist()
    except Exception as e:
        logger.exception("Failed to generate embeddings")
        raise RuntimeError("Embedding generation failed") from e
