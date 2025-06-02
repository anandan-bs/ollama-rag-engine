"""
reranker.py

This module re-ranks retrieved documents based on semantic similarity
between the query and each document using a sentence transformer.
Useful for improving relevance in a RAG system after initial vector retrieval.
"""

import logging
from typing import List, Dict

from sentence_transformers import CrossEncoder
from app.config import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load cross-encoder model
_cross_encoder = CrossEncoder(settings.reranker_model)


def rerank_results(query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Rerank retrieved candidate documents using a cross-encoder model.

    Args:
        query (str): The user query.
        candidates (List[Dict]): List of retrieved document dictionaries with "text" keys.
        top_k (int): Number of top-ranked results to return.

    Returns:
        List[Dict]: Re-ranked list of documents based on semantic relevance.
    """
    try:
        logger.info("Starting reranking of retrieved documents")

        pairs = [(query, doc["text"]) for doc in candidates]
        scores = _cross_encoder.predict(pairs)

        # Attach scores to each candidate
        for doc, score in zip(candidates, scores):
            doc["score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
        logger.info("Completed reranking")
        return reranked[:top_k]

    except Exception as e:
        logger.exception("Failed to rerank results")
        return candidates[:top_k]
