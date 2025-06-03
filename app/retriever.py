"""
retriever.py

Module for retrieving relevant documents from a local Chroma vector store using local embeddings.
Supports sentence-transformers or Ollama embeddings, and performs top-k vector similarity search.

This module assumes documents are already ingested into Chroma.
"""

import logging
from typing import List

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from app.embedder import get_embedding

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize persistent Chroma vector database
chroma_client = PersistentClient(path="./chroma_storage")
collection = chroma_client.get_or_create_collection(
    name="rag_documents",
    embedding_function=DefaultEmbeddingFunction()
)


def retrieve_chunks(query: str, k: int = 5) -> List[dict]:
    """
    Retrieve top-k most relevant documents for a given query string using vector similarity.

    Embedding is generated locally using get_embedding().

    Args:
        query (str): Query text.
        k (int): Number of top documents to retrieve.

    Returns:
        List[dict]: List of documents with metadata and text.
    """
    try:
        query_vec = get_embedding([query])[0]
        results = collection.query(query_embeddings=[query_vec], n_results=k)

        documents = []
        for i in range(len(results["documents"][0])):
            documents.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        logger.info(f"Retrieved {len(documents)} documents for query: '{query}'")
        return documents

    except Exception as e:
        logger.exception("Failed to retrieve documents: %s", str(e))
        return []
