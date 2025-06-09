"""
Chroma vector store client.

This module provides a simple interface to a Chroma vector store for storing and querying document embeddings.

The `ChromaStore` class provides methods for adding documents to the store, and querying the store for documents
relevant to a given query string.

The `add_documents` method takes a list of document IDs and a list of corresponding document text,
and adds them to the store.

The `query` method takes a query string and an optional `top_k` parameter,
and returns the top-K documents most relevant
to the query.

The documents are stored in a Chroma collection, which is created automatically if it does not exist.
"""

import logging
from chromadb import PersistentClient

from ragify_docs.config import settings
from ragify_docs.core.embed import SmartBatchedEmbedder, ChromaEmbeddingFunction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChromaStore:

    def __init__(self):
        """
        Initialize the Chroma store client.

        The store is created automatically if it does not exist.
        """
        self.client = PersistentClient(path=str(settings.chroma_storage))
        self.embedder = SmartBatchedEmbedder()
        self.embedding_fn = ChromaEmbeddingFunction(self.embedder)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=self.embedding_fn,
        )
        self.max_batch_size = 5000  # Slightly lower than 5461 to be safe

    def add_documents(self, ids, documents):
        """
        Add documents to the store.

        This method takes a list of document IDs and a list of corresponding document text, and adds them to the store.
        The documents are stored in a Chroma collection, which is created automatically if it does not exist.

        Args:
            ids (list): List of document IDs.
            documents (list): List of document text.
        """
        for i in range(0, len(documents), self.max_batch_size):
            batch_ids = ids[i:i+self.max_batch_size]
            batch_docs = documents[i:i+self.max_batch_size]

            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
            )

    def query(self, query, top_k=5):
        """
        Query the store for documents relevant to a given query string.

        This method takes a query string and an optional `top_k` parameter,
        and returns the top-K documents most relevant
        to the query.

        Args:
            query (str): Query string.
            top_k (int, optional): Number of documents to return. Defaults to 5.

        Returns:
            list: List of top-K documents.
        """
        query_embedding = self.embedding_fn.embed_query([query])[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results['documents'][0]
