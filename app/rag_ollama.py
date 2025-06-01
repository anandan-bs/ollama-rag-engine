"""
A class that wraps a Chroma database and uses it to answer questions using a
RAG (Retrieval-Augmented Generation) model.

The class provides a simple interface for building an index of documents
and querying the index with a question. The query method returns the answer
to the question as a string.

The class uses the Chroma library to build and query the index, and the
llama3.2 model to generate the answer.
"""

import time
import logging
import requests
from typing import Any
from functools import lru_cache
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.utils import extract_chunks_from_dir
from app.config import OLLAMA_CONFIG, EMBEDDING_CONFIG, DB_CONFIG

logger = logging.getLogger(__name__)


def rate_limit(max_per_minute: int):
    """Decorator to limit the rate of function calls.

    Args:
        max_per_minute (int): Maximum number of calls allowed per minute
    """
    def decorator(func: Any) -> Any:
        last_reset = time.time()
        calls = 0

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal last_reset, calls
            now = time.time()
            if now - last_reset >= 60:
                calls = 0
                last_reset = now
            if calls >= max_per_minute:
                msg = f"Rate limit exceeded: {max_per_minute} calls per minute"
                logger.warning(msg)
                raise Exception(msg)
            calls += 1
            return func(*args, **kwargs)

        return wrapper

    return decorator


class RAGChatbot:
    def __init__(self, data_dir: str):
        """
        Initialize the RAGChatbot.

        Args:
            data_dir (str): The directory containing the documents to index.
        """
        self.data_dir = data_dir

        # Initialize embedding function
        self.embedder = SentenceTransformerEmbeddingFunction(
            EMBEDDING_CONFIG["model_name"]
        )

        # Initialize database
        self.chroma = PersistentClient(path=DB_CONFIG["path"])
        self.collection = self.chroma.get_or_create_collection(
            name=DB_CONFIG["collection_name"], embedding_function=self.embedder
        )

    def build_index(self) -> None:
        """Build the search index from documents."""
        logger.info(f"Building index from directory: {self.data_dir}")
        try:
            """
            Build the index of documents.
            """
            texts = extract_chunks_from_dir(self.data_dir)
            logger.info(f"Found {len(texts)} text chunks to index")
            for i, chunk in enumerate(texts):
                self.collection.add(documents=[chunk], ids=[f"doc_{i}"])
            logger.info("Successfully built search index")
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            raise

    @rate_limit(max_per_minute=30)
    def query(self, question: str) -> str:
        """
        Query the index with a question.

        Args:
            question (str): The question to ask.

        Returns:
            str: The answer to the question.
        """
        results = self.collection.query(query_texts=[question], n_results=3)
        documents = results.get("documents", [[]])[0]
        context = "\n---\n".join(documents)

        # Construct prompt based on available context
        if context.strip() and any(documents):
            self.prompt = (
                "You are an expert assistant. Use the context below if relevant "
                "to answer the user's question.\n"
                "\n"
                "[Context]\n"
                f"{context}\n"
                "\n"
                "[Question]\n"
                f"{question}\n"
                "\n"
                "[Answer]"
            )
        else:
            self.prompt = (
                "You are a helpful assistant. Answer the question below "
                "concisely and clearly.\n"
                "\n"
                "[Question]\n"
                f"{question}\n"
                "\n"
                "[Answer]"
            )

        try:
            logger.info(f"Processing query: {question[:50]}...")
            return self._get_cached_response(question)
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"[Error] Failed to process query: {str(e)}"

    @lru_cache(maxsize=100)
    def _get_cached_response(self, question: str) -> str:
        """Get cached response for a question."""
        return self._query_llm(question)

    def _query_llm(self, question: str) -> str:
        """Make actual API call to LLM.

        Args:
            question (str): The question to ask the LLM

        Returns:
            str: The LLM's response

        Raises:
            requests.exceptions.RequestException: If the API call fails
        """
        try:
            logger.debug(f"Making API call to LLM for question: {question[:50]}...")
            response = requests.post(
                OLLAMA_CONFIG["api_url"],
                json={
                    "model": OLLAMA_CONFIG["model_name"],
                    "prompt": self.prompt,
                    "stream": False,
                    "temperature": OLLAMA_CONFIG["temperature"],
                },
                timeout=30,
            )
            response.raise_for_status()
            answer = response.json().get("response", "")
            logger.debug(f"Received response from LLM: {answer[:50]}...")
            return answer
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API error: {str(e)}")
            raise
