"""
ollama.py

Handles interaction with Ollama local models for embedding generation.
This implementation avoids batching, sending one prompt per request.
"""

import requests
import json
import logging
from typing import List, Optional

from pytaskexec import TaskRunner, taskify
import multiprocessing

from ragify_docs.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@taskify
def get_embedding_one(text: str) -> List[float]:
    """
    Get a single embedding from the Ollama API.

    Args:
        text (str): Text to embed.

    Returns:
        List[float]: The embedding.
    """
    url = f"{settings.ollama_base_url}/api/embeddings"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": settings.embedding_model.replace("ollama/", ""),
        "prompt": text
    }
    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result["embedding"]
    except Exception as e:
        logger.exception(f"Embedding failed for text: {text[:60]}...")
        raise e


def is_ollama_available() -> bool:
    """
    Check if the Ollama API is available.

    Returns:
        bool: Whether the Ollama API is available.
    """
    try:
        response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Ollama not available: %s", e)
        return False


def call_ollama(prompt: str, model: Optional[str] = None, stream: bool = False) -> str:
    """
    Call the Ollama API for text generation.

    Args:
        prompt (str): Prompt to send.
        model (Optional[str]): Model name (optional override).
        stream (bool): Whether to stream output.

    Returns:
        str: Generated response.
    """
    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": (model or settings.ollama_model).replace("ollama/", ""),
        "prompt": prompt,
        "stream": stream
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        logger.exception("Ollama call failed")
        raise RuntimeError("Failed to call Ollama") from e


def get_ollama_embedding(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings from Ollama one by one (since most models don't support batch).

    Args:
        texts (List[str]): List of strings to embed.

    Returns:
        List[List[float]]: List of embeddings.
    """
    vectors = []
    max_workers = multiprocessing.cpu_count() * 10
    logger.info(f"Using {max_workers} workers to embed {len(texts)} texts")
    with TaskRunner(max_workers=max_workers) as runner:
        tids = [runner.schedule(get_embedding_one(text)) for text in texts]
        for tid in tids:
            try:
                vectors.append(runner.get_result(tid))
            except Exception as e:
                logger.exception(f"Embedding failed for text: {runner.get_input(tid)[:60]}...")
                raise e
    logger.info(f"Embedding completed for {len(texts)} texts, total embeddings: {len(vectors)}")
    return vectors
