"""
ollama.py

Handles interaction with Ollama local models for embedding generation.
This implementation avoids batching, sending one prompt per request.
"""

import requests
import json
import logging
from typing import List, Optional
from app.config import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def is_ollama_available() -> bool:
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
    url = f"{settings.ollama_base_url}/api/embeddings"
    headers = {"Content-Type": "application/json"}
    vectors = []

    for text in texts:
        try:
            payload = {
                "model": settings.embedding_model.replace("ollama/", ""),
                "prompt": text
            }
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=60
            )
            response.raise_for_status()
            result = response.json()
            vectors.append(result["embedding"])
        except Exception as e:
            logger.exception(f"Embedding failed for text: {text[:60]}...")
            raise e

    return vectors
