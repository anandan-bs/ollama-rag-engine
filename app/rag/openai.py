"""
openai.py

Handles OpenAI-based ChatCompletion for fallback inference when Ollama is unavailable.
Only used for generating answers. Embeddings are never handled by OpenAI.
"""

import openai
import logging
from app.config import settings

openai.api_key = settings.openai_api_key

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def call_openai(prompt: str) -> str:
    """
    Use OpenAI ChatCompletion to generate a response to the given prompt.

    Args:
        prompt (str): Full prompt containing context and user question.

    Returns:
        str: Answer text from the model.
    """
    try:
        response = openai.ChatCompletion.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:  # noqa: F841
        logger.exception("OpenAI call failed")
        return "[OpenAI model failed to respond.]"
