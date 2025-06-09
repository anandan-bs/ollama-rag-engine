"""
Downloads and caches the Hugging Face tokenizer and SentenceTransformer model specified in the settings.
"""

import logging
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from ragify_docs.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_and_cache_model():
    """
    Downloads and caches the Hugging Face tokenizer and SentenceTransformer model specified in the settings.
    """
    logger.info("Downloading and saving tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_repo, token=settings.huggingface_token)
    tokenizer.save_pretrained(str(settings.local_model_dir))

    model = SentenceTransformer(settings.embedding_repo, use_auth_token=settings.huggingface_token)
    model.save(str(settings.local_model_dir))
    logger.info("Model and tokenizer saved to: " + str(settings.local_model_dir))


if __name__ == "__main__":
    download_and_cache_model()
