"""
config.py

Centralized configuration module for managing environment variables, default settings,
and toggles for models, API keys, directories, and features.
Compatible with Pydantic v2+ using `pydantic-settings`.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

base_data_dir = Path(__file__).parent.parent / ".data"
base_data_dir.mkdir(exist_ok=True)


class Settings(BaseSettings):
    """
    Centralized configuration for the application.
    """

    # Logging configuration
    log_dir: str = str(base_data_dir / "logs")
    local_model_dir: str = str(base_data_dir / "models")
    tokenizers_parallelism: str = os.getenv("TOKENIZERS_PARALLELISM", "false")
    nltk_data_dir: str = str(Path(local_model_dir) / "nltk_data")

    # Huggingface configuration
    huggingface_token: str = Field(default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN", ""))

    # OpenAI configuration
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = "gpt-4"

    # Ollama configuration
    use_ollama: bool = True
    ollama_base_url: str = "http://localhost:11434"  # Added base URL for Ollama
    ollama_model: str = "llama3.2"
    ollama_embed_model: str = "nomic-embed-text"

    # Embedding and reranking
    embedding_model: str = "gte-large"
    embedding_repo: str = "thenlper/gte-large"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # used in app/reranker.py
    enable_rerank: bool = True
    collection_name: str = 'rag_documents'
    max_document_token: int = 10000

    # App configuration
    session_dir: str = str(base_data_dir / "sessions")
    export_dir: str = str(base_data_dir / "exports")
    chroma_storage: str = str(base_data_dir / "chroma_storage")

    # Generation configuration
    temperature: float = 0.2

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
Path(settings.local_model_dir).mkdir(exist_ok=True)
Path(settings.log_dir).mkdir(exist_ok=True)
Path(settings.session_dir).mkdir(exist_ok=True)
Path(settings.export_dir).mkdir(exist_ok=True)
Path(settings.chroma_storage).mkdir(exist_ok=True)
Path(settings.nltk_data_dir).mkdir(exist_ok=True)
