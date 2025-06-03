"""
config.py

Centralized configuration module for managing environment variables, default settings,
and toggles for models, API keys, directories, and features.
Compatible with Pydantic v2+ using `pydantic-settings`.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # OpenAI configuration
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = "gpt-4"

    # Ollama configuration
    use_ollama: bool = True
    ollama_base_url: str = "http://localhost:11434"  # Added base URL for Ollama
    ollama_model: str = "llama3.2"
    ollama_embed_model: str = "nomic-embed-text"

    # Embedding and reranking
    embedding_model: str = "ollama/nomic-embed-text"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # used in app/reranker.py
    enable_rerank: bool = True

    # App configuration
    session_dir: str = "sessions"
    export_dir: str = "exports"
    chroma_storage: str = "./chroma_storage"

    # Generation configuration
    temperature: float = 0.2

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
