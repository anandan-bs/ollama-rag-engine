"""
Configuration management for the application.
Loads environment variables and provides configuration objects.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Model Configuration
OLLAMA_CONFIG = {
    "api_url": os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate"),
    "model_name": os.getenv("OLLAMA_MODEL_NAME", "llama3.2"),
    "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.8")),
}

# Embedding Configuration
EMBEDDING_CONFIG = {
    "model_name": os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
}

# Database Configuration
DB_CONFIG = {
    "collection_name": os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base"),
    "path": os.getenv("DB_PATH", "embeddings/chroma_db"),
}

# File Storage Configuration
STORAGE_CONFIG = {
    "upload_dir": os.getenv("UPLOAD_DIR", "data/uploads"),
    "chatlog_dir": os.getenv("CHATLOG_DIR", "chatlogs"),
}

# Text Processing Configuration
PROCESSING_CONFIG = {
    "min_chunk_size": int(os.getenv("MIN_CHUNK_SIZE", 100)),
}

# Environment Configuration
ENV = os.getenv("ENV", "development")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# Create necessary directories
for dir_path in [
    STORAGE_CONFIG["upload_dir"],
    STORAGE_CONFIG["chatlog_dir"],
    DB_CONFIG["path"]
]:
    os.makedirs(dir_path, exist_ok=True)
