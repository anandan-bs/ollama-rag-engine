"""
Health check utilities for the application.
"""

import logging
import requests
from typing import Dict, Tuple
from chromadb import PersistentClient

from app.config import OLLAMA_CONFIG, DB_CONFIG, STORAGE_CONFIG

logger = logging.getLogger(__name__)


def check_llm_health() -> Tuple[bool, str]:
    """Check if the LLM service is healthy."""
    try:
        response = requests.get(
            OLLAMA_CONFIG["api_url"].replace("/generate", "/health"), timeout=5
        )
        response.raise_for_status()
        return True, "LLM service is healthy"
    except Exception as e:
        logger.error(f"LLM health check failed: {str(e)}")
        return False, f"LLM service is unhealthy: {str(e)}"


def check_db_health() -> Tuple[bool, str]:
    """Check if the database is healthy."""
    try:
        db = PersistentClient(path=DB_CONFIG["path"])
        db.heartbeat()
        return True, "Database is healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return False, f"Database is unhealthy: {str(e)}"


def check_storage_health() -> Tuple[bool, str]:
    """Check if the storage directories are accessible."""
    try:
        import os

        for dir_name, dir_path in STORAGE_CONFIG.items():
            if not os.path.exists(dir_path):
                return False, f"Storage directory {dir_name} does not exist"
            if not os.access(dir_path, os.W_OK):
                return False, f"Storage directory {dir_name} is not writable"

        return True, "Storage is healthy"
    except Exception as e:
        logger.error(f"Storage health check failed: {str(e)}")
        return False, f"Storage check failed: {str(e)}"


def get_system_health() -> Dict[str, Dict[str, str]]:
    """Get the health status of all system components."""
    health_checks = {
        "llm": check_llm_health(),
        "database": check_db_health(),
        "storage": check_storage_health(),
    }

    return {
        name: {"status": "healthy" if status else "unhealthy", "message": message}
        for name, (status, message) in health_checks.items()
    }
