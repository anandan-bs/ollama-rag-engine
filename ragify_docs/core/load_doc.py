"""
Load text content from a document file.

Supported formats are PDF, DOC(X), TXT, and MD.
"""

import os
import fitz  # PyMuPDF
import docx2txt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_document(path: str) -> list[str]:
    """
    Load text content from a document file.

    Supported formats are PDF, DOC(X), TXT, and MD.

    Args:
        path (str): The path to the document to load.

    Returns:
        list[str]: A list of text paragraphs.
    """
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".pdf":
        with fitz.open(path) as doc:
            text = "\n".join(page.get_text() for page in doc)

    elif ext in [".doc", ".docx"]:
        text = docx2txt.process(path)

    elif ext in [".txt", ".md"]:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Normalize and split by paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
    logger.info(f"✅ Loaded {len(paragraphs)} paragraphs")
    return paragraphs
