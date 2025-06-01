"""
Utility functions for the chatbot.

This module contains functions for extracting text from files and extracting
chunks from text. The chunks are used to build the index of documents that the
chatbot uses to answer questions.

"""

import os
import fitz
from tqdm import tqdm


def extract_text_from_file(filepath):
    """
    Extract text from a file, supporting PDF, TXT, MD, and RST files.

    Parameters
    ----------
    filepath : str
        The path to the file to extract text from.

    Returns
    -------
    str
        The extracted text.
    """
    ext = os.path.splitext(filepath)[-1].lower()
    text = ""
    if ext == ".pdf":
        doc = fitz.open(filepath)
        for page in doc:
            text += page.get_text()
    elif ext in [".txt", ".md", ".rst"]:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    return text


def extract_chunks_from_dir(directory):
    """
    Extract text chunks from all files in a directory, supporting PDF, TXT, MD,
    and RST files.

    Parameters
    ----------
    directory : str
        The path to the directory to extract text chunks from.

    Returns
    -------
    list
        A list of text chunks, where each chunk is a string of text separated by
        at least two newline characters.
    """
    chunks = []
    for file in tqdm(os.listdir(directory), desc="Indexing files"):
        path = os.path.join(directory, file)
        if os.path.isfile(path):
            raw = extract_text_from_file(path)
            chunks.extend(
                [c.strip() for c in raw.split("\n\n") if len(c.strip()) > 100]
            )
    return chunks
