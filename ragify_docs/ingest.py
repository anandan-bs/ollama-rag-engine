"""
Ingest a document into the Chroma vector database.

This script is responsible for:

1. Loading a document from disk using PyMuPDF.
2. Chunking the text into suitable lengths for embedding.
3. Filtering out chunks that are too long for the model.
4. Embedding the valid chunks using the SmartBatchedEmbedder.
5. Storing the embeddings in the Chroma vector database.
"""

import os
import logging

from ragify_docs.core.load_doc import load_document
from ragify_docs.core.chunk import chunk_text
from ragify_docs.core.store import ChromaStore
from ragify_docs.core.embed import SmartBatchedEmbedder


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

embedder = SmartBatchedEmbedder()
store = ChromaStore()


def ingest_document(path: str):
    """
    Ingest a document into the Chroma vector database.

    Args:
        path (str): The path to the document to ingest.

    Returns:
        None
    """
    logger.info(f" Loading: {path}")
    doc = load_document(path)

    logger.info(" Chunking text")
    raw_chunks = chunk_text(doc)

    logger.info(" Filtering valid chunks")
    valid_chunks = []
    valid_ids = []

    skipped = list()
    for i, chunk in enumerate(raw_chunks):
        tokens = embedder._count_tokens(chunk)
        if tokens <= embedder.model_max_length:
            valid_chunks.append(chunk)
            valid_ids.append(f"{os.path.basename(path)}-{i}")
        else:
            skipped.append(tokens)

    if skipped:
        logger.info(f"Skipped {len(skipped)} chunks")
        logger.info(f"Max size of skipped chunk: {max(skipped)} tokens")
        logger.info(f"Avg size of skipped chunk: {sum(skipped) / len(skipped)} tokens")
        logger.info(f"Min size of skipped chunk: {min(skipped)} tokens")

    logger.info(f"Storing {len(valid_chunks)} chunks")

    if not valid_chunks:
        logger.warning(" No valid chunks to embed!")
        return

    logger.info(f" Storing {len(valid_chunks)} items to vector DB")
    store.add_documents(ids=valid_ids, documents=valid_chunks)
    logger.info(" Done")


if __name__ == "__main__":
    ingest_document("/Users/Anandan/Downloads/Core_v6.1.pdf")
