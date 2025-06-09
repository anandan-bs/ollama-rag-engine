"""
chunk.py

Contains functions for chunking text into suitable lengths for embedding.

This module provides one public function, `chunk_text`, which takes a list of
paragraphs and returns a list of chunked text. The chunking is done by
splitting the text into SpaCy sentences, and then truncating the sentences to
fit within the model's maximum sequence length.

The `chunk_text` function takes an optional `overlap_tokens` parameter, which
specifies the number of tokens to overlap between chunks. This can be used to
ensure that the model sees some context when generating embeddings.
"""

import logging
from typing import List
from transformers import AutoTokenizer
import spacy

from ragify_docs.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load SpaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("⚠️ SpaCy English model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'")
    raise

tokenizer = AutoTokenizer.from_pretrained(settings.local_model_dir)
model_max_length = tokenizer.model_max_length


def chunk_text(paragraphs: List[str], overlap_tokens: int = 50) -> List[str]:
    """
    Chunk text into suitable lengths for embedding.

    :param paragraphs: A list of paragraphs to chunk.
    :param overlap_tokens: The number of tokens to overlap between chunks.
    :return: A list of chunked text.
    """
    chunks = []

    for para in paragraphs:
        tokens = tokenizer(para, truncation=False).input_ids

        # If paragraph is short enough, use as is
        if len(tokens) <= model_max_length:
            chunks.append(para)
            continue

        logger.debug(f"Paragraph exceeds model limit ({len(tokens)} tokens). Splitting by SpaCy sentences.")

        # Sentence-level splitting
        sentence_group = []
        sentence_group_tokens = []

        for sentence in nlp(para).sents:
            sent = sentence.text.strip()
            sent_tokens = tokenizer(sent, truncation=False).input_ids

            # Truncate single oversized sentence
            if len(sent_tokens) > model_max_length:
                logger.debug(f"Truncating long sentence with {len(sent_tokens)} tokens")
                sent_tokens = sent_tokens[:model_max_length]
                sent = tokenizer.decode(sent_tokens, skip_special_tokens=True)

            # Finalize chunk if adding this sentence would overflow
            if sum(len(t) for t in sentence_group_tokens) + len(sent_tokens) > model_max_length:
                chunk_text = " ".join(sentence_group)
                chunks.append(chunk_text)

                # Optionally add overlap
                if overlap_tokens > 0:
                    all_tokens = tokenizer(chunk_text, truncation=False).input_ids
                    overlap = all_tokens[-overlap_tokens:]
                    overlap_text = tokenizer.decode(overlap, skip_special_tokens=True)
                    sentence_group = [overlap_text]
                    sentence_group_tokens = [tokenizer(overlap_text, truncation=False).input_ids]
                else:
                    sentence_group, sentence_group_tokens = [], []

            sentence_group.append(sent)
            sentence_group_tokens.append(sent_tokens)

        if sentence_group:
            chunks.append(" ".join(sentence_group))

    return chunks
