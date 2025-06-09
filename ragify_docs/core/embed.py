"""
SmartBatchedEmbedder: a SentenceTransformer wrapper for efficient embedding of lists of strings.

The embedder is optimized for performance by batching,
truncating input strings to the model's maximum sequence length.
"""

import os
import torch
import logging
from typing import List, Generator
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from ragify_docs.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SmartBatchedEmbedder:
    """
    A SentenceTransformer wrapper for efficient embedding of lists of strings.
    The embedder is optimized for performance by batching,
    truncating input strings to the model's maximum sequence length.
    """

    def __init__(self):
        """
        Initialize the embedder.

        The device is determined automatically based on availability of CUDA and MPS devices.
        """
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Detected device: {self.device}")

        # Load model and tokenizer from local directory
        self.model = SentenceTransformer(settings.local_model_dir)
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(settings.local_model_dir)

        # Derive limits based on model type
        self.model_max_length = self.tokenizer.model_max_length
        self.max_tokens_per_batch = self._infer_optimal_batch_size()

        logger.info(
            f"Embedder initialized with max sequence length {self.model_max_length} "
            f"and max batch tokens {self.max_tokens_per_batch}"
        )

    def _infer_optimal_batch_size(self):
        """
        Infer an optimal batch size based on the device type.

        :return: The optimal batch size.
        """
        # Use smaller batch size for CPU
        if self.device == "cpu":
            return min(2048, self.model_max_length * 4)
        # Larger for GPU/MPS
        elif self.device == "cuda" or self.device == "mps":
            return min(8192, self.model_max_length * 8)
        else:
            return 4096

    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a given string.

        :param text: The string to count tokens from.
        :return: The number of tokens in the string.
        """
        return len(self.tokenizer(text, truncation=False).input_ids)

    def truncate_to_max_tokens(self, text: str) -> str:
        """
        Truncate a given string to the model's maximum sequence length.

        :param text: The string to truncate.
        :return: The truncated string.
        """
        tokens = self.tokenizer(text, truncation=False).input_ids[:self.model_max_length]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _split_batches(self, texts: List[str]) -> Generator[List[str], None, None]:
        """
        Split a list of strings into batches based on the model's maximum sequence length and the optimal batch size.

        :param texts: The list of strings to split.
        :return: A generator yielding batches of strings.
        """
        batch, total_tokens = [], 0
        for text in texts:
            tokens = self._count_tokens(text)

            if tokens > self.model_max_length:
                logger.warning(f"Truncating chunk from {tokens} to {self.model_max_length} tokens")
                text = self.truncate_to_max_tokens(text)
                tokens = self.model_max_length

            if total_tokens + tokens > self.max_tokens_per_batch:
                yield batch
                batch, total_tokens = [], 0

            batch.append(text)
            total_tokens += tokens

        if batch:
            yield batch

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings using the SentenceTransformer model.

        :param texts: The list of strings to embed.
        :return: A list of embeddings for the input strings.
        """
        all_embeddings = []

        def encode_batch(batch):
            return self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                device=self.device
            ).tolist()

        # Parallel encoding using multiprocessing if device is CPU
        if self.device == "cpu":
            import multiprocessing as mp
            with mp.Pool(processes=os.cpu_count()) as pool:
                results = pool.map(encode_batch, self._split_batches(texts))
                for result in results:
                    all_embeddings.extend(result)
        else:
            for batch in self._split_batches(texts):
                all_embeddings.extend(encode_batch(batch))

        return all_embeddings


class ChromaEmbeddingFunction:
    """
    A wrapper around the SmartBatchedEmbedder for use with Chroma.
    """

    def __init__(self, embedder: SmartBatchedEmbedder):
        self.embedder = embedder

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embedder.embed(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self.embedder.embed(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self.embedder.embed(input)

    def name(self) -> str:
        return "smart_batched_embedder"
