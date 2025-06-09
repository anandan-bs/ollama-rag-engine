# tokenize.py
from transformers import AutoTokenizer

from ragify_docs.config import settings

tokenizer = AutoTokenizer.from_pretrained(
    settings.local_model_dir,
    token=settings.huggingface_token
)


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a given string.

    :param text: The string to count tokens from.
    :return: The number of tokens in the string.
    """
    return len(tokenizer(text).input_ids)


def truncate(text: str, max_tokens: int) -> str:
    """
    Truncate a given string to the model's maximum sequence length.

    :param text: The string to truncate.
    :param max_tokens: The maximum number of tokens to keep.
    :return: The truncated string.
    """
    tokens = tokenizer(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens.input_ids, skip_special_tokens=True)
