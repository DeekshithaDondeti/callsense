"""Local embeddings with bge-small-en-v1.5, run through fastembed (ONNX). Free; no API calls.

The first use downloads the model (about 70-130 MB) from Hugging Face into a local cache.
The model reads at most 512 tokens, so the text starts with the call's summary: even when a long
transcript is cut off, the embedding still covers how the call went.
"""

from pathlib import Path

import numpy as np

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
CACHE_DIR = Path.home() / ".cache" / "callsense" / "fastembed"

_model = None


def get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(EMBEDDING_MODEL, cache_dir=str(CACHE_DIR))
    return _model


def text_to_embed(summary: str, transcript: str) -> str:
    return f"{summary}\n\n{transcript}"


def embed_documents(texts: list[str]) -> list[np.ndarray]:
    return list(get_model().embed(texts))


def embed_query(query: str) -> np.ndarray:
    """Embed a search query (bge models expect queries to be prefixed; fastembed handles it)."""
    return next(iter(get_model().query_embed(query)))
