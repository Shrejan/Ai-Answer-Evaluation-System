"""
Local embedding logic: model loading (once), text splitting, and cosine
similarity. No network calls happen in this file — everything runs on
your machine (GPU if available, otherwise CPU).
"""

import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import config

_model = None  # loaded lazily, once, and reused for every call


def get_model() -> SentenceTransformer:
    """Load the Sentence Transformer model once and cache it in memory."""
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Embedding device: {device}")
        _model = SentenceTransformer(config.MODEL_NAME, device=device)
    return _model


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentence-level chunks.

    This is intentionally simple (regex on sentence-ending punctuation).
    For the reference answer, each resulting sentence is treated as one
    "concept". For the student answer, each sentence is a unit that gets
    compared against each reference concept.
    """
    text = text.strip()
    if not text:
        return []

    raw_parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = [p.strip() for p in raw_parts if p.strip()]
    return sentences


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts in one call (faster than one-by-one)."""
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so cosine similarity = dot product
        show_progress_bar=False,
    )
    return embeddings


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between every row of `a` and every row of `b`.
    Embeddings are already L2-normalized, so this is just a dot product.
    """
    return a @ b.T
