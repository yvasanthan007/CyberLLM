"""Embedding backends for the retrieval layer.

Two backends ship with CyberLLM:

``hashing`` (default)
    A deterministic, dependency-free bag-of-words projection to a fixed
    vector. Requires no model download, runs instantly on CPU, and is good
    enough for lexical threat-intel search (CVE IDs, ATT&CK technique IDs,
    product names). It is **not** a semantic model and does not claim to be.

``sentence-transformers``
    A real dense encoder (default ``all-MiniLM-L6-v2``). Selected when the
    package is installed; falls back to ``hashing`` with a logged warning
    rather than crashing.

Both implement :class:`Embedder` so the vector store is backend-agnostic.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from ..core.logging_config import get_logger

logger = get_logger("rag.embeddings")

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-_/][a-z0-9]+)*")

#: Lightweight suffix folding so "exploited"/"exploiting" share a stem.
_SUFFIXES = ("ing", "ed", "es", "s")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, and apply suffix folding."""
    tokens = _TOKEN_RE.findall(text.lower())
    out: list[str] = []
    for token in tokens:
        if len(token) > 4:
            for suffix in _SUFFIXES:
                if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                    out.append(token[: -len(suffix)])
                    break
            else:
                out.append(token)
        else:
            out.append(token)
    return out


class Embedder(ABC):
    """Common interface for embedding backends."""

    name: str = "base"
    dimension: int = 384

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts into an ``(n, dimension)`` float array."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def describe(self) -> dict[str, str | int]:
        return {"backend": self.name, "dimension": self.dimension}


class HashingEmbedder(Embedder):
    """Deterministic hashed bag-of-words with sublinear TF and L2 norm.

    Uses the hashing trick (no vocabulary to persist) with signed hashing to
    reduce collision bias, then L2-normalises so cosine similarity is a plain
    dot product.
    """

    name = "hashing"

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % self.dimension
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return index, sign

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for token in tokenize(text):
                index, sign = self._bucket(token)
                counts[index] = counts.get(index, 0.0) + sign
            for index, value in counts.items():
                # Sublinear TF damps repeated boilerplate in long documents.
                magnitude = 1.0 + math.log(abs(value)) if value else 0.0
                matrix[row, index] = magnitude * (1.0 if value >= 0 else -1.0)
            norm = float(np.linalg.norm(matrix[row]))
            if norm > 0:
                matrix[row] /= norm
        return matrix


class SentenceTransformerEmbedder(Embedder):
    """Dense encoder backed by ``sentence-transformers``."""

    name = "sentence-transformers"

    def __init__(self, model_name: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts), batch_size=32, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def describe(self) -> dict[str, str | int]:
        return {"backend": self.name, "dimension": self.dimension, "model": self._model_name}


def get_embedder(backend: str = "hashing", *, dimension: int = 384,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str | None = None) -> Embedder:
    """Build the requested embedder, degrading gracefully when unavailable."""
    backend = (backend or "hashing").lower()
    if backend in {"sentence-transformers", "st", "dense"}:
        try:
            return SentenceTransformerEmbedder(model_name, device=device)
        except Exception as exc:  # ImportError, network failure, missing weights
            logger.warning(
                "sentence-transformers backend unavailable (%s: %s); falling back to "
                "the hashing embedder. Retrieval will be lexical, not semantic.",
                type(exc).__name__, exc,
            )
    return HashingEmbedder(dimension=dimension)


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and many document vectors.

    Inputs are expected to be L2-normalised, but the function normalises
    defensively so callers cannot silently get wrong scores.
    """
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(q))
    if norm > 0:
        q = q / norm
    docs = np.asarray(matrix, dtype=np.float32)
    if docs.ndim == 1:
        docs = docs.reshape(1, -1)
    norms = np.linalg.norm(docs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (docs / norms) @ q


__all__ = [
    "Embedder", "HashingEmbedder", "SentenceTransformerEmbedder",
    "get_embedder", "cosine_similarity_matrix", "tokenize",
]
