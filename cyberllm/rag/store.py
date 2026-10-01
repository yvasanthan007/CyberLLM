"""Persistent vector store for the CyberLLM knowledge base.

Design rationale
----------------
ChromaDB is a supported backend (it is already a project dependency and
handles persistence well), but a **self-contained NumPy store** is the
default. Why: it has no native-wheel requirement, works on every platform
this project targets, keeps the whole knowledge base in one auditable file,
and lets tests run with zero setup. Both satisfy the same :class:`VectorStore`
interface, so the choice is a configuration detail, not an architectural one.

Persistence layout (built-in backend)::

    artifacts/vectorstore/
        cyberllm_kb.jsonl     # one record per chunk (text + metadata)
        cyberllm_kb.npy       # float32 embedding matrix
        cyberllm_kb.meta.json # backend/embedding metadata
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core.errors import VectorStoreError
from ..core.logging_config import get_logger
from .chunking import Document
from .embeddings import Embedder, cosine_similarity_matrix
from .identifiers import extract_identifiers

logger = get_logger("rag.store")


@dataclass
class SearchResult:
    """One retrieved chunk plus its score and provenance."""

    document: Document
    score: float
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(float(self.score), 6),
            "doc_id": self.document.doc_id,
            "title": self.document.title,
            "source": self.document.source,
            "url": self.document.url,
            "text": self.document.text,
            "metadata": self.document.metadata,
        }


class VectorStore(ABC):
    """Minimal vector-store contract used by the RAG pipeline."""

    @abstractmethod
    def add(self, documents: Sequence[Document], embeddings: np.ndarray) -> int:
        """Insert documents; returns the number actually added."""

    @abstractmethod
    def search(self, embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        """Return the ``top_k`` most similar documents."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored chunks."""

    @abstractmethod
    def delete_source(self, source: str) -> int:
        """Remove every chunk from one source; returns rows removed."""

    @abstractmethod
    def reset(self) -> None:
        """Drop all data."""

    def sources(self) -> list[str]:
        return []

    def find_by_identifier(self, identifier: str) -> list[Document]:
        """Exact lookup of documents containing a security identifier.

        Identifiers (CVE-2021-44228, T1059.001, CWE-79) are keys, not fuzzy
        terms. Approximate nearest-neighbour search is the wrong tool: at a few
        hundred dimensions, hash collisions bury the one document that
        literally contains the id. Backends that can index exactly should
        override this; the default returns nothing so behaviour degrades
        gracefully rather than silently.
        """
        return []

    def describe(self) -> dict[str, Any]:
        return {"backend": type(self).__name__, "documents": self.count()}


class NumpyVectorStore(VectorStore):
    """Self-contained persistent store: JSONL metadata + NumPy vectors.

    Adequate for the tens-of-thousands of chunks a threat-intel corpus
    produces. Documents are keyed by ``doc_id`` so re-ingesting the same
    content updates in place rather than duplicating.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.path / "cyberllm_kb.jsonl"
        self.npy_path = self.path / "cyberllm_kb.npy"
        self.meta_path = self.path / "cyberllm_kb.meta.json"
        self._documents: dict[str, Document] = {}
        self._keys: list[str] = []
        self._embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._index: dict[str, list[str]] = {}
        self._meta: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "updated_at": "", "embedding_backend": "", "dimension": 0,
        }
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not (self.jsonl_path.exists() and self.npy_path.exists()):
            return
        try:
            with self.jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    doc_id = raw.get("doc_id", "")
                    if not doc_id:
                        continue
                    self._documents[doc_id] = Document(
                        text=raw.get("text", ""), doc_id=doc_id,
                        title=raw.get("title", ""), source=raw.get("source", ""),
                        source_type=raw.get("source_type", "dataset"),
                        url=raw.get("url", ""), metadata=raw.get("metadata", {}),
                        chunk_index=raw.get("chunk_index", -1),
                        chunk_total=raw.get("chunk_total", 1),
                    )
            self._keys = list(self._documents)
            self._rebuild_identifier_index()
            self._embeddings = np.load(self.npy_path)
            if self.meta_path.exists():
                self._meta.update(json.loads(self.meta_path.read_text(encoding="utf-8")))
            logger.info("loaded %d vectors from %s", len(self._documents), self.path)
        except (OSError, ValueError) as exc:
            raise VectorStoreError(
                f"Could not load vector store at {self.path}: {exc}"
            ) from exc

    def _persist(self) -> None:
        tmp = self.jsonl_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for doc_id in self._keys:
                payload = self._documents[doc_id].as_dict()
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        tmp.replace(self.jsonl_path)

        if self._embeddings.size:
            tmp_npy = self.path / "vectors.tmp.npy"
            np.save(tmp_npy, self._embeddings)
            tmp_npy.replace(self.npy_path)
        self._meta["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._meta["documents"] = len(self._documents)
        self._meta["dimension"] = int(self._embeddings.shape[1]) if self._embeddings.size else 0
        self.meta_path.write_text(json.dumps(self._meta, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    def _rebuild_identifier_index(self) -> None:
        """Recompute the identifier -> doc_id map from stored documents."""
        self._index = {}
        for key, doc in self._documents.items():
            for identifier in self._doc_identifiers(doc):
                self._index.setdefault(identifier, []).append(key)
        logger.debug("identifier index rebuilt: %d keys", len(self._index))

    @staticmethod
    def _doc_identifiers(doc: Document) -> set[str]:
        """Identifiers in a document's text plus its structured metadata."""
        found = extract_identifiers(doc.text)
        for field in ("cve_id", "technique_id", "stix_id"):
            value = (doc.metadata or {}).get(field)
            if value:
                found |= extract_identifiers(str(value))
        return found

    def find_by_identifier(self, identifier: str) -> list[Document]:
        """Exact identifier lookup. O(1) via the in-memory index."""
        key = (identifier or "").strip().upper()
        if not key:
            return []
        return [self._documents[doc_id] for doc_id in self._index.get(key, [])
                if doc_id in self._documents]

    # ------------------------------------------------------------------
    def add(self, documents: Sequence[Document], embeddings: np.ndarray) -> int:
        if len(documents) != len(embeddings):
            raise VectorStoreError(
                f"Mismatched batch: {len(documents)} documents vs {len(embeddings)} embeddings"
            )
        if not documents:
            return 0

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        added = 0
        index_of = {key: i for i, key in enumerate(self._keys)}
        for doc, vector in zip(documents, vectors):
            existing = index_of.get(doc.doc_id)
            if existing is not None:
                # Re-ingest: refresh content and vector in place.
                if existing < len(self._embeddings):
                    self._embeddings[existing] = vector
                self._documents[doc.doc_id] = doc
                continue
            self._documents[doc.doc_id] = doc
            self._keys.append(doc.doc_id)
            self._embeddings = (
                vector.reshape(1, -1) if self._embeddings.size == 0
                else np.vstack([self._embeddings, vector.reshape(1, -1)])
            )
            added += 1

        self._persist()
        return added

        return {"backend": type(self).__name__, "documents": self.count()}

    def search(self, embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        if not self._keys or self._embeddings.size == 0:
            return []
        n = min(self._embeddings.shape[0], len(self._keys))
        if n == 0:
            return []
        scores = cosine_similarity_matrix(embedding, self._embeddings[:n])
        limit = min(max(top_k, 1), len(scores))
        if limit < len(scores):
            top_idx = np.argpartition(-scores, limit - 1)[:limit]
        else:
            top_idx = np.arange(len(scores))
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        results: list[SearchResult] = []
        for rank, idx in enumerate(top_idx, start=1):
            doc = self._documents[self._keys[int(idx)]]
            doc.metadata["citation_index"] = rank
            results.append(SearchResult(document=doc, score=float(scores[idx]), rank=rank))
        return results

    def count(self) -> int:
        return len(self._keys)

    def delete_source(self, source: str) -> int:
        keep = [(i, k) for i, k in enumerate(self._keys)
                if self._documents[k].source != source]
        removed = len(self._keys) - len(keep)
        if not removed:
            return 0
        self._keys = [k for _, k in keep]
        self._documents = {k: self._documents[k] for _, k in keep}
        self._embeddings = (
            self._embeddings[[i for i, _ in keep]]
            if keep and self._embeddings.size else np.zeros((0, 0), dtype=np.float32)
        )
        self._persist()
        logger.info("removed %d chunks from source %s", removed, source)
        return removed

    def reset(self) -> None:
        self._keys = []
        self._documents = {}
        self._embeddings = np.zeros((0, 0), dtype=np.float32)
        self._persist()

    def sources(self) -> list[str]:
        return sorted({d.source for d in self._documents.values() if d.source})

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "numpy",
            "documents": self.count(),
            "dimension": int(self._meta.get("dimension", 0)),
            "embedding_backend": self._meta.get("embedding_backend", ""),
            "path": str(self.path),
            "updated_at": self._meta.get("updated_at", ""),
        }



class ChromaVectorStore(VectorStore):
    """ChromaDB-backed store for larger corpora.

    Selected with ``rag.store_backend=chroma``. Chroma persists to the same
    directory, so switching backends is a config change followed by a
    re-index (``rag reindex``).
    """

    def __init__(self, path: Path, collection: str = "cyberllm_kb",
                 embedder: Embedder | None = None) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise VectorStoreError(
                "ChromaDB is not installed. Install it or use the numpy backend."
            ) from exc

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._client = chromadb.PersistentClient(
            path=str(self.path), settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, documents: Sequence[Document], embeddings: np.ndarray) -> int:
        if not documents:
            return 0
        self._collection.upsert(
            ids=[d.doc_id for d in documents],
            embeddings=np.asarray(embeddings, dtype=np.float32).tolist(),
            documents=[d.text for d in documents],
            metadatas=[
                {
                    "title": d.title[:500], "source": d.source,
                    "source_type": d.source_type, "url": d.url[:1000],
                    "chunk_index": d.chunk_index, "chunk_total": d.chunk_total,
                    "metadata": json.dumps(d.metadata, default=str)[:4000],
                }
                for d in documents
            ],
        )
        return len(documents)

    def search(self, embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        if self._collection.count() == 0:
            return []
        got = self._collection.query(
            query_embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
            n_results=min(top_k, self._collection.count()),
        )
        ids = (got.get("ids") or [[]])[0]
        docs = (got.get("documents") or [[]])[0]
        metas = (got.get("metadatas") or [[]])[0]
        dists = (got.get("distances") or [[]])[0]

        results: list[SearchResult] = []
        for rank, (doc_id, text, meta, dist) in enumerate(
            zip(ids, docs, metas, dists), start=1
        ):
            try:
                extra = json.loads((meta or {}).get("metadata", "{}"))
            except (json.JSONDecodeError, TypeError):
                extra = {}
            results.append(SearchResult(
                document=Document(
                    text=text or "", doc_id=doc_id, title=(meta or {}).get("title", ""),
                    source=(meta or {}).get("source", ""), url=(meta or {}).get("url", ""),
                    source_type=(meta or {}).get("source_type", "dataset"),
                    chunk_index=(meta or {}).get("chunk_index", -1),
                    metadata={**extra, "citation_index": rank},
                ),
                # Chroma returns cosine distance; convert back to similarity.
                score=float(1.0 - dist), rank=rank,
            ))
        return results

    def count(self) -> int:
        return int(self._collection.count())

    def delete_source(self, source: str) -> int:
        before = self.count()
        self._collection.delete(where={"source": source})
        return before - self.count()

    def reset(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def sources(self) -> list[str]:
        got = self._collection.get(include=["metadatas"])
        return sorted({
            (m or {}).get("source", "") for m in (got.get("metadatas") or [])
            if (m or {}).get("source")
        })

    def describe(self) -> dict[str, Any]:
        return {"backend": "chroma", "documents": self.count(), "path": str(self.path)}


def build_store(path: Path, backend: str = "numpy", collection: str = "cyberllm_kb",
                embedder: Embedder | None = None) -> VectorStore:
    """Instantiate the configured vector store."""
    if backend.lower() == "chroma":
        return ChromaVectorStore(path, collection=collection, embedder=embedder)
    return NumpyVectorStore(path)


__all__ = [
    "VectorStore", "NumpyVectorStore", "ChromaVectorStore", "SearchResult", "build_store",
]

