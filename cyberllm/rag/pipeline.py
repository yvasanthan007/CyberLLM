"""The RAG pipeline: ingest sources, retrieve context, and answer questions.

The pipeline enforces the project's core safety property: **retrieved text is
data, never instructions**. Every retrieved chunk passes through
:func:`~cyberllm.security.guards.scan_prompt_injection` before it is placed
near a prompt, and is wrapped in explicit delimiters.

It also enforces the epistemic property the UI depends on: an answer states
which parts came from a cited source and which are model reasoning, and it
declines to answer when the retrieval evidence is weak rather than inventing
content.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..core.config import AppConfig, get_settings
from ..core.errors import VectorStoreError
from ..core.logging_config import get_logger
from ..security.guards import scan_prompt_injection, wrap_untrusted
from .chunking import Document, prepare_documents
from .embeddings import Embedder, get_embedder
from .identifiers import extract_identifiers
from .loaders import load_source
from .store import SearchResult, VectorStore, build_store

logger = get_logger("rag.pipeline")

#: Below this cosine score a hit is reported as weak evidence, not fact.
WEAK_SCORE = 0.12


@dataclass
class IngestResult:
    """Outcome of indexing one or more sources."""

    source: str
    documents_loaded: int = 0
    chunks_indexed: int = 0
    injection_flagged: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "documents_loaded": self.documents_loaded,
            "chunks_indexed": self.chunks_indexed,
            "injection_flagged": self.injection_flagged,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "errors": self.errors,
            "path": self.path,
        }


@dataclass
class RetrievedContext:
    """Assembled context for a question, with citation metadata."""

    question: str
    results: list[SearchResult] = field(default_factory=list)
    context_text: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    weak_evidence: bool = False
    injection_warnings: list[str] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        return bool(self.results)

    def citation_block(self) -> str:
        """Numbered reference list appended to an answer."""
        if not self.citations:
            return ""
        lines = ["", "Sources:"]
        for cite in self.citations:
            url = cite.get("url") or ""
            suffix = f" - {url}" if url else ""
            lines.append(
                f"[{cite['index']}] {cite['source']}: {cite['title']}{suffix} "
                f"(retrieval score {cite['score']:.3f})"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "results": [r.as_dict() for r in self.results],
            "citations": self.citations,
            "weak_evidence": self.weak_evidence,
            "injection_warnings": self.injection_warnings,
            "context_chars": len(self.context_text),
        }


class RAGPipeline:
    """High-level retrieval facade used by the chat service and the API."""

    def __init__(self, settings: AppConfig | None = None,
                 store: VectorStore | None = None,
                 embedder: Embedder | None = None) -> None:
        self.settings = settings or get_settings()
        self.rag = self.settings.rag
        self.embedder = embedder or get_embedder(
            self.rag.embedding_backend,
            dimension=self.rag.embedding_dim,
            model_name=self.rag.embedding_model,
        )
        self._store = store
        self._store_backend = "numpy"

    # ------------------------------------------------------------------
    @property
    def store(self) -> VectorStore:
        """Lazily open the store so import time stays cheap in the API."""
        if self._store is None:
            self._store = build_store(
                self.settings.paths.vector_dir,
                backend=self._store_backend,
                collection=self.rag.collection_name,
                embedder=self.embedder,
            )
        return self._store

    def use_chroma(self) -> None:
        """Switch to the Chroma backend (requires a re-index)."""
        self._store_backend = "chroma"
        self._store = None

    # ------------------------------------------------------------------
    def ingest(self, source: str, path: Path, *, limit: int | None = None,
               replace: bool = True, batch_size: int = 512) -> IngestResult:
        """Load, clean, chunk, embed and index one dataset file.

        ``replace=True`` removes prior chunks from the same source so a
        re-index after a dataset update does not leave stale entries behind.
        """
        started = time.monotonic()
        result = IngestResult(source=source, path=str(path))
        try:
            documents = load_source(source, path, limit=limit)
        except (FileNotFoundError, ValueError) as exc:
            result.errors.append(str(exc))
            logger.error("ingest %s failed: %s", source, exc)
            return result

        result.documents_loaded = len(documents)
        if not documents:
            result.errors.append("loader produced no documents")
            return result

        chunks, stats = prepare_documents(
            documents, chunk_size=self.rag.chunk_size, overlap=self.rag.chunk_overlap
        )
        result.injection_flagged = stats["injection_flagged"]

        if replace:
            removed = self.store.delete_source(source)
            if removed:
                logger.info("replaced %d existing chunks for %s", removed, source)

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = self.embedder.embed([d.text for d in batch])
            result.chunks_indexed += self.store.add(batch, vectors)

        result.elapsed_seconds = time.monotonic() - started
        logger.info(
            "ingested %s: %d docs -> %d chunks in %.1fs",
            source, result.documents_loaded, result.chunks_indexed, result.elapsed_seconds,
        )
        return result

    def ingest_paths(self, items: Sequence[tuple[str, Path]],
                      **kwargs: Any) -> list[IngestResult]:
        return [self.ingest(name, path, **kwargs) for name, path in items]

    def reindex(self, items: Sequence[tuple[str, Path]], **kwargs: Any) -> list[IngestResult]:
        """Drop the whole store and rebuild it from scratch."""
        self.store.reset()
        return self.ingest_paths(items, **kwargs)

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Exact-identifier lookup first, then vector similarity.

        Identifiers are *keys*: asking about ``CVE-2021-44228`` must return
        that record. Approximate nearest-neighbour search is the wrong tool for
        that - at a few hundred dimensions, hash collisions rank unrelated
        documents above the one literally containing the id. So the pipeline
        resolves identifiers through an exact index and only falls back to
        similarity (excluding duplicates) for the remainder of the results.
        """
        query = (query or "").strip()
        if not query:
            return []
        if self.store.count() == 0:
            logger.warning("retrieval requested but the knowledge base is empty")
            return []

        limit = top_k or self.rag.top_k
        seen: set[str] = set()
        results: list[SearchResult] = []

        for identifier in sorted(extract_identifiers(query)):
            for doc in self.store.find_by_identifier(identifier)[:limit]:
                if doc.doc_id in seen:
                    continue
                seen.add(doc.doc_id)
                results.append(SearchResult(document=doc, score=1.0, rank=0))
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break

        vector = self.embedder.embed_one(query)
        for hit in self.store.search(vector, max(limit * 2, limit)):
            if len(results) >= limit:
                break
            if hit.document.doc_id in seen:
                continue
            seen.add(hit.document.doc_id)
            results.append(hit)

        for rank, result in enumerate(results, start=1):
            result.rank = rank
            result.document.metadata["citation_index"] = rank
        if results and results[0].score == 1.0:
            logger.debug("exact identifier match for %s", query[:80])
        return results

    def build_context(self, question: str, top_k: int | None = None) -> RetrievedContext:
        """Retrieve and assemble a safe, cited context block for a question."""
        results = self.search(question, top_k)
        context = RetrievedContext(question=question, results=results)
        if not results:
            return context

        selected = results[: self.rag.max_context_docs]
        blocks: list[str] = []
        budget = self.settings.llm.context_chars
        used = 0

        for result in selected:
            scan = scan_prompt_injection(result.document.text)
            if scan.suspicious:
                context.injection_warnings.append(
                    f"{result.document.doc_id}: content matched a prompt-injection "
                    "pattern and was neutralised before reaching the model."
                )
                logger.warning("injection pattern neutralised in %s", result.document.doc_id)
            text = scan.sanitized_text if scan.suspicious else result.document.text
            if used + len(text) > budget:
                remaining = budget - used
                if remaining <= 200:
                    break
                text = text[:remaining]

            label = re.sub(r"[^a-z_]", "", (result.document.source or "doc").lower()) or "doc"
            blocks.append(wrap_untrusted(text, label))
            used += len(text)
            context.citations.append({
                "index": result.rank,
                "doc_id": result.document.doc_id,
                "source": result.document.source,
                "title": result.document.title,
                "url": result.document.url,
                "score": round(float(result.score), 6),
                "metadata": result.document.metadata,
            })

        context.context_text = "\n\n".join(blocks)
        top_score = selected[0].score if selected else 0.0
        context.weak_evidence = top_score < max(WEAK_SCORE, self.rag.relevance_floor)
        return context

    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Knowledge-base health for /health and the Streamlit inventory."""
        try:
            info = self.store.describe()
            return {
                "available": True,
                "documents": info.get("documents", 0),
                "sources": self.store.sources(),
                "backend": info.get("backend", "unknown"),
                "embedding": self.embedder.describe(),
                "empty": info.get("documents", 0) == 0,
            }
        except VectorStoreError as exc:
            return {"available": False, "error": str(exc), "documents": 0, "empty": True}
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("vector store status failed")
            return {"available": False, "error": f"{type(exc).__name__}: {exc}",
                    "documents": 0, "empty": True}


__all__ = ["RAGPipeline", "IngestResult", "RetrievedContext", "WEAK_SCORE"]

