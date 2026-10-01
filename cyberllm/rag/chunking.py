"""Document representation and chunking for RAG ingestion.

A :class:`Document` is the atomic unit that flows from a source loader into
the vector store. Every document carries its provenance (``source``,
``doc_id``, ``url``) so an answer can always be traced back to a citable
source - this is what separates *verified source information* from
*model-generated explanation* in the UI.

Chunking is section-aware where possible: threat-intel records (CVE
descriptions, ATT&CK techniques) are naturally short and are kept whole,
while long narrative documents are split on paragraph boundaries with
overlap so a fact near a boundary is not lost.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

from ..security.guards import sanitize_text, scan_prompt_injection

_WS_RE = re.compile(r"[ \t]+")
_BLANK_RE = re.compile(r"\n{3,}")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n{2,}")


@dataclass
class Document:
    """A single retrievable unit with full provenance."""

    text: str
    doc_id: str
    title: str = ""
    source: str = ""                 # e.g. "cisa_kev", "mitre_attack", "nvd_cve"
    source_type: str = "dataset"     # dataset | upload | manual
    url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = -1
    chunk_total: int = 1

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = make_doc_id(self.source, self.text)
        if not self.title:
            self.title = self.text.split("\n", 1)[0][:120]

    def citation(self) -> str:
        """Human-readable citation line, e.g. ``[1] cisa_kev: CVE-2021-1234``."""
        label = f"[{self.metadata.get('citation_index', '?')}]"
        return f"{label} {self.source}: {self.title}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def content_hash(self) -> str:
        return make_doc_id(self.source, self.text)


def make_doc_id(source: str, text: str) -> str:
    """Stable content-addressed id so re-ingesting is idempotent."""
    digest = hashlib.sha256(f"{source}\x00{text}".encode("utf-8")).hexdigest()
    return f"{source or 'doc'}:{digest[:16]}"


def clean_text(text: str, *, max_chars: int = 200_000) -> str:
    """Normalise raw document text for storage and embedding."""
    text = sanitize_text(text, max_chars=max_chars)
    text = _WS_RE.sub(" ", text)
    return _BLANK_RE.sub("\n\n", text).strip()


def chunk_document(
    doc: Document, *, chunk_size: int = 900, overlap: int = 120,
    max_chunks: int = 512,
) -> list[Document]:
    """Split a document into overlapping chunks on sentence boundaries.

    Short documents (a CVE description, an ATT&CK technique) are returned
    whole: splitting them would fragment the citation and reduce answer
    quality.
    """
    text = doc.text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [doc]

    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        if length + len(sentence) + 1 > chunk_size and current:
            chunks.append(" ".join(current))
            # Overlap: carry the tail of the previous chunk forward.
            carried, carried_len = [], 0
            for prev in reversed(current):
                if carried_len + len(prev) > overlap:
                    break
                carried.insert(0, prev)
                carried_len += len(prev) + 1
            current, length = carried, carried_len
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))

    chunks = [c.strip() for c in chunks if c.strip()][:max_chunks]
    if len(chunks) <= 1:
        return [doc]

    total = len(chunks)
    return [
        Document(
            text=chunk,
            doc_id=f"{doc.doc_id}#{i}",
            title=doc.title,
            source=doc.source,
            source_type=doc.source_type,
            url=doc.url,
            chunk_index=i,
            chunk_total=total,
            metadata={**doc.metadata, "parent_doc_id": doc.doc_id},
        )
        for i, chunk in enumerate(chunks)
    ]


def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, falling back to paragraphs then words."""
    parts = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    out: list[str] = []
    for part in parts:
        if len(part) <= 2000:
            out.append(part)
            continue
        # Very long paragraph: break on words so chunks stay bounded.
        buf: list[str] = []
        for word in part.split():
            buf.append(word)
            if len(" ".join(buf)) > 1600:
                out.append(" ".join(buf))
                buf = []
        if buf:
            out.append(" ".join(buf))
    return out or [text[:2000]]


def prepare_documents(
    docs: Iterable[Document], *, chunk_size: int = 900, overlap: int = 120,
) -> tuple[list[Document], dict[str, Any]]:
    """Clean, injection-scan and chunk a batch of documents.

    Returns the prepared chunks plus a stats dict recording how many
    chunks were produced and how many triggered the injection filter - this
    audit trail is surfaced in the ingestion report.
    """
    prepared: list[Document] = []
    scanned = 0
    empty = 0
    for doc in docs:
        cleaned = clean_text(doc.text)
        if not cleaned:
            empty += 1
            continue
        scan = scan_prompt_injection(cleaned)
        if scan.suspicious:
            scanned += 1
            cleaned = scan.sanitized_text
        doc.text = cleaned
        prepared.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))

    stats = {
        "chunks": len(prepared),
        "injection_flagged": scanned,
        "empty_skipped": empty,
        "sources": sorted({d.source for d in prepared}),
    }
    return prepared, stats


__all__ = ["Document", "chunk_document", "clean_text", "prepare_documents", "make_doc_id"]
