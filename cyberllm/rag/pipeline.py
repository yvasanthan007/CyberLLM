"""
RAG pipeline (Phase 5).

Retrieves documents from the existing 2,427-document knowledge base
(artifacts/vectorstore/cyberllm_kb.jsonl) via the existing BM25 lexical
search (cyberllm.retrieval.lexical) and supplies concise, source-attributed
context to the local LLM.

Design points:
- Never rebuilds the index; reuses load_documents()/search_documents().
- Handles irrelevant retrieval (below-score results dropped), missing
  sources, context limits (truncate long docs, cap total chars), and
  unsupported questions (returns no_context).
- Retrieved evidence stays clearly separated from model explanation:
  callers receive "context" plus per-doc metadata and must attribute.
"""

from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KB_PATH = ROOT / "artifacts" / "vectorstore" / "cyberllm_kb.jsonl"

MIN_DOC_SCORE = 1.0          # BM25 below this = irrelevant query for the KB
MAX_DOCS = 3                 # docs passed to the model at most
MAX_DOC_CHARS = 900          # truncate a single doc's text
MAX_TOTAL_CHARS = 3500       # total context budget


@lru_cache(maxsize=1)
def _documents():
    from cyberllm.retrieval.lexical import load_documents
    if not KB_PATH.is_file():
        return []
    return load_documents(KB_PATH)


def retrieve(query, top_k=MAX_DOCS, min_score=MIN_DOC_SCORE):
    """Return list of relevant docs with metadata, or [] if none are relevant."""
    documents = _documents()
    if not documents or not query or not query.strip():
        return []
    from cyberllm.retrieval.lexical import search_documents
    try:
        results = search_documents(query, documents, top_k=top_k)
    except Exception:
        return []
    relevant = []
    for doc in results:
        score = doc.get("score", 0.0)
        if score < min_score:
            continue
        relevant.append({
            "doc_id": doc.get("doc_id", ""),
            "title": doc.get("title", "") or doc.get("doc_id", ""),
            "source": doc.get("source", ""),
            "url": doc.get("url", ""),
            "score": round(score, 3),
            "text": (doc.get("text", "") or "")[:MAX_DOC_CHARS],
        })
        if sum(len(d["text"]) for d in relevant) > MAX_TOTAL_CHARS:
            relevant.pop()
            break
    return relevant


def build_context(docs):
    """Assemble a compact, source-attributed context block."""
    if not docs:
        return ""
    blocks = []
    for doc in docs:
        header = doc.get("title") or doc.get("doc_id") or "document"
        source = doc.get("source") or doc.get("url") or ""
        blocks.append(f"[KB: {header}" + (f" | source: {source}" if source else "") + f"]\n{doc['text']}")
    return "\n\n".join(blocks)


def rag_context(query):
    """Convenience: returns (context_str, docs, status)."""
    docs = retrieve(query)
    if not docs:
        return "", [], "no_relevant_documents"
    return build_context(docs), docs, "ok"
