"""Local lexical retrieval for the CyberLLM knowledge base."""

import argparse
import json
import math
import re
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KB = PROJECT_ROOT / "artifacts" / "vectorstore" / "cyberllm_kb.jsonl"

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", re.IGNORECASE)


def tokenize(text):
    """Convert text to lowercase search terms."""
    return TOKEN_PATTERN.findall(str(text).lower())


def load_documents(path=DEFAULT_KB):
    """Load JSONL documents, reporting malformed records clearly."""
    documents = []

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}, line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}, line {line_number}"
                )

            if str(record.get("text", "")).strip():
                documents.append(record)

    return documents


def search_documents(query, documents, top_k=5):
    """Rank documents using BM25-style lexical relevance scoring."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    query_terms = tokenize(query)
    if not query_terms or not documents:
        return []

    indexed = []
    document_frequency = Counter()
    total_length = 0

    for document in documents:
        title_terms = tokenize(document.get("title", ""))
        text_terms = tokenize(document.get("text", ""))
        terms = title_terms + text_terms
        frequencies = Counter(terms)

        indexed.append((document, frequencies, len(terms)))
        document_frequency.update(set(terms))
        total_length += len(terms)

    average_length = total_length / len(indexed) if indexed else 0
    if average_length == 0:
        return []

    k1 = 1.5
    b = 0.75
    results = []

    for document, frequencies, length in indexed:
        score = 0.0

        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue

            df = document_frequency[term]
            inverse_frequency = math.log(
                1 + (len(indexed) - df + 0.5) / (df + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * length / average_length
            )
            score += inverse_frequency * (
                frequency * (k1 + 1) / denominator
            )

            # Give terms in the title extra weight.
            title_frequency = tokenize(document.get("title", "")).count(term)
            score += 1.5 * title_frequency * inverse_frequency

        if score > 0:
            results.append({
                "score": score,
                "title": document.get("title") or document.get("doc_id", "Untitled"),
                "text": document.get("text", ""),
                "source": document.get("source", "unknown"),
                "url": document.get("url", ""),
                "doc_id": document.get("doc_id", ""),
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def main():
    parser = argparse.ArgumentParser(
        description="Search the local CyberLLM cybersecurity knowledge base."
    )
    parser.add_argument("--query", required=True, help="Cybersecurity search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB, help="JSONL knowledge base")
    args = parser.parse_args()

    documents = load_documents(args.kb)
    results = search_documents(args.query, documents, args.top_k)

    print(f"Knowledge-base records loaded: {len(documents)}")
    print(f"Query: {args.query}")
    print()

    if not results:
        print("No matching documents found.")
        return

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result['title']}")
        print(f"   Relevance score: {result['score']:.3f}")
        print(f"   Source: {result['source']}")
        print(f"   Document ID: {result['doc_id']}")
        print(f"   {result['text'][:700].strip()}")
        if result["url"]:
            print(f"   URL: {result['url']}")
        print()


if __name__ == "__main__":
    main()
