"""Exact security-identifier extraction for retrieval boosting.

Threat-intel identifiers (CVE ids, MITRE ATT&CK technique ids, CWE ids) are
*lookup keys*, not fuzzy search terms. If a user asks about
``CVE-2021-44228`` they want that record, and a purely lexical vector score is
both noisy (hash collisions) and semantically wrong for this case.

This module extracts those identifiers from a query so the RAG pipeline can
promote documents that contain them verbatim. Documents are still ranked by
embedding similarity within each boost tier - the boost only decides *tier*,
it does not fabricate a score.
"""

from __future__ import annotations

import re
from typing import Iterable

#: CVE identifier, e.g. CVE-2021-44228.
CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)

#: ATT&CK technique / sub-technique, e.g. T1059 or T1059.001.
TECHNIQUE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)

#: ATT&CK software identifier, e.g. S0266.
SOFTWARE_RE = re.compile(r"\bS\d{4}\b")

#: CWE identifier, e.g. CWE-79.
CWE_RE = re.compile(r"\bCWE-\d+\b", re.IGNORECASE)

_ALL_PATTERNS = (CVE_RE, TECHNIQUE_RE, SOFTWARE_RE, CWE_RE)


def extract_identifiers(text: str) -> set[str]:
    """Return every canonical identifier found in ``text`` (upper-cased)."""
    found: set[str] = set()
    for pattern in _ALL_PATTERNS:
        for match in pattern.findall(text or ""):
            found.add(match.upper())
    return found


def document_identifiers(documents: Iterable) -> set[str]:
    """Collect the identifiers present across a set of documents.

    Checks both the document text and structured metadata (``cve_id``,
    ``technique_id``), so a record indexed from CISA KEV matches on its id even
    if the id appears only in metadata.
    """
    found: set[str] = set()
    for doc in documents:
        found |= extract_identifiers(getattr(doc, "text", "") or "")
        metadata = getattr(doc, "metadata", None) or {}
        for key in ("cve_id", "technique_id", "stix_id"):
            value = metadata.get(key)
            if value:
                found |= extract_identifiers(str(value))
    return found


def exact_match_boost(query: str, document) -> float:
    """Score how strongly a document matches the identifiers in a query.

    Returns a boost in ``[0, 1]``:
      * ``1.0`` - every identifier in the query appears in the document;
      * a fraction - only some of them do;
      * ``0.0`` - the query has no identifiers, or none match.
    """
    query_ids = extract_identifiers(query)
    if not query_ids:
        return 0.0
    doc_ids = extract_identifiers(getattr(document, "text", "") or "")
    metadata = getattr(document, "metadata", None) or {}
    for key in ("cve_id", "technique_id", "stix_id"):
        if metadata.get(key):
            doc_ids |= extract_identifiers(str(metadata[key]))
    hits = query_ids & doc_ids
    return len(hits) / len(query_ids) if hits else 0.0


__all__ = [
    "extract_identifiers", "document_identifiers", "exact_match_boost",
    "CVE_RE", "TECHNIQUE_RE", "SOFTWARE_RE", "CWE_RE",
]
