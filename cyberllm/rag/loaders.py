"""Source loaders: turn raw dataset files into citable :class:`Document` objects.

Each loader is a pure function from a file on disk to documents, with no
network access. That separation means re-indexing works offline and the
provenance chain is auditable: file -> loader -> Document(source, url, id).

Loaders implemented
-------------------
``cisa_kev``   CISA Known Exploited Vulnerabilities catalog
``nvd_cve``    NVD CVE API 2.0 sample
``mitre_attack``  MITRE ATT&CK Enterprise STIX 2.1 bundle
``jsonl``      Generic JSONL (used for the in-repo SFT corpus)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..core.logging_config import get_logger
from .chunking import Document, clean_text

logger = get_logger("rag.loaders")

KEV_URL_TEMPLATE = "https://nvd.nist.gov/vuln/detail/{cve_id}"
ATTACK_URL_TEMPLATE = "https://attack.mitre.org/techniques/{technique_id}/"


def _read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def load_cisa_kev(path: Path, *, limit: int | None = None) -> list[Document]:
    """Parse the CISA KEV catalog into one document per vulnerability.

    KEV records are short and highly structured, so each stays as a single
    document - splitting them would damage the citation.
    """
    data = _read_json(path)
    records = data.get("vulnerabilities", []) if isinstance(data, dict) else data
    if not isinstance(records, list):
        logger.warning("%s: expected a list of vulnerabilities", path)
        return []

    docs: list[Document] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        cve_id = str(record.get("cveID", "")).strip()
        if not cve_id:
            continue
        vendor = record.get("vendorProject", "") or "unknown vendor"
        product = record.get("product", "") or "unknown product"
        due = record.get("dueDate", "") or "unspecified"
        added = record.get("dateAdded", "") or "unknown"
        ransomware = "Yes" if record.get("knownRansomwareCampaignUse") == "Known" else "No"
        vuln_name = record.get("vulnerabilityName", "") or "Unnamed vulnerability"
        cwes = record.get("cwes", []) or []
        cwe_text = ", ".join(str(c) for c in cwes) if cwes else "not specified"

        text = clean_text(
            f"{cve_id}: {vuln_name}\n"
            f"Vendor/Project: {vendor}. Product: {product}.\n"
            f"This vulnerability is known to be exploited in the wild and was added to "
            f"the CISA Known Exploited Vulnerabilities catalog on {added}.\n"
            f"Federal civilian agencies were directed to remediate by {due}.\n"
            f"Weaknesses (CWE): {cwe_text}.\n"
            f"Known to be used in ransomware campaigns: {ransomware}."
        )
        docs.append(Document(
            text=text,
            doc_id=f"cisa_kev:{cve_id}",
            title=f"{cve_id} - {vuln_name}",
            source="cisa_kev",
            source_type="dataset",
            url=KEV_URL_TEMPLATE.format(cve_id=cve_id),
            metadata={
                "cve_id": cve_id,
                "vendor": vendor,
                "product": product,
                "date_added": added,
                "due_date": due,
                "ransomware_use": ransomware,
                "cwes": [str(c) for c in cwes],
                "vulnerability_name": vuln_name,
                "authority": "CISA KEV (evidence of active exploitation)",
            },
        ))
        if limit and len(docs) >= limit:
            break

    logger.info("loaded %d CISA KEV records from %s", len(docs), path.name)
    return docs


def _pick_description(entry: dict) -> tuple[str, str]:
    """Return ``(description, source)`` preferring the English CVE text."""
    descriptions = entry.get("descriptions") or []
    for desc in descriptions:
        if desc.get("lang") == "en":
            return str(desc.get("value", "")), "NVD"
    for desc in descriptions:
        if str(desc.get("value", "")).strip():
            return str(desc.get("value", "")), "NVD"
    return "", "NVD"


def load_nvd_cve(path: Path, *, limit: int | None = None) -> list[Document]:
    """Parse an NVD CVE API 2.0 payload into one document per CVE.

    Also records CVSS base score/vector when present, and cross-references the
    CISA KEV flag so retrieval can distinguish "bad" from "actively exploited".
    """
    data = _read_json(path)
    records = data.get("vulnerabilities", []) if isinstance(data, dict) else data
    if not isinstance(records, list):
        logger.warning("%s: expected a list of vulnerabilities", path)
        return []

    docs: list[Document] = []
    for wrapper in records:
        if not isinstance(wrapper, dict):
            continue
        cve = wrapper.get("cve") or {}
        cve_id = str(cve.get("id", "")).strip()
        if not cve_id:
            continue
        description, desc_source = _pick_description(cve)

        severity, score, vector = "", None, ""
        for key, value in (cve.get("metrics") or {}).items():
            if not isinstance(value, list) or not value:
                continue
            entry = value[0].get("cvssData", {}) if isinstance(value[0], dict) else {}
            if entry:
                severity = str(entry.get("baseSeverity", "") or "")
                score = entry.get("baseScore")
                vector = str(entry.get("vectorString", "") or "")
                break

        refs = cve.get("references") or []
        ref_urls = [r.get("url") for r in refs[:5] if isinstance(r, dict) and r.get("url")]
        weaknesses = [
            str(w.get("value", ""))
            for w in (cve.get("weaknesses") or [])
            if isinstance(w, dict) and w.get("value")
        ]

        lines = [f"{cve_id}", description]
        if severity or score is not None:
            score_text = f"{score}" if score is not None else "unscored"
            lines.append(f"CVSS base score: {score_text} ({severity or 'severity unknown'}).")
        if vector:
            lines.append(f"CVSS vector: {vector}.")
        if weaknesses:
            lines.append(f"Weaknesses: {', '.join(weaknesses[:6])}.")
        if ref_urls:
            lines.append("References: " + "; ".join(ref_urls[:3]))

        docs.append(Document(
            text=clean_text("\n".join(x for x in lines if x)),
            doc_id=f"nvd_cve:{cve_id}",
            title=f"{cve_id} (NVD)",
            source="nvd_cve",
            source_type="dataset",
            url=KEV_URL_TEMPLATE.format(cve_id=cve_id),
            metadata={
                "cve_id": cve_id,
                "published": cve.get("published", ""),
                "last_modified": cve.get("lastModified", ""),
                "cvss_score": score,
                "cvss_severity": severity,
                "cvss_vector": vector,
                "cwes": weaknesses,
                "references": ref_urls,
                "description_source": desc_source,
                "authority": "NVD (National Vulnerability Database)",
            },
        ))
        if limit and len(docs) >= limit:
            break

    logger.info("loaded %d NVD CVE records from %s", len(docs), path.name)
    return docs


def _attack_ref_url(stix_id: str) -> str:
    """Map a STIX id to its ATT&CK page.

    STIX ids embed the technique as ``TA0001/T1059.001``; the public site
    uses ``T1059/001`` (the sub-technique suffix zero-padded to 3 digits).
    """
    if not stix_id:
        return ""
    tail = stix_id.split("--", 1)[-1]
    if not tail.startswith("T"):
        return ""
    if "." in tail:
        outer, _, inner = tail.partition(".")
        return ATTACK_URL_TEMPLATE.format(technique_id=f"{outer}/{inner.zfill(3)}")
    if len(tail) >= 5:
        return ATTACK_URL_TEMPLATE.format(technique_id=tail)
    return ""


def _stix_descriptions(obj: dict) -> str:
    for desc in obj.get("description", "") or []:
        value = str(desc).strip()
        if value:
            return value
    return ""


def load_mitre_attack(path: Path, *, limit: int | None = None,
                      include_software: bool = False) -> list[Document]:
    """Parse a MITRE ATT&CK Enterprise STIX 2.1 bundle into documents.

    Emits one document per *active* attack-pattern (technique) and, when
    ``include_software`` is set, per malware/tool. Revoked and deprecated
    objects are skipped - stale entries would otherwise be retrieved as if
    current, which is exactly the "conflicting information" failure mode the
    RAG design is meant to avoid.
    """
    bundle = _read_json(path)
    objects = bundle.get("objects", []) if isinstance(bundle, dict) else bundle
    if not isinstance(objects, list):
        logger.warning("%s: expected a STIX bundle with 'objects'", path)
        return []

    tactics_by_id: dict[str, str] = {}
    for obj in objects:
        if isinstance(obj, dict) and obj.get("type") == "x-mitre-tactic":
            name = obj.get("name") or (obj.get("x_mitre_shortname") or "")
            tactics_by_id[obj.get("id", "")] = str(name)

    docs: list[Document] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        obj_type = obj.get("type")
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        if obj_type == "attack-pattern":
            if obj.get("x_mitre_is_subtechnique") and not obj.get("description"):
                continue
        elif obj_type == "malware":
            # Malware/software objects are useful but secondary; only include
            # them when explicitly requested so techniques dominate the corpus.
            if not include_software:
                continue
        else:
            continue

        stix_id = obj.get("id", "")
        name = str(obj.get("name", "")).strip()
        description = _stix_descriptions(obj)
        if not name and not description:
            continue

        tactics = [
            tactics_by_id.get(r, r.split("--")[-1])
            for r in (obj.get("kill_chain_phases") or [])
            if isinstance(r, str)
        ]
        platforms = obj.get("x_mitre_platforms") or []
        # Prefer the publisher's own external reference (canonical URL);
        # fall back to deriving it from the external_id.
        display_id, reference_url = "", ""
        for ext in (obj.get("external_references") or []):
            if not isinstance(ext, dict):
                continue
            if ext.get("external_id"):
                display_id = str(ext["external_id"])
            if ext.get("url", "").startswith("https://attack.mitre.org/"):
                reference_url = str(ext["url"])
            if display_id and reference_url:
                break
        technique_id = display_id or _attack_ref_url(stix_id).rstrip("/").split("/")[-1]
        url = reference_url or _attack_ref_url(stix_id) or "https://attack.mitre.org/"

        header = f"{display_id or technique_id}: {name}".strip(": ")
        lines = [header]
        if tactics:
            lines.append(f"Tactic/phase: {', '.join(tactics)}.")
        if platforms:
            lines.append(f"Platforms: {', '.join(str(p) for p in platforms[:8])}.")
        if description:
            lines.append(description)

        docs.append(Document(
            text=clean_text("\n".join(lines)),
            doc_id=f"mitre_attack:{stix_id}",
            title=f"{display_id} {name}".strip(),
            source="mitre_attack",
            source_type="dataset",
            url=url,
            metadata={
                "technique_id": display_id or technique_id,
                "name": name,
                "tactics": tactics,
                "platforms": [str(p) for p in platforms[:12]],
                "stix_id": stix_id,
                "type": obj_type,
                "authority": "MITRE ATT&CK (knowledge base of adversary tactics and techniques)",
            },
        ))
        if limit and len(docs) >= limit:
            break

    logger.info("loaded %d MITRE ATT&CK objects from %s", len(docs), path.name)
    return docs


def load_jsonl(path: Path, *, limit: int | None = None,
               source: str = "jsonl") -> list[Document]:
    """Load a generic JSONL corpus of ``{instruction, response}`` pairs."""
    docs: list[Document] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            question = str(record.get("instruction") or record.get("question") or "").strip()
            answer = str(record.get("response") or record.get("answer") or "").strip()
            if not question or not answer:
                continue
            docs.append(Document(
                text=clean_text(f"Q: {question}\nA: {answer}"),
                title=question[:100],
                source=source,
                source_type="dataset",
                metadata={"question": question, "answer": answer},
            ))
            if limit and len(docs) >= limit:
                break
    logger.info("loaded %d JSONL records from %s", len(docs), path.name)
    return docs


#: Dispatch table used by the ingestion CLI/API.
LOADERS = {
    "cisa_kev": load_cisa_kev,
    "nvd_cve": load_nvd_cve,
    "mitre_attack": load_mitre_attack,
    "cyberllm_sft": load_jsonl,
}


def load_source(name: str, path: Path, *, limit: int | None = None) -> list[Document]:
    """Load documents for a registered dataset name."""
    loader = LOADERS.get(name)
    if loader is None:
        raise ValueError(
            f"No loader registered for '{name}'. Available: {sorted(LOADERS)}"
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    if loader is load_jsonl:
        return loader(path, limit=limit, source=name)
    return loader(path, limit=limit)


__all__ = [
    "load_cisa_kev", "load_nvd_cve", "load_mitre_attack", "load_jsonl",
    "load_source", "LOADERS",
]

