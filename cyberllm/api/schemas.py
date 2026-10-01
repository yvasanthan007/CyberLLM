"""Pydantic request/response schemas for the CyberLLM API.

Validation is deliberately strict and the limits are explicit, because this
API accepts untrusted input: a URL from an analyst, a flow record lifted from
a SIEM, an uploaded CSV. Pydantic enforces shape and bounds before any handler
runs, so handlers never have to defend themselves ad hoc.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

MAX_URL_LENGTH = 2048
MAX_QUESTION_CHARS = 16_000
MAX_BATCH_RECORDS = 100
MAX_TOP_K = 25


class HealthResponse(BaseModel):
    """Overall service health and per-capability readiness."""

    status: str = Field(description="'ok' or 'degraded'")
    version: str
    demo_mode: bool
    config: str
    llm: dict[str, Any]
    knowledge_base: dict[str, Any]
    intrusion_detection: dict[str, Any]
    phishing: dict[str, Any]
    limitations: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    """One retrieved source backing part of an answer."""

    index: int
    source: str
    title: str
    url: str = ""
    score: float
    doc_id: str


class ChatRequest(BaseModel):
    """A question for the cybersecurity assistant."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)
    use_rag: bool = True
    session_id: str | None = Field(default=None, max_length=64)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class ChatResponse(BaseModel):
    """A cited answer, with the mode that produced it."""

    answer: str
    mode: str = Field(description="transformers | ollama | openai | demo")
    model: str
    is_demo: bool
    citations: list[Citation] = Field(default_factory=list)
    retrieved_count: int = 0
    weak_evidence: bool = False
    warnings: list[str] = Field(default_factory=list)
    injection_warnings: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


class RetrieveRequest(BaseModel):
    """A retrieval-only query (no generation)."""

    query: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


class RetrieveResponse(BaseModel):
    """Retrieved chunks with provenance and scores."""

    query: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    weak_evidence: bool = False
    injection_warnings: list[str] = Field(default_factory=list)


class VulnerabilityResponse(BaseModel):
    """A vulnerability record assembled from the indexed threat-intel sources."""

    cve_id: str
    found: bool
    in_kev: bool = Field(
        default=False, description="Listed in CISA KEV (actively exploited)")
    ransomware_use: str = ""
    title: str = ""
    description: str = ""
    cvss_score: float | None = None
    cvss_severity: str = ""
    cwes: list[str] = Field(default_factory=list)
    due_date: str = ""
    date_added: str = ""
    vendor: str = ""
    product: str = ""
    references: list[str] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    disclaimer: str = (
        "Assembled from indexed source records. Verify against the linked "
        "primary source before acting on severity or exploitability."
    )


class FlowRecord(BaseModel):
    """An authorised network-flow record submitted for classification.

    Extra columns are allowed on purpose: real flow exports carry many more
    fields than any model consumes, and rejecting them would make the endpoint
    unusable. Unused columns are reported back as ``ignored_columns``.
    """

    model_config = {"extra": "allow"}

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class DetectRequest(BaseModel):
    """One or more flow records to classify."""

    records: list[FlowRecord] = Field(min_length=1, max_length=MAX_BATCH_RECORDS)


class DetectResponse(BaseModel):
    """Classification results for the submitted records."""

    predictions: list[dict[str, Any]]
    model: str
    is_synthetic_model: bool
    count: int


class PhishingRequest(BaseModel):
    """A URL to analyse. It is never fetched."""

    url: str = Field(min_length=8, max_length=MAX_URL_LENGTH)

    @field_validator("url")
    @classmethod
    def _looks_like_url(cls, value: str) -> str:
        value = value.strip()
        if not value.lower().startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return value


class PhishingResponse(BaseModel):
    """An offline phishing assessment."""

    url: str
    is_phishing: bool
    score: float
    threshold: float
    risk_signals: list[dict[str, Any]] = Field(default_factory=list)
    features: dict[str, Any] = Field(default_factory=dict)
    model: str
    method: str
    is_synthetic_model: bool = False
    network_access: str
    disclaimer: str


class MalwareFeatures(BaseModel):
    """Volatile-memory feature row for offline malware classification."""

    model_config = {"extra": "allow"}

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class MalwareRequest(BaseModel):
    """A memory-feature row. Malware samples are never executed."""

    features: list[MalwareFeatures] = Field(min_length=1, max_length=50)


class MalwareResponse(BaseModel):
    """Malware classification results from pre-extracted features only."""

    predictions: list[dict[str, Any]]
    model: str
    execution_policy: str = (
        "No sample was executed, loaded or emulated. Classification uses only "
        "the pre-extracted numeric features supplied with the request."
    )
    available: bool = True
    setup_hint: str = ""


class ErrorResponse(BaseModel):
    """Uniform error body for every failure mode."""

    error: str
    code: str
    details: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "HealthResponse", "ChatRequest", "ChatResponse", "Citation",
    "RetrieveRequest", "RetrieveResponse", "VulnerabilityResponse",
    "FlowRecord", "DetectRequest", "DetectResponse",
    "PhishingRequest", "PhishingResponse",
    "MalwareFeatures", "MalwareRequest", "MalwareResponse", "ErrorResponse",
]

