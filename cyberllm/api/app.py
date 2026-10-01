"""FastAPI application exposing the CyberLLM services.

Security posture
----------------
The service is designed for **localhost**. It ships without authentication or
TLS, so binding it to a routable interface is refused unless
``CYBERLLM_API_KEY`` is set, and even then a reverse proxy with TLS is
strongly recommended. The reasoning is stated in the startup log rather than
left for the operator to discover.

Other controls that apply to every endpoint:

* strict Pydantic validation with explicit upper bounds on body size, array
  length and string length;
* a declared ``Content-Length`` limit checked in middleware, so an oversized
  body is rejected before it is read into memory;
* a uniform error envelope, with domain errors mapped onto their semantic
  HTTP status and no internal traceback ever returned to a client.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Header, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from ..core.config import AppConfig, get_settings
from ..core.errors import CyberLLMError, NotConfiguredError, PayloadTooLargeError
from ..core.logging_config import configure_logging, get_logger
from ..security.guards import check_content_length
from .schemas import (
    ChatRequest, ChatResponse, DetectRequest, DetectResponse, HealthResponse,
    MalwareRequest, MalwareResponse, PhishingRequest, PhishingResponse,
    RetrieveRequest, RetrieveResponse, VulnerabilityResponse,
)

logger = get_logger("api")
settings = get_settings()
configure_logging(settings.log_level, settings.log_format)

app = FastAPI(
    title="CyberLLM API",
    version=__version__,
    description=(
        "Defensive-security LLM platform: retrieval-augmented chat over threat "
        "intelligence, network intrusion classification and offline phishing "
        "URL analysis. Classifier outputs are probabilistic analyst aids, not "
        "guarantees of detection."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.security.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# ---------------------------------------------------------------------------
# Dependencies: one instance per process, created lazily
# ---------------------------------------------------------------------------
class Services:
    """Lazily-constructed singletons shared across requests.

    Construction is deferred so importing the app never loads model weights or
    opens the vector store, which keeps startup and the test suite fast.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._chat: Any = None
        self._detector: Any = None
        self._phishing: Any = None
        self._malware: Any = None

    @property
    def chat(self) -> Any:
        if self._chat is None:
            from ..service.chat import ChatService
            self._chat = ChatService(self.config)
        return self._chat

    def _nids_dir(self) -> Any:
        root = self.config.paths.models_dir / "nids"
        for name in ("lightgbm", "random_forest"):
            candidate = root / name
            if (candidate / "model.pkl").exists():
                return candidate
        return None

    @property
    def detector(self) -> Any:
        if self._detector is None:
            from ..detection.inference import IntrusionDetector
            self._detector = IntrusionDetector(self._nids_dir())
        return self._detector

    @property
    def phishing(self) -> Any:
        if self._phishing is None:
            from ..phishing import PhishingDetector
            self._phishing = PhishingDetector()
        return self._phishing

    @property
    def malware(self) -> Any:
        if self._malware is None:
            from ..malware import MalwareAnalyzer
            self._malware = MalwareAnalyzer()
        return self._malware


services = Services(settings)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforce the shared secret when one is configured.

    No-op when ``CYBERLLM_API_KEY`` is unset, because the documented default
    deployment is localhost-only. The comparison is constant-time so the key
    cannot be recovered from response timing.
    """
    import hmac

    expected = settings.security.api_key
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise NotConfiguredError(
            "Missing or invalid X-API-Key header.",
            details={"setup_hint": "Send the value of CYBERLLM_API_KEY."},
        )


# ---------------------------------------------------------------------------
# Middleware and error handling
# ---------------------------------------------------------------------------
@app.middleware("http")
async def limit_body_size(request: Request, call_next: Any) -> Any:
    """Reject oversized bodies from the declared Content-Length, before reading."""
    try:
        check_content_length(
            request.headers.get("content-length"), settings.security.max_request_bytes
        )
    except PayloadTooLargeError as exc:
        return JSONResponse(
            status_code=413,
            content={"error": exc.message, "code": exc.code, "details": exc.details},
        )
    return await call_next(request)


@app.exception_handler(CyberLLMError)
async def domain_error_handler(_: Request, exc: CyberLLMError) -> JSONResponse:
    """Map domain errors onto semantic HTTP codes with a uniform envelope."""
    logger.warning("%s -> HTTP %s: %s", exc.code, exc.status_code, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "code": exc.code, "details": exc.details},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Never leak a traceback; log it server-side with a correlation id."""
    reference = f"req-{id(exc):x}"
    logger.exception("unhandled error on %s %s (%s)", request.method,
                     request.url.path, reference)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error.", "code": "internal_error",
                 "details": {"reference": reference}},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["meta"], summary="Service banner")
def root() -> dict[str, Any]:
    """Identify the service and point at the useful entry points."""
    return {
        "service": "CyberLLM API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "notice": (
            "Defensive use only. Classifier outputs are probabilistic analyst "
            "aids and must be corroborated with the cited sources."
        ),
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"],
         summary="Capability and readiness report")
def health() -> HealthResponse:
    """Report per-capability status.

    Returns 200 with ``status="degraded"`` when a capability is missing rather
    than failing the request, so a monitor can distinguish "the service is
    broken" from "the service is fine but unconfigured".
    """
    from ..llm import probe

    chat_status = services.chat.status()
    kb = chat_status["knowledge_base"]
    nids = services.detector.status()
    phishing = services.phishing.status()
    malware = services.malware.status()

    ready = bool(kb.get("documents")) and bool(nids.get("available"))
    return HealthResponse(
        status="ok" if ready else "degraded",
        version=__version__,
        demo_mode=chat_status["demo_mode"],
        config=settings.redacted_summary(),
        llm=chat_status["llm"],
        knowledge_base=kb,
        intrusion_detection=nids,
        phishing=phishing,
        limitations=chat_status["limitations"] + [
            "Malware analysis is feature-based only: no sample is ever executed.",
            "Phishing analysis is offline: submitted URLs are never fetched.",
        ],
    )



@app.post("/chat", response_model=ChatResponse, tags=["assistant"],
          dependencies=[Depends(require_api_key)], summary="Cited cybersecurity Q&A")
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question using the knowledge base, with numbered citations.

    The response always states which mode produced the text. When no language
    model is configured the answer is explicitly marked ``is_demo`` so a client
    cannot mistake a template for model reasoning.
    """
    turn = services.chat.ask(
        request.question, top_k=request.top_k, use_rag=request.use_rag
    )
    return ChatResponse(
        answer=turn.answer,
        mode=turn.mode,
        model=turn.model,
        is_demo=turn.is_demo,
        citations=turn.citations,
        retrieved_count=turn.retrieved_count,
        weak_evidence=turn.weak_evidence,
        warnings=turn.warnings,
        injection_warnings=turn.injection_warnings,
        elapsed_seconds=turn.elapsed_seconds,
    )


@app.post("/retrieve", response_model=RetrieveResponse, tags=["knowledge-base"],
          dependencies=[Depends(require_api_key)], summary="Retrieval only")
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """Return matching source documents without generating an answer.

    Useful for building a UI that shows raw evidence, and for testing
    retrieval quality independently of the language model.
    """
    context = services.chat.rag.build_context(request.query, request.top_k)
    return RetrieveResponse(
        query=request.query,
        results=[r.as_dict() for r in context.results],
        citations=context.citations,
        weak_evidence=context.weak_evidence,
        injection_warnings=context.injection_warnings,
    )


@app.get("/vulnerability/{cve_id}", response_model=VulnerabilityResponse,
         tags=["knowledge-base"], dependencies=[Depends(require_api_key)],
         summary="Look up a CVE from indexed threat intelligence")
def vulnerability(cve_id: str) -> VulnerabilityResponse:
    """Assemble a vulnerability record from the indexed sources.

    Served entirely from the local knowledge base - this endpoint does not
    call out to NVD, so it stays fast and works offline. ``found=false`` means
    the id is not in the indexed data, which is stated plainly rather than
    answered with a guess.
    """
    from ..core.errors import ValidationError as _ValidationError

    cve_id = cve_id.strip().upper()
    if not cve_id.startswith("CVE-") or not cve_id[4:8].isdigit():
        raise _ValidationError(
            f"'{cve_id}' is not a CVE identifier.",
            details={"expected_format": "CVE-YYYY-NNNN"},
        )

    documents = services.chat.rag.store.find_by_identifier(cve_id)
    record = VulnerabilityResponse(cve_id=cve_id, found=bool(documents))
    for doc in documents:
        meta = doc.metadata or {}
        record.sources.append({
            "source": doc.source, "title": doc.title, "url": doc.url,
            "authority": meta.get("authority", ""),
        })
        if not record.title:
            record.title = doc.title
        if doc.source == "cisa_kev":
            record.in_kev = True
            record.due_date = meta.get("due_date", "")
            record.date_added = meta.get("date_added", "")
            record.vendor = meta.get("vendor", "")
            record.product = meta.get("product", "")
            record.ransomware_use = meta.get("ransomware_use", "")
            record.cwes = [str(c) for c in meta.get("cwes", [])]
        elif doc.source == "nvd_cve":
            score = meta.get("cvss_score")
            record.cvss_score = float(score) if isinstance(score, (int, float)) else None
            record.cvss_severity = meta.get("cvss_severity", "")
            record.references = [str(r) for r in meta.get("references", [])][:5]
            record.description = " ".join(doc.text.split())[:1200]
            record.cwes = record.cwes or [str(c) for c in meta.get("cwes", [])]
    return record



@app.post("/detect/intrusion", response_model=DetectResponse, tags=["detection"],
          dependencies=[Depends(require_api_key)],
          summary="Classify authorised network-flow records")
def detect_intrusion(request: DetectRequest) -> DetectResponse:
    """Classify one or more flow records.

    Requires a trained model; without one the endpoint returns HTTP 503 with
    the training command rather than a placeholder score.
    """
    predictions = [services.detector.predict(record.to_dict()).as_dict()
                   for record in request.records]
    return DetectResponse(
        predictions=predictions,
        model=predictions[0]["model"] if predictions else "",
        is_synthetic_model=any(p["is_synthetic_model"] for p in predictions),
        count=len(predictions),
    )


@app.post("/phishing/analyse", response_model=PhishingResponse, tags=["detection"],
          dependencies=[Depends(require_api_key)], summary="Offline phishing analysis")
def phishing_analyse(request: PhishingRequest) -> PhishingResponse:
    """Score a URL without ever visiting it.

    The handler performs no DNS lookup, HTTP request or WHOIS query; the
    response states this explicitly so the guarantee is auditable.
    """
    return PhishingResponse(**services.phishing.analyse(request.url).as_dict())


@app.post("/malware/analyse", response_model=MalwareResponse, tags=["detection"],
          dependencies=[Depends(require_api_key)],
          summary="Classify pre-extracted malware features")
def malware_analyse(request: MalwareRequest) -> MalwareResponse:
    """Classify pre-extracted memory features.

    No sample is executed, loaded or emulated. Without a trained model this
    returns HTTP 503 with the acquisition and training steps.
    """
    results = [services.malware.analyse(row.to_dict()).as_dict()
               for row in request.features]
    return MalwareResponse(
        predictions=results, model=results[0]["model"] if results else "",
    )


@app.get("/datasets", tags=["meta"], dependencies=[Depends(require_api_key)],
         summary="Dataset inventory and acquisition status")
def datasets() -> dict[str, Any]:
    """Report every registered dataset with its real on-disk status."""
    from ..datasets.manager import DatasetManager

    manager = DatasetManager()
    return {
        "summary": manager.manifest.summary(),
        "datasets": manager.inventory(),
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

