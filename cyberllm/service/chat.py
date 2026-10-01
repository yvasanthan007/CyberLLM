"""The cybersecurity chat service: retrieval + generation + citations.

This module owns the answer contract that the API and UI render:

* every factual claim is tied to a numbered citation;
* the response states which mode produced it (``demo`` vs a real model);
* weak or absent retrieval evidence is reported as such instead of being papered
  over with model prose;
* prompt-injection attempts found in retrieved documents are surfaced to the
  caller as warnings, not silently swallowed.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..core.config import AppConfig, get_settings
from ..core.errors import LLMUnavailableError, ValidationError
from ..core.logging_config import get_logger
from ..llm.base import SYSTEM_PREAMBLE, LLMBackend, LLMUnavailable, build_backend
from ..rag.chunking import Document
from ..rag.pipeline import RAGPipeline, RetrievedContext
from ..security.guards import sanitize_text

logger = get_logger("chat.service")

#: The instruction block wrapped around retrieved material.
ANSWER_INSTRUCTIONS = """Answer the question using ONLY the reference material below.

Rules:
- Quote or closely paraphrase the sources for factual claims and cite them
  by their number, e.g. [1].
- Add a short section "Analysis (not from sources)" for your own reasoning,
  clearly labelled as such.
- If the sources do not answer the question, say exactly what is missing
  instead of guessing. Never invent CVE ids, CVSS scores or technique ids.
- If the question asks you to execute something, scan a network, or modify a
  system, explain that you cannot and describe the authorised alternative.

Reference material:
{context}

Question: {question}"""


@dataclass
class ChatTurn:
    """One question/answer exchange, ready for the API and the history store."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    question: str = ""
    answer: str = ""
    mode: str = "unknown"
    model: str = ""
    is_demo: bool = False
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieved_count: int = 0
    weak_evidence: bool = False
    injection_warnings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "mode": self.mode,
            "model": self.model,
            "is_demo": self.is_demo,
            "citations": self.citations,
            "retrieved_count": self.retrieved_count,
            "weak_evidence": self.weak_evidence,
            "injection_warnings": self.injection_warnings,
            "warnings": self.warnings,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "created_at": self.created_at,
        }


class ChatService:
    """Retrieval-augmented chat over the CyberLLM knowledge base."""

    def __init__(self, settings: AppConfig | None = None,
                 pipeline: RAGPipeline | None = None,
                 backend: LLMBackend | None = None) -> None:
        self.settings = settings or get_settings()
        self.rag = pipeline or RAGPipeline(self.settings)
        self._backend = backend

    # ------------------------------------------------------------------
    @property
    def backend(self) -> LLMBackend:
        """Lazily build the backend so importing the service is cheap."""
        if self._backend is None:
            self._backend = build_backend(self.settings)
        return self._backend

    def is_demo(self) -> bool:
        return not self.backend.is_real_model

    def _validate(self, question: str) -> str:
        limit = self.settings.security.max_prompt_chars
        cleaned = sanitize_text(question or "", max_chars=limit + 1)
        if not cleaned:
            raise ValidationError("Question must not be empty.")
        if len(cleaned) > limit:
            raise ValidationError(
                f"Question is too long ({len(cleaned)} chars); the limit is {limit}."
            )
        return cleaned

    # ------------------------------------------------------------------
    def retrieve(self, question: str, top_k: int | None = None) -> RetrievedContext:
        """Expose retrieval on its own (used by ``/retrieve``)."""
        return self.rag.build_context(self._validate(question), top_k)

    def ask(self, question: str, *, history: Sequence[tuple[str, str]] = (),
            top_k: int | None = None,
            use_rag: bool = True) -> ChatTurn:
        """Answer a question, with citations and an explicit mode label."""
        started = time.monotonic()
        question = self._validate(question)
        turn = ChatTurn(question=question, created_at=_now())

        context = (self.rag.build_context(question, top_k) if use_rag
                   else RetrievedContext(question=question))
        turn.retrieved_count = len(context.results)
        turn.citations = context.citations
        turn.weak_evidence = context.weak_evidence
        turn.injection_warnings = list(context.injection_warnings)
        turn.warnings.extend(context.injection_warnings)

        if not context.has_context:
            turn.warnings.append(
                "No indexed source matched this question. The answer below is "
                "general guidance, not sourced from a retrieved document."
            )

        prompt = ANSWER_INSTRUCTIONS.format(
            context=context.context_text or "(no reference material was retrieved)",
            question=question,
        )

        try:
            result = self.backend.generate(
                prompt, system=SYSTEM_PREAMBLE, history=history
            )
        except LLMUnavailableError as exc:
            # Re-raise with the backend's setup hint so the UI can show it.
            hint = getattr(exc, "setup_hint", "")
            raise LLMUnavailable(
                f"{exc.message}", setup_hint=hint or "No LLM backend is configured."
            ) from exc

        turn.answer = result.text
        turn.mode = result.mode
        turn.model = result.model
        turn.is_demo = result.is_demo
        turn.warnings.extend(result.warnings)
        if turn.is_demo:
            turn.warnings.append(
                "Response produced in demo mode by a template, not by a language model."
            )
        if context.has_context:
            turn.answer += context.citation_block()
        turn.elapsed_seconds = time.monotonic() - started
        logger.info(
            "answered in %.2fs via %s (%d citations, weak=%s)",
            turn.elapsed_seconds, turn.mode, len(turn.citations), turn.weak_evidence,
        )
        return turn

    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Capability report for the health endpoint and the UI banner."""
        try:
            backend = self.backend
            backend_info = backend.describe()
        except LLMUnavailable as exc:
            backend_info = {
                "provider": "unavailable", "available": False,
                "error": exc.message, "setup_hint": exc.setup_hint,
                "is_real_model": False,
            }
        kb = self.rag.status()
        return {
            "llm": backend_info,
            "demo_mode": self.is_demo(),
            "knowledge_base": kb,
            "ready": kb.get("documents", 0) > 0 and not self.is_demo(),
            "limitations": [
                "Retrieval uses a lexical hashing embedder unless "
                "sentence-transformers is configured; recall is keyword-driven.",
                "Classifier outputs are probabilistic - always corroborate with "
                "the cited source before acting.",
            ],
        }


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = ["ChatService", "ChatTurn", "ANSWER_INSTRUCTIONS"]

