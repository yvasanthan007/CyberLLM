"""Provider-independent LLM interface and concrete backends.

CyberLLM never hardcodes a model. Four backends implement one interface:

``transformers``  Local Hugging Face weights (small models preferred)
``ollama``         A local Ollama daemon (no API key, no cloud egress)
``openai``         Hosted OpenAI-compatible endpoint (key from the env only)
``demo``           **No model at all** - a clearly labelled, retrieval-backed
                   template response

The distinction that matters for honesty: ``demo`` mode is *not* a language
model. It restates retrieved source material and says so. The API surfaces
``mode`` on every response and the UI renders a banner, so a reader can never
mistake templated output for model reasoning.
"""

from __future__ import annotations

from .base import (
    GenerationResult,
    LLMBackend,
    LLMUnavailable,
    SYSTEM_PREAMBLE,
    build_backend,
    probe,
)
from .demo import DemoBackend

__all__ = [
    "LLMBackend",
    "GenerationResult",
    "LLMUnavailable",
    "SYSTEM_PREAMBLE",
    "build_backend",
    "probe",
    "DemoBackend",
]
