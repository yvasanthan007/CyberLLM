"""Typed exception hierarchy for CyberLLM.

Every error raised by the platform derives from :class:`CyberLLMError` so the
API layer can map failures onto meaningful HTTP status codes without leaking
implementation details to clients.
"""

from __future__ import annotations


class CyberLLMError(Exception):
    """Base class for all CyberLLM errors."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(CyberLLMError):
    """Raised when configuration is missing or invalid."""

    status_code = 500
    code = "configuration_error"


class NotConfiguredError(CyberLLMError):
    """A capability was requested that has not been configured.

    The API maps this to HTTP 503 so the UI can show a setup message instead
    of fabricating a result.
    """

    status_code = 503
    code = "not_configured"


class ValidationError(CyberLLMError):
    """User supplied input failed validation."""

    status_code = 422
    code = "validation_error"


class DatasetError(CyberLLMError):
    """Base class for dataset problems."""


class DatasetNotFoundError(DatasetError):
    status_code = 404
    code = "dataset_not_found"


class DatasetDownloadError(DatasetError):
    """A download failed after exhausting retries."""

    status_code = 502
    code = "dataset_download_failed"


class DatasetValidationError(DatasetError):
    """A downloaded artefact failed schema / signature checks."""

    status_code = 422
    code = "dataset_validation_failed"


class ModelNotAvailableError(CyberLLMError):
    """A required model artefact is missing or unloadable."""

    status_code = 503
    code = "model_not_available"


class LLMUnavailableError(CyberLLMError):
    """No language model provider is reachable or configured."""

    status_code = 503
    code = "llm_unavailable"


class VectorStoreError(CyberLLMError):
    """The retrieval store could not be opened or queried."""

    status_code = 503
    code = "vector_store_unavailable"


class PayloadTooLargeError(CyberLLMError):
    """An upload or request body exceeded the configured limit."""

    status_code = 413
    code = "payload_too_large"


__all__ = [
    "CyberLLMError",
    "ConfigurationError",
    "NotConfiguredError",
    "ValidationError",
    "DatasetError",
    "DatasetNotFoundError",
    "DatasetDownloadError",
    "DatasetValidationError",
    "ModelNotAvailableError",
    "LLMUnavailableError",
    "VectorStoreError",
    "PayloadTooLargeError",
]
