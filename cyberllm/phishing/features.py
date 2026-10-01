"""Lexical URL features and the heuristic risk score.

Pure functions, no I/O, no network. This module is the safety-critical core
of phishing analysis: it must be impossible for it to cause a request to be
made, so it imports nothing that can open a socket.

``heuristic_score`` provides a transparent, model-free risk score. It exists
so the platform can still give a defensible answer before any classifier is
trained, and so a classifier's output can be cross-checked against an
explainable baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..security.guards import extract_url_features, url_risk_signals

DISCLAIMER = (
    "Offline lexical analysis. The URL was not visited, resolved or submitted "
    "to any third party. A phishing score is a triage signal from a model "
    "trained on historical URL data - it is neither a guarantee nor a "
    "substitute for analyst review, and false positives on unusual but "
    "legitimate URLs are expected."
)

#: Feature vector order. Persisted with the model so inference cannot drift.
FEATURE_NAMES: tuple[str, ...] = (
    "url_length", "hostname_length", "path_length", "subdomain_depth",
    "digit_ratio_host", "special_char_ratio", "suspicious_token_count",
    "has_https", "has_at_symbol", "has_ip_host", "has_punycode",
    "has_homoglyph", "is_shortener",
)


def url_to_vector(url: str) -> tuple[list[float], list[dict[str, Any]]]:
    """Validate a URL and turn it into a numeric feature vector.

    Raises :class:`ValidationError` for anything that is not a well-formed
    http(s) URL - malformed input must fail loudly rather than be scored.
    """
    features = extract_url_features(url)
    signals = url_risk_signals(features)
    data = features.as_dict()
    return [float(data[name]) for name in FEATURE_NAMES], signals


def heuristic_score(signals: list[dict[str, Any]]) -> float:
    """Transparent risk score in ``[0, 1]`` from the rule signals.

    The rule weights are combined with a saturating sum (``1 - prod(1-w)``)
    so several weak indicators cannot exceed certainty, and the result is
    capped at 0.99: a lexical heuristic can raise suspicion but cannot prove
    intent, so it never claims to.
    """
    if not signals:
        return 0.0
    complement = 1.0
    for signal in signals:
        weight = float(signal.get("weight", 0.0))
        complement *= max(0.0, 1.0 - min(weight, 0.95))
    return round(min(0.99, 1.0 - complement), 6)


@dataclass
class PhishingVerdict:
    """A phishing assessment with its reasoning."""

    url: str
    is_phishing: bool
    score: float
    threshold: float
    signals: list[dict[str, Any]] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    model: str = "lexical-heuristic"
    is_synthetic_model: bool = False
    method: str = "heuristic"
    analysed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    disclaimer: str = DISCLAIMER
    #: Recorded explicitly so the UI can state that nothing left the process.
    network_access: str = "none (fully offline lexical analysis)"

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "is_phishing": self.is_phishing,
            "score": round(self.score, 6),
            "threshold": round(self.threshold, 6),
            "risk_signals": self.signals,
            "features": self.features,
            "model": self.model,
            "method": self.method,
            "is_synthetic_model": self.is_synthetic_model,
            "analysed_at": self.analysed_at,
            "disclaimer": self.disclaimer,
            "network_access": self.network_access,
        }


def analyse_heuristically(url: str, threshold: float = 0.5) -> PhishingVerdict:
    """Score a URL with no model at all - the always-available baseline."""
    vector, signals = url_to_vector(url)
    score = heuristic_score(signals)
    return PhishingVerdict(
        url=url,
        is_phishing=score >= threshold,
        score=score,
        threshold=threshold,
        signals=signals,
        features=dict(zip(FEATURE_NAMES, vector)),
        method="heuristic",
    )


__all__ = [
    "FEATURE_NAMES", "DISCLAIMER", "PhishingVerdict",
    "url_to_vector", "heuristic_score", "analyse_heuristically",
]
