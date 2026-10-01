"""Offline phishing URL analysis.

**The submitted URL is never fetched.** No DNS lookup, no HTTP request, no
redirect following, no WHOIS query, no third-party reputation service. Every
feature is computed lexically from the URL string alone. This is both a
privacy control (a user pasting a suspicious URL does not leak it to us or to
a third party) and a safety control (an analyst must never detonate a live
phishing page by clicking it).

The classifier is a baseline over those lexical features. Its precision and
recall are bounded by the training data's era and by the arms race against
detection; a score is a signal for triage, not a verdict.
"""

from .features import (
    DISCLAIMER,
    FEATURE_NAMES,
    PhishingVerdict,
    url_to_vector,
    heuristic_score,
)
from .service import PhishingDetector, train_phishing, evaluate_phishing

__all__ = [
    "url_to_vector", "heuristic_score", "PhishingVerdict", "FEATURE_NAMES",
    "DISCLAIMER", "PhishingDetector", "train_phishing", "evaluate_phishing",
]

