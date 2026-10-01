"""Inference facade for network intrusion detection.

Takes an authorised network-flow record (or a batch), validates it, maps the
supplied columns onto the exact feature order the model was trained with, and
returns a prediction with an explanation.

Three properties matter here:

* **Feature-order safety.** Tree ensembles are order-sensitive; a record whose
  columns are silently reordered produces a confident, wrong answer. Columns
  are therefore matched by name and anything missing is reported.
* **Calibrated uncertainty.** The response carries the predicted class, the
  class probabilities, and an explicit statement that this is a probabilistic
  estimate requiring analyst review.
* **Missing-model honesty.** With no trained model the service raises a typed
  error carrying setup instructions instead of returning a made-up result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..core.errors import ModelNotAvailableError, ValidationError
from ..core.logging_config import get_logger
from ..security.guards import validate_flow_record
from .preprocess import map_columns
from .train import LoadedModel, load_bundle

logger = get_logger("detection.inference")

#: Shown with every prediction so a result is never read as a verdict.
DISCLAIMER = (
    "This is a probabilistic classification from a supervised model trained on "
    "labelled lab traffic. It is an analyst aid, not a detection guarantee. "
    "Corroborate with the raw flow before escalating or blocking."
)


@dataclass
class Prediction:
    """One flow record's verdict, with evidence."""

    predicted_class: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    top_features: list[dict[str, Any]] = field(default_factory=list)
    missing_features: list[str] = field(default_factory=list)
    ignored_columns: list[str] = field(default_factory=list)
    model: str = ""
    algorithm: str = ""
    is_synthetic_model: bool = False
    disclaimer: str = DISCLAIMER

    def as_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 6),
            "probabilities": {k: round(v, 6) for k, v in self.probabilities.items()},
            "top_features": self.top_features,
            "missing_features": self.missing_features,
            "ignored_columns": self.ignored_columns,
            "model": self.model,
            "algorithm": self.algorithm,
            "is_synthetic_model": self.is_synthetic_model,
            "disclaimer": self.disclaimer,
        }


class IntrusionDetector:
    """Loads a trained bundle once and answers flow queries."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory else None
        self._model: LoadedModel | None = None

    # ------------------------------------------------------------------
    @property
    def model(self) -> LoadedModel:
        """Lazily load so importing this module never requires an artefact."""
        if self._model is None:
            if self.directory is None:
                raise ModelNotAvailableError(
                    "No intrusion-detection model is configured.",
                    details={"setup_hint":
                             "Train one first: `python -m cyberllm.cli models train "
                             "--task nids --data <flow.csv>`"},
                )
            self._model = load_bundle(self.directory)
        return self._model

    def is_ready(self) -> bool:
        return bool(self._model or (self.directory and
                                    (self.directory / "model.pkl").exists()))

    def _align(self, record: dict) -> tuple[list[float], list[str], list[str]]:
        """Project a user record onto the trained feature order.

        Returns ``(vector, missing, ignored)``. Missing features become 0.0
        *and are reported*, so a caller can see the prediction rests on a
        partial record instead of assuming completeness.
        """
        model = self.model
        mapping = map_columns(record.keys())
        by_canonical: dict[str, Any] = {}
        consumed: set[str] = set()
        for canonical, original in mapping.items():
            by_canonical[canonical] = record.get(original)
            consumed.add(original)

        vector: list[float] = []
        missing: list[str] = []
        for feature in model.features:
            value = by_canonical.get(feature)
            if value is None:
                vector.append(0.0)
                missing.append(feature)
                continue
            try:
                vector.append(float(value))
            except (TypeError, ValueError):
                vector.append(0.0)
                missing.append(feature)

        return vector, missing, sorted(set(record) - consumed)

    def _explain(self, vector: list[float], k: int = 5) -> list[dict[str, Any]]:
        """Per-feature contributions, from tree importances where available."""
        model = self.model
        importances = getattr(model.estimator, "feature_importances_", None)
        if importances is None or not model.features:
            return []
        rows = [
            {"feature": name, "value": float(value),
             "importance": round(float(imp), 6)}
            for name, value, imp in zip(model.features, vector, importances)
        ]
        rows.sort(key=lambda item: -item["importance"])
        return rows[:k]

    # ------------------------------------------------------------------
    def predict(self, record: dict) -> Prediction:
        """Classify a single authorised flow record."""
        cleaned = validate_flow_record(record)
        model = self.model
        if not model.features:
            raise ModelNotAvailableError(
                "The loaded model has no recorded feature list; retrain it."
            )

        vector, missing, ignored = self._align(cleaned)
        if len(missing) == len(model.features):
            raise ValidationError(
                "None of the submitted fields match the model's features. "
                f"Expected any of: {', '.join(model.features[:10])}",
                details={"expected_features": model.features[:30]},
            )

        import numpy as np

        matrix = np.asarray([vector], dtype="float32")
        predicted = int(model.estimator.predict(matrix)[0])
        classes = model.classes or [str(i) for i in range(len(model.estimator.classes_))]
        label = classes[predicted] if predicted < len(classes) else str(predicted)

        probabilities: dict[str, float] = {}
        if hasattr(model.estimator, "predict_proba"):
            for name, prob in zip(classes, model.estimator.predict_proba(matrix)[0]):
                probabilities[name] = float(prob)

        return Prediction(
            predicted_class=label,
            confidence=probabilities.get(label, 0.0),
            probabilities=probabilities,
            top_features=self._explain(vector),
            missing_features=missing,
            ignored_columns=ignored,
            model=model.card.name,
            algorithm=model.card.algorithm,
            is_synthetic_model=model.card.is_synthetic,
        )

    def predict_batch(self, records: Sequence[dict]) -> list[Prediction]:
        """Classify a list of records."""
        if not records:
            raise ValidationError("No records supplied.")
        return [self.predict(record) for record in records]

    def status(self) -> dict[str, Any]:
        """Capability report for /health and the UI."""
        if not self.is_ready():
            return {
                "available": False,
                "reason": "no trained model artefact",
                "setup_hint": (
                    "Train one: `python -m cyberllm.cli models train --task nids "
                    "--data <flow.csv>`"
                ),
            }
        try:
            info = self.model.describe()
        except ModelNotAvailableError as exc:
            return {"available": False, "reason": exc.message}
        info["available"] = True
        return info


__all__ = ["IntrusionDetector", "Prediction", "DISCLAIMER"]

