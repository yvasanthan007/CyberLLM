"""Training, evaluation and inference for the intrusion detector.

Two baselines are compared deliberately, because "we used a random forest" is
not an evaluation:

``random_forest``  Strong, fast, interpretable feature importances.
``lightgbm``       Gradient-boosted trees; usually the better accuracy, but
                   only if ``lightgbm`` is installed (it is in this
                   environment - the code degrades gracefully if not).

Every training run writes a self-describing artefact bundle: the fitted model,
the preprocessing summary, the metrics, and reproducibility metadata (seed,
library versions, source hash). Nothing is claimed that is not in there.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.errors import ModelNotAvailableError
from ..core.logging_config import get_logger
from .preprocess import PreparedData, SchemaReport

logger = get_logger("detection.train")

ARTIFACT_VERSION = "1.0"

#: Written into every model card. Accuracy on one lab dataset is not a claim
#: that the model detects every attack.
DEFAULT_LIMITATIONS = [
    "A classifier estimates a probability; it does not guarantee detection. "
    "Expect false positives and false negatives on real traffic.",
    "Metrics reflect the training distribution. A lab dataset is not a "
    "production network: expect degradation under concept drift.",
    "Verify every alert against the underlying evidence before acting.",
    "Classifiers trained on encrypted or anonymised flows (e.g. some "
    "published CSV releases) see less signal than production captures.",
]


@dataclass
class ModelCard:
    """Everything needed to interpret and reproduce one trained model."""

    name: str
    algorithm: str
    task: str
    dataset: str
    dataset_sha256: str = ""
    features: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    label_encoder: dict[str, int] = field(default_factory=dict)
    seed: int = 42
    split_strategy: str = ""
    rows_total: int = 0
    rows_train: int = 0
    rows_test: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    is_synthetic: bool = False
    limitations: list[str] = field(default_factory=list)
    artifact_path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def environment_info() -> dict[str, str]:
    """Versions needed to reproduce a run."""
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
    }


def build_estimator(algorithm: str, seed: int, *, class_weight: str | None = "balanced"):
    """Construct a classifier by name.

    ``class_weight="balanced"`` is the default because IDS datasets are
    heavily imbalanced; without it the majority class dominates accuracy and
    minority attacks are missed.
    """
    key = algorithm.lower()
    if key in {"random_forest", "rf", "randomforest"}:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=1,
            n_jobs=-1, random_state=seed, class_weight=class_weight,
        )
    if key in {"lightgbm", "lgbm", "gbm", "gradient_boosting"}:
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier

            logger.warning("lightgbm unavailable; using HistGradientBoostingClassifier")
            return HistGradientBoostingClassifier(
                max_iter=200, random_state=seed, class_weight=class_weight
            )
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.1, num_leaves=31,
            n_jobs=-1, random_state=seed, class_weight=class_weight, verbose=-1,
        )
    if key in {"logistic_regression", "logreg"}:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            max_iter=1000, random_state=seed, class_weight=class_weight
        )
    raise ValueError(
        f"Unknown algorithm '{algorithm}'. Choose random_forest, lightgbm or "
        "logistic_regression."
    )


def compute_metrics(y_true: Any, y_pred: Any, *, average: str = "weighted",
                    class_names: list[str] | None = None) -> dict[str, Any]:
    """Precision, recall, F1, accuracy and a per-class breakdown.

    Macro-averaged scores are reported alongside weighted ones: on imbalanced
    IDS data a high weighted F1 can coexist with useless per-attack recall,
    and only the macro view exposes that.
    """
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score,
        precision_score, recall_score,
    )

    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision_weighted": round(float(precision_score(
            y_true, y_pred, average=average, zero_division=0)), 6),
        "recall_weighted": round(float(recall_score(
            y_true, y_pred, average=average, zero_division=0)), 6),
        "f1_weighted": round(float(f1_score(
            y_true, y_pred, average=average, zero_division=0)), 6),
        "precision_macro": round(float(precision_score(
            y_true, y_pred, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(
            y_true, y_pred, average="macro", zero_division=0)), 6),
        "f1_macro": round(float(f1_score(
            y_true, y_pred, average="macro", zero_division=0)), 6),
    }
    try:
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        metrics["per_class"] = {
            name: {k: round(float(v), 6) for k, v in scores.items()
                   if k in ("precision", "recall", "f1-score", "support")}
            for name, scores in report.items()
            if isinstance(scores, dict)
        }
    except ValueError:  # a single class present in y_true
        logger.warning("classification_report failed - likely a single-class split")

    matrix = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = matrix.tolist()
    metrics["confusion_matrix_labels"] = class_names or [
        str(i) for i in range(matrix.shape[0])
    ]
    return metrics


def train_one(data: PreparedData, algorithm: str, seed: int, *,
              dataset: str = "unknown", source_hash: str = "",
              is_synthetic: bool = False) -> tuple[Any, ModelCard]:
    """Fit one classifier and build its model card."""
    import time

    started = time.monotonic()
    estimator = build_estimator(algorithm, seed)
    estimator.fit(data.X_train, data.y_train)
    elapsed = time.monotonic() - started

    predictions = estimator.predict(data.X_test)
    class_names = [
        name for name, _ in sorted(data.label_encoder.items(), key=lambda kv: kv[1])
    ]
    metrics = compute_metrics(
        data.y_test, predictions, class_names=class_names
    )
    metrics["fit_seconds"] = round(elapsed, 3)
    metrics["n_features"] = len(data.feature_names)

    importances = getattr(estimator, "feature_importances_", None)
    if importances is not None and len(data.feature_names):
        order = sorted(
            zip(data.feature_names, [float(v) for v in importances]),
            key=lambda kv: -kv[1],
        )[:15]
        metrics["top_features"] = [{"feature": f, "importance": round(v, 6)}
                                   for f, v in order]

    card = ModelCard(
        name=f"nids_{algorithm}",
        algorithm=algorithm,
        task="network intrusion detection (tabular flow classification)",
        dataset=dataset,
        dataset_sha256=source_hash,
        features=list(data.feature_names),
        classes=class_names,
        label_encoder=dict(data.label_encoder),
        seed=seed,
        split_strategy=(
            f"train_test_split(test_size={data.test_size}, random_state={seed}"
            f"{', stratified' if data.stratify_applied else ', unstratified'})"
        ),
        rows_total=data.rows_out,
        rows_train=int(len(data.X_train)),
        rows_test=int(len(data.X_test)),
        class_distribution=dict(data.class_distribution),
        metrics=metrics,
        preprocessing=data.as_dict(),
        environment=environment_info(),
        is_synthetic=is_synthetic,
        limitations=list(DEFAULT_LIMITATIONS) + (
            ["Metrics were computed on a small synthetic fixture and are NOT a "
             "real-world benchmark. Do not quote these numbers as field accuracy."]
            if is_synthetic else []
        ),
    )
    logger.info(
        "trained %s: acc=%.4f f1_macro=%.4f in %.1fs",
        algorithm, metrics["accuracy"], metrics["f1_macro"], elapsed,
    )
    return estimator, card



def save_bundle(model: Any, card: ModelCard, directory: Path) -> Path:
    """Persist a model plus its card and preprocessing summary.

    The bundle is self-describing: ``model.pkl`` alone is not enough to use
    the model correctly, so the feature order and label encoding live beside it
    and are re-applied on load.
    """
    import joblib

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.pkl"
    joblib.dump(model, model_path)

    card.artifact_path = str(model_path)
    (directory / "model_card.json").write_text(
        json.dumps(card.as_dict(), indent=2), encoding="utf-8"
    )
    (directory / "preprocessing.json").write_text(
        json.dumps(card.preprocessing, indent=2, default=str), encoding="utf-8"
    )
    logger.info("saved model bundle to %s", directory)
    return model_path


@dataclass
class LoadedModel:
    """A model restored from disk, with everything inference needs."""

    estimator: Any
    card: ModelCard
    directory: Path

    @property
    def features(self) -> list[str]:
        return self.card.features

    @property
    def classes(self) -> list[str]:
        return self.card.classes

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.card.name,
            "algorithm": self.card.algorithm,
            "dataset": self.card.dataset,
            "is_synthetic": self.card.is_synthetic,
            "classes": self.card.classes,
            "n_features": len(self.card.features),
            "metrics": {
                k: v for k, v in self.card.metrics.items()
                if k in ("accuracy", "f1_macro", "f1_weighted", "recall_macro")
            },
            "path": str(self.directory),
        }


def load_bundle(directory: Path) -> LoadedModel:
    """Restore a saved model bundle."""
    import joblib

    directory = Path(directory)
    model_path = directory / "model.pkl"
    card_path = directory / "model_card.json"
    if not model_path.exists() or not card_path.exists():
        raise ModelNotAvailableError(
            f"No trained model found at {directory}.",
            details={"expected": [str(model_path), str(card_path)]},
        )
    try:
        estimator = joblib.load(model_path)
        card = ModelCard(**json.loads(card_path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as exc:
        raise ModelNotAvailableError(
            f"Model bundle at {directory} could not be loaded: {type(exc).__name__}. "
            "It may have been produced by a different library version - retrain it.",
            details={"path": str(directory)},
        ) from exc
    logger.info("loaded model %s from %s", card.name, directory)
    return LoadedModel(estimator=estimator, card=card, directory=directory)


def compare_models(cards: list[ModelCard]) -> dict[str, Any]:
    """Tabulate several model cards for side-by-side comparison."""
    rows = []
    for card in cards:
        metrics = card.metrics
        rows.append({
            "algorithm": card.algorithm,
            "dataset": card.dataset,
            "is_synthetic": card.is_synthetic,
            "accuracy": metrics.get("accuracy"),
            "f1_weighted": metrics.get("f1_weighted"),
            "f1_macro": metrics.get("f1_macro"),
            "recall_macro": metrics.get("recall_macro"),
            "fit_seconds": metrics.get("fit_seconds"),
        })
    real = [r for r in rows if not r["is_synthetic"]]
    # Synthetic models are excluded from ranking when any real model exists:
    # a near-perfect score on generated data is not evidence of anything and
    # must never be presented as the "best" model.
    pool = real or rows
    best = max(pool, key=lambda r: r.get("f1_macro") or 0.0, default=None)
    return {
        "results": rows,
        "best_by_f1_macro": best["algorithm"] if best else None,
        "ranked_on": "real-data models only" if real else (
            "SYNTHETIC models only - not a valid benchmark"
            if rows else "no models"),
        "note": (
            "All comparisons are on the same held-out split. Macro-F1 is the "
            "primary metric because IDS datasets are imbalanced; weighted F1 "
            "alone can flatter a model that only ever predicts the majority class."
        ),
    }

