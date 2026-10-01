"""End-to-end intrusion-detection workflow: inspect, prepare, train, evaluate.

This is the layer the CLI and the API call. It keeps the ordering guarantees
in one place - inspect before training, de-duplicate before splitting, record
every transformation - so a caller cannot accidentally skip a step.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ValidationError
from ..core.logging_config import get_logger
from .fixtures import write_synthetic_flows
from .preprocess import (
    PreparedData, SchemaReport, file_sha256, inspect_csv, prepare_flow_data,
)
from .train import ModelCard, compare_models, save_bundle, train_one

logger = get_logger("detection.service")

#: Filename markers that mean "this file was generated locally".
SYNTHETIC_MARKERS = ("synthetic", "fixture", "demo")


@dataclass
class TrainingReport:
    """Result of a full train-and-compare run."""

    dataset: str
    data_path: str
    is_synthetic: bool
    schema: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    models: list[dict[str, Any]] = field(default_factory=list)
    comparison: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "data_path": self.data_path,
            "is_synthetic": self.is_synthetic,
            "schema": self.schema,
            "preprocessing": self.preprocessing,
            "models": self.models,
            "comparison": self.comparison,
            "artifacts": self.artifacts,
            "errors": self.errors,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def detect_synthetic(path: Path, explicit: bool | None = None) -> bool:
    """Decide whether a data file is synthetic.

    Explicit caller intent wins; otherwise the filename is inspected. This
    matters because synthetic results must never be presented as benchmarks.
    """
    if explicit is not None:
        return explicit
    return any(marker in Path(path).name.lower() for marker in SYNTHETIC_MARKERS)


def ensure_training_data(path: Path, *, make_fixture_if_missing: bool = True,
                         n_per_class: int = 300,
                         seed: int = 42) -> tuple[Path, bool]:
    """Return a usable CSV path, generating a labelled fixture if allowed."""
    path = Path(path)
    if path.exists():
        return path, detect_synthetic(path)
    if not make_fixture_if_missing:
        raise ValidationError(
            f"Training data not found: {path}. Download the dataset first "
            "(`datasets download --name <id>`) or point --data at an existing CSV."
        )
    logger.warning("%s not found; generating a SYNTHETIC fixture for testing", path)
    return write_synthetic_flows(path, n_per_class=n_per_class, seed=seed), True


def train_nids(
    data_path: Path,
    *,
    algorithms: tuple[str, ...] = ("random_forest", "lightgbm"),
    label_column: str | None = None,
    seed: int = 42,
    test_size: float = 0.2,
    max_rows: int | None = 60_000,
    out_dir: Path | None = None,
    is_synthetic: bool | None = None,
    make_fixture_if_missing: bool = True,
) -> TrainingReport:
    """Inspect, prepare, train and compare intrusion-detection models."""
    from ..core.config import get_settings

    started = time.monotonic()
    out_root = Path(out_dir) if out_dir else get_settings().paths.models_dir / "nids"
    data_path, synthetic = ensure_training_data(
        data_path, make_fixture_if_missing=make_fixture_if_missing, seed=seed
    )
    is_syn = detect_synthetic(data_path, is_synthetic) or synthetic

    report = TrainingReport(
        dataset=data_path.stem, data_path=str(data_path), is_synthetic=is_syn
    )

    frame, schema = inspect_csv(data_path, label_column=label_column)
    report.schema = schema.as_dict()
    logger.info("schema:\n%s", schema.summary())

    if schema.label_column is None:
        report.errors.append(
            "No label column found; cannot train. Expected one of: "
            "label, class, attack."
        )
        return report
    if len(schema.label_distribution) < 2:
        report.errors.append(
            "Fewer than two distinct classes present; there is nothing to learn."
        )
        return report

    data: PreparedData = prepare_flow_data(
        frame, label_column=schema.label_column, seed=seed,
        test_size=test_size, max_rows=max_rows,
    )
    data.source_hash = file_sha256(data_path)
    report.preprocessing = data.as_dict()

    cards: list[ModelCard] = []
    for algorithm in algorithms:
        try:
            model, card = train_one(
                data, algorithm, seed, dataset=report.dataset,
                source_hash=data.source_hash, is_synthetic=is_syn,
            )
        except Exception as exc:  # one algorithm failing must not hide the rest
            logger.error("training %s failed: %s", algorithm, exc)
            report.errors.append(f"{algorithm}: {type(exc).__name__}: {exc}")
            continue
        directory = out_root / algorithm
        save_bundle(model, card, directory)
        report.artifacts.append(str(directory))
        report.models.append({
            "algorithm": algorithm,
            "metrics": card.metrics,
            "path": str(directory),
        })

    if cards:
        report.comparison = compare_models(cards)
        (out_root / "comparison.json").write_text(
            json.dumps(report.comparison, indent=2), encoding="utf-8"
        )
        report.artifacts.append(str(out_root / "comparison.json"))
        logger.info("best model by macro-F1: %s",
                    report.comparison.get("best_by_f1_macro"))

    report.elapsed_seconds = time.monotonic() - started
    if is_syn:
        report.errors.append(
            "NOTE: trained on a SYNTHETIC fixture. These metrics validate the "
            "pipeline only and must NOT be reported as real-world accuracy."
        )
    return report


def evaluate_nids(directory: Path) -> dict[str, Any]:
    """Re-read a saved model card for the evaluation report."""
    card_path = Path(directory) / "model_card.json"
    if not card_path.exists():
        raise ValidationError(
            f"No model card at {card_path}. Train a model first: `models train --task nids`."
        )
    card = json.loads(card_path.read_text(encoding="utf-8"))
    return {
        "model": card.get("name"),
        "algorithm": card.get("algorithm"),
        "dataset": card.get("dataset"),
        "dataset_sha256": card.get("dataset_sha256"),
        "is_synthetic": card.get("is_synthetic"),
        "seed": card.get("seed"),
        "split_strategy": card.get("split_strategy"),
        "rows": {"total": card.get("rows_total"), "train": card.get("rows_train"),
                 "test": card.get("rows_test")},
        "class_distribution": card.get("class_distribution"),
        "metrics": card.get("metrics"),
        "environment": card.get("environment"),
        "limitations": card.get("limitations"),
        "created_at": card.get("created_at"),
    }


__all__ = [
    "train_nids", "evaluate_nids", "TrainingReport",
    "ensure_training_data", "detect_synthetic",
]

