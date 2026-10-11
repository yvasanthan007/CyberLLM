"""Reproducible UNSW-NB15 intrusion-detection benchmark.

What this module guarantees
---------------------------
* **The test split is never touched.** All imputation, encoding and any
  statistic the model depends on is fitted on training rows only, and the
  test rows are carried through untouched until scoring.
* **Duplicates are removed before the split**, never after, so no row can
  appear on both sides of the boundary.
* **The target cannot be read off the features.** Every candidate column is
  tested for target purity first; anything that determines the label is
  dropped and named in the report.
* **Baselines are reported next to the models.** A macro-F1 of 0.9 means
  nothing without knowing what majority-class and stratified-random baselines
  score on the same split.
* **Synthetic and real results are never mixed.** This module only ever
  produces ``is_synthetic=False`` results, and refuses a synthetic input.

Split strategies
----------------
``dedup_random``
    Global de-duplication, then a stratified random split. The conventional
    choice, and the one comparable with most published UNSW-NB15 numbers.

``grouped``
    De-duplication, then a split that keeps whole *conversation groups* on
    one side. Stricter: it removes the "the same conversation appears in
    both halves" advantage that a random split leaves in place.

``publisher``
    The dataset authors' own train/test partition, untouched. Reported for
    comparability with the literature, with its measured duplicate overlap
    attached - the audit found that partition is *not* duplicate-free.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..core.errors import ValidationError
from ..core.logging_config import get_logger
from .preprocess import LEAKAGE_COLUMNS, canonical_column
from .train import ModelCard, build_estimator, compute_metrics, save_bundle
from .unsw import CATEGORY_TARGET, audit_partition, audit_split, conversation_key

logger = get_logger("detection.benchmark")

#: Strategies this module will build, in the order they are reported.
SPLIT_STRATEGIES = ("dedup_random", "grouped", "publisher")

#: Reference points every benchmark must be compared against.
BASELINES = ("majority_class", "uniform_random")


@dataclass
class BenchmarkConfig:
    """Every knob that affects the result, recorded in the report."""

    data_dir: Path = Path("data/raw/unsw_nb15")
    target_column: str = CATEGORY_TARGET
    strategies: tuple[str, ...] = ("dedup_random", "grouped", "publisher")
    algorithms: tuple[str, ...] = ("random_forest", "lightgbm")
    seed: int = 42
    test_size: float = 0.3
    max_rows: int | None = None
    out_dir: Path = Path("artifacts/reports/unsw_nb15")
    model_dir: Path = Path("artifacts/models/nids")
    register_model: bool = True

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("data_dir", "out_dir", "model_dir"):
            payload[key] = str(self.__dict__[key])
        payload["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload["environment"] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        return payload


def _resolve_core_files(data_dir: Path) -> dict[str, Path]:
    """Locate the publisher's released partitions under ``data_dir``."""
    from ..datasets.unsw_nb15 import core_files_present

    found = core_files_present(Path(data_dir))
    missing = [role for role in ("train", "test") if role not in found]
    if missing:
        raise ValidationError(
            f"UNSW-NB15 partition(s) missing from {data_dir}: {', '.join(missing)}.",
            details={"setup_hint": "python -m cyberllm.cli datasets download "
                                   "--name unsw_nb15"},
        )
    return found


def _feature_columns(frame: Any, target: str, dropped: Sequence[str]) -> list[str]:
    """Columns usable as features, after leakage removal and aliasing."""
    skip = {canonical_column(c) for c in dropped} | {canonical_column(target)}
    columns: list[str] = []
    seen: set[str] = set()
    for column in frame.columns:
        canonical = canonical_column(column)
        if canonical in skip or canonical in LEAKAGE_COLUMNS or canonical in seen:
            continue
        seen.add(canonical)
        columns.append(column)
    return columns


def _fit_categories(train_frame: Any, test_frame: Any,
                   columns: list[str]) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Learn category -> code mappings from the training rows only.

    Fitted on train and *applied* to test. Encoding the two frames separately
    would give ``tcp`` different codes in each half, silently turning a
    categorical feature into noise and corrupting every downstream number.
    An unseen test category maps to ``-1`` and is reported, never invented.
    """
    mappings: dict[str, dict[str, int]] = {}
    unseen: list[str] = []
    for column in columns:
        series = train_frame[column]
        if not (series.dtype == object or str(series.dtype) in {"str", "string"}):
            continue
        values = sorted(series.astype(str).unique())
        mappings[column] = {value: index for index, value in enumerate(values)}
        if column in test_frame.columns:
            missing = sorted(set(test_frame[column].astype(str)) - set(values))
            if missing:
                unseen.append(
                    f"{column}: {', '.join(missing[:5])}"
                    f"{' ...' if len(missing) > 5 else ''}"
                )
    return mappings, unseen


def _encode(frame: Any, columns: list[str],
            mappings: dict[str, dict[str, int]]) -> Any:
    """Apply the fitted encodings and coerce the rest to numbers."""
    import pandas as pd

    block = frame[columns].copy()
    for column in columns:
        mapping = mappings.get(column)
        if mapping is not None:
            block[column] = (
                block[column].astype(str).map(mapping).fillna(-1).astype("float64")
            )
        else:
            block[column] = pd.to_numeric(block[column], errors="coerce")
    return block.to_numpy(dtype="float32")


def _impute(train_matrix: Any, test_matrix: Any) -> tuple[Any, Any, Any]:
    """Replace non-finite values using training-split medians only."""
    import numpy as np

    medians = np.nanmedian(
        np.where(np.isfinite(train_matrix), train_matrix, np.nan),
        axis=0,
    )
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train = np.where(np.isfinite(train_matrix), train_matrix, medians)
    test = np.where(np.isfinite(test_matrix), test_matrix, medians)
    return train.astype("float32"), test.astype("float32"), medians


def _dedupe(frame: Any, features: list[str], target: str) -> tuple[Any, int]:
    """Drop exact feature duplicates, keeping the first occurrence.

    Done *before* any split, so a duplicated flow can never appear on both
    sides of the train/test boundary. De-duplication is on the feature vector
    plus the target, so a flow that appears twice with two *different* labels
    is kept once per label rather than silently discarded - discarding it
    would delete evidence that the features are ambiguous.
    """
    import pandas as pd

    from .unsw import row_hashes

    block = pd.DataFrame({
        "hash": row_hashes(frame, features),
        "label": frame[target].astype(str).to_numpy(),
    })
    keep = ~block.duplicated()
    removed = int((~keep).sum())
    return frame[keep.to_numpy()].reset_index(drop=True), removed


def _split_dedup_random(pool: Any, features: list[str], target: str, *,
                        test_size: float, seed: int) -> tuple[Any, Any, dict[str, Any]]:
    """Stratified random split over the de-duplicated pool."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    labels = pool[target].astype(str)
    can_stratify = labels.value_counts().min() >= 2
    train_idx, test_idx = train_test_split(
        np.arange(len(pool)), test_size=test_size, random_state=seed,
        stratify=labels if can_stratify else None,
    )
    meta = {
        "method": "deduplicated stratified random split",
        "test_size": test_size,
        "seed": seed,
        "stratified": bool(can_stratify),
    }
    return pool.iloc[train_idx], pool.iloc[test_idx], meta


def _split_grouped(pool: Any, features: list[str], target: str, *,
                   test_size: float, seed: int) -> tuple[Any, Any, dict[str, Any]]:
    """Split that keeps whole conversation groups on one side.

    Uses ``StratifiedGroupKFold`` rather than ``GroupShuffleSplit``: a plain
    group shuffle moves every group containing a rare class wholesale to one
    side, which can leave the test split with no examples of that class at all
    and makes macro-F1 quietly meaningless for it. Stratifying on the target
    while still keeping groups intact avoids that, and the fold closest to the
    requested size is chosen deterministically.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    labels = pool[target].astype(str)
    counts = labels.value_counts()
    meta: dict[str, Any] = {
        "method": "de-duplicated stratified conversation-group split",
        "test_size": test_size,
        "seed": seed,
    }
    if int(counts.min()) < 2:
        logger.warning(
            "a class has fewer than 2 members; stratifying on the rest only"
        )

    groups = conversation_key(pool, target, features)
    unique_groups = groups.nunique()
    meta["groups"] = int(unique_groups)
    if unique_groups < 2:
        logger.warning("group proxy degenerate; falling back to a random split")
        train, test, _fallback = _split_dedup_random(
            pool, features, target, test_size=test_size, seed=seed,
        )
        meta["fallback"] = "group proxy degenerate; used random split"
        return train, test, meta

    n_splits = max(2, min(10, int(round(1 / max(test_size, 1e-6)))))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed)
    all_classes = set(counts.index)
    best: tuple[float, Any, Any] | None = None
    for train_idx, test_idx in splitter.split(np.arange(len(pool)), labels,
                                              groups=groups):
        if not len(test_idx) or not len(train_idx):
            continue
        drift = abs(len(test_idx) / len(pool) - test_size)
        if set(labels.iloc[test_idx]) != all_classes:
            drift += 1.0  # a class missing from test makes its F1 undefined
        if best is None or drift < best[0]:
            best = (drift, train_idx, test_idx)
        if drift == 0.0:
            break

    if best is None:
        logger.warning("group split produced no usable fold; using random split")
        train, test, _fallback = _split_dedup_random(
            pool, features, target, test_size=test_size, seed=seed,
        )
        meta["fallback"] = "group split degenerate; used random split"
        return train, test, meta

    _drift, train_idx, test_idx = best
    missing = sorted(all_classes - set(labels.iloc[test_idx]))
    missing_train = sorted(all_classes - set(labels.iloc[train_idx]))
    meta["stratified"] = True
    meta["n_splits_considered"] = n_splits
    meta["classes_missing_from_test"] = missing
    meta["classes_missing_from_train"] = missing_train
    if missing or missing_train:
        parts = []
        if missing:
            parts.append(
                f"absent from test ({', '.join(missing)}) - their F1 is undefined"
            )
        if missing_train:
            parts.append(
                f"absent from train ({', '.join(missing_train)}) - they can never "
                "be predicted, so their F1 is undefined"
            )
        meta["warning"] = (
            f"{len(missing) + len(missing_train)} class(es) not present on both "
            f"sides: " + "; ".join(parts)
        )
        logger.warning(meta["warning"])
    return pool.iloc[train_idx], pool.iloc[test_idx], meta


def _label_per_class(metrics: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    """Replace numeric class codes with their real names in ``per_class``.

    ``classification_report`` keys by the encoded integer, so the report would
    otherwise list "0..9" while the model card says "Normal..Worms" - same
    rows, unreadable side by side.
    """
    per_class = metrics.get("per_class")
    if not per_class or not class_names:
        return metrics
    labelled: dict[str, Any] = {}
    for key, value in per_class.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            labelled[key] = value
            continue
        name = class_names[index] if 0 <= index < len(class_names) else key
        labelled[name] = value
    metrics["per_class"] = labelled
    return metrics


def baseline_predictions(y_train: Any, y_test: Any, seed: int,
                         class_names: list[str] | None = None) -> dict[str, Any]:
    """Reference points any classifier must beat to mean anything.

    On a 10-class imbalanced problem a model that never learns can still post
    a respectable accuracy, so the floor has to be stated explicitly.
    """
    from collections import Counter

    import numpy as np

    train_counts = Counter(int(v) for v in np.asarray(y_train).ravel())
    y_test = np.asarray(y_test)
    majority = train_counts.most_common(1)[0][0] if train_counts else 0
    majority_pred = np.full_like(y_test, majority)

    classes = np.array(sorted(train_counts))
    rng = np.random.default_rng(seed)
    random_pred = (
        rng.choice(classes, size=len(y_test)) if len(classes) else majority_pred
    )

    predictions = {
        "majority_class": majority_pred,
        "uniform_random": random_pred,
    }
    results: dict[str, Any] = {}
    for name in BASELINES:
        metrics = _label_per_class(
            compute_metrics(y_test, predictions[name]), class_names or []
        )
        results[name] = {
            key: metrics[key]
            for key in ("accuracy", "f1_macro", "f1_weighted", "recall_macro",
                        "precision_macro", "per_class")
        }
    return results


def _materialise(train_frame: Any, test_frame: Any, features: list[str],
                 target: str) -> dict[str, Any]:
    """Build train/test matrices with every transform fitted on train only."""
    mappings, unseen = _fit_categories(train_frame, test_frame, features)
    raw_train = _encode(train_frame, features, mappings)
    raw_test = _encode(test_frame, features, mappings)
    X_train, X_test, medians = _impute(raw_train, raw_test)

    labels = sorted(set(train_frame[target].astype(str))
                    | set(test_frame[target].astype(str)))
    encoder = {name: index for index, name in enumerate(labels)}
    y_train = train_frame[target].astype(str).map(encoder).to_numpy(dtype="int64")
    y_test = test_frame[target].astype(str).map(encoder).to_numpy(dtype="int64")
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "label_encoder": encoder, "features": features,
        "medians": medians, "category_maps": mappings,
        "unseen_test_categories": unseen,
    }


def _fit_and_score(material: dict[str, Any], algorithm: str,
                   seed: int) -> tuple[Any, dict[str, Any]]:
    """Fit one estimator and score it on the held-out side."""
    started = time.monotonic()
    estimator = build_estimator(algorithm, seed)
    estimator.fit(material["X_train"], material["y_train"])
    elapsed = time.monotonic() - started
    predictions = estimator.predict(material["X_test"])
    class_names = [
        name for name, _ in sorted(material["label_encoder"].items(),
                                   key=lambda kv: kv[1])
    ]
    metrics = _label_per_class(
        compute_metrics(material["y_test"], predictions, class_names=class_names),
        class_names,
    )
    metrics["fit_seconds"] = round(elapsed, 3)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is not None and material["features"]:
        ranked = sorted(
            zip(material["features"], [float(v) for v in importances]),
            key=lambda kv: -kv[1],
        )[:15]
        metrics["top_features"] = [
            {"feature": f, "importance": round(v, 6)} for f, v in ranked
        ]
    return estimator, metrics


def run_benchmark(config: BenchmarkConfig | None = None) -> dict[str, Any]:
    """Run the full UNSW-NB15 benchmark and return a self-describing report.

    The report contains the data audit, the leakage measurements, the split
    metadata, per-class metrics, confusion matrices, baselines, the model
    cards and the complete configuration - everything needed to re-run and
    check the numbers, and nothing that is not in there.
    """
    import pandas as pd

    from ..datasets.unsw_nb15 import (
        CITATION, DATASET_CAVEAT, provenance,
    )

    config = config or BenchmarkConfig()
    started = time.monotonic()
    report: dict[str, Any] = {
        "benchmark": "UNSW-NB15 network intrusion detection",
        "config": config.as_dict(),
        "is_synthetic": False,
        "provenance": provenance(),
        "citation": list(CITATION),
        "dataset_caveat": DATASET_CAVEAT,
        "audits": {},
        "splits": {},
        "models": {},
        "errors": [],
    }

    files = _resolve_core_files(config.data_dir)
    partitions: dict[str, Any] = {}
    for role in ("train", "test"):
        frame, audit = audit_partition(files[role],
                                      target_column=config.target_column)
        partitions[role] = frame
        report["audits"][role] = audit.as_dict()
        logger.info("audit %s\n%s", role, audit.summary())
        if not audit.ok:
            report["errors"].append(f"{role}: " + "; ".join(audit.errors))

    if report["errors"]:
        return report

    target = partitions["train"].columns[
        [canonical_column(c) for c in partitions["train"].columns]
        .index(canonical_column(config.target_column))
    ]
    dropped = sorted({c for role in partitions
                      for c in report["audits"][role]["dropped_columns"]
                      if c in partitions[role].columns})
    features = _feature_columns(partitions["train"], target, dropped)
    report["feature_columns"] = features
    report["dropped_columns"] = dropped
    if not features:
        report["errors"].append("no usable feature columns remain")
        return report

    # The publisher's partition, audited before anything is done to it.
    report["publisher_partition_audit"] = audit_split(
        partitions["train"], partitions["test"],
        target_column=target, dropped=dropped,
    ).as_dict()

    pool = pd.concat([partitions["train"], partitions["test"]], ignore_index=True)
    pool, removed = _dedupe(pool, features, target)
    report["pool"] = {
        "rows": int(len(pool)),
        "duplicates_removed": removed,
        "note": "exact feature duplicates removed once, globally, before splitting",
    }
    if config.max_rows and len(pool) > config.max_rows:
        pool = pool.sample(
            n=config.max_rows, random_state=config.seed
        ).reset_index(drop=True)
        report["pool"]["sampled_to"] = int(len(pool))

    estimators: dict[str, dict[str, Any]] = {}
    materials: dict[str, dict[str, Any]] = {}
    for strategy in config.strategies:
        if strategy == "publisher":
            train_frame, test_frame = partitions["train"], partitions["test"]
            split_meta = {
                "method": "publisher's own train/test partition, used as released",
                "seed": config.seed,
                "note": "NOT de-duplicated across the boundary by the publisher; "
                        "see leakage audit",
            }
        else:
            builder = _split_dedup_random if strategy == "dedup_random" else _split_grouped
            train_frame, test_frame, split_meta = builder(
                pool, features, target, test_size=config.test_size,
                seed=config.seed,
            )
        entry: dict[str, Any] = {
            "split": split_meta,
            "rows": {"train": int(len(train_frame)), "test": int(len(test_frame))},
            "leakage": audit_split(train_frame, test_frame,
                                   target_column=target, dropped=dropped).as_dict(),
        }
        material = _materialise(train_frame, test_frame, features, target)
        entry["unseen_test_categories"] = material["unseen_test_categories"]
        entry["class_distribution"] = {
            "train": {k: int(v) for k, v in
                      train_frame[target].astype(str).value_counts().items()},
            "test": {k: int(v) for k, v in
                     test_frame[target].astype(str).value_counts().items()},
        }
        entry["baselines"] = baseline_predictions(
            material["y_train"], material["y_test"], config.seed,
            class_names=[
                name for name, _ in sorted(material["label_encoder"].items(),
                                           key=lambda kv: kv[1])
            ],
        )
        models: dict[str, Any] = {}
        fitted: dict[str, Any] = {}
        for algorithm in config.algorithms:
            try:
                estimator, metrics = _fit_and_score(material, algorithm, config.seed)
            except (ValueError, TypeError, ImportError, MemoryError) as exc:
                logger.error("%s failed on %s: %s", algorithm, strategy, exc)
                report["errors"].append(f"{strategy}/{algorithm}: "
                                        f"{type(exc).__name__}: {exc}")
                continue
            models[algorithm] = metrics
            fitted[algorithm] = estimator
        entry["models"] = models
        report["splits"][strategy] = entry
        estimators[strategy] = {"models": models, "fitted": fitted}
        materials[strategy] = material
        logger.info("split %s done: %s", strategy,
                    {k: v.get("f1_macro") for k, v in models.items()})

    report["artifacts"] = _persist(
        config, report, estimators, materials, _source_hash(config, files)
    )
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    _write_report(config.out_dir, report)
    return report


def _source_hash(config: BenchmarkConfig, files: dict[str, Path]) -> str:
    """A single content hash covering every partition used.

    One hash for the whole dataset (not one per file) so a model card can be
    tied to exactly the bytes that produced it.
    """
    import hashlib

    digest = hashlib.sha256()
    for role in sorted(files):
        path = files[role]
        digest.update(role.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _model_card(config: BenchmarkConfig, strategy: str, algorithm: str,
                entry: dict[str, Any], metrics: dict[str, Any],
                material: dict[str, Any], source_hash: str,
                dropped: list[str]) -> ModelCard:
    """Assemble the model card written next to the saved estimator."""
    from ..datasets.unsw_nb15 import DATASET_CAVEAT
    from .train import DEFAULT_LIMITATIONS

    class_names = [
        name for name, _ in sorted(material["label_encoder"].items(),
                                   key=lambda kv: kv[1])
    ]
    return ModelCard(
        name=f"nids_{algorithm}_unsw_nb15",
        algorithm=algorithm,
        task=f"UNSW-NB15 {config.target_column} classification "
              "(tabular network-flow features)",
        dataset="unsw_nb15",
        dataset_sha256=source_hash,
        features=list(material["features"]),
        classes=class_names,
        label_encoder=dict(material["label_encoder"]),
        seed=config.seed,
        split_strategy=(
            f"{strategy}: {entry['split'].get('method')} "
            f"(test_size={entry['split'].get('test_size', 'as-published')}, "
            f"seed={config.seed})"
        ),
        rows_total=entry["rows"]["train"] + entry["rows"]["test"],
        rows_train=entry["rows"]["train"],
        rows_test=entry["rows"]["test"],
        class_distribution=entry["class_distribution"]["test"],
        metrics=metrics,
        preprocessing={
            "strategy": strategy,
            "features": list(material["features"]),
            "dropped_columns": dropped,
            "category_maps": material["category_maps"],
            "imputation": "median fitted on training rows only",
            "unseen_test_categories": material["unseen_test_categories"],
            "baselines": entry["baselines"],
            "leakage_audit": entry["leakage"],
        },
        is_synthetic=False,
        limitations=list(DEFAULT_LIMITATIONS) + [
            DATASET_CAVEAT,
            "UNSW-NB15 releases flow *statistics*, not payloads and not IP "
            "addresses or ports, so the model cannot see the indicators an "
            "analyst would actually use.",
            "Scores are on one lab capture. They are not a production "
            "detection rate and must not be presented as one.",
            "The publisher's terms restrict this dataset to academic research "
            "use; commercial deployment is out of scope.",
        ],
    )


def _persist(config: BenchmarkConfig, report: dict[str, Any],
             estimators: dict[str, dict[str, Any]],
             materials: dict[str, dict[str, Any]],
             source_hash: str) -> list[str]:
    """Write provenance and save the deployable model bundle.

    The bundle is the best model on the primary split, written to the same
    location the API and CLI already read, so the real-data model replaces
    the synthetic one with no configuration change.
    """
    from .train import save_bundle

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    primary = "dedup_random" if "dedup_random" in estimators else next(
        iter(estimators), None
    )
    if primary:
        scored = {
            algorithm: metrics.get("f1_macro") or 0.0
            for algorithm, metrics in estimators[primary]["models"].items()
        }
        if scored:
            best = max(scored, key=lambda key: scored[key])
            entry = report["splits"][primary]
            card = _model_card(
                config, primary, best, entry, entry["models"][best],
                materials[primary], source_hash, report["dropped_columns"],
            )
            directory = Path(config.model_dir) / best
            save_bundle(estimators[primary]["fitted"][best], card, directory)
            written.append(str(directory))
            report["registered_model"] = {
                "algorithm": best,
                "split": primary,
                "f1_macro": scored[best],
                "path": str(directory),
                "is_synthetic": False,
            }

    provenance_path = out_dir / "PROVENANCE.json"
    provenance_path.write_text(
        json.dumps(report["provenance"], indent=2), encoding="utf-8"
    )
    written.append(str(provenance_path))
    return written


def _write_report(out_dir: Path, report: dict[str, Any]) -> Path:
    """Write the machine-readable benchmark report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "benchmark.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("benchmark report written to %s", path)
    return path


def summarise(report: dict[str, Any]) -> str:
    """A short human-readable digest of a benchmark report."""
    lines = [report["benchmark"],
             f"  dataset   : {report['provenance']['dataset']} (real data)"]
    for strategy, entry in report.get("splits", {}).items():
        rows = entry["rows"]
        lines.append(
            f"\n[{strategy}] {entry['split'].get('method')} | "
            f"train {rows['train']:,} / test {rows['test']:,}"
        )
        leakage = entry["leakage"]
        lines.append(
            f"  leakage: {leakage['cross_split_duplicate_rows']:,} duplicated test "
            f"row(s) ({leakage['cross_split_duplicate_share']:.2%}), "
            f"{leakage['conflicting_label_rows']:,} conflicting-label vector(s)"
        )
        for name, base in entry["baselines"].items():
            lines.append(f"  baseline {name:<16} acc={base['accuracy']:.4f} "
                         f"f1_macro={base['f1_macro']:.4f}")
        for name, metrics in entry["models"].items():
            lines.append(f"  model    {name:<16} acc={metrics['accuracy']:.4f} "
                         f"f1_macro={metrics['f1_macro']:.4f} "
                         f"recall_macro={metrics['recall_macro']:.4f}")
    if report.get("errors"):
        lines.append("\nerrors:")
        lines.extend(f"  - {error}" for error in report["errors"])
    return "\n".join(lines)


__all__ = [
    "BenchmarkConfig", "run_benchmark", "summarise",
    "SPLIT_STRATEGIES", "BASELINES",
]
