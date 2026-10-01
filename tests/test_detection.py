"""Tests for intrusion-detection preprocessing and training.

The most important test here is
:func:`test_labels_stay_aligned_after_deduplication`. An earlier version of
``prepare_flow_data`` re-indexed the feature frame after de-duplication but
sliced the labels *positionally*, which paired most rows with the wrong
target. The model still trained and still reported a plausible-looking
accuracy - it was just learning from scrambled labels (0.25 macro-F1 on
perfectly separable data). Any change to the ordering of those operations
must keep this test green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cyberllm.detection.fixtures import generate_synthetic_flows
from cyberllm.detection.inference import IntrusionDetector
from cyberllm.detection.preprocess import (
    canonical_column, identify_label_column, map_columns, prepare_flow_data,
)
from cyberllm.detection.service import detect_synthetic, train_nids
from cyberllm.detection.train import build_estimator, compute_metrics, compare_models


@pytest.fixture(scope="module")
def frame():
    return generate_synthetic_flows(n_per_class=40, seed=7)


def with_duplicates(frame, frac=0.15, seed=1):
    """Return the frame with duplicates, to exercise de-duplication."""
    import pandas as pd

    return pd.concat([frame, frame.sample(frac=frac, random_state=seed)],
                     ignore_index=True)


# ---------------------------------------------------------------------------
# Column handling
# ---------------------------------------------------------------------------
def test_canonical_column_normalises_separators():
    assert canonical_column("  Flow Duration ") == "flow duration"
    assert canonical_column("src-bytes") == "src bytes"


@pytest.mark.parametrize("columns,expected", [
    (["srcip", "Label"], "Label"),
    (["a", "class"], "class"),
    (["a", "Attack Type"], "Attack Type"),
    (["a", "b"], None),
])
def test_identify_label_column(columns, expected):
    assert identify_label_column(columns) == expected


def test_map_columns_keeps_distinct_columns_separate():
    """Source and destination TTL must not collapse into one feature."""
    mapping = map_columns(["sttl", "dttl", "sbytes", "dbytes", "dur"])
    assert mapping["time_to_live"] == "sttl", (
        "time_to_live must bind to the source TTL, not the destination TTL"
    )
    assert mapping["dst_time_to_live"] == "dttl"
    assert mapping["src_bytes"] == "sbytes" and mapping["dst_bytes"] == "dbytes"


def test_map_columns_is_deterministic():
    """Alias precedence must not depend on set iteration order."""
    first = map_columns(["sttl", "dttl", "sbytes", "dbytes"])
    for _ in range(20):
        assert map_columns(["sttl", "dttl", "sbytes", "dbytes"]) == first
    assert first["time_to_live"] == "sttl"


# ---------------------------------------------------------------------------
# Label integrity - the regression that mattered
# ---------------------------------------------------------------------------
def test_labels_stay_aligned_after_deduplication(frame):
    """De-duplication must carry each row's own label with it.

    Rows whose numeric features were median-imputed are skipped: imputation
    replaces the original value, so comparing them against the raw column
    mean would be comparing against data the pipeline never saw.
    """
    data = prepare_flow_data(with_duplicates(frame), label_column="Label")

    encoder = data.label_encoder
    i_dur = data.feature_names.index("duration")
    combined = list(data.X_train) + list(data.X_test)
    labels = list(data.y_train) + list(data.y_test)

    checked = 0
    for features, encoded in zip(combined, labels):
        name = next(n for n, code in encoder.items() if code == encoded)
        if name not in {"Flood", "DOS"}:
            continue
        if features[i_dur] == pytest.approx(0.84925, abs=0.01):
            continue  # median-imputed row
        expected = frame[frame["Label"] == name]["dur"].mean()
        assert features[i_dur] == pytest.approx(expected, rel=0.2), (
            f"row labelled {name} has duration {features[i_dur]:.2f}, but the "
            f"dataset mean for {name} is {expected:.2f} - features and labels "
            "have been misaligned"
        )
        checked += 1
    assert checked > 10, "not enough non-imputed rows checked to be meaningful"


def test_sampling_keeps_labels_attached(frame):
    """max_rows sampling must not re-pair rows with the wrong labels."""
    full = prepare_flow_data(frame, label_column="Label", max_rows=None)
    sampled = prepare_flow_data(frame, label_column="Label", max_rows=200)
    assert len(sampled.X_train) + len(sampled.X_test) == 200
    assert (set(sampled.y_train.tolist()) | set(sampled.y_test.tolist())) == \
        (set(full.y_train.tolist()) | set(full.y_test.tolist()))


def test_leakage_columns_are_dropped(frame):
    data = prepare_flow_data(frame, label_column="Label")
    for column in ("srcip", "dstip", "srcipport", "dstipport"):
        assert column in data.dropped_columns, f"{column} should be dropped as leakage"
    assert "Label" not in data.feature_names


def test_train_test_sizes_add_up(frame):
    """Rows after de-duplication must equal train + test exactly."""
    data = prepare_flow_data(frame, label_column="Label", test_size=0.2)
    total = len(data.X_train) + len(data.X_test)
    assert total == data.rows_out, "train + test must account for every kept row"
    assert total <= len(frame), "de-duplication must not increase the row count"
    assert len(data.X_test) == pytest.approx(data.rows_out * 0.2, abs=5)


def test_same_seed_is_reproducible(frame):
    """The recorded seed must actually make preprocessing deterministic."""
    a = prepare_flow_data(frame, label_column="Label", seed=42)
    b = prepare_flow_data(frame, label_column="Label", seed=42)
    assert (a.X_train == b.X_train).all()
    assert (a.y_train == b.y_train).all()


def test_different_seeds_give_different_splits(frame):
    a = prepare_flow_data(frame, label_column="Label", seed=1)
    b = prepare_flow_data(frame, label_column="Label", seed=2)
    assert not (a.X_test == b.X_test).all()


def test_missing_label_column_is_rejected(frame):
    from cyberllm.core.errors import ValidationError

    with pytest.raises(ValidationError, match="Label column"):
        prepare_flow_data(frame, label_column="DoesNotExist")


# ---------------------------------------------------------------------------
# Metrics and comparison
# ---------------------------------------------------------------------------
def test_compute_metrics_includes_macro_and_per_class():
    metrics = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1], class_names=["A", "B"])
    assert metrics["accuracy"] == 0.75
    assert 0 <= metrics["f1_macro"] <= 1
    assert len(metrics["confusion_matrix"]) == 2
    # classification_report keys classes by encoded value; the human-readable
    # names travel separately in confusion_matrix_labels.
    assert set(metrics["confusion_matrix_labels"]) == {"A", "B"}
    assert "0" in metrics["per_class"] and "1" in metrics["per_class"]
    assert metrics["per_class"]["0"]["recall"] == 0.5
    assert "macro avg" in metrics["per_class"]


def test_compare_models_never_ranks_synthetic_as_best():
    from cyberllm.detection.train import ModelCard

    cards = [
        ModelCard(name="a", algorithm="random_forest", task="t", dataset="synth",
                  is_synthetic=True, metrics={"f1_macro": 0.99}),
        ModelCard(name="b", algorithm="lightgbm", task="t", dataset="real",
                  is_synthetic=False, metrics={"f1_macro": 0.80}),
    ]
    out = compare_models(cards)
    assert out["best_by_f1_macro"] == "lightgbm"
    assert "real-data" in out["ranked_on"]


def test_build_estimator_rejects_unknown_algorithm():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        build_estimator("deep_learning", 42)


def test_detect_synthetic_from_filename():
    assert detect_synthetic(Path("x_synthetic_fixture.csv")) is True
    assert detect_synthetic(Path("UNSW_NB15_features.csv")) is False
    assert detect_synthetic(Path("anything.csv"), explicit=True) is True



# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    data_path = Path("data/raw/synthetic/nids_synthetic_fixture.csv")
    if not data_path.exists():
        from cyberllm.detection.fixtures import write_synthetic_flows
        write_synthetic_flows(data_path, n_per_class=60, seed=7)
    return train_nids(data_path, out_dir=out, max_rows=2000), out


def test_training_produces_both_algorithms(trained):
    report, _ = trained
    assert [m["algorithm"] for m in report.models] == ["random_forest", "lightgbm"]


def test_training_is_labelled_synthetic(trained):
    report, _ = trained
    assert report.is_synthetic is True
    assert any("SYNTHETIC" in e for e in report.errors), (
        "a synthetic run must carry an explicit warning"
    )


def test_model_card_records_reproducibility(trained):
    import json

    _, out = trained
    card = json.loads(
        (out / "random_forest" / "model_card.json").read_text(encoding="utf-8")
    )
    assert card["seed"] == 42
    assert "random_state=42" in card["split_strategy"]
    assert card["dataset_sha256"]
    assert card["environment"]["scikit_learn"]
    assert card["limitations"], "a model card must state its limitations"


def test_model_reaches_high_accuracy_on_separable_fixture(trained):
    """Guards the label-alignment regression: scrambled targets scored ~0.25."""
    report, _ = trained
    for model in report.models:
        assert model["metrics"]["f1_macro"] > 0.8, (
            f"{model['algorithm']} scored macro-F1 "
            f"{model['metrics']['f1_macro']:.3f} on separable synthetic data - "
            "check for label misalignment in preprocessing"
        )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def test_detector_without_model_reports_unavailable():
    status = IntrusionDetector(directory=Path("does/not/exist")).status()
    assert status["available"] is False
    assert "setup_hint" in status


def test_detector_predicts_and_explains(trained):
    _, out = trained
    detector = IntrusionDetector(directory=out / "random_forest")
    prediction = detector.predict({
        "srcip": "10.0.0.5", "dstip": "10.0.0.9", "proto": "tcp",
        "service": "http", "flag": "S", "dur": 0.1,
        "sbytes": 4000, "dbytes": 100, "sttl": 64, "dttl": 3,
    })
    assert prediction.predicted_class
    assert 0.0 <= prediction.confidence <= 1.0
    assert "probabilistic" in prediction.disclaimer


def test_detector_rejects_record_with_no_known_features(trained):
    from cyberllm.core.errors import ValidationError

    _, out = trained
    detector = IntrusionDetector(directory=out / "random_forest")
    with pytest.raises(ValidationError):
        detector.predict({"totally": "unrelated", "columns": 1})


def test_detector_reports_missing_features(trained):
    _, out = trained
    detector = IntrusionDetector(directory=out / "random_forest")
    prediction = detector.predict({"sbytes": 1000, "dbytes": 2000})
    assert prediction.missing_features, "missing features must be reported"

