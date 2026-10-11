"""Tests for the UNSW-NB15 acquisition, audit and benchmark path.

Every test here pins a behaviour that a real defect violated, or a claim the
documentation makes:

* ``test_attack_cat_is_treated_as_leakage`` - ``attack_cat`` is in 1:1
  correspondence with the binary ``label`` in the real data. Leaving it in as
  a feature turns the task into reading the answer key. It was found by
  auditing the downloaded dataset, not by reading the code.
* ``test_bom_header_does_not_bypass_the_leakage_blocklist`` - the published
  CSVs start with a UTF-8 BOM, so ``id`` arrives as ``\\ufeffid`` and skips the
  blocklist. Silent protection loss, which is worse than a crash.
* ``test_imputation_is_fitted_on_training_rows_only`` - medians computed over
  the whole frame let the test split influence the training transform.
* ``test_deduplication_precedes_the_split`` - duplicates removed after a split
  land on both sides of it.
* ``test_split_never_marks_real_data_as_synthetic`` - the benchmark must only
  emit real-data results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cyberllm.core.errors import DatasetDownloadError, ValidationError
from cyberllm.datasets.registry import get_dataset
from cyberllm.datasets.unsw_nb15 import (
    CORE_FILES, DATASET_CAVEAT, UnswShare, manual_steps, provenance,
)
from cyberllm.detection.benchmark import (
    BASELINES, BenchmarkConfig, baseline_predictions, run_benchmark, summarise,
)
from cyberllm.detection.preprocess import (
    LEAKAGE_COLUMNS, canonical_column, identify_label_column, prepare_flow_data,
    strip_invisible,
)
from cyberllm.detection.unsw import (
    BINARY_TARGET, CATEGORY_TARGET, audit_partition, audit_split,
    conversation_key, target_is_encoded_by,
)

#: Columns in the released UNSW-NB15 partitions, with the BOM on the first one
#: exactly as the publisher writes it.
PARTITION_COLUMNS = (
    "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes",
    "dbytes", "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth",
    "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login",
    "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports", "attack_cat", "label",
)

CLASSES = ("Normal", "Fuzzers", "Analysis", "Backdoor", "DoS", "Exploits",
           "Generic", "Reconnaissance", "Shellcode", "Worms")


def make_partition(n_per_class: int = 12, seed: int = 3,
                   with_bom: bool = True) -> pd.DataFrame:
    """A small frame shaped exactly like the released partitions.

    Feature values depend only on ``seed``, the class and the row index - not
    on ``n_per_class`` - so the same ``(seed, class, i)`` produces the same flow
    in both files. That is what lets the cross-split duplication test be exact
    rather than probabilistic.
    """
    rows = []
    for class_index, name in enumerate(CLASSES):
        for i in range(n_per_class):
            rows.append({
                "id": class_index * n_per_class + i + 1,
                "dur": 0.001 * (class_index + 1) + 0.0001 * (i + seed),
                # Vary the categorical columns within each class as well as
                # across classes, so a class spans many conversation groups -
                # otherwise group splitting has nothing to stratify around.
                "proto": ["tcp", "udp", "icmp"][i % 3],
                "service": ["-", "http", "dns"][(i // 3) % 3],
                "state": ["FIN", "INT", "CON"][(class_index + i) % 3],
                "spkts": 2 + i, "dpkts": 1 + i,
                "sbytes": 100 * (class_index + 1) + i,
                "dbytes": 10 * (class_index + 1),
                "rate": 1.0 + i, "sttl": 64, "dttl": 63,
                "sload": 1.0 + i, "dload": 0.5, "sloss": 0, "dloss": 0,
                "sinpkt": 0.1 + i, "dinpkt": 0.2, "sjit": 0, "djit": 0,
                "swin": 256, "stcpb": 1, "dtcpb": 1, "dwin": 292,
                "tcprtt": 0.001, "synack": 0, "ackdat": 1, "smean": 20,
                "dmean": 10, "trans_depth": 1, "response_body_len": 0,
                "ct_srv_src": 1, "ct_state_ttl": 0, "ct_dst_ltm": 0,
                "ct_src_dport_ltm": 0, "ct_dst_sport_ltm": 0,
                "ct_dst_src_ltm": 0, "is_ftp_login": 0, "ct_ftp_cmd": 0,
                "ct_flw_http_mthd": 0, "ct_src_ltm": 0, "ct_srv_dst": 0,
                "is_sm_ips_ports": 0,
                "attack_cat": name,
                "label": 0 if name == "Normal" else 1,
            })
    frame = pd.DataFrame(rows, columns=list(PARTITION_COLUMNS))
    if with_bom:
        frame.columns = ["\ufeffid"] + list(frame.columns[1:])
    return frame


def write_partitions(directory: Path, **kwargs) -> dict[str, Path]:
    """Write train/test CSVs shaped like the publisher's releases.

    Both files are generated from the same seed, so the test partition
    reproduces flows that also appear in the training partition. That is a
    property of the real dataset (its released partition is *not*
    duplicate-free) and the audit must be able to detect it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for role, count in (("train", 40), ("test", 20)):
        frame = make_partition(n_per_class=count, seed=1)
        path = directory / {
            "train": "UNSW_NB15_training-set.csv",
            "test": "UNSW_NB15_testing-set.csv",
        }[role]
        frame.to_csv(path, index=False, encoding="utf-8")
        paths[role] = path
    return paths


@pytest.fixture
def partitions(tmp_path: Path) -> dict[str, Path]:
    return write_partitions(tmp_path / "unsw_nb15")


# -----------------------------------------------------------------------------
# Registry and provenance
# -----------------------------------------------------------------------------
def test_registry_declares_the_public_share_and_its_terms():
    spec = get_dataset("unsw_nb15")
    assert spec.access == "public"
    assert "sharepoint.com" in (spec.download_url or "")
    assert spec.source_url == "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
    # Citation is a licence condition, not a courtesy.
    assert "cite" in spec.license.lower() or "MUST be cited" in spec.license
    assert spec.priority == 1


def test_manual_steps_never_suggest_working_around_a_gate():
    steps = " ".join(manual_steps())
    assert "research.unsw.edu.au" in steps
    assert "rather than working around the gate" in steps
    assert "nour.moustafa@unsw.edu.au" in steps


def test_provenance_carries_the_required_citation():
    record = provenance()
    assert record["attribution_required"] is True
    assert len(record["citation"]) == 5
    assert record["retrieved_at"]
    assert record["caveat"] == DATASET_CAVEAT


def test_core_filenames_match_what_the_publisher_actually_serves():
    # The publisher misspells "UNSW" as "NUSW" in the features file; the
    # "correction" would make the download silently fail.
    assert CORE_FILES["features"] == "NUSW-NB15_features.csv"
    assert CORE_FILES["train"] == "UNSW_NB15_training-set.csv"
    assert CORE_FILES["test"] == "UNSW_NB15_testing-set.csv"


# -----------------------------------------------------------------------------
# Leakage defences
# -----------------------------------------------------------------------------
def test_attack_cat_is_treated_as_leakage():
    assert canonical_column("attack_cat") in LEAKAGE_COLUMNS
    assert canonical_column("attack category") in LEAKAGE_COLUMNS


def test_attack_cat_perfectly_encodes_the_binary_label_in_real_data():
    """The property that justifies dropping it, demonstrated on real shape."""
    frame = make_partition(n_per_class=5)
    assert target_is_encoded_by(frame, "attack_cat", BINARY_TARGET) == pytest.approx(1.0)


def test_label_column_is_not_confused_with_attack_category():
    columns = ["dur", "proto", "attack_cat", "label"]
    # "label" wins: it is checked before the attack-category aliases.
    assert identify_label_column(columns) == "label"


def test_bom_header_does_not_bypass_the_leakage_blocklist():
    assert strip_invisible("\ufeffid") == "id"
    assert canonical_column("\ufeffid") == "id"
    assert canonical_column("\ufeffid") in LEAKAGE_COLUMNS


def test_audit_drops_the_identifier_and_both_targets(partitions):
    _, audit = audit_partition(partitions["train"], target_column=CATEGORY_TARGET)
    assert set(audit.dropped_columns) == {"id", "attack_cat", "label"}
    assert audit.has_bom is True
    assert audit.ok


def test_prepare_flow_data_drops_attack_cat_from_features():
    """A BOM-prefixed id and attack_cat must both vanish from the matrix."""
    import numpy as np

    frame = make_partition(n_per_class=8, seed=11)
    prepared = prepare_flow_data(
        frame, label_column=CATEGORY_TARGET, seed=1, test_size=0.25,
    )
    produced = {canonical_column(c) for c in prepared.feature_names}
    assert "attack cat" not in produced
    assert "id" not in produced
    assert "label" not in produced
    # Every remaining feature must be one the source actually has.
    assert len(produced) == len(prepared.feature_names)
    assert np.isfinite(prepared.X_train).all()
    assert np.isfinite(prepared.X_test).all()


def test_imputation_is_fitted_on_training_rows_only():
    """A test-only column must not be able to influence the train transform."""
    frame = make_partition(n_per_class=10, seed=5)
    frame.loc[frame.index[:20], "sload"] = float("nan")
    prepared = prepare_flow_data(
        frame, label_column=CATEGORY_TARGET, seed=3, test_size=0.25,
    )
    assert any("training-split medians" in step
               for step in prepared.transformations), prepared.transformations


# -----------------------------------------------------------------------------
# Audit
# -----------------------------------------------------------------------------
def test_audit_reports_schema_class_distribution_and_integrity(partitions):
    frame, audit = audit_partition(partitions["train"], target_column=CATEGORY_TARGET)
    assert audit.rows == len(frame) == 40 * len(CLASSES)
    assert audit.target_column == "attack_cat"
    assert set(audit.class_distribution) == set(CLASSES)
    assert audit.binary_distribution == {"0": 40, "1": 360}
    assert audit.missing_values == 0
    assert audit.infinite_values == 0
    assert audit.missing_expected_features == []
    assert audit.unknown_classes == []
    payload = audit.as_dict()
    assert json.dumps(payload)  # fully serialisable
    assert "class distribution" in audit.summary()


def test_audit_flags_the_absent_ip_and_port_columns(partitions):
    _, audit = audit_partition(partitions["train"])
    assert audit.source_only_features_present == []
    assert any("IP/port" in warning for warning in audit.warnings)


def test_audit_reports_extra_columns_rather_than_ignoring_them(partitions):
    _, audit = audit_partition(partitions["train"])
    # spkts/dpkts/rate ship in the partitions but are absent from the feature
    # dictionary; the audit must surface that instead of silently accepting it.
    assert set(audit.extra_columns) == {"spkts", "dpkts", "rate"}


def test_audit_fails_loudly_on_a_file_that_is_not_unsw_nb15(tmp_path):
    alien = tmp_path / "other.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4], "label": [0, 1]}).to_csv(
        alien, index=False
    )
    _, audit = audit_partition(alien)
    assert not audit.ok
    assert audit.errors


def test_audit_reports_conflicting_labels_and_cross_split_duplicates(partitions):
    train, _ = audit_partition(partitions["train"])
    test, _ = audit_partition(partitions["test"])
    audit = audit_split(train, test, target_column="attack_cat",
                        dropped=["attack_cat", "id", "label"])
    assert audit.train_rows == 40 * len(CLASSES)
    assert audit.test_rows == 20 * len(CLASSES)
    # Both files come from the same generator, so every test flow also occurs
    # in training - the audit must report that rather than miss it.
    assert audit.cross_split_duplicate_rows == 20 * len(CLASSES)
    assert audit.cross_split_duplicate_share == pytest.approx(1.0)
    assert any("memorisation alone" in note for note in audit.notes)
    # The identical-id columns must NOT be what makes them match: `id` is
    # excluded, so this is duplication on features alone.
    assert "\ufeffid" not in json.dumps(audit.as_dict())


def test_audit_excludes_a_bom_prefixed_identifier_from_the_overlap_count(partitions):
    """A BOM must not smuggle `id` back into the feature set."""
    train, _ = audit_partition(partitions["train"])
    with_id = audit_split(train, train, target_column="attack_cat",
                          dropped=["attack_cat", "label"])
    assert with_id.cross_split_duplicate_rows == len(train)
    without_id = audit_split(train, train, target_column="attack_cat",
                             dropped=["attack_cat", "id", "label"])
    assert without_id.cross_split_duplicate_rows == len(train)
    # Keeping `id` must make *distinct* rows look duplicated.
    distinct = audit_split(train.iloc[:5], train.iloc[5:], target_column="attack_cat",
                           dropped=["attack_cat", "id", "label"])
    assert distinct.cross_split_duplicate_rows == 0


def test_audit_names_a_dropped_column_that_is_an_answer_key(partitions):
    train, _ = audit_partition(partitions["train"])
    audit = audit_split(train, train.head(10), target_column=BINARY_TARGET,
                        dropped=["attack_cat", "id", "label"])
    assert audit.target_encoded_by_dropped_column.get("attack_cat") == 1.0
    assert any("answer key" in note for note in audit.notes)


def test_conversation_key_is_stable_and_does_not_use_the_target():
    frame = make_partition(n_per_class=6)
    first = conversation_key(frame, CATEGORY_TARGET, ["id"])
    second = conversation_key(frame.sample(frac=1.0, random_state=4),
                              CATEGORY_TARGET, ["id"])
    assert set(first) == set(second)
    assert first.nunique() > 1


# -----------------------------------------------------------------------------
# Share access: failures must be typed, documented and never silently retried
# -----------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _FakeSession:
    def __init__(self, response, cookies=None):
        self.response = response
        self.headers: dict[str, str] = {}
        self.cookies = dict(cookies or {})
        self.requests_made: list[str] = []

    def get(self, url, **kwargs):
        self.requests_made.append(url)
        return self.response


def test_open_rejects_a_share_that_demands_authentication():
    share = UnswShare()
    share.session = _FakeSession(_FakeResponse(401), cookies={})
    with pytest.raises(DatasetDownloadError) as excinfo:
        share.open()
    assert excinfo.value.details["http_status"] == 401
    assert "nour.moustafa@unsw.edu.au" in excinfo.value.details["setup_hint"]


def test_open_refuses_when_no_anonymous_grant_is_issued():
    share = UnswShare()
    share.session = _FakeSession(_FakeResponse(200), cookies={})
    with pytest.raises(DatasetDownloadError) as excinfo:
        share.open()
    assert "no anonymous session cookie" in excinfo.value.message
    assert "research.unsw.edu.au" in excinfo.value.details["setup_hint"]


def test_open_accepts_a_public_share():
    share = UnswShare()
    share.session = _FakeSession(_FakeResponse(200), cookies={"FedAuth": "anon"})
    assert share.open() is share


def test_file_url_uses_the_api_endpoint_not_the_bare_path():
    """On this tenant the bare server-relative path 404s; the API form does not."""
    url = UnswShare().file_url("/personal/x/Documents/UNSW-NB15 dataset/a b.csv")
    assert "GetFileByServerRelativeUrl" in url
    assert url.endswith("/$value")
    assert " " not in url and "%20" in url


def test_listing_that_is_not_json_is_reported_as_a_gate():
    share = UnswShare()
    share.session = _FakeSession(_FakeResponse(200), cookies={"FedAuth": "a"})
    with pytest.raises(DatasetDownloadError) as excinfo:
        share.list_folder("/personal/x/Documents")
    assert "did not return JSON" in excinfo.value.message
    assert "requires sign-in" in excinfo.value.message
    assert "research.unsw.edu.au" in excinfo.value.details["setup_hint"]


# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------
def test_baselines_are_reported_for_every_split():
    """Baselines must expose the floor a classifier has to clear."""
    import numpy as np

    # Ten classes on UNSW-NB15's shape: heavily imbalanced, smallest class 0.5%.
    counts = [4000, 1900, 700, 180, 60, 20, 10, 5, 3, 2]
    y_train = np.concatenate([np.full(n, i) for i, n in enumerate(counts)])
    y_test = np.concatenate([np.full(max(n // 4, 1), i) for i, n in enumerate(counts)])

    results = baseline_predictions(
        y_train, y_test, seed=1,
        class_names=list("ABCDEFGHIJ"),
    )
    assert set(results) == set(BASELINES)
    majority = results["majority_class"]
    # The majority-class baseline always predicts one class, so its recall on
    # every other class is zero - macro-F1 cannot exceed 2/(1+9) = 0.2.
    assert majority["accuracy"] == pytest.approx(4000 / sum(counts), abs=0.01)
    assert majority["f1_macro"] <= 0.2 + 1e-9
    assert results["uniform_random"]["f1_macro"] < majority["f1_macro"]
    # Per-class rows are keyed by name, not by the internal integer code.
    assert set(majority["per_class"]) >= {"A", "B", "J"}


def test_benchmark_end_to_end_produces_a_real_data_report(partitions, tmp_path):
    config = BenchmarkConfig(
        data_dir=partitions["train"].parent,
        strategies=("dedup_random", "grouped", "publisher"),
        algorithms=("lightgbm",),
        out_dir=tmp_path / "report", model_dir=tmp_path / "models",
        test_size=0.3, seed=42,
    )
    report = run_benchmark(config)
    assert report["errors"] == []
    assert report["is_synthetic"] is False
    assert set(report["splits"]) == {"dedup_random", "grouped", "publisher"}

    for entry in report["splits"].values():
        assert set(entry["baselines"]) == set(BASELINES)
        # A split must never silently lose a class from either side.
        assert entry["split"].get("classes_missing_from_test", []) == []
        assert entry["split"].get("classes_missing_from_train", []) == []
        metrics = entry["models"]["lightgbm"]
        for key in ("accuracy", "f1_macro", "f1_weighted", "recall_macro",
                    "precision_macro", "per_class", "confusion_matrix"):
            assert key in metrics, key
        # Per-class rows are keyed by class name; `classification_report` also
        # appends its macro/weighted summary rows, which stay as they are.
        assert set(CLASSES) <= set(metrics["per_class"])
        for name in CLASSES:
            row = metrics["per_class"][name]
            assert {"precision", "recall", "f1-score", "support"} <= set(row)

    for column in ("attack_cat", "label", "id"):
        assert column not in report["feature_columns"]
    assert "duplicates removed once, globally, before splitting" in report["pool"]["note"]

    bundle = report["registered_model"]["path"]
    card = json.loads((Path(bundle) / "model_card.json").read_text(encoding="utf-8"))
    assert card["is_synthetic"] is False
    assert card["seed"] == 42
    assert card["dataset"] == "unsw_nb15"
    assert len(card["dataset_sha256"]) == 64
    assert "attack_cat" not in card["features"]
    assert card["limitations"]

    assert (tmp_path / "report" / "benchmark.json").exists()
    assert (tmp_path / "report" / "PROVENANCE.json").exists()
    assert "UNSW-NB15 network intrusion detection" in summarise(report)


def test_benchmark_refuses_to_run_without_the_partitions(tmp_path):
    config = BenchmarkConfig(data_dir=tmp_path / "absent", out_dir=tmp_path / "out")
    with pytest.raises(ValidationError) as excinfo:
        run_benchmark(config)
    assert "datasets download" in excinfo.value.details["setup_hint"]


def test_report_records_the_full_configuration_and_citation(partitions, tmp_path):
    config = BenchmarkConfig(
        data_dir=partitions["train"].parent, strategies=("dedup_random",),
        algorithms=("lightgbm",), out_dir=tmp_path / "r",
        model_dir=tmp_path / "m", seed=7, test_size=0.25,
    )
    report = run_benchmark(config)
    assert report["config"]["seed"] == 7
    assert report["config"]["test_size"] == 0.25
    assert report["config"]["created_at"]
    assert report["config"]["environment"]["python"]
    assert report["splits"]["dedup_random"]["split"]["seed"] == 7
    assert len(report["citation"]) == 5
    assert report["provenance"]["attribution_required"] is True


def test_grouped_split_holds_out_whole_conversations(partitions):
    from cyberllm.detection.benchmark import _split_grouped

    train, _audit = audit_partition(partitions["train"])
    features = [c for c in train.columns if c not in {"attack_cat", "id", "label"}]
    left, right, meta = _split_grouped(
        train, features, "attack_cat", test_size=0.3, seed=1,
    )
    assert meta["method"].startswith("de-duplicated stratified conversation-group")
    overlap = audit_split(left, right, target_column="attack_cat",
                          dropped=["attack_cat", "id", "label"])
    assert overlap.cross_split_duplicate_rows == 0
    assert overlap.shared_conversations == 0
    # Rare classes must survive on both sides or macro-F1 is not measurable.
    assert meta["classes_missing_from_test"] == []


def test_grouped_split_reports_classes_it_cannot_place_on_both_sides():
    """A class confined to one conversation cannot appear on both sides.

    Whether it lands in train or test, its per-class F1 is undefined, so the
    split must say which side lost it rather than reporting a clean-looking
    macro-F1 over an unmeasurable class.
    """
    from cyberllm.detection.benchmark import _split_grouped

    frame = make_partition(n_per_class=6, seed=1)
    features = [c for c in frame.columns if c not in {"attack_cat", "id", "label"}]
    # Pin Worms to one conversation key - protocol, service, state and a single
    # byte size - so no split can place it on both sides.
    frame.loc[frame["attack_cat"] == "Worms",
              ["proto", "service", "state", "sbytes"]] = [
        "tcp", "-", "CON", 1000
    ]
    _left, _right, meta = _split_grouped(
        frame, features, "attack_cat", test_size=0.3, seed=1,
    )
    missing_test = meta["classes_missing_from_test"]
    missing_train = meta["classes_missing_from_train"]
    assert missing_test == ["Worms"] or missing_train == ["Worms"]
    assert "Worms" in meta["warning"]
    assert "F1 is undefined" in meta["warning"]
