"""UNSW-NB15 specific loading, schema validation and leakage auditing.

Why this module exists
----------------------
The generic path in :mod:`cyberllm.detection.preprocess` is enough to *train*
on a flow CSV, but it cannot answer the questions a benchmark has to answer
before its numbers mean anything:

* Is the file the dataset the registry claims, with the documented columns?
* Which columns encode the target, and would leave-in inflate the score?
* How many rows are duplicated, and how many duplicates straddle the
  train/test boundary?
* Are related flows (the same conversation) spread across both splits?

Every function here is dataset-specific on purpose: a general "IDS dataset"
helper would have to guess, and guessing is what produced the two real defects
recorded in :data:`LEAKAGE_COLUMNS` and :data:`CANONICAL_TARGETS`.

Schema facts, read from the publisher's own feature dictionary and both
released partitions (verified, not assumed)::

    columns        45 per row
    identifier     `id`          - unique per flow, restarts at 1 per file
    features       40 Argus/Bro-IDS flow statistics
    categorical    proto, service, state
    binary target  `label`       - 0 = normal, 1 = attack
    category       `attack_cat`  - Normal or one of 9 attack families
    missing values none
    BOM            present on the first header cell (`id`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ValidationError
from ..core.logging_config import get_logger
from .preprocess import LEAKAGE_COLUMNS, canonical_column, strip_invisible

logger = get_logger("detection.unsw")

#: Columns that hold a target. Both are leakage with respect to *either*
#: target, because ``attack_cat == "Normal"`` if and only if ``label == 0``.
CANONICAL_TARGETS: tuple[str, ...] = ("label", "attack_cat")

#: Columns the publisher's feature dictionary defines. Used to prove a file
#: really is UNSW-NB15 and not some other CSV dropped in the same folder.
#:
#: Note the four IP/port columns are absent here on purpose. They exist in the
#: feature dictionary and in the four ~2M-row source files, but the released
#: train/test *partitions* do not carry them - verified, not assumed. They are
#: listed in :data:`SOURCE_ONLY_FEATURES` and reported separately, because
#: treating their absence as a schema error would make a perfectly valid file
#: unusable, and ignoring it would hide a real limitation.
EXPECTED_FEATURES: tuple[str, ...] = (
    "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss",
    "dloss", "service", "sload", "dload", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
    "ct_srv_dst", "is_sm_ips_ports",
)

#: Documented in the feature dictionary and present in the ~2M-row source
#: files, but deliberately not distributed in the released partitions.
#:
#: This is why conversation grouping has to use a proxy: without addresses
#: and ports there is no 5-tuple to group on.
SOURCE_ONLY_FEATURES: tuple[str, ...] = ("srcip", "sport", "dstip", "dsport")

#: Categorical columns in the released partitions.
CATEGORICAL_FEATURES: tuple[str, ...] = ("proto", "service", "state")

#: The nine attack families plus the normal class, as published.
EXPECTED_CLASSES: tuple[str, ...] = (
    "Normal", "Fuzzers", "Analysis", "Backdoor", "DoS", "Exploits", "Generic",
    "Reconnaissance", "Shellcode", "Worms",
)

#: Column used as the supervised target for the multiclass benchmark.
CATEGORY_TARGET = "attack_cat"

#: Column used as the supervised target for the binary benchmark.
BINARY_TARGET = "label"

#: The identifier column. Unique per flow, so it can only memorise rows.
IDENTIFIER_COLUMN = "id"


def read_unsw_csv(path: Path, **kwargs: Any) -> Any:
    """Read one UNSW-NB15 CSV, tolerating the publisher's encoding quirks.

    The files are UTF-8 *with a BOM* and the feature dictionary contains at
    least one cp1252 byte. Both are handled explicitly rather than by hoping
    a later reader is lenient.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise ValidationError(
            f"UNSW-NB15 file not found: {path}",
            details={"setup_hint": "python -m cyberllm.cli datasets download "
                                   "--name unsw_nb15"},
        )
    frame = pd.read_csv(
        path, encoding="utf-8", encoding_errors="replace",
        low_memory=False, **kwargs,
    )
    frame.columns = [strip_invisible(c).strip() for c in frame.columns]
    return frame


@dataclass
class SchemaAudit:
    """What a UNSW-NB15 partition actually contains.

    This is the artefact that answers "can we trust a number computed on this
    file?" - it is deliberately verbose and is written next to the results.
    """

    file: str
    rows: int = 0
    columns: list[str] = field(default_factory=list)
    missing_expected_features: list[str] = field(default_factory=list)
    source_only_features_present: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    target_column: str = ""
    dropped_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    class_distribution: dict[str, int] = field(default_factory=dict)
    binary_distribution: dict[str, int] = field(default_factory=dict)
    missing_values: int = 0
    infinite_values: int = 0
    duplicate_rows: int = 0
    duplicate_feature_rows: int = 0
    unknown_classes: list[str] = field(default_factory=list)
    has_bom: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "rows": self.rows,
            "column_count": len(self.columns),
            "columns": self.columns,
            "target_column": self.target_column,
            "missing_expected_features": self.missing_expected_features,
            "source_only_features_present": self.source_only_features_present,
            "extra_columns": self.extra_columns,
            "dropped_columns": self.dropped_columns,
            "categorical_columns": self.categorical_columns,
            "class_distribution": self.class_distribution,
            "binary_distribution": self.binary_distribution,
            "missing_values": self.missing_values,
            "infinite_values": self.infinite_values,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_feature_rows": self.duplicate_feature_rows,
            "unknown_classes": self.unknown_classes,
            "utf8_bom_present": self.has_bom,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        lines = [
            f"{self.file}: {self.rows:,} rows x {len(self.columns)} columns",
            f"  target        : {self.target_column or '(none found)'}",
            f"  dropped       : {', '.join(self.dropped_columns) or 'none'}",
            f"  missing values: {self.missing_values:,}"
            f" | infinite: {self.infinite_values:,}",
            f"  duplicates    : {self.duplicate_rows:,} full-row, "
            f"{self.duplicate_feature_rows:,} feature-only",
        ]
        if self.class_distribution:
            total = sum(self.class_distribution.values()) or 1
            lines.append("  class distribution:")
            for name, count in sorted(self.class_distribution.items(),
                                      key=lambda kv: -kv[1]):
                lines.append(f"    {name:<18} {count:>9,}  ({count / total:6.2%})")
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        for error in self.errors:
            lines.append(f"  x {error}")
        return "\n".join(lines)


def audit_partition(path: Path, *, target_column: str = CATEGORY_TARGET,
                    label_column: str = BINARY_TARGET) -> tuple[Any, SchemaAudit]:
    """Load one partition and audit it without altering the data."""
    import numpy as np
    import pandas as pd

    path = Path(path)
    raw_head = path.open("rb").read(3) if path.exists() else b""
    frame = read_unsw_csv(path)
    audit = SchemaAudit(file=path.name, rows=int(len(frame)),
                        columns=list(frame.columns),
                        has_bom=raw_head.startswith(b"\xef\xbb\xbf"))
    normalised = {canonical_column(c): c for c in frame.columns}

    # 1. Schema: prove this is UNSW-NB15 and not a look-alike CSV.
    documented = {canonical_column(f) for f in EXPECTED_FEATURES}
    targets = {canonical_column(t) for t in CANONICAL_TARGETS}
    audit.missing_expected_features = [
        f for f in EXPECTED_FEATURES if canonical_column(f) not in normalised
    ]
    if audit.missing_expected_features:
        audit.errors.append(
            f"{len(audit.missing_expected_features)} documented feature(s) absent: "
            f"{', '.join(audit.missing_expected_features[:8])}"
        )
    audit.source_only_features_present = [
        f for f in SOURCE_ONLY_FEATURES if canonical_column(f) in normalised
    ]
    if not audit.source_only_features_present:
        audit.warnings.append(
            "no IP/port columns in this file (expected in the released "
            "partitions) - conversation grouping falls back to a feature proxy "
            "and the model cannot see host or service-port identity"
        )
    audit.extra_columns = [
        c for c in frame.columns
        if canonical_column(c) not in documented | targets
        and canonical_column(c) != IDENTIFIER_COLUMN
    ]
    if audit.extra_columns:
        audit.warnings.append(
            f"undocumented column(s) present: {', '.join(audit.extra_columns)}"
        )

    # 2. Targets and class distribution.
    target = normalised.get(canonical_column(target_column))
    audit.target_column = target or ""
    if target is None:
        audit.errors.append(f"target column '{target_column}' not present")
    else:
        counts = frame[target].astype(str).value_counts()
        audit.class_distribution = {str(k): int(v) for k, v in counts.items()}
        unknown = sorted(set(audit.class_distribution) - set(EXPECTED_CLASSES))
        if unknown:
            audit.unknown_classes = unknown
            audit.warnings.append(f"unexpected class label(s): {', '.join(unknown)}")

    binary = normalised.get(canonical_column(label_column))
    if binary is not None:
        audit.binary_distribution = {
            str(k): int(v)
            for k, v in frame[binary].astype(str).value_counts().items()
        }
    else:
        audit.warnings.append(f"binary column '{label_column}' not present")

    # 3. Columns that must never become features.
    leakage = [c for c in frame.columns if canonical_column(c) in LEAKAGE_COLUMNS]
    if IDENTIFIER_COLUMN in frame.columns and IDENTIFIER_COLUMN not in leakage:
        leakage.append(IDENTIFIER_COLUMN)
    audit.dropped_columns = sorted(set(leakage))
    audit.categorical_columns = [
        c for c in frame.columns
        if canonical_column(c) in {canonical_column(x) for x in CATEGORICAL_FEATURES}
    ]

    # 4. Integrity.
    audit.missing_values = int(frame.isna().sum().sum())
    numeric = frame.select_dtypes(include="number")
    if not numeric.empty:
        audit.infinite_values = int(
            np.isinf(numeric.to_numpy(dtype="float64")).sum()
        )
    if audit.missing_values:
        audit.warnings.append(f"{audit.missing_values:,} missing value(s) present")
    if audit.infinite_values:
        audit.warnings.append(f"{audit.infinite_values:,} infinite value(s) present")

    audit.duplicate_rows = int(frame.duplicated().sum())
    keep = [c for c in frame.columns if c not in audit.dropped_columns]
    # Hashed rather than a 42-column pandas groupby: at 175k rows the object
    # comparison is minutes of CPU, and the hash answers the same question.
    audit.duplicate_feature_rows = (
        int(len(frame) - len(pd.unique(row_hashes(frame, keep))))
        if keep else 0
    )
    if audit.duplicate_rows:
        audit.warnings.append(
            f"{audit.duplicate_rows:,} exact duplicate row(s) - remove before "
            "splitting or they straddle the train/test boundary"
        )
    return frame, audit


@dataclass
class LeakageAudit:
    """Evidence about how much the split can be trusted.

    ``cross_split_duplicate_rows`` is the headline number: rows whose feature
    vector appears in *both* the training and the test partition. Even a small
    share inflates the test score, because the model is being asked about
    rows it has already seen.
    """

    train_rows: int = 0
    test_rows: int = 0
    train_duplicate_rows: int = 0
    test_duplicate_rows: int = 0
    cross_split_duplicate_rows: int = 0
    cross_split_duplicate_share: float = 0.0
    shared_distinct_feature_vectors: int = 0
    shared_conversations: int = 0
    total_conversations: int = 0
    conflicting_label_rows: int = 0
    target_encoded_by_dropped_column: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "train_duplicate_rows": self.train_duplicate_rows,
            "test_duplicate_rows": self.test_duplicate_rows,
            "cross_split_duplicate_rows": self.cross_split_duplicate_rows,
            "cross_split_duplicate_share": round(self.cross_split_duplicate_share, 6),
            "shared_distinct_feature_vectors": self.shared_distinct_feature_vectors,
            "shared_conversations": self.shared_conversations,
            "total_conversations": self.total_conversations,
            "conflicting_label_rows": self.conflicting_label_rows,
            "target_encoded_by_dropped_column": self.target_encoded_by_dropped_column,
            "notes": self.notes,
        }


def conversation_key(frame: Any, target_column: str,
                     dropped: list[str]) -> Any:
    """Group flows that plausibly belong to the same conversation.

    UNSW-NB15 omits IP addresses and ports from the released feature tables,
    so a true 5-tuple is unavailable. What remains that is stable within a
    conversation is the protocol/service/state triple plus a coarse size
    bucket. This is a *proxy*: it over-groups in places and under-groups in
    others, which is exactly why the benchmark reports it as a limitation
    rather than as a guarantee of independence.
    """
    import pandas as pd

    sizes = frame["sbytes"]
    # Bucket the *values*, not their ranks: UNSW-NB15 has tens of thousands of
    # flows with byte-identical sizes, and ranking first would scatter those
    # twins across different buckets - splitting one conversation into several
    # groups and quietly weakening the grouped split.
    if sizes.nunique(dropna=True) < 2:
        buckets = pd.Series("0", index=frame.index)
    else:
        buckets = pd.qcut(sizes, q=min(20, int(sizes.nunique())),
                          labels=False, duplicates="drop").fillna(0).astype(int)
    return (
        frame.get("proto", pd.Series(index=frame.index, dtype=object)).astype(str)
        + "|" + frame.get("service", pd.Series(index=frame.index, dtype=object)).astype(str)
        + "|" + frame.get("state", pd.Series(index=frame.index, dtype=object)).astype(str)
        + "|" + buckets.astype(str)
    )


def target_is_encoded_by(frame: Any, column: str, target_column: str) -> float:
    """Share of rows where ``column`` alone determines the target.

    1.0 means the column is a perfect label proxy: leaving it in as a feature
    turns the task into reading the answer key, and any score obtained is
    meaningless. This is how ``attack_cat`` was caught against ``label``.
    """
    if column not in frame.columns or target_column not in frame.columns:
        return 0.0
    grouped = frame.groupby(frame[column].astype(str), observed=True)[
        target_column
    ].nunique(dropna=False)
    return float((grouped <= 1).mean())


def row_hashes(frame: Any, columns: list[str]) -> Any:
    """A 64-bit content hash per row over ``columns``.

    Grouping 42 object columns across 250k rows is prohibitively slow, and
    the only question being asked is "are these two rows identical?". A 64-bit
    content hash answers it: at 250k rows the birthday collision probability is
    about 2e-9. A collision could only ever merge two distinct rows, which
    would make the reported duplication figure an *under*-estimate, so the
    leakage numbers stay conservative.

    Only the object/categorical columns are stringified. Casting all 40+ numeric
    columns to Python strings first turns ~10M objects into garbage and costs
    minutes of CPU for no extra correctness; numerics are normalised to
    float64 instead, which also stops an int64 column in one partition and a
    float64 column in the other from hashing differently.
    """
    import numpy as np
    import pandas as pd

    block = pd.DataFrame(index=frame.index)
    for column in columns:
        series = frame[column]
        if series.dtype == object or str(series.dtype) in {"str", "string"}:
            block[column] = series.astype(str)
        else:
            block[column] = series.to_numpy(dtype="float64", na_value=np.nan)
    return pd.util.hash_pandas_object(block, index=False).to_numpy()


def _split_conversation_keys(train_frame: Any, test_frame: Any,
                            target_column: str,
                            dropped: list[str]) -> tuple[set[str], set[str]]:
    """Conversation keys for both halves, bucketed over the *combined* frames.

    ``conversation_key`` assigns quantile buckets, so calling it separately on
    each half makes bucket boundaries depend on which rows are in that half -
    a key that means "small flow" in train can mean "large flow" in test. That
    silently invents shared conversations out of nothing. Computing the key
    once over the concatenation keeps the two sides comparable, which is the
    only way the "groups straddling the split" number means anything.
    """
    import pandas as pd

    combined = pd.concat([train_frame, test_frame], ignore_index=True)
    keys = conversation_key(combined, target_column, dropped)
    return set(keys.iloc[: len(train_frame)]), set(keys.iloc[len(train_frame):])


def audit_split(train_frame: Any, test_frame: Any, *,
                target_column: str = CATEGORY_TARGET,
                dropped: list[str] | None = None) -> LeakageAudit:
    """Measure duplication, cross-split overlap and target proxies.

    Runs on raw frames (before any splitting) so it can report what the
    *existing* partition already leaks, and on freshly built ones so a new
    split can be shown to leak less.
    """
    import numpy as np
    import pandas as pd

    dropped = list(dropped or [])
    # Compare canonically, not literally: a BOM-prefixed `id` is the same
    # column as `id`, and a raw-name comparison would quietly leave the
    # identifier in the hash and the caller would never know.
    excluded = {canonical_column(c) for c in dropped}
    excluded.add(canonical_column(target_column))
    keep = [c for c in train_frame.columns
            if canonical_column(c) not in excluded]
    shared = [c for c in keep if c in test_frame.columns]
    audit = LeakageAudit(
        train_rows=int(len(train_frame)),
        test_rows=int(len(test_frame)),
    )

    if not shared:
        audit.notes.append("no shared feature columns between the two frames")
        return audit

    train_hashes = row_hashes(train_frame, shared)
    test_hashes = row_hashes(test_frame, shared)
    audit.train_duplicate_rows = int(
        len(train_hashes) - len(pd.unique(train_hashes))
    )
    audit.test_duplicate_rows = int(len(test_hashes) - len(pd.unique(test_hashes)))

    # Count *test rows* that have a twin in training. Using
    # `pd.Index(train_hashes).isin(test_hashes)` here would answer a different
    # question - how many distinct training vectors recur in test - and report
    # it under a name that claims otherwise.
    train_unique = pd.Index(pd.unique(train_hashes))
    position = train_unique.get_indexer(test_hashes)
    duplicated_in_test = position >= 0
    audit.cross_split_duplicate_rows = int(duplicated_in_test.sum())
    audit.cross_split_duplicate_share = (
        float(duplicated_in_test.mean()) if len(duplicated_in_test) else 0.0
    )
    audit.shared_distinct_feature_vectors = int(
        train_unique[position[duplicated_in_test]].nunique()
    ) if duplicated_in_test.any() else 0
    if audit.cross_split_duplicate_rows:
        audit.notes.append(
            f"{audit.cross_split_duplicate_rows:,} test row(s) "
            f"({audit.cross_split_duplicate_share:.2%}) are identical to a "
            f"training row on all {len(shared)} feature columns; a model can "
            f"score these by memorisation alone"
        )

    # Rows whose features are identical but whose labels disagree are
    # irreducible: no feature-based model can get them right.
    combined = pd.DataFrame({
        "hash": np.concatenate([train_hashes, test_hashes]),
        "label": pd.concat(
            [train_frame[target_column].astype(str),
             test_frame[target_column].astype(str)], ignore_index=True,
        ).to_numpy(),
    })
    audit.conflicting_label_rows = int(
        combined.groupby("hash", observed=True)["label"].nunique().gt(1).sum()
    )
    if audit.conflicting_label_rows:
        audit.notes.append(
            f"{audit.conflicting_label_rows:,} feature vector(s) carry more than "
            "one target value; they cap achievable accuracy"
        )

    train_groups, test_groups = _split_conversation_keys(
        train_frame, test_frame, target_column, dropped
    )
    audit.total_conversations = len(train_groups | test_groups)
    audit.shared_conversations = len(train_groups & test_groups)
    if audit.shared_conversations:
        audit.notes.append(
            f"{audit.shared_conversations:,} of {audit.total_conversations:,} "
            "conversation groups appear in both splits"
        )

    for column in dropped:
        if column in train_frame.columns and column != target_column:
            score = target_is_encoded_by(train_frame, column, target_column)
            if score > 0:
                audit.target_encoded_by_dropped_column[column] = round(score, 6)
    for column, score in audit.target_encoded_by_dropped_column.items():
        if score >= 0.999:
            audit.notes.append(
                f"dropped column '{column}' perfectly determines the target "
                f"(purity {score:.3f}) - it is an answer key, not a feature"
            )
    return audit


__all__ = [
    "SchemaAudit", "LeakageAudit", "audit_partition", "audit_split",
    "conversation_key", "target_is_encoded_by", "read_unsw_csv",
    "CANONICAL_TARGETS", "EXPECTED_FEATURES", "SOURCE_ONLY_FEATURES",
    "EXPECTED_CLASSES", "CATEGORICAL_FEATURES", "CATEGORY_TARGET",
    "BINARY_TARGET", "IDENTIFIER_COLUMN",
]
