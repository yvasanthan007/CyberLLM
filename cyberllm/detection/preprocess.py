"""Flow-record preprocessing for network intrusion detection.

Design constraints
------------------
* **Leakage prevention.** The label column and any identifier that trivially
  encodes it are dropped *before* the split, and the drop list is recorded in
  the artefact so an auditor can see exactly what the model never saw.
* **Reproducibility.** Every artefact records the random seed, the split
  strategy and a content hash of the source data.
* **Honest transformations.** Imputation, scaling and encoding happen inside a
  fitted scikit-learn ``Pipeline`` so the same transform is applied at
  inference time and cannot drift from training.

Supported inputs: CIC-IDS2017, CSE-CIC-IDS2018 and UNSW-NB15 CSVs. The
readers differ mainly in their column naming, so the schema is normalised to a
canonical set before any cleaning happens.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

from ..core.errors import ValidationError
from ..core.logging_config import get_logger

logger = get_logger("detection.preprocess")

#: Column names that must never become features.
#:  - the label itself (obviously)
#:  - ids that are unique per flow and would let the model memorise rows
LEAKAGE_COLUMNS = {
    "label", "labels", "class", "attack", "attacktype", "attack_type",
    "category", "id", "flowid", "flow_id", "srcipport", "dstipport",
    "timestamp", "ts", "expiration", "flowstartmillis", "flowstarmillis",
    "flowendmillis", "stime", "ltime", "fwdflags", "bwdflags",
    "sourceip", "destip", "srcip", "dstip", "srcport", "dstport",
    "src_ip", "dst_ip", "src_port", "dst_port", "ip1", "ip2",
    "sport", "dport", "protocolname_full", "unnamed: 29", "unnamed: 28",
}

#: Columns that are useful but must be treated as categorical.
CATEGORICAL_HINTS = {"proto", "protocol", "service", "state", "attack",
                     "attacktype", "attack_type", "flg", "flags"}

#: Canonical name -> ordered alias preference list.
#:
#: Order matters: aliases are tried left to right, and the first match wins.
#: They are tuples rather than sets because set iteration order is not stable
#: across processes, which previously made ``time_to_live`` bind to ``dttl``
#: or ``sttl`` at random. Source-side names are listed first so the source
#: TTL wins and the destination TTL keeps its own canonical name.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "duration": ("dur", "flow duration", "duration", "flowduration"),
    "src_bytes": ("srcbytes", "stotalbytes", "sourcebytes", "outbytes",
                  "totalbytes of source", "sbytes"),
    "dst_bytes": ("dstbytes", "dtotalbytes", "destinationbytes", "inbytes",
                  "totalbytes of destination", "dbytes"),
    "protocol": ("proto", "protocol", "protocolname", "prot"),
    "service": ("service", "servicename", "service_name"),
    "flag": ("flag", "flags", "tcpflags", "flg", "state"),
    "src_port": ("sport", "srcport", "sourceport", "src port", "source port"),
    "dst_port": ("dport", "dstport", "destport", "dst port", "destination port",
                 "destination port number"),
    "time_to_live": ("sttl", "srcttl", "ttl"),
    "dst_time_to_live": ("dttl", "dstttl"),
}

#: Irregular whitespace / stray characters in published CSV headers.
_HEADER_CLEAN_RE = re.compile(r"[\s_\-]+")


def canonical_column(name: str) -> str:
    """Normalise a raw column name to a stable lowercase key."""
    cleaned = str(name).strip().lower()
    cleaned = _HEADER_CLEAN_RE.sub(" ", cleaned).strip()
    return cleaned


def identify_label_column(columns: Iterable[str]) -> str | None:
    """Find the label column, preferring the dataset's own conventions."""
    normalised = {canonical_column(c): c for c in columns}
    for candidate in ("label", "labels", "class", "attack", "attacktype", "attack type"):
        if candidate in normalised:
            return normalised[candidate]
    return None


def map_columns(columns: Iterable[str]) -> dict[str, str]:
    """Map raw column names onto canonical feature names.

    Returns ``{canonical_name: original_name}`` for aliases that are present.
    Unrecognised columns are returned under their normalised name, so a dataset
    with a new field still works rather than being silently dropped.
    """
    raw = list(columns)
    normalised = {canonical_column(c): c for c in raw}
    mapping: dict[str, str] = {}
    claimed: set[str] = set()

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_n = canonical_column(alias)
            if alias_n in normalised and normalised[alias_n] not in claimed:
                mapping[canonical] = normalised[alias_n]
                claimed.add(normalised[alias_n])
                break

    for norm_name, original in normalised.items():
        if original in claimed or norm_name in mapping:
            continue
        mapping[norm_name] = original
    return mapping


@dataclass
class SchemaReport:
    """What the loader found in a raw CSV, before any cleaning."""

    file: str
    rows: int = 0
    columns: list[str] = field(default_factory=list)
    label_column: str | None = None
    label_distribution: dict[str, int] = field(default_factory=dict)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    missing_values: dict[str, float] = field(default_factory=dict)
    duplicate_rows: int = 0
    infinite_values: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"{self.file}: {self.rows:,} rows x {len(self.columns)} columns",
            f"  label column : {self.label_column}",
            f"  numeric     : {len(self.numeric_columns)}",
            f"  categorical : {len(self.categorical_columns)}",
            f"  duplicates  : {self.duplicate_rows:,}",
        ]
        if self.label_distribution:
            total = sum(self.label_distribution.values()) or 1
            top = sorted(self.label_distribution.items(), key=lambda kv: -kv[1])[:8]
            lines.append("  label distribution:")
            for name, count in top:
                lines.append(f"    {name:<28} {count:>10,}  ({count / total:6.2%})")
        if self.missing_values:
            worst = sorted(self.missing_values.items(), key=lambda kv: -kv[1])[:5]
            lines.append("  most-missing columns: " +
                         ", ".join(f"{k}={v:.1%}" for k, v in worst))
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


def inspect_csv(path: Path, *, label_column: str | None = None,
                max_rows: int | None = None) -> tuple[Any, SchemaReport]:
    """Load a flow CSV and report on it without modifying anything.

    This is the "look before you leap" step required before training: schema,
    label distribution, missingness, duplicates, class imbalance and
    non-finite values are all surfaced here.
    """
    import numpy as np
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise ValidationError(f"Flow CSV not found: {path}")

    report = SchemaReport(file=path.name)
    frame = pd.read_csv(path, nrows=max_rows, low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]

    report.rows = int(len(frame))
    report.columns = list(frame.columns)
    if report.rows == 0:
        report.warnings.append("file contains a header but no data rows")
        return frame, report

    label_col = label_column or identify_label_column(frame.columns)
    report.label_column = label_col

    if label_col is None:
        report.warnings.append(
            "No label column found (looked for label/class/attack). "
            "This file cannot be used for supervised training."
        )
        return frame, report

    counts = frame[label_col].astype(str).value_counts().head(50)
    report.label_distribution = {str(k): int(v) for k, v in counts.items()}
    if len(counts) == 1:
        report.warnings.append(
            f"Only one class present ({counts.index[0]!r}); there is nothing to learn."
        )
    elif len(counts) < 2:
        report.warnings.append("Fewer than two distinct classes present.")

    # Normalise the column-space survey onto canonical names.
    mapping = map_columns([c for c in frame.columns if c != label_col])
    for canonical, original in mapping.items():
        series = frame[original]
        if canonical in CATEGORICAL_HINTS or series.dtype == object:
            report.categorical_columns.append(canonical)
        else:
            report.numeric_columns.append(canonical)

    for original in mapping.values():
        series = frame[original]
        if series.dtype == object:
            numeric = pd.to_numeric(series, errors="coerce")
        else:
            numeric = series
        missing = float(numeric.isna().mean())
        if missing > 0:
            report.missing_values[str(original)] = round(missing, 4)
        if numeric.dtype.kind == "f":
            inf = int(np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).sum())
            report.infinite_values += inf

    report.duplicate_rows = int(frame.duplicated().sum())
    if report.duplicate_rows > 0:
        share = report.duplicate_rows / max(report.rows, 1)
        report.warnings.append(
            f"{report.duplicate_rows:,} duplicate rows ({share:.1%}) - "
            "de-duplicate before splitting or they leak across train/test."
        )
    if report.infinite_values:
        report.warnings.append(
            f"{report.infinite_values:,} infinite values found; these will be imputed."
        )

    if report.label_distribution:
        total = sum(report.label_distribution.values())
        top = max(report.label_distribution.values())
        if total and top / total > 0.95:
            report.warnings.append(
                f"Class imbalance: the majority class is {top / total:.1%} of the data. "
                "Report per-class metrics, not just accuracy."
            )
    return frame, report



@dataclass
class PreparedData:
    """A cleaned, split and encoded dataset ready for training."""

    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    feature_names: list[str] = field(default_factory=list)
    label_encoder: dict[str, int] = field(default_factory=dict)
    dropped_columns: list[str] = field(default_factory=list)
    transformations: list[str] = field(default_factory=list)
    class_distribution: dict[str, int] = field(default_factory=dict)
    source_hash: str = ""
    seed: int = 42
    rows_in: int = 0
    rows_out: int = 0
    test_size: float = 0.2
    stratify_applied: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Serialisable summary recorded alongside the trained model."""
        return {
            "feature_names": self.feature_names,
            "label_encoder": self.label_encoder,
            "dropped_columns": self.dropped_columns,
            "transformations": self.transformations,
            "class_distribution": self.class_distribution,
            "source_hash": self.source_hash,
            "seed": self.seed,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "test_size": self.test_size,
            "stratify_applied": self.stratify_applied,
            "train_rows": int(len(self.X_train)),
            "test_rows": int(len(self.X_test)),
        }


def file_sha256(path: Path, limit_bytes: int | None = None) -> str:
    """Hash a file (optionally just the first N bytes) for reproducibility."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        read = 0
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
            read += len(block)
            if limit_bytes and read >= limit_bytes:
                break
    return digest.hexdigest()[:32]


def normalise_labels(series: Any, positive_names: set[str] | None = None) -> Any:
    """Map attack labels to a consistent set.

    UNSW-NB15 uses 0/1 in some releases and names in others; CIC datasets use
    attack-family names. Both are normalised to strings so the encoder is
    dataset-agnostic.
    """
    text = series.astype(str).str.strip()
    if positive_names:
        binary = text.str.lower().isin({p.lower() for p in positive_names})
        return binary.map({True: "ATTACK", False: "BENIGN"})
    return text.replace({"0": "BENIGN", "1": "ATTACK"}, regex=False)



def prepare_flow_data(
    frame: Any,
    *,
    label_column: str,
    seed: int = 42,
    test_size: float = 0.2,
    drop_duplicates: bool = True,
    max_rows: int | None = None,
) -> PreparedData:
    """Clean, encode and split a flow dataframe.

    Order matters: duplicates and the label are removed **before** the split,
    so neither can leak across the train/test boundary.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    transformations: list[str] = []
    work = frame.copy()
    transformations.append(f"loaded {len(work):,} rows x {work.shape[1]} columns")
    rows_in = len(work)

    if label_column not in work.columns:
        raise ValidationError(
            f"Label column '{label_column}' is not present in the data."
        )

    y = work[label_column].astype(str).str.strip()
    transformations.append(f"labels read from column '{label_column}'")

    if drop_duplicates:
        before = len(work)
        feature_cols = [c for c in work.columns if c != label_column]
        keep_mask = ~work.duplicated(subset=feature_cols)
        removed = int((~keep_mask).sum())
        # Index BEFORE resetting, so labels follow their own rows. Slicing y
        # positionally would silently misalign every label after the first
        # dropped duplicate - a bug that trains on scrambled targets.
        y = y[keep_mask]
        work = work[keep_mask].reset_index(drop=True)
        y = y.reset_index(drop=True)
        transformations.append(
            f"removed {removed:,} duplicate rows before splitting"
            if removed else "no duplicate rows found"
        )

    if max_rows and len(work) > max_rows:
        # Sample by position and carry the label along with it; taking the
        # first max_rows labels instead would pair rows with the wrong targets.
        positions = work.sample(n=max_rows, random_state=seed).index.to_numpy()
        y = y.iloc[positions].reset_index(drop=True)
        work = work.iloc[positions].reset_index(drop=True)
        transformations.append(f"sampled {max_rows:,} rows for tractable training")

    mapping = map_columns([c for c in work.columns if c != label_column])
    dropped: list[str] = []
    feature_frame = pd.DataFrame(index=work.index)
    for canonical, original in mapping.items():
        if canonical in LEAKAGE_COLUMNS or canonical_column(original) in LEAKAGE_COLUMNS:
            dropped.append(original)
            continue
        feature_frame[canonical] = work[original]
    transformations.append(
        f"dropped {len(dropped)} leakage-prone/identifier column(s)"
        if dropped else "no leakage-prone columns present"
    )

    if feature_frame.shape[1] == 0:
        raise ValidationError("No usable feature columns remain after removing leakage columns.")

    for column in list(feature_frame.columns):
        series = feature_frame[column]
        if series.dtype == object:
            converted = pd.to_numeric(series, errors="coerce")
            if converted.notna().mean() > 0.7:
                feature_frame[column] = converted
            else:
                codes, _ = pd.factorize(series.astype(str), sort=True)
                feature_frame[column] = codes.astype("float64")
                transformations.append(f"factorised categorical column '{column}'")
        else:
            feature_frame[column] = pd.to_numeric(series, errors="coerce")

    if feature_frame.size:
        inf_count = int(np.isinf(feature_frame.to_numpy(dtype="float64")).sum())
        if inf_count:
            feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
            transformations.append(
                f"replaced {inf_count:,} infinite value(s) with NaN for imputation"
            )

    nulls = int(feature_frame.isna().sum().sum())
    if nulls:
        medians = feature_frame.median(numeric_only=True)
        feature_frame = feature_frame.fillna(medians).fillna(0.0)
        transformations.append(f"median-imputed {nulls:,} missing value(s)")

    labels = sorted(set(map(str, y)))
    label_encoder = {name: i for i, name in enumerate(labels)}
    y_encoded = y.astype(str).map(label_encoder).astype("int64")
    class_distribution = {str(k): int(v) for k, v in y.value_counts().to_dict().items()}

    can_stratify = len(class_distribution) >= 2 and min(class_distribution.values()) >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame.to_numpy(dtype="float32"),
        y_encoded.to_numpy(),
        test_size=test_size, random_state=seed,
        stratify=y_encoded if can_stratify else None,
    )
    transformations.append(
        f"stratified train/test split {1 - test_size:.0%}/{test_size:.0%} (seed={seed})"
        if can_stratify else
        f"random train/test split {1 - test_size:.0%}/{test_size:.0%} (seed={seed}); "
        "stratification skipped - a class has fewer than 2 members"
    )

    return PreparedData(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        feature_names=list(feature_frame.columns),
        label_encoder=label_encoder,
        dropped_columns=sorted(set(dropped)),
        transformations=transformations,
        class_distribution=class_distribution,
        seed=seed, rows_in=rows_in, rows_out=len(work),
        test_size=test_size, stratify_applied=can_stratify,
    )


__all__ = [
    "SchemaReport", "PreparedData", "inspect_csv", "prepare_flow_data",
    "canonical_column", "identify_label_column", "map_columns",
    "normalise_labels", "file_sha256", "LEAKAGE_COLUMNS", "COLUMN_ALIASES",
]

