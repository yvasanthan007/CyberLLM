"""Synthetic fixture generation for tests and demos.

**Everything produced here is synthetic and is labelled as such** in the
model card, the manifest and the API response. Synthetic numbers are useful
for exercising a pipeline end-to-end and for CI, but they are *not* a
benchmark: no claim about real-world detection accuracy may be derived from
them. The project enforces this by setting ``is_synthetic=True`` on any model
trained on this data and refusing to rank it above a real-data model.

The generator is shaped like UNSW-NB15 (one of the supported schemas) so the
same preprocessing code path runs against synthetic and real data alike.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.logging_config import get_logger

logger = get_logger("detection.fixtures")

#: Column layout modelled on UNSW-NB15's feature CSVs.
SYNTHETIC_COLUMNS: list[str] = [
    "srcip", "dstip", "proto", "service", "flag",
    "srcipport", "dstipport", "sttl", "dttl", "dur",
    "sbytes", "dbytes", "sbytes_packets", "dbytes_packets",
    "sload", "dload", "scaled", "stcp", "dtcp", "swin", "dwin",
    "Label",
]

#: Normal-traffic feature profile, roughly matching a benign TCP session.
_BENIGN = {
    "sttl": 64, "dttl": 64, "sbytes": 1400, "dbytes": 3600,
    "sbytes_packets": 12, "dbytes_packets": 16, "dur": 1.2, "sload": 0.9,
    "dload": 0.7, "scaled": 0.8, "stcp": 1, "dtcp": 1, "swin": 256, "dwin": 292,
}

#: Per-class deviations. Each attack family shifts a different subset of the
#: feature space, so a tree model has something real to learn rather than a
#: trivially separable synthetic label.
_ATTACK_PROFILES: dict[str, dict[str, float]] = {
    "Flood": {"dur": 90.0, "sbytes": 4200, "dbytes": 120, "dttl": 3, "sload": 9.5,
              "stcp": 1, "dtcp": 0, "sbytes_packets": 95, "dbytes_packets": 4},
    "DOS": {"dur": 120.0, "sbytes": 6400, "dbytes": 80, "dttl": 2, "sload": 14.0,
            "sbytes_packets": 180, "dbytes_packets": 2},
    "Reconnaissance": {"dur": 0.05, "sbytes": 90, "dbytes": 0, "dttl": 255,
                       "sbytes_packets": 2, "dbytes_packets": 0, "swin": 1024},
    "Shellcode": {"dur": 0.2, "sbytes": 180, "dbytes": 40, "dttl": 128,
                  "stcp": 3, "dtcp": 0},
    "Analysis": {"dur": 3.0, "sbytes": 220, "dbytes": 900, "dttl": 64,
                 "sbytes_packets": 3, "dbytes_packets": 20},
    "Fuzzers": {"dur": 0.4, "sbytes": 5200, "dbytes": 30, "dttl": 4,
                "sbytes_packets": 130},
    "Exploits": {"dur": 0.6, "sbytes": 3100, "dbytes": 700, "dttl": 32,
                 "sbytes_packets": 70},
    "Generic": {"dur": 0.8, "sbytes": 2600, "dbytes": 500, "dttl": 48},
    "Worms": {"dur": 30.0, "sbytes": 12000, "dbytes": 200, "dttl": 2,
              "sbytes_packets": 400},
    "Heartbleed": {"dur": 0.1, "sbytes": 300, "dbytes": 5200, "dttl": 64,
                   "dbytes_packets": 90},
}


def make_flow_record(rng: np.random.Generator, label: str) -> dict[str, Any]:
    """Build one synthetic flow record for a given label."""
    profile = _BENIGN if label == "Normal" else _ATTACK_PROFILES.get(
        label, _ATTACK_PROFILES["Generic"]
    )
    record: dict[str, Any] = {
        "srcip": f"192.168.{rng.integers(0, 255)}.{rng.integers(1, 254)}",
        "dstip": f"10.0.{rng.integers(0, 255)}.{rng.integers(1, 254)}",
        "proto": str(rng.choice(["tcp", "udp"])),
        "service": str(rng.choice(["http", "https", "dns", "ftp", "-"])),
        "flag": str(rng.choice(["S", "SF", "PA", "FA", "-"])),
        "srcipport": int(rng.integers(1024, 65535)),
        "dstipport": int(rng.choice([80, 443, 53, 22, 445, 8080])),
    }
    for column, base in _BENIGN.items():
        value = profile.get(column, base)
        noise = base * 0.15 if base else 0.15
        record[column] = round(float(max(value + rng.normal(0, noise), 0.0)), 4)
    record["Label"] = label
    return record


def generate_synthetic_flows(
    n_per_class: int = 300, seed: int = 42, *, include_normal: bool = True
) -> Any:
    """Build a labelled synthetic flow dataframe.

    Duplicates and NaNs are injected on purpose: the preprocessing path must
    be exercised against them, and a pipeline that has never seen a missing
    value has not been tested.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    labels = list(_ATTACK_PROFILES)
    if include_normal:
        labels = ["Normal"] + labels

    rows: list[dict[str, Any]] = []
    for label in labels:
        for _ in range(n_per_class):
            rows.append(make_flow_record(rng, label))

    frame = pd.DataFrame(rows, columns=SYNTHETIC_COLUMNS)

    # Inject ~1% missing values and ~0.5% exact duplicates, deterministically.
    for column in ("sbytes", "dbytes", "dur", "sttl"):
        mask = rng.random(len(frame)) < 0.01
        frame.loc[mask, column] = np.nan
    duplicate_rows = frame.sample(frac=0.005, random_state=seed)
    frame = pd.concat([frame, duplicate_rows], ignore_index=True)
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info(
        "generated SYNTHETIC flow fixture: %d rows x %d cols, %d classes",
        len(frame), frame.shape[1], labels and len(labels),
    )
    return frame


def write_synthetic_flows(path: Path, n_per_class: int = 300, seed: int = 42) -> Path:
    """Write a synthetic fixture CSV to ``path``."""
    frame = generate_synthetic_flows(n_per_class=n_per_class, seed=seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.warning(
        "wrote SYNTHETIC fixture to %s - not real network data", path
    )
    return path


__all__ = [
    "generate_synthetic_flows", "write_synthetic_flows", "make_flow_record",
    "SYNTHETIC_COLUMNS",
]
