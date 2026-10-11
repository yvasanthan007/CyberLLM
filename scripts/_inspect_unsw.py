"""Inspect the downloaded UNSW-NB15 files: schema, labels, dtypes, integrity.

Writes a UTF-8 report so the Windows console codepage cannot corrupt it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

base = Path(sys.argv[1] if len(sys.argv) > 1 else r"d:\CyberLLM\data\raw\unsw_nb15")
out = Path(__file__).with_name("_unsw_inspect.json")
lines: list[str] = []


def emit(text: str = "") -> None:
    lines.append(str(text))


def read(path: Path) -> pd.DataFrame:
    # The publisher's CSVs are UTF-8 with occasional cp1252 bytes; replace
    # rather than crash, and report the substitution count.
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


emit("=== raw header of the first data line of each file ===")
for name in ("NUSW-NB15_features.csv", "UNSW_NB15_training-set.csv",
             "UNSW_NB15_testing-set.csv"):
    with (base / name).open("r", encoding="utf-8", errors="replace") as handle:
        emit(f"{name}: {handle.readline().rstrip()[:400]}")

feat = read(base / "NUSW-NB15_features.csv")
emit("\n=== feature dictionary ===")
emit(feat.to_string(max_rows=60))

report: dict[str, dict] = {}
frames: dict[str, object] = {}
for name in ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"):
    df = read(base / name)
    frames[name] = df
    label = "label" if "label" in df.columns else "Label"
    cat = "attack_cat" if "attack_cat" in df.columns else "Attack"
    emit(f"\n=== {name} ===")
    emit(f"shape: {df.shape}")
    emit(f"columns: {list(df.columns)}")
    emit(f"dtypes: {df.dtypes.astype(str).to_dict()}")
    emit(f"nulls: {int(df.isna().sum().sum())}")
    emit(f"{label}: {df[label].value_counts().to_dict()}")
    emit(f"{cat}: {df[cat].value_counts().to_dict()}")
    emit(f"exact duplicate rows (all cols): {int(df.duplicated().sum())}")
    feat_cols = [c for c in df.columns if c not in (label, cat)]
    emit(f"duplicates on features only: {int(df.duplicated(subset=feat_cols).sum())}")
    report[name] = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "nulls": int(df.isna().sum().sum()),
        "label": df[label].value_counts().to_dict(),
        "attack_cat": df[cat].value_counts().to_dict(),
        "dup_all": int(df.duplicated().sum()),
        "dup_features": int(df.duplicated(subset=feat_cols).sum()),
    }

# Cross-split leakage: how many feature rows are identical in both files?
train_df = frames["UNSW_NB15_training-set.csv"]
test_df = frames["UNSW_NB15_testing-set.csv"]
label, cat = "label", "attack_cat"
feature_cols = [c for c in train_df.columns if c not in (label, cat, "id")]
train_keys = set(map(tuple, train_df[feature_cols].astype(str).to_numpy()))
test_keys = set(map(tuple, test_df[feature_cols].astype(str).to_numpy()))
overlap = train_keys & test_keys
emit("\n=== cross-split overlap ===")
emit(f"feature-only columns: {len(feature_cols)}")
emit(f"distinct feature tuples: train={len(train_keys):,} test={len(test_keys):,}")
emit(f"identical feature tuples in BOTH splits: {len(overlap):,}")
emit(f"share of test rows matching a train row exactly: "
     f"{len(overlap) / max(len(test_keys), 1):.4%}")

for name, df in frames.items():
    emit(f"\n{name}: attack_cat x label crosstab\n"
         f"{pd.crosstab(df[cat], df[label]).to_string()}")

out.write_text("\n".join(lines), encoding="utf-8")
out.with_suffix(".summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("written:", out)
