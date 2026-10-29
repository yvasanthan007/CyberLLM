"""
Preprocess conversational records: validate, dedupe, and split.

Reads raw JSONL files from data/conversation/, validates schema,
removes duplicates, and produces train/validation/test splits into
data/conversation/processed and data/conversation/splits.

Never touches data/raw/, dataset/, or phishing data.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "conversation"
PROCESSED_PATH = RAW_DIR / "processed" / "conversations_clean.jsonl"
SPLITS_DIR = RAW_DIR / "splits"

SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10

REQUIRED_KEYS = {"instruction", "input", "output"}
OPTIONAL_KEYS = {"category", "source"}
ALLOWED_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS

INVALID_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def validate_record(record, line_number, path):
    problems = []
    if not isinstance(record, dict):
        problems.append("not a JSON object")
        return problems
    missing = REQUIRED_KEYS - record.keys()
    if missing:
        problems.append(f"missing keys: {missing}")
        return problems
    unknown = record.keys() - ALLOWED_KEYS
    if unknown:
        problems.append(f"unknown keys: {unknown}")
    for key in REQUIRED_KEYS:
        value = record[key]
        if not isinstance(value, str):
            problems.append(f"{key} is not a string")
            continue
        if INVALID_CHARS.search(value):
            problems.append(f"{key} contains control characters")
            continue
        if key == "input" and not value.strip():
            continue  # empty optional input field is valid
        if not value.strip():
            problems.append(f"{key} is empty")
    if isinstance(record.get("instruction"), str) and len(record["instruction"]) < 5:
        problems.append("instruction too short (<5 chars)")
    if isinstance(record.get("output"), str) and len(record["output"]) < 20:
        problems.append("output too short (<20 chars)")
    return problems


def dedupe_key(record):
    normalized = (
        record["instruction"].strip().lower()
        + "||"
        + record["input"].strip().lower()
        + "||"
        + record["output"].strip()
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                        help="Directory containing raw JSONL files")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report_path = args.report or (PROCESSED_PATH.parent / "preprocess_report.json")

    files = sorted(args.raw_dir.glob("*.jsonl"))
    files = [f for f in files if "processed" not in f.parts and "splits" not in f.parts]

    valid_records = []
    seen_hashes = set()
    stats = {
        "files_scanned": [str(f.relative_to(ROOT)) for f in files],
        "total_lines": 0,
        "invalid": 0,
        "invalid_details": [],
        "duplicates": 0,
        "valid": 0,
    }

    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stats["total_lines"] += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    stats["invalid"] += 1
                    stats["invalid_details"].append(
                        f"{path.name}:{line_number} JSON error: {exc}")
                    continue

                problems = validate_record(record, line_number, path)
                if problems:
                    stats["invalid"] += 1
                    stats["invalid_details"].append(
                        f"{path.name}:{line_number}: {'; '.join(problems)}")
                    continue

                key = dedupe_key(record)
                if key in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                seen_hashes.add(key)

                clean = {
                    "instruction": record["instruction"].strip(),
                    "input": record.get("input", "").strip(),
                    "output": record["output"].strip(),
                    "category": record.get("category", "general"),
                    "source": record.get("source", path.name),
                }
                valid_records.append(clean)
                stats["valid"] += 1

    # Reproducible split
    rng = random.Random(SEED)
    rng.shuffle(valid_records)
    n = len(valid_records)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    splits = {
        "train": valid_records[:n_train],
        "validation": valid_records[n_train:n_train + n_val],
        "test": valid_records[n_train + n_val:],
    }

    # Write processed file
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        for record in valid_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write splits
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        with open(SPLITS_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write report
    report = {
        "seed": SEED,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "processed_path": str(PROCESSED_PATH.relative_to(ROOT)),
        "splits_dir": str(SPLITS_DIR.relative_to(ROOT)),
        "counts": {
            "total_lines_scanned": stats["total_lines"],
            "invalid_rejected": stats["invalid"],
            "duplicates_removed": stats["duplicates"],
            "valid_unique": stats["valid"],
            "train": len(splits["train"]),
            "validation": len(splits["validation"]),
            "test": len(splits["test"]),
        },
        "category_counts": {},
        "invalid_examples": stats["invalid_details"][:20],
        "phishing_data_touched": False,
    }
    category_counts = {}
    for r in valid_records:
        category_counts[r["category"]] = category_counts.get(r["category"], 0) + 1
    report["category_counts"] = dict(sorted(category_counts.items()))

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report["counts"], indent=2))
    print(f"Processed file: {PROCESSED_PATH}")
    print(f"Splits dir:     {SPLITS_DIR}")
    print(f"Report:         {report_path}")


if __name__ == "__main__":
    main()
