"""
CyberLLM
Dataset Merger
"""

import json
import random
from pathlib import Path

from configs.config import (
    CVE_INSTRUCTION,
    MITRE_INSTRUCTION,
    SIGMA_INSTRUCTION,
    CYBERLLM_TRAIN,
)


def load_dataset(file_path):

    records = []

    if not Path(file_path).exists():

        print(f"[WARNING] Missing: {file_path}")
        return records

    with open(file_path, "r", encoding="utf-8") as infile:

        for line in infile:

            line = line.strip()

            if not line:
                continue

            try:

                records.append(json.loads(line))

            except json.JSONDecodeError:

                continue

    print(f"Loaded {len(records)} records from {file_path.name}")

    return records


def save_dataset(records):

    CYBERLLM_TRAIN.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        CYBERLLM_TRAIN,
        "w",
        encoding="utf-8"
    ) as outfile:

        for record in records:

            outfile.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
            )

            outfile.write("\n")


def main():

    print("=" * 60)
    print("CyberLLM Dataset Merger")
    print("=" * 60)

    cve = load_dataset(CVE_INSTRUCTION)

    mitre = load_dataset(MITRE_INSTRUCTION)

    sigma = load_dataset(SIGMA_INSTRUCTION)

    merged = cve + mitre + sigma

    print()

    print(f"CVE Records    : {len(cve)}")
    print(f"MITRE Records  : {len(mitre)}")
    print(f"Sigma Records  : {len(sigma)}")

    print("-" * 60)

    print(f"Total Before Shuffle : {len(merged)}")

    random.shuffle(merged)

    save_dataset(merged)

    print()

    print("=" * 60)
    print("Merge Completed")
    print("=" * 60)

    print(f"Saved Dataset : {CYBERLLM_TRAIN}")

    print(f"Total Records : {len(merged)}")


if __name__ == "__main__":
    main()