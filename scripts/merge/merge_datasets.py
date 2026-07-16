"""
CyberLLM Dataset Merger

Merges all instruction datasets into one dataset
and removes duplicate records.
"""

import json
import random

from configs.config import (
    INSTRUCTION_DATASET,
    CYBERLLM_TRAIN,
)

all_records = []
seen = set()

instruction_files = sorted(
    INSTRUCTION_DATASET.glob("*_instruction.jsonl")
)

print("=" * 60)
print("CyberLLM Dataset Merger")
print("=" * 60)

for file in instruction_files:

    print(f"Loading: {file.name}")

    with open(file, "r", encoding="utf-8") as f:

        count = 0

        for line in f:

            record = json.loads(line)

            key = (
                record["instruction"].strip(),
                record["input"].strip(),
                record["output"].strip(),
            )

            if key not in seen:
                seen.add(key)
                all_records.append(record)

            count += 1

    print(f"Records Read : {count}")

print("\nRemoving duplicates...")
print(f"Unique Records : {len(all_records)}")

print("\nShuffling dataset...")
random.shuffle(all_records)

with open(CYBERLLM_TRAIN, "w", encoding="utf-8") as outfile:

    for record in all_records:

        outfile.write(
            json.dumps(record, ensure_ascii=False)
            + "\n"
        )

print("=" * 60)
print("Merge Completed")
print("=" * 60)
print(f"Final Records : {len(all_records)}")
print(f"Saved To      : {CYBERLLM_TRAIN}")