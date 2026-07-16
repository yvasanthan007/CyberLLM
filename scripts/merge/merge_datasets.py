"""
CyberLLM Dataset Merger

Merge all instruction datasets into one training dataset.
"""
print("merge_datasets.py started")
import json
import random

from configs.config import (
    INSTRUCTION_DATASET,
    CYBERLLM_TRAIN,
)

DATASETS = [
    "cve_instruction.jsonl",
    "mitre_instruction.jsonl",
]


def load_dataset(file_path):
    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def main():

    print("=" * 60)
    print("CyberLLM Dataset Merger")
    print("=" * 60)

    all_records = []

    for dataset in DATASETS:

        path = INSTRUCTION_DATASET / dataset

        print(f"Loading: {dataset}")

        records = load_dataset(path)

        print(f"Records: {len(records)}")

        all_records.extend(records)

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
    print(f"Total Records : {len(all_records)}")
    print(f"Saved To      : {CYBERLLM_TRAIN}")


if __name__ == "__main__":
    main()