"""
CyberLLM
Instruction Dataset Validator
"""

import json
from pathlib import Path

from configs.config import PROJECT_ROOT

DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "instruction"
    / "cve_instruction.jsonl"
)


def validate():

    total = 0

    with open(DATASET, "r", encoding="utf-8") as f:

        for line in f:

            total += 1

            record = json.loads(line)

            assert "instruction" in record
            assert "input" in record
            assert "output" in record

            assert isinstance(record["instruction"], str)
            assert isinstance(record["input"], str)
            assert isinstance(record["output"], str)

            assert len(record["instruction"]) > 0
            assert len(record["output"]) > 0

    print("=" * 50)
    print("Validation Successful")
    print("Examples:", total)
    print("=" * 50)


if __name__ == "__main__":
    validate()