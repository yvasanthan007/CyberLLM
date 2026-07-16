"""
CyberLLM
MITRE Instruction Dataset Validator
"""

import json
from pathlib import Path

from configs.config import PROJECT_ROOT

INPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "instruction"
    / "mitre_instruction.jsonl"
)


def validate():

    total = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile:

        for line in infile:

            total += 1

            record = json.loads(line)

            required = [
                "instruction",
                "input",
                "output"
            ]

            for field in required:
                assert field in record, f"Missing {field}"

            assert record["instruction"].strip()
            assert record["output"].strip()

    print("=" * 50)
    print("MITRE Instruction Dataset Valid")
    print("Examples:", total)
    print("=" * 50)


if __name__ == "__main__":
    validate()