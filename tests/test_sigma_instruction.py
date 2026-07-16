"""
CyberLLM
Sigma Instruction Dataset Validator
"""

import json
from pathlib import Path

from configs.config import SIGMA_INSTRUCTION


def validate_record(record):

    required = [
        "instruction",
        "input",
        "output"
    ]

    for field in required:

        if field not in record:
            return False, f"Missing field: {field}"

        if not str(record[field]).strip():
            return False, f"Empty field: {field}"

    return True, ""


def main():

    print("=" * 60)
    print("CyberLLM Sigma Instruction Validator")
    print("=" * 60)

    total = 0
    valid = 0
    invalid = 0

    invalid_records = []

    with open(SIGMA_INSTRUCTION, "r", encoding="utf-8") as infile:

        for line in infile:

            total += 1

            try:

                record = json.loads(line)

            except json.JSONDecodeError:

                invalid += 1
                invalid_records.append(
                    (
                        total,
                        "Invalid JSON"
                    )
                )

                continue

            status, reason = validate_record(record)

            if status:

                valid += 1

            else:

                invalid += 1

                invalid_records.append(
                    (
                        total,
                        reason,
                        record
                    )
                )

    print("=" * 60)
    print("Validation Report")
    print("=" * 60)

    print(f"Total Records : {total}")
    print(f"Valid Records : {valid}")
    print(f"Invalid Records : {invalid}")

    if invalid_records:

        print("\nInvalid Examples")

        for line, reason, record in invalid_records:

            print("=" * 60)
            print(f"Line : {line}")
            print(f"Reason : {reason}")
            print("Record:")
            print(json.dumps(record, indent=4, ensure_ascii=False))

    else:

        print("\nDataset is valid.")

    print("=" * 60)


if __name__ == "__main__":
    main()