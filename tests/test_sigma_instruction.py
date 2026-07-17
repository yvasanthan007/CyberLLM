"""
CyberLLM
Sigma Instruction Dataset Validator
"""

import json

from configs.config import SIGMA_INSTRUCTION


def validate_record(record):
    """
    Validate a single instruction record.
    """

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
                        "Invalid JSON",
                        line.strip()
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

        print("\n")
        print("=" * 60)
        print("Invalid Examples")
        print("=" * 60)

        for line_no, reason, record in invalid_records:

            print(f"\nLine : {line_no}")
            print(f"Reason : {reason}")
            print("Record :")

            if isinstance(record, dict):

                print(
                    json.dumps(
                        record,
                        indent=4,
                        ensure_ascii=False
                    )
                )

            else:

                print(record)

    else:

        print("\nDataset is valid.")

    print("=" * 60)


if __name__ == "__main__":
    main()