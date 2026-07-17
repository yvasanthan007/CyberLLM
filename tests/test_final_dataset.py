"""
CyberLLM
Final Dataset Validator
"""

import json
import hashlib

from configs.config import CYBERLLM_TRAIN


def validate_record(record):
    """
    Validate one instruction record.
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


def record_hash(record):
    """
    Generate a hash for duplicate detection.
    """

    text = json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False
    )

    return hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()


def main():

    print("=" * 60)
    print("CyberLLM Final Dataset Validator")
    print("=" * 60)

    total = 0
    valid = 0
    invalid = 0
    duplicates = 0

    seen = set()

    invalid_examples = []

    with open(
        CYBERLLM_TRAIN,
        "r",
        encoding="utf-8"
    ) as infile:

        for line in infile:

            total += 1

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(line)

            except json.JSONDecodeError:

                invalid += 1

                invalid_examples.append(
                    (
                        total,
                        "Invalid JSON"
                    )
                )

                continue

            status, reason = validate_record(record)

            if not status:

                invalid += 1

                invalid_examples.append(
                    (
                        total,
                        reason
                    )
                )

                continue

            h = record_hash(record)

            if h in seen:

                duplicates += 1

                continue

            seen.add(h)

            valid += 1

    print("=" * 60)
    print("Validation Report")
    print("=" * 60)

    print(f"Total Records : {total}")
    print(f"Valid Records : {valid}")
    print(f"Invalid Records : {invalid}")
    print(f"Duplicate Records : {duplicates}")

    print()

    if invalid_examples:

        print("Invalid Examples")

        for line_no, reason in invalid_examples[:20]:

            print(f"Line {line_no}: {reason}")

    else:

        print("Dataset is valid.")

    print("=" * 60)


if __name__ == "__main__":
    main()