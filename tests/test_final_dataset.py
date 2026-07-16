"""
CyberLLM Final Dataset Validator
"""

import json
from configs.config import CYBERLLM_TRAIN


def validate():

    print("=" * 60)
    print("CyberLLM Final Dataset Validator")
    print("=" * 60)

    total = 0
    duplicates = 0
    empty_instruction = 0
    empty_output = 0
    invalid_json = 0

    seen = set()

    with open(CYBERLLM_TRAIN, "r", encoding="utf-8") as infile:

        for line in infile:

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue

            instruction = record.get("instruction", "").strip()
            output = record.get("output", "").strip()
            user_input = record.get("input", "").strip()

            if instruction == "":
                empty_instruction += 1

            if output == "":
                empty_output += 1

            key = (
                instruction,
                user_input,
                output,
            )

            if key in seen:
                duplicates += 1
            else:
                seen.add(key)

    print("\nValidation Report")
    print("-" * 60)
    print(f"Total Records      : {total}")
    print(f"Duplicate Records  : {duplicates}")
    print(f"Empty Instructions : {empty_instruction}")
    print(f"Empty Outputs      : {empty_output}")
    print(f"Invalid JSON       : {invalid_json}")

    if (
        duplicates == 0
        and empty_instruction == 0
        and empty_output == 0
        and invalid_json == 0
    ):
        print("\n✅ Dataset Validation PASSED")
    else:
        print("\n❌ Dataset Validation FAILED")


if __name__ == "__main__":
    validate()