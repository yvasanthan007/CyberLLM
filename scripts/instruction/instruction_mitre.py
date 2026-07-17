"""
CyberLLM
MITRE ATT&CK Instruction Generator

Input:
dataset/processed/mitre_clean.jsonl

Output:
dataset/instruction/mitre_instruction.jsonl
"""

import json

from configs.config import (
    MITRE_OUTPUT,
    MITRE_INSTRUCTION
)


def clean(value):
    """
    Convert None to empty string and strip whitespace.
    """
    return str(value or "").strip()


def generate_examples(record):

    examples = []

    technique = clean(record.get("technique_id"))
    name = clean(record.get("name"))
    description = clean(record.get("description"))

    platforms = record.get("platforms") or []

    if isinstance(platforms, list):
        platforms = ", ".join(platforms)
    else:
        platforms = clean(platforms)

    # Skip invalid records
    if not technique or not description:
        return examples

    # ----------------------------------------------------
    # Example 1
    # ----------------------------------------------------
    examples.append({
        "instruction": "What is this MITRE ATT&CK technique?",
        "input": technique,
        "output": description
    })

    # ----------------------------------------------------
    # Example 2
    # ----------------------------------------------------
    explain = description

    if name:
        explain = f"{name}\n\n{description}"

    examples.append({
        "instruction": "Explain this MITRE ATT&CK technique.",
        "input": technique,
        "output": explain
    })

    # ----------------------------------------------------
    # Example 3
    # ----------------------------------------------------
    if platforms:

        examples.append({
            "instruction": "Which platforms are affected by this MITRE ATT&CK technique?",
            "input": technique,
            "output": platforms
        })

    # ----------------------------------------------------
    # Example 4
    # ----------------------------------------------------
    if name:

        examples.append({
            "instruction": "Describe this attack technique.",
            "input": name,
            "output": description
        })

    return examples


def main():

    print("=" * 60)
    print("CyberLLM MITRE Instruction Generator")
    print("=" * 60)

    total_examples = 0
    skipped = 0

    with open(
        MITRE_OUTPUT,
        "r",
        encoding="utf-8"
    ) as infile, open(
        MITRE_INSTRUCTION,
        "w",
        encoding="utf-8"
    ) as outfile:

        for line in infile:

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(line)

            except json.JSONDecodeError:

                skipped += 1
                continue

            examples = generate_examples(record)

            if not examples:
                skipped += 1
                continue

            for example in examples:

                outfile.write(
                    json.dumps(
                        example,
                        ensure_ascii=False
                    )
                )

                outfile.write("\n")

                total_examples += 1

    print("=" * 60)
    print("Instruction Dataset Created")
    print("=" * 60)
    print("Examples Generated :", total_examples)
    print("Skipped Records    :", skipped)
    print("Saved File         :", MITRE_INSTRUCTION)
    print("=" * 60)


if __name__ == "__main__":
    main()