"""
CyberLLM
CVE Instruction Generator

Input:
dataset/processed/cve_clean.jsonl

Output:
dataset/instruction/cve_instruction.jsonl
"""

import json

from configs.config import (
    CVE_PROCESSED,
    CVE_INSTRUCTION
)


def clean(value):
    """
    Convert None to empty string and strip whitespace.
    """
    return str(value or "").strip()


def generate_examples(cve):

    examples = []

    cve_id = clean(cve.get("cve_id"))
    description = clean(cve.get("description"))
    severity = clean(cve.get("severity"))
    score = clean(cve.get("cvss_score"))
    published = clean(cve.get("published"))

    # Skip invalid records
    if not cve_id or not description:
        return examples

    # -----------------------------
    # Example 1
    # -----------------------------
    examples.append({
        "instruction": "What is this CVE?",
        "input": cve_id,
        "output": description
    })

    # -----------------------------
    # Example 2
    # -----------------------------
    if severity:

        text = severity

        if score:
            text += f" (CVSS Score: {score})"

        examples.append({
            "instruction": "What is the severity of this CVE?",
            "input": cve_id,
            "output": text
        })

    # -----------------------------
    # Example 3
    # -----------------------------
    if published:

        examples.append({
            "instruction": "When was this CVE published?",
            "input": cve_id,
            "output": published
        })

    # -----------------------------
    # Example 4
    # -----------------------------
    explanation = description

    if severity:
        explanation += f"\n\nSeverity: {severity}"

    if score:
        explanation += f"\nCVSS Score: {score}"

    examples.append({
        "instruction": "Explain this vulnerability.",
        "input": cve_id,
        "output": explanation
    })

    return examples


def main():

    print("=" * 60)
    print("CyberLLM CVE Instruction Generator")
    print("=" * 60)

    total_examples = 0
    skipped = 0

    with open(
        CVE_PROCESSED,
        "r",
        encoding="utf-8"
    ) as infile, open(
        CVE_INSTRUCTION,
        "w",
        encoding="utf-8"
    ) as outfile:

        for line in infile:

            line = line.strip()

            if not line:
                continue

            try:

                cve = json.loads(line)

            except json.JSONDecodeError:

                skipped += 1
                continue

            examples = generate_examples(cve)

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
    print("Saved File         :", CVE_INSTRUCTION)
    print("=" * 60)


if __name__ == "__main__":
    main()