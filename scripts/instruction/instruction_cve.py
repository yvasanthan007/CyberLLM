"""
CyberLLM
CVE Instruction Generator

Input:
dataset/processed/cve_clean.jsonl

Output:
dataset/instruction/cve_instruction.jsonl
"""

import json
from pathlib import Path

from configs.config import PROJECT_ROOT

INPUT_FILE = PROJECT_ROOT / "dataset" / "processed" / "cve_clean.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "instruction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "cve_instruction.jsonl"


def generate_examples(cve):

    examples = []

    cve_id = cve.get("cve_id", "")
    description = cve.get("description", "")
    severity = cve.get("severity", "")
    score = cve.get("cvss_score", "")
    published = cve.get("published", "")

    # Example 1
    examples.append({
        "instruction": f"What is {cve_id}?",
        "input": "",
        "output": description
    })

    # Example 2
    examples.append({
        "instruction": f"What is the severity of {cve_id}?",
        "input": "",
        "output": f"{severity} (CVSS Score: {score})"
    })

    # Example 3
    examples.append({
        "instruction": f"When was {cve_id} published?",
        "input": "",
        "output": published
    })

    # Example 4
    examples.append({
        "instruction": f"Explain the vulnerability {cve_id}.",
        "input": "",
        "output":
            f"{description}\n\n"
            f"Severity: {severity}\n"
            f"CVSS Score: {score}"
    })

    return examples


def main():

    total = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in infile:

            cve = json.loads(line)

            examples = generate_examples(cve)

            for example in examples:

                outfile.write(
                    json.dumps(example, ensure_ascii=False)
                    + "\n"
                )

                total += 1

    print("=" * 60)
    print("Instruction Dataset Created")
    print("Output:", OUTPUT_FILE)
    print("Examples:", total)
    print("=" * 60)


if __name__ == "__main__":
    main()