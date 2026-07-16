"""
CyberLLM
MITRE Instruction Generator
"""

import json

from configs.config import (
    PROJECT_ROOT,
    MITRE_INSTRUCTION
)

INPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "mitre_clean.jsonl"
)


def generate_examples(record):

    technique = record["technique_id"]
    name = record["name"]
    description = record["description"]
    platforms = ", ".join(record["platforms"])

    examples = []

    examples.append({
        "instruction": f"What is technique {technique}?",
        "input": "",
        "output": description
    })

    examples.append({
        "instruction": f"Explain MITRE ATT&CK technique {technique}.",
        "input": "",
        "output":
            f"{name}\n\n{description}"
    })

    examples.append({
        "instruction": f"Which platforms are affected by {technique}?",
        "input": "",
        "output": platforms
    })

    examples.append({
        "instruction": f"What is {name}?",
        "input": "",
        "output": description
    })

    return examples


def main():

    total = 0

    with open(INPUT_FILE, encoding="utf-8") as infile, \
         open(MITRE_INSTRUCTION, "w", encoding="utf-8") as outfile:

        for line in infile:

            record = json.loads(line)

            examples = generate_examples(record)

            for example in examples:

                outfile.write(
                    json.dumps(example, ensure_ascii=False)
                    + "\n"
                )

                total += 1

    print("=" * 60)
    print("MITRE Instruction Dataset Created")
    print("=" * 60)
    print("Output:", MITRE_INSTRUCTION)
    print("Examples:", total)


if __name__ == "__main__":
    main()