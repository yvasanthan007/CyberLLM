"""
CyberLLM
Sigma Instruction Generator
"""

import json

from configs.config import (
    SIGMA_CLEAN,
    SIGMA_INSTRUCTION
)


def generate_examples(rule):

    title = rule.get("title", "")
    description = rule.get("description", "")
    level = rule.get("level", "")
    tags = ", ".join(rule.get("tags", []))

    logsource = json.dumps(
        rule.get("logsource", {}),
        ensure_ascii=False,
        indent=2
    )

    detection = json.dumps(
        rule.get("detection", {}),
        ensure_ascii=False,
        indent=2
    )

    examples = []

    # Example 1
    examples.append({
        "instruction": "Explain this Sigma rule.",
        "input": title,
        "output": description
    })

    # Example 2
    examples.append({
        "instruction": "What severity level does this Sigma rule have?",
        "input": title,
        "output": level
    })

    # Example 3
    examples.append({
        "instruction": "Which MITRE ATT&CK tags are associated with this Sigma rule?",
        "input": title,
        "output": tags
    })

    # Example 4
    examples.append({
        "instruction": "Describe the log source of this Sigma rule.",
        "input": title,
        "output": logsource
    })

    # Example 5
    examples.append({
        "instruction": "Explain how this Sigma rule detects suspicious activity.",
        "input": title,
        "output": detection
    })

    return examples


def main():

    print("=" * 60)
    print("CyberLLM Sigma Instruction Generator")
    print("=" * 60)

    total = 0

    with open(SIGMA_CLEAN, "r", encoding="utf-8") as infile, \
         open(SIGMA_INSTRUCTION, "w", encoding="utf-8") as outfile:

        for line in infile:

            rule = json.loads(line)

            examples = generate_examples(rule)

            for example in examples:

                outfile.write(
                    json.dumps(
                        example,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                total += 1

    print("=" * 60)
    print("Completed")
    print("=" * 60)
    print("Examples:", total)
    print("Saved:", SIGMA_INSTRUCTION)


if __name__ == "__main__":
    main()