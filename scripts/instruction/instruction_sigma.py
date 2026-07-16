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

    title = str(rule.get("title") or "").strip()

    if not title:
        return []

    description = str(rule.get("description") or "").strip()
    if not description:
        description = "No description available."

    level = str(rule.get("level") or "").strip()
    if not level:
        level = "unknown"

    # ---------- FIX FOR TAGS ----------
    tags = rule.get("tags", [])

    if not tags:
        tags_text = "No MITRE ATT&CK tags available."
    else:
        tags_text = ", ".join(str(tag) for tag in tags)
    # ----------------------------------

    logsource = rule.get("logsource", {})

    if not logsource:
        logsource_text = "No log source available."
    else:
        logsource_text = json.dumps(
            logsource,
            ensure_ascii=False,
            indent=2
        )

    detection = rule.get("detection", {})

    if not detection:
        detection_text = "No detection logic available."
    else:
        detection_text = json.dumps(
            detection,
            ensure_ascii=False,
            indent=2
        )

    examples = [

        {
            "instruction": "Explain this Sigma rule.",
            "input": title,
            "output": description
        },

        {
            "instruction": "What severity level does this Sigma rule have?",
            "input": title,
            "output": level
        },

        {
            "instruction": "Which MITRE ATT&CK tags are associated with this Sigma rule?",
            "input": title,
            "output": tags_text
        },

        {
            "instruction": "Describe the log source.",
            "input": title,
            "output": logsource_text
        },

        {
            "instruction": "Explain the detection logic.",
            "input": title,
            "output": detection_text
        }

    ]

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