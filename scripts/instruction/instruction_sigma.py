"""
CyberLLM
Sigma Instruction Generator

Input:
dataset/processed/sigma_clean.jsonl

Output:
dataset/instruction/sigma_instruction.jsonl
"""

import json

from configs.config import (
    SIGMA_CLEAN,
    SIGMA_INSTRUCTION
)


def clean(value):
    """
    Convert None to empty string and strip whitespace.
    """
    return str(value or "").strip()


def generate_examples(rule):

    examples = []

    title = clean(rule.get("title"))

    if not title:
        return examples

    description = clean(rule.get("description"))

    if not description:
        description = "No description available."

    level = clean(rule.get("level"))

    if not level:
        level = "Unknown"

    # -----------------------------
    # MITRE ATT&CK Tags
    # -----------------------------
    tags = rule.get("tags") or []

    if isinstance(tags, list):

        tags = [
            clean(tag)
            for tag in tags
            if clean(tag)
        ]

        if tags:
            tags_text = ", ".join(tags)
        else:
            tags_text = "No MITRE ATT&CK tags available."

    else:

        tags_text = clean(tags)

        if not tags_text:
            tags_text = "No MITRE ATT&CK tags available."

    # -----------------------------
    # Log Source
    # -----------------------------
    logsource = rule.get("logsource") or {}

    if logsource:

        logsource_text = json.dumps(
            logsource,
            ensure_ascii=False,
            indent=2,
            default=str
        )

    else:

        logsource_text = "No log source available."

    # -----------------------------
    # Detection Logic
    # -----------------------------
    detection = rule.get("detection") or {}

    if detection:

        detection_text = json.dumps(
            detection,
            ensure_ascii=False,
            indent=2,
            default=str
        )

    else:

        detection_text = "No detection logic available."

    # -----------------------------
    # Instruction Examples
    # -----------------------------
    examples.append({

        "instruction": "Explain this Sigma rule.",
        "input": title,
        "output": description

    })

    examples.append({

        "instruction": "What severity level does this Sigma rule have?",
        "input": title,
        "output": level

    })

    examples.append({

        "instruction": "Which MITRE ATT&CK tags are associated with this Sigma rule?",
        "input": title,
        "output": tags_text

    })

    examples.append({

        "instruction": "Describe the log source for this Sigma rule.",
        "input": title,
        "output": logsource_text

    })

    examples.append({

        "instruction": "Explain the detection logic of this Sigma rule.",
        "input": title,
        "output": detection_text

    })

    return examples


def main():

    print("=" * 60)
    print("CyberLLM Sigma Instruction Generator")
    print("=" * 60)

    total_examples = 0
    skipped = 0

    with open(
        SIGMA_CLEAN,
        "r",
        encoding="utf-8"
    ) as infile, open(
        SIGMA_INSTRUCTION,
        "w",
        encoding="utf-8"
    ) as outfile:

        for line in infile:

            line = line.strip()

            if not line:
                continue

            try:

                rule = json.loads(line)

            except json.JSONDecodeError:

                skipped += 1
                continue

            examples = generate_examples(rule)

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
    print("Saved File         :", SIGMA_INSTRUCTION)
    print("=" * 60)


if __name__ == "__main__":
    main()