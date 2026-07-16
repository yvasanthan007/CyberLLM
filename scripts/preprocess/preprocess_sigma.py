"""
CyberLLM
Sigma Rule Preprocessor
"""

import json
import yaml
from pathlib import Path

from configs.config import (
    SIGMA_RAW,
    SIGMA_PROCESSED
)


def extract_rule(rule):

    return {
        "title": rule.get("title", ""),
        "id": rule.get("id", ""),
        "status": rule.get("status", ""),
        "description": rule.get("description", ""),
        "author": rule.get("author", ""),
        "date": rule.get("date", ""),
        "modified": rule.get("modified", ""),
        "tags": rule.get("tags", []),
        "level": rule.get("level", ""),
        "logsource": rule.get("logsource", {}),
        "detection": rule.get("detection", {})
    }


def main():

    print("=" * 60)
    print("CyberLLM Sigma Preprocessor")
    print("=" * 60)

    rules = []

    yaml_files = list(SIGMA_RAW.rglob("*.yml"))
    yaml_files += list(SIGMA_RAW.rglob("*.yaml"))

    print(f"Found {len(yaml_files)} Sigma rules")

    for file in yaml_files:

        try:

            with open(file, encoding="utf-8") as f:

                rule = yaml.safe_load(f)

            if not isinstance(rule, dict):
                continue

            rules.append(extract_rule(rule))

        except Exception:
            continue

    with open(SIGMA_PROCESSED, "w", encoding="utf-8") as outfile:

        for rule in rules:

            outfile.write(
    json.dumps(
        rule,
        ensure_ascii=False,
        default=str
    )
    + "\n"
)

    print("=" * 60)
    print("Processing Complete")
    print("=" * 60)
    print("Rules:", len(rules))
    print("Saved:", SIGMA_PROCESSED)


if __name__ == "__main__":
    main()