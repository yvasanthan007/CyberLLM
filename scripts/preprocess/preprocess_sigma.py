"""
CyberLLM
Sigma Rule Preprocessor
"""

import json
from datetime import date, datetime
import yaml

from configs.config import (
    SIGMA_RAW,
    SIGMA_PROCESSED
)


RULE_FOLDERS = [
    "rules",
    "rules-threat-hunting",
    "rules-emerging-threats",
    "rules-compliance",
    "rules-dfir",
]


def convert(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [convert(v) for v in obj]

    return obj


def extract_rule(rule):

    if not isinstance(rule, dict):
        return None

    # Only process real Sigma rules
    if "title" not in rule:
        return None

    if "detection" not in rule:
        return None

    return {
        "title": str(rule.get("title", "")).strip(),
        "id": str(rule.get("id", "")).strip(),
        "status": str(rule.get("status", "")).strip(),
        "description": str(rule.get("description", "")).strip(),
        "author": str(rule.get("author", "")).strip(),
        "date": convert(rule.get("date", "")),
        "modified": convert(rule.get("modified", "")),
        "tags": convert(rule.get("tags", [])),
        "level": str(rule.get("level", "")).strip(),
        "logsource": convert(rule.get("logsource", {})),
        "detection": convert(rule.get("detection", {}))
    }


def main():

    print("=" * 60)
    print("CyberLLM Sigma Preprocessor")
    print("=" * 60)

    rules = []

    yaml_files = []

    for folder in RULE_FOLDERS:
        path = SIGMA_RAW / folder

        if path.exists():
            yaml_files.extend(path.rglob("*.yml"))
            yaml_files.extend(path.rglob("*.yaml"))

    print(f"Found {len(yaml_files)} Sigma rule files")

    for file in yaml_files:

        try:

            with open(file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            rule = extract_rule(data)

            if rule is None:
                continue

            rules.append(rule)

        except Exception:
            continue

    with open(SIGMA_PROCESSED, "w", encoding="utf-8") as outfile:

        for rule in rules:
            outfile.write(
                json.dumps(rule, ensure_ascii=False, default=str)
            )
            outfile.write("\n")

    print("=" * 60)
    print("Processing Complete")
    print("=" * 60)
    print(f"Rules: {len(rules)}")
    print(f"Saved: {SIGMA_PROCESSED}")


if __name__ == "__main__":
    main()