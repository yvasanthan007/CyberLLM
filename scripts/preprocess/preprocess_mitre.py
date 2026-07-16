"""
CyberLLM
MITRE ATT&CK Preprocessor
"""

import json

from configs.config import (
    PROJECT_ROOT,
    MITRE_OUTPUT,
)

RAW_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "mitre"
    / "enterprise-attack.json"
)


def clean_attack_pattern(obj):

    external_id = ""

    for ref in obj.get("external_references", []):

        if ref.get("source_name") == "mitre-attack":

            external_id = ref.get("external_id", "")

    return {
        "technique_id": external_id,
        "name": obj.get("name", ""),
        "description": obj.get("description", ""),
        "platforms": obj.get("x_mitre_platforms", []),
        "tactics": obj.get("kill_chain_phases", [])
    }


def main():

    print("=" * 60)
    print("MITRE Preprocessor")
    print("=" * 60)

    with open(RAW_FILE, encoding="utf-8") as f:

        data = json.load(f)

    total = 0

    with open(MITRE_OUTPUT, "w", encoding="utf-8") as outfile:

        for obj in data["objects"]:

            if obj.get("type") != "attack-pattern":
                continue

            record = clean_attack_pattern(obj)

            outfile.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

            total += 1

    print(f"Techniques: {total}")
    print("Saved:", MITRE_OUTPUT)


if __name__ == "__main__":
    main()