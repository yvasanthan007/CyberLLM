"""
CyberLLM
CVE Preprocessor

Reads raw CVE batches from:
dataset/raw/cve/

Outputs:
dataset/processed/cve_clean.jsonl
"""

import json
from pathlib import Path

from configs.config import PROJECT_ROOT

RAW_DIR = PROJECT_ROOT / "dataset" / "raw" / "cve"

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "cve_clean.jsonl"


def get_description(cve):
    descriptions = cve.get("descriptions", [])

    for d in descriptions:
        if d.get("lang") == "en":
            return d.get("value", "")

    return ""


def get_cvss(metrics):

    score = None
    severity = None

    if "cvssMetricV31" in metrics:

        cvss = metrics["cvssMetricV31"][0]

        score = cvss["cvssData"]["baseScore"]
        severity = cvss["cvssData"]["baseSeverity"]

    elif "cvssMetricV30" in metrics:

        cvss = metrics["cvssMetricV30"][0]

        score = cvss["cvssData"]["baseScore"]
        severity = cvss["cvssData"]["baseSeverity"]

    elif "cvssMetricV2" in metrics:

        cvss = metrics["cvssMetricV2"][0]

        score = cvss["cvssData"]["baseScore"]
        severity = cvss["baseSeverity"]

    return score, severity


def clean_record(record):

    cve = record["cve"]

    metrics = cve.get("metrics", {})

    score, severity = get_cvss(metrics)

    references = []

    for ref in cve.get("references", []):

        references.append(ref["url"])

    return {
        "cve_id": cve.get("id"),
        "description": get_description(cve),
        "published": cve.get("published"),
        "last_modified": cve.get("lastModified"),
        "cvss_score": score,
        "severity": severity,
        "references": references,
    }


def main():

    files = sorted(RAW_DIR.glob("*.json"))

    total = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for file in files:

            print(f"Processing {file.name}")

            with open(file, encoding="utf-8") as f:

                data = json.load(f)

            vulnerabilities = data.get("vulnerabilities", [])

            for record in vulnerabilities:

                cleaned = clean_record(record)

                outfile.write(
                    json.dumps(cleaned, ensure_ascii=False)
                    + "\n"
                )

                total += 1

    print("=" * 60)
    print("Finished")
    print("Output:", OUTPUT_FILE)
    print("Records:", total)
    print("=" * 60)


if __name__ == "__main__":
    main()