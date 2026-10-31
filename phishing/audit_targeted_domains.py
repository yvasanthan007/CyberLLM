from pathlib import Path
import json

import pandas as pd

root = Path.cwd()
data_path = root / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
report_path = root / "artifacts" / "reports" / "phishing" / "targeted_domain_audit.json"

targets = {
    "youtube.com": [
        "https://www.youtube.com/",
        "https://www.youtube.com/watch?v=example",
    ],
    "wikipedia.org": [
        "https://en.wikipedia.org/",
        "https://en.wikipedia.org/wiki/Phishing",
    ],
    "microsoft.com": [
        "https://www.microsoft.com/",
        "https://login.microsoft.com/",
    ],
    "example.com": [
        "https://example.com/",
        "https://example.com/login",
    ],
}

print("Loading dataset...")
df = pd.read_csv(data_path, usecols=["URL", "label"])
df = df.dropna(subset=["URL", "label"]).copy()
df["URL"] = df["URL"].astype(str).str.strip()
df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

report = {
    "dataset_rows": int(len(df)),
    "label_mapping": {"0": "PHISHING", "1": "LEGITIMATE"},
    "targets": {},
}

for domain, test_urls in targets.items():
    # This is a substring audit, not an exact-hostname audit.
    mask = df["URL"].str.contains(domain, case=False, regex=False, na=False)
    matched = df.loc[mask, ["URL", "label"]]
    counts = matched["label"].value_counts().to_dict()

    report["targets"][domain] = {
        "substring_match_rows": int(len(matched)),
        "label_counts": {
            str(int(label)): int(count)
            for label, count in counts.items()
        },
        "examples": [
            {"url": row.URL, "label": int(row.label)}
            for row in matched.head(5).itertuples(index=False)
        ],
        "test_urls": test_urls,
    }

    print(f"\n{domain}: {len(matched):,} substring matches")
    print(f"Label counts (0=phishing, 1=legitimate): {counts}")

    if not matched.empty:
        print(matched.head(5).to_string(index=False))

report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nAudit report saved to:", report_path)
print("No models or dataset rows were modified.")
