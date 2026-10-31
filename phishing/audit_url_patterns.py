from pathlib import Path
from urllib.parse import urlsplit
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
OUT = ROOT / "artifacts" / "reports" / "phishing" / "url_pattern_audit.json"

df = pd.read_csv(DATASET, usecols=["URL", "label"])
df = df.dropna(subset=["URL", "label"]).copy()
df["URL"] = df["URL"].astype(str).str.strip()
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)
df = df[df["label"].isin([0, 1])]
df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)

def parse_parts(url):
    try:
        p = urlsplit(url)
        return {
            "hostname": (p.hostname or "").lower().rstrip("."),
            "scheme": p.scheme.lower(),
            "path": p.path or "",
            "query": p.query or "",
        }
    except ValueError:
        return {
            "hostname": "",
            "scheme": "",
            "path": "",
            "query": "",
        }

parts = pd.DataFrame(df["URL"].map(parse_parts).tolist())
df = pd.concat([df, parts], axis=1)

patterns = {
    "google_host": r"(^|\.)google\.com$",
    "youtube_host": r"(^|\.)youtube\.com$",
    "wikipedia_host": r"(^|\.)wikipedia\.org$",
    "microsoft_host": r"(^|\.)microsoft\.com$",
    "office_host": r"(^|\.)office\.com$",
    "example_host": r"(^|\.)example\.com$",
    "has_login_word": r"login|signin|verify|account",
    "has_redirect_word": r"redirect|return=|continue=",
    "has_at_symbol": r"@",
    "has_percent_encoding": r"%",
    "has_ip_address_text": r"(?:\d{1,3}\.){3}\d{1,3}",
}

results = {}

print("===== URL PATTERN LABEL AUDIT =====")
print("Label 0 = phishing; label 1 = legitimate\n")

for name, pattern in patterns.items():
    if name.endswith("_host"):
        mask = df["hostname"].str.contains(pattern, regex=True, na=False)
    elif name == "has_login_word":
        mask = df["URL"].str.contains(pattern, case=False, regex=True, na=False)
    elif name == "has_redirect_word":
        mask = df["URL"].str.contains(pattern, case=False, regex=True, na=False)
    else:
        mask = df["URL"].str.contains(pattern, case=False, regex=True, na=False)

    subset = df[mask]
    counts = subset["label"].value_counts().to_dict()

    results[name] = {
        "rows": int(len(subset)),
        "phishing_label_0": int(counts.get(0, 0)),
        "legitimate_label_1": int(counts.get(1, 0)),
        "sample_urls": subset["URL"].head(5).tolist(),
    }

    print(f"{name}: {results[name]}")

report = {
    "dataset_rows_after_deduplication": int(len(df)),
    "label_mapping": {"0": "PHISHING", "1": "LEGITIMATE"},
    "patterns": results,
    "limitations": [
        "Pattern matches are not proof of maliciousness.",
        "URL query parameters can contain external destinations.",
        "Dataset label distributions may reflect collection bias.",
    ],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nSaved report:", OUT)
print("No model or dataset was modified.")
