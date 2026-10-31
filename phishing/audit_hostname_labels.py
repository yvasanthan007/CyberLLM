from pathlib import Path
from urllib.parse import urlsplit
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
OUT = ROOT / "artifacts" / "reports" / "phishing" / "hostname_label_audit.json"

print("Loading URL and label columns...")
df = pd.read_csv(DATASET, usecols=["URL", "label"])
df = df.dropna(subset=["URL", "label"]).copy()
df["URL"] = df["URL"].astype(str).str.strip()
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)
df = df.drop_duplicates(subset=["URL"])

def get_host(url):
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""

df["hostname"] = df["URL"].map(get_host)
df = df[df["hostname"] != ""].copy()

# Labels: 0 = phishing, 1 = legitimate.
df["domain_family"] = df["hostname"].map(
    lambda h: ".".join(h.split(".")[-2:]) if len(h.split(".")) >= 2 else h
)

print("\n===== DATASET LABEL SUMMARY =====")
print(df["label"].value_counts().sort_index().rename(
    index={0: "PHISHING", 1: "LEGITIMATE"}
).to_string())

print("\n===== COMMON HOSTNAMES BY LABEL =====")
for label, name in [(0, "PHISHING"), (1, "LEGITIMATE")]:
    print(f"\n{name}:")
    print(
        df.loc[df["label"] == label, "hostname"]
        .value_counts()
        .head(20)
        .to_string()
    )

print("\n===== SELECTED HOSTNAME LABEL COUNTS =====")
targets = [
    "youtube.com",
    "www.youtube.com",
    "en.wikipedia.org",
    "wikipedia.org",
    "microsoft.com",
    "www.microsoft.com",
    "login.microsoft.com",
    "office.com",
    "www.office.com",
    "example.com",
    "www.example.com",
    "google.com",
    "www.google.com",
]

audit = {}
for host in targets:
    subset = df[df["hostname"] == host]
    counts = subset["label"].value_counts().to_dict()
    audit[host] = {
        "rows": int(len(subset)),
        "phishing_label_0": int(counts.get(0, 0)),
        "legitimate_label_1": int(counts.get(1, 0)),
        "sample_urls": subset["URL"].head(5).tolist(),
    }
    print(f"\n{host}: {audit[host]}")

report = {
    "dataset": str(DATASET),
    "unique_url_rows": int(len(df)),
    "label_mapping": {"0": "PHISHING", "1": "LEGITIMATE"},
    "selected_hostnames": audit,
    "limitations": [
        "Dataset labels are not independent proof that a URL is malicious or safe.",
        "Hostname patterns and label frequencies do not establish live-site reputation.",
    ],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nAudit saved to:", OUT)
print("No models or dataset records were modified.")
