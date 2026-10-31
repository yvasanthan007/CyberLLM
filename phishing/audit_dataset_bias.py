from pathlib import Path
import pandas as pd

dataset = Path(
    "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
)

print("=" * 75)
print("CYBERLLM DATASET BIAS AUDIT")
print("=" * 75)

df = pd.read_csv(dataset, low_memory=False)

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# Find likely URL and label columns.
url_candidates = [
    c for c in df.columns
    if c.lower() in {"url", "urls", "domain", "website"}
]

label_candidates = [
    c for c in df.columns
    if c.lower() in {"label", "status", "result", "class"}
]

if not url_candidates:
    url_candidates = [
        c for c in df.columns
        if "url" in c.lower()
    ]

if not label_candidates:
    label_candidates = [
        c for c in df.columns
        if "label" in c.lower() or "status" in c.lower()
    ]

print("\nPossible URL columns:", url_candidates)
print("Possible label columns:", label_candidates)

# Use the known dataset schema if available.
url_col = "URL" if "URL" in df.columns else (
    url_candidates[0] if url_candidates else None
)

label_col = "label" if "label" in df.columns else (
    label_candidates[0] if label_candidates else None
)

if url_col is None or label_col is None:
    print(
        "\nSTOP: Could not safely identify URL and label columns."
    )
    raise SystemExit(1)

work = df[[url_col, label_col]].copy()
work.columns = ["url", "label"]

work["url"] = work["url"].astype("string").str.lower()
work["label"] = pd.to_numeric(
    work["label"], errors="coerce"
)

work = work.dropna(subset=["url", "label"])
work["url"] = work["url"].str.strip()

# Dataset convention established in previous checks:
# label 0 = phishing; label 1 = legitimate.
print("\nLabel counts:")
print(work["label"].value_counts().sort_index())

patterns = [
    ".com/",
    ".org/",
    "youtube.com",
    "wikipedia.org",
    "github.com",
    "microsoft.com",
    "example.com",
    "paypal.com",
    "login",
    "verify",
    "secure",
]

print("\n" + "=" * 75)
print("PATTERN COUNTS BY LABEL")
print("=" * 75)

for pattern in patterns:
    matched = work[
        work["url"].str.contains(
            pattern, regex=False, na=False
        )
    ]

    print(f"\nPattern: {pattern}")
    print("Total matches:", len(matched))

    if len(matched):
        print(
            matched["label"]
            .value_counts()
            .sort_index()
            .rename(index={
                0: "PHISHING",
                1: "LEGITIMATE"
            })
            .to_string()
        )

print("\n" + "=" * 75)
print("COMMON HOSTNAME LABEL CONFLICTS")
print("=" * 75)

from urllib.parse import urlsplit

work["hostname"] = work["url"].map(
    lambda value: (
        urlsplit(
            value if "://" in value else "//" + value
        ).hostname or ""
    )
)

host_stats = (
    work[work["hostname"] != ""]
    .groupby("hostname")["label"]
    .agg(["count", "nunique"])
    .reset_index()
)

conflicts = host_stats[
    (host_stats["count"] >= 2) &
    (host_stats["nunique"] > 1)
].sort_values("count", ascending=False)

print("Hostnames with multiple labels:", len(conflicts))

if conflicts.empty:
    print("No conflicting hostnames found in this raw-data check.")
else:
    print(conflicts.head(30).to_string(index=False))

print("\n" + "=" * 75)
print("SELECTED KNOWN HOSTNAMES")
print("=" * 75)

known_hosts = [
    "youtube.com",
    "wikipedia.org",
    "github.com",
    "microsoft.com",
    "example.com",
    "google.com",
    "paypal.com",
]

for host in known_hosts:
    subset = work[
        work["hostname"].str.endswith(
            host, na=False
        )
    ]

    print(f"\n{host}: {len(subset)} matching rows")

    if len(subset):
        print(
            subset["label"]
            .value_counts()
            .sort_index()
            .rename(index={
                0: "PHISHING",
                1: "LEGITIMATE"
            })
            .to_string()
        )

print("\nAUDIT COMPLETE — no model was trained or changed.")
