from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

source = Path("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")

df = pd.read_csv(source, low_memory=False)
work = df[["URL", "label"]].copy()

work["label"] = pd.to_numeric(work["label"], errors="coerce")
work["URL"] = work["URL"].astype("string").str.strip()
work = work.dropna(subset=["URL", "label"]).copy()

def get_scheme(value):
    try:
        raw = str(value).strip()
        parsed = urlsplit(raw if "://" in raw else "//" + raw)
        return parsed.scheme.lower() or "missing"
    except (ValueError, TypeError):
        return "invalid"

work["scheme"] = work["URL"].map(get_scheme)

print("=" * 70)
print("FULL DATASET: PROTOCOL AND LABEL AUDIT")
print("=" * 70)

print("\nLabel counts:")
print(work["label"].value_counts().sort_index().to_string())

print("\nProtocol counts:")
print(work["scheme"].value_counts().to_string())

print("\nProtocol by label:")
table = pd.crosstab(
    work["scheme"],
    work["label"],
    margins=True
)
print(table.to_string())

print("\nLabel percentages within each protocol:")
percentages = pd.crosstab(
    work["scheme"],
    work["label"],
    normalize="index"
) * 100
print(percentages.round(2).to_string())

print("\nRead-only audit complete.")
