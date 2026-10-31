from pathlib import Path
import pandas as pd
from urllib.parse import urlsplit

path = Path("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
df = pd.read_csv(path, low_memory=False)

work = df[["URL", "label"]].copy()
work["URL"] = work["URL"].astype("string").str.strip()
work["label"] = pd.to_numeric(work["label"], errors="coerce")
work = work.dropna(subset=["URL", "label"]).copy()

def normalize_url(value):
    return str(value).strip().lower()

def get_hostname(value):
    try:
        value = str(value).strip()
        parsed = urlsplit(value if "://" in value else "//" + value)
        return (parsed.hostname or "").lower().rstrip(".")
    except (ValueError, TypeError):
        return ""

work["normalized_url"] = work["URL"].map(normalize_url)
work["hostname"] = work["URL"].map(get_hostname)

print("=" * 75)
print("RECHECKING URL AND HOSTNAME LABEL CONFLICTS")
print("=" * 75)

url_groups = work.groupby("normalized_url")["label"].agg(
    row_count="size",
    unique_labels="nunique"
)

url_conflicts = url_groups[url_groups["unique_labels"] > 1]

print("\nExact normalized URL conflicts:", len(url_conflicts))

if len(url_conflicts):
    conflict_values = set(url_conflicts.index)
    print(
        work[work["normalized_url"].isin(conflict_values)]
        [["URL", "normalized_url", "hostname", "label"]]
        .head(30)
        .to_string(index=False)
    )

hostname_groups = work[work["hostname"] != ""].groupby(
    "hostname"
)["label"].agg(
    row_count="size",
    unique_labels="nunique"
)

hostname_conflicts = hostname_groups[
    hostname_groups["unique_labels"] > 1
].sort_values("row_count", ascending=False)

print("\nHostnames with conflicting labels:", len(hostname_conflicts))

if len(hostname_conflicts):
    print("\nTop 15 conflicting hostnames:")
    print(hostname_conflicts.head(15).to_string())

    selected = list(hostname_conflicts.head(5).index)

    print("\nSample rows for the first five hostnames:")
    print(
        work[work["hostname"].isin(selected)]
        [["URL", "hostname", "label"]]
        .sort_values(["hostname", "label", "URL"])
        .to_string(index=False)
    )

print("\n" + "=" * 75)
print("DUPLICATE URL COUNTS")
print("=" * 75)

print("Total usable rows:", len(work))
print("Unique normalized URLs:", work["normalized_url"].nunique())
print(
    "Rows beyond first occurrence:",
    len(work) - work["normalized_url"].nunique()
)

print("\nRead-only diagnostic complete.")
