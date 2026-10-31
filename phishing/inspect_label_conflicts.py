from pathlib import Path
from urllib.parse import urlsplit
import pandas as pd

dataset = Path(
    "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
)

output_dir = Path(
    "artifacts/reports/phishing"
)
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(dataset, low_memory=False)

work = df[["URL", "label"]].copy()
work["URL"] = work["URL"].astype("string").str.strip()
work["label"] = pd.to_numeric(
    work["label"], errors="coerce"
)

work = work.dropna(subset=["URL", "label"])
work["url_lower"] = work["URL"].str.lower()

def get_hostname(value):
    try:
        parsed = urlsplit(
            value if "://" in value else "//" + value
        )
        return (parsed.hostname or "").lower().rstrip(".")
    except (ValueError, TypeError):
        return ""

work["hostname"] = work["URL"].map(get_hostname)

# Find URLs that contain trusted brand text.
brands = [
    "youtube.com",
    "google.com",
    "microsoft.com",
    "github.com",
    "paypal.com",
    "wikipedia.org",
    "example.com",
]

brand_mask = work["url_lower"].apply(
    lambda value: any(brand in value for brand in brands)
)

brand_records = work.loc[
    brand_mask,
    ["URL", "hostname", "label"]
].copy()

# Find repeated URL strings with different labels.
url_label_counts = (
    work.groupby("url_lower")["label"]
    .nunique()
)

conflicting_urls = set(
    url_label_counts[url_label_counts > 1].index
)

duplicate_conflicts = work[
    work["url_lower"].isin(conflicting_urls)
][["URL", "hostname", "label"]].copy()

# Find hostnames appearing with both labels.
hostname_label_counts = (
    work[work["hostname"] != ""]
    .groupby("hostname")["label"]
    .nunique()
)

conflicting_hosts = set(
    hostname_label_counts[
        hostname_label_counts > 1
    ].index
)

hostname_conflicts = work[
    work["hostname"].isin(conflicting_hosts)
][["URL", "hostname", "label"]].copy()

# Inspect a manageable sample of the .com/ pattern.
com_mask = work["url_lower"].str.contains(
    ".com/", regex=False, na=False
)

com_records = work.loc[
    com_mask,
    ["URL", "hostname", "label"]
].head(1000).copy()

brand_path = output_dir / "brand_label_audit.csv"
url_path = output_dir / "duplicate_url_label_conflicts.csv"
host_path = output_dir / "hostname_label_conflicts.csv"
com_path = output_dir / "com_pattern_sample.csv"

brand_records.to_csv(brand_path, index=False)
duplicate_conflicts.to_csv(url_path, index=False)
hostname_conflicts.to_csv(host_path, index=False)
com_records.to_csv(com_path, index=False)

print("=" * 75)
print("CYBERLLM LABEL INVESTIGATION")
print("=" * 75)

print("\nBrand-containing URL records:", len(brand_records))
print("Exact URL strings with conflicting labels:",
      len(conflicting_urls))
print("Rows in exact URL conflicts:", len(duplicate_conflicts))
print("Hostnames with conflicting labels:",
      len(conflicting_hosts))
print("Rows for conflicting hostnames:", len(hostname_conflicts))
print("Sampled .com/ rows:", len(com_records))

print("\n" + "=" * 75)
print("BRAND-CONTAINING URL EXAMPLES")
print("=" * 75)

for brand in brands:
    subset = brand_records[
        brand_records["URL"].str.lower().str.contains(
            brand, regex=False, na=False
        )
    ]

    print(f"\n{brand}: {len(subset)} rows")

    if len(subset):
        print(
            subset[["URL", "hostname", "label"]]
            .head(8)
            .to_string(index=False)
        )

print("\n" + "=" * 75)
print("EXACT URL CONFLICT EXAMPLES")
print("=" * 75)

if duplicate_conflicts.empty:
    print("No exact URL strings have conflicting labels.")
else:
    print(
        duplicate_conflicts.head(20).to_string(index=False)
    )

print("\n" + "=" * 75)
print("HOSTNAME CONFLICT EXAMPLES")
print("=" * 75)

if hostname_conflicts.empty:
    print("No hostname conflicts found.")
else:
    print(
        hostname_conflicts.head(30).to_string(index=False)
    )

print("\n" + "=" * 75)
print("REPORT FILES")
print("=" * 75)

for path in [
    brand_path,
    url_path,
    host_path,
    com_path,
]:
    print(path.resolve())

print("\nInspection complete. No model was trained or changed.")
