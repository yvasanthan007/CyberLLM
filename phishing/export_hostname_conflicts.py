from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

source = Path("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
output = Path("artifacts/reports/phishing/hostname_conflict_details.csv")

df = pd.read_csv(source, low_memory=False)

work = df[["URL", "label"]].copy()
work["URL"] = work["URL"].astype("string").str.strip()
work["label"] = pd.to_numeric(work["label"], errors="coerce")
work = work.dropna(subset=["URL", "label"]).copy()

def parse_url(value):
    try:
        raw = str(value).strip()
        parsed = urlsplit(raw if "://" in raw else "//" + raw)

        return pd.Series({
            "hostname": (parsed.hostname or "").lower().rstrip("."),
            "scheme": parsed.scheme.lower(),
            "path": parsed.path,
            "query_length": len(parsed.query),
        })
    except (ValueError, TypeError):
        return pd.Series({
            "hostname": "",
            "scheme": "",
            "path": "",
            "query_length": 0,
        })

parts = work["URL"].apply(parse_url)
work = pd.concat([work, parts], axis=1)

hostname_labels = (
    work[work["hostname"] != ""]
    .groupby("hostname")["label"]
    .nunique()
)

conflicting_hosts = hostname_labels[
    hostname_labels > 1
].index

details = work[work["hostname"].isin(conflicting_hosts)].copy()

details["hostname_row_count"] = (
    details.groupby("hostname")["URL"].transform("size")
)

details = details.sort_values(
    ["hostname", "label", "URL"]
)

output.parent.mkdir(parents=True, exist_ok=True)
details.to_csv(output, index=False)

print("Conflicting hostnames:", len(conflicting_hosts))
print("Rows in conflict groups:", len(details))
print("Report saved to:", output.resolve())

print("\nSample records:")
print(
    details[
        ["hostname", "scheme", "path", "query_length", "label", "URL"]
    ].head(30).to_string(index=False)
)
