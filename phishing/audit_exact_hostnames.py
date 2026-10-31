from pathlib import Path
from urllib.parse import urlsplit
import json

import pandas as pd

root = Path.cwd()
data_path = root / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
report_path = root / "artifacts" / "reports" / "phishing" / "exact_hostname_audit.json"

targets = [
    "youtube.com",
    "wikipedia.org",
    "microsoft.com",
    "example.com",
]

print("Loading dataset...")
df = pd.read_csv(data_path, usecols=["URL", "label"])
df = df.dropna(subset=["URL", "label"]).copy()
df["URL"] = df["URL"].astype(str).str.strip()
df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

def get_hostname(value):
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        return hostname.lower().rstrip(".") if hostname else ""
    except (ValueError, UnicodeError):
        return ""

df["hostname"] = df["URL"].map(get_hostname)

report = {
    "dataset_rows": int(len(df)),
    "label_mapping": {"0": "PHISHING", "1": "LEGITIMATE"},
    "domains": {},
}

for domain in targets:
    # Exact domain or a true subdomain.
    # Excludes deceptive hosts such as youtube.com.attacker.example.
    mask = df["hostname"].map(
        lambda host: host == domain or host.endswith("." + domain)
    )
    matched = df.loc[mask, ["URL", "hostname", "label"]]
    counts = matched["label"].value_counts().to_dict()

    report["domains"][domain] = {
        "matching_rows": int(len(matched)),
        "label_counts": {
            str(int(label)): int(count)
            for label, count in counts.items()
        },
        "examples": [
            {
                "url": row.URL,
                "hostname": row.hostname,
                "label": int(row.label),
            }
            for row in matched.head(10).itertuples(index=False)
        ],
    }

    print(f"\n{'=' * 65}")
    print(f"Domain: {domain}")
    print(f"Exact domain/subdomain rows: {len(matched):,}")
    print(f"Label counts (0=phishing, 1=legitimate): {counts}")

    if not matched.empty:
        print(matched.head(10).to_string(index=False))
    else:
        print("No exact domain or subdomain matches.")

report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(f"\nReport saved to: {report_path}")
print("Audit only: no dataset rows or models were modified.")
