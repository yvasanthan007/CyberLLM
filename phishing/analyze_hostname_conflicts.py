from pathlib import Path
import pandas as pd

source = Path(
    "artifacts/reports/phishing/hostname_conflict_details.csv"
)

df = pd.read_csv(source, low_memory=False)

print("=" * 70)
print("HOSTNAME CONFLICT ANALYSIS")
print("=" * 70)

print("\nRows by protocol and label:")
print(
    pd.crosstab(
        df["scheme"],
        df["label"],
        margins=True
    ).to_string()
)

print("\nRows by label:")
print(df["label"].value_counts().sort_index().to_string())

print("\nRecords with non-root paths:")
non_root = df[
    df["path"].fillna("").astype(str).ne("")
    & df["path"].fillna("").astype(str).ne("/")
]

print("Non-root path records:", len(non_root))

print("\nConflicting hostnames with HTTP and HTTPS records:")
protocol_counts = df.groupby("hostname")["scheme"].nunique()
both_protocols = protocol_counts[protocol_counts > 1].index

print("Hostnames using both protocols:", len(both_protocols))

protocol_details = df[df["hostname"].isin(both_protocols)]

print("\nHostnames with multiple labels and multiple protocols:")
label_counts = df.groupby("hostname")["label"].nunique()
both_dimensions = protocol_details[
    protocol_details["hostname"].isin(
        label_counts[label_counts > 1].index
    )
]

for hostname, group in both_dimensions.groupby("hostname"):
    print(f"\n{hostname}")
    print(
        group[["scheme", "path", "label", "URL"]]
        .to_string(index=False)
    )

print("\nRead-only analysis complete.")
