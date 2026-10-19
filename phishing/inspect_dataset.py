from pathlib import Path
import pandas as pd

path = Path("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")

if not path.exists():
    raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

df = pd.read_csv(path, nrows=5)
print("Columns:", len(df.columns))
print("\nColumn names:")
for col in df.columns:
    print(repr(col))

# Read the full dataset to count labels
full = pd.read_csv(path, usecols=["URL", "label"])
print(f"\nTotal records: {len(full):,}")
print("\nLabel counts:")
print(full["label"].value_counts(dropna=False).sort_index())
print("\nMissing URLs:", full["URL"].isna().sum())
print("Duplicate URLs:", full["URL"].duplicated().sum())
