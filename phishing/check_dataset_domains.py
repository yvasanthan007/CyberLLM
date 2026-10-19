
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATA_PATH, usecols=["URL", "label"])

youtube = df[
    df["URL"].str.contains(
        r"(?i)(?:^|//)(?:[^/]+\.)?youtube\.com(?:[/:?#]|$)",
        regex=True,
        na=False,
    )
]

print(f"YouTube URL records: {len(youtube)}")

if youtube.empty:
    print("No matching YouTube URLs found in the dataset.")
else:
    print("\nLabel counts (0=phishing, 1=legitimate):")
    print(youtube["label"].value_counts().sort_index())

    print("\nExample records:")
    print(youtube.head(15).to_string(index=False))
