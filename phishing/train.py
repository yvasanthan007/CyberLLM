
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "phishing_url_model.joblib"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, usecols=["URL", "label"])

df = df.dropna(subset=["URL", "label"])
df["URL"] = df["URL"].astype(str).str.strip()
df = df[df["URL"] != ""]
df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

# PhiUSIIL label convention:
# 0 = phishing, 1 = legitimate
if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError(f"Unexpected labels: {df['label'].unique()}")

# Remove duplicate URLs to reduce train/test leakage.
df = df.drop_duplicates(subset=["URL"])

print(f"Unique URL records: {len(df):,}")
print("Labels (0=phishing, 1=legitimate):")
print(df["label"].value_counts().sort_index())

X = df["URL"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=300_000,
            sublinear_tf=True,
            dtype="float32",
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

print(f"\nTraining on {len(X_train):,} URLs...")
model.fit(X_train, y_train)

print(f"Evaluating on {len(X_test):,} unseen URLs...")
predictions = model.predict(X_test)

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["phishing (0)", "legitimate (1)"],
    digits=4,
    zero_division=0,
))

print("Confusion matrix (rows=true, columns=predicted; labels 0, 1):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print("Remember: good test performance does not guarantee detection")
print("of new phishing campaigns or previously unseen domains.")
