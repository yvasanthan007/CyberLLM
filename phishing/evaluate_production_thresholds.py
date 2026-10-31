from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "raw"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)

MODEL_PATH = ROOT / "models" / "phishing_url_model.joblib"

OUT_PATH = (
    ROOT / "artifacts" / "reports" / "phishing"
    / "production_threshold_evaluation.json"
)

THRESHOLDS = [
    0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.95, 0.99,
]

started = time.time()

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, usecols=["URL", "label"])

df = df.dropna(subset=["URL", "label"]).copy()
df["URL"] = df["URL"].astype(str).str.strip()
df = df[df["URL"] != ""]

df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected labels found.")

df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)

X = df["URL"]
y = df["label"]

# Match the split used by phishing/train.py.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training rows (not used for fitting): {len(X_train):,}")
print(f"Evaluation rows: {len(X_test):,}")

print("\nLoading production model...")
model = joblib.load(MODEL_PATH)

print("Scoring evaluation URLs...")
probabilities = model.predict_proba(X_test)

classes = list(model.classes_)

if 0 not in classes:
    raise ValueError(f"Phishing class 0 not found. Classes: {classes}")

phishing_index = classes.index(0)
phishing_scores = probabilities[:, phishing_index]

# Convert labels to a phishing-positive target:
# 1 = phishing; 0 = legitimate.
y_phishing = (y_test.to_numpy() == 0).astype(int)

results = []

for threshold in THRESHOLDS:
    # Positive prediction means phishing.
    predictions = (phishing_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_phishing,
        predictions,
        labels=[0, 1],
    ).ravel()

    legitimate_false_positive_rate = (
        fp / (fp + tn) if (fp + tn) else 0.0
    )

    phishing_recall = (
        tp / (tp + fn) if (tp + fn) else 0.0
    )

    legitimate_recall = (
        tn / (tn + fp) if (tn + fp) else 0.0
    )

    results.append({
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_phishing, predictions)),
        "phishing_precision": float(
            precision_score(y_phishing, predictions, zero_division=0)
        ),
        "phishing_recall": float(
            recall_score(y_phishing, predictions, zero_division=0)
        ),
        "phishing_f1": float(
            f1_score(y_phishing, predictions, zero_division=0)
        ),
        "legitimate_recall": float(legitimate_recall),
        "legitimate_false_positive_rate": float(
            legitimate_false_positive_rate
        ),
        "true_legitimate": int(tn),
        "false_phishing_alerts": int(fp),
        "missed_phishing": int(fn),
        "detected_phishing": int(tp),
    })

print("\n===== THRESHOLD RESULTS =====")
print(
    "Threshold | Accuracy | Phishing recall | "
    "Legitimate FPR | False alerts | Missed phishing"
)

for item in results:
    print(
        f"{item['threshold']:9.2f} | "
        f"{item['accuracy']:.4f} | "
        f"{item['phishing_recall']:.4f} | "
        f"{item['legitimate_false_positive_rate']:.4f} | "
        f"{item['false_phishing_alerts']:12d} | "
        f"{item['missed_phishing']:15d}"
    )

report = {
    "model_path": str(MODEL_PATH),
    "dataset_path": str(DATA_PATH),
    "evaluation_method": (
        "Stratified random 80/20 split matching phishing/train.py; "
        "existing production model evaluated without retraining."
    ),
    "random_state": 42,
    "unique_rows": int(len(df)),
    "evaluation_rows": int(len(X_test)),
    "evaluation_label_counts": {
        "phishing": int((y_test == 0).sum()),
        "legitimate": int((y_test == 1).sum()),
    },
    "score_is_calibrated_probability": False,
    "threshold_results": results,
    "limitations": [
        "This is a random URL-level split, not an unseen-hostname test.",
        "The model was trained using this same split recipe, but its "
        "artifact may have been trained at a different time or on "
        "different data; verify artifact provenance before interpreting "
        "these metrics as held-out performance.",
        "Dataset collection and label biases may remain.",
        "Threshold selection requires an appropriate validation set.",
        "No model or production API settings were modified.",
    ],
    "elapsed_seconds": round(time.time() - started, 2),
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nReport saved to:")
print(OUT_PATH)
print("\nProduction model and API settings remain unchanged.")
