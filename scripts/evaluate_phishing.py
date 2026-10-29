"""
Evaluate the EXISTING phishing classifier separately.

This uses the phishing dataset's held-out test split (train_test_split
with random_state=42, matching phishing/train.py) so it does not
conflate conversational-model metrics with classifier metrics.

Never touches conversational data or models.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_PATH = ROOT / "models" / "phishing_url_model.joblib"
REPORT_DIR = ROOT / "artifacts" / "reports" / "phishing"


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Phishing dataset missing: {DATA_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Phishing model missing: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH, usecols=["URL", "label"]).dropna(subset=["URL", "label"])
    df["URL"] = df["URL"].astype(str).str.strip()
    df = df[df["URL"] != ""]
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df = df.drop_duplicates(subset=["URL"])

    X = df["URL"]
    y = df["label"]
    # Same split recipe as phishing/train.py (random_state=42)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    report_text = classification_report(
        y_test, predictions, labels=[0, 1],
        target_names=["phishing (0)", "legitimate (1)"],
        digits=4, zero_division=0,
    )
    cm = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": "phishing_url_tfidf_logreg",
        "artifact": str(MODEL_PATH.relative_to(ROOT)),
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "test_size": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_phishing": round(float(precision_score(y_test, predictions, pos_label=0, zero_division=0)), 4),
        "recall_phishing": round(float(recall_score(y_test, predictions, pos_label=0, zero_division=0)), 4),
        "f1_phishing": round(float(f1_score(y_test, predictions, pos_label=0, zero_division=0)), 4),
        "confusion_matrix_labels_0_1": cm,
        "classification_report": report_text,
        "disclaimer": "Metrics are on the held-out split of PhiUSIIL; "
                      "real-world performance on unseen campaigns differs.",
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "phishing_evaluation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(report_text)
    print(f"Report: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
