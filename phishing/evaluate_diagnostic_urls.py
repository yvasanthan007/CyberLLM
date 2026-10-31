from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "phishing_diagnostic_urls.csv"
MODEL_PATH = ROOT / "models" / "phishing_url_model.joblib"
REPORT_PATH = (
    ROOT
    / "artifacts"
    / "reports"
    / "phishing"
    / "diagnostic_evaluation.json"
)

# Existing model convention: 0 = phishing, 1 = legitimate.
df = pd.read_csv(DATA_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Production model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

required_columns = {"URL", "label", "expected_class", "test_category"}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Diagnostic dataset missing columns: {sorted(missing)}")

if df["URL"].isna().any():
    raise ValueError("Diagnostic dataset contains missing URLs.")

if not df["label"].isin([0, 1]).all():
    raise ValueError("Labels must be 0 (phishing) or 1 (legitimate).")

if not hasattr(model, "predict"):
    raise TypeError("Loaded artifact does not expose predict().")

X = df["URL"].astype(str)
y_true = df["label"].astype(int)
y_pred = model.predict(X).astype(int)

# Class 0 is phishing. Retrieve its score without assuming class order.
phishing_score = None
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 0 in classes:
        phishing_score = probabilities[:, classes.index(0)]

results = df.copy()
results["predicted_label"] = y_pred
results["predicted_class"] = results["predicted_label"].map(
    {0: "phishing", 1: "legitimate"}
)
results["correct"] = results["label"].astype(int) == y_pred

if phishing_score is not None:
    results["phishing_score"] = phishing_score

print("\n=== CYBERLLM PRODUCTION MODEL DIAGNOSTIC ===")
print(f"Model: {MODEL_PATH}")
print(f"Test URLs: {len(results)}")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print("\nConfusion matrix (rows=true, columns=predicted; labels [0, 1]):")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["phishing", "legitimate"],
        zero_division=0,
    )
)

print("\nIndividual predictions:")
columns = [
    "URL",
    "expected_class",
    "predicted_class",
    "correct",
]
if phishing_score is not None:
    columns.append("phishing_score")

print(results[columns].to_string(index=False))

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

report = {
    "model_path": str(MODEL_PATH),
    "test_count": int(len(results)),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "confusion_matrix_labels": [0, 1],
    "confusion_matrix": confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).tolist(),
    "classification_report": classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["phishing", "legitimate"],
        output_dict=True,
        zero_division=0,
    ),
    "predictions": results.to_dict(orient="records"),
    "warning": (
        "Small synthetic diagnostic set; not a representative estimate "
        "of real-world performance. Scores are not calibrated probabilities."
    ),
}

# Convert NumPy scalar values to standard JSON-compatible values.
REPORT_PATH.write_text(
    json.dumps(report, indent=2, default=str),
    encoding="utf-8",
)

print(f"\nReport saved to: {REPORT_PATH}")
print("Production model was loaded for evaluation only; it was not modified.")
