from pathlib import Path
import hashlib
import json

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "phishing_url_model.joblib"
DATA_PATH = (
    ROOT / "data" / "raw"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)

REPORT_PATH = (
    ROOT / "artifacts" / "reports" / "phishing"
    / "production_model_audit.json"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

print("Loading production model...")
model = joblib.load(MODEL_PATH)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, usecols=["URL", "label"])

df = df.dropna(subset=["URL", "label"]).copy()
df["URL"] = df["URL"].astype(str).str.strip()
df = df[df["URL"] != ""]
df["label"] = pd.to_numeric(
    df["label"], errors="raise"
).astype(int)

original_rows = len(df)
unique_urls = df["URL"].nunique()
duplicate_rows = original_rows - unique_urls

label_counts = {
    str(int(label)): int(count)
    for label, count in df["label"].value_counts().items()
}

model_hash = hashlib.sha256(
    MODEL_PATH.read_bytes()
).hexdigest()

pipeline_steps = []

if hasattr(model, "named_steps"):
    for name, step in model.named_steps.items():
        pipeline_steps.append({
            "name": name,
            "class": (
                f"{step.__class__.__module__}."
                f"{step.__class__.__name__}"
            ),
            "parameters": {
                key: value
                for key, value in step.get_params(deep=False).items()
                if key in [
                    "ngram_range",
                    "max_features",
                    "min_df",
                    "max_iter",
                    "class_weight",
                    "C",
                ]
            },
        })

classes = getattr(model, "classes_", None)

report = {
    "model_path": str(MODEL_PATH),
    "model_size_bytes": MODEL_PATH.stat().st_size,
    "model_sha256": model_hash,
    "model_classes": (
        [int(value) for value in classes]
        if classes is not None
        else None
    ),
    "pipeline_steps": pipeline_steps,
    "dataset_path": str(DATA_PATH),
    "dataset_rows_after_basic_cleaning": int(original_rows),
    "unique_urls": int(unique_urls),
    "duplicate_url_rows": int(duplicate_rows),
    "label_counts": label_counts,
    "label_mapping": {
        "0": "PHISHING",
        "1": "LEGITIMATE",
    },
    "limitations": [
        "This script inspects metadata and dataset counts only.",
        "It does not establish when or on which exact rows the model was trained.",
        "It does not change the model, dataset, or API.",
    ],
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(
    json.dumps(report, indent=2, default=str),
    encoding="utf-8",
)

print("\n===== PRODUCTION MODEL AUDIT =====")
print("Model:", MODEL_PATH)
print("Model size:", MODEL_PATH.stat().st_size, "bytes")
print("SHA-256:", model_hash)
print("Model classes:", classes)
print("Pipeline steps:", pipeline_steps)
print("Dataset rows:", original_rows)
print("Unique URLs:", unique_urls)
print("Duplicate URL rows:", duplicate_rows)
print("Label counts:", label_counts)
print("\nReport saved to:", REPORT_PATH)
print("No production files were modified.")
