from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phishing.url_feature_extractor import URLFeatureExtractor, parse_url

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "raw"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)

PRODUCTION_MODEL_PATH = (
    ROOT / "models" / "phishing_url_model.joblib"
)

CANDIDATE_PATH = (
    ROOT / "models" / "phishing_url_improved_candidate.joblib"
)

REPORT_PATH = (
    ROOT / "artifacts" / "reports" / "phishing"
    / "improved_candidate_evaluation.json"
)

TARGETED_TESTS = [
    ("Google homepage", "https://www.google.com/", 1),
    ("YouTube homepage", "https://www.youtube.com/", 1),
    (
        "YouTube watch URL",
        "https://www.youtube.com/watch?v=example",
        1,
    ),
    ("Wikipedia homepage", "https://en.wikipedia.org/", 1),
    (
        "Wikipedia article",
        "https://en.wikipedia.org/wiki/Phishing",
        1,
    ),
    ("Microsoft homepage", "https://www.microsoft.com/", 1),
    ("Microsoft login", "https://login.microsoft.com/", 1),
    ("Office homepage", "https://www.office.com/", 1),
    ("Example reserved domain", "https://example.com/", 1),
    ("Example login path", "https://example.com/login", 1),
    (
        "PayPal lookalike",
        "http://paypal.com.security-check.example/",
        0,
    ),
    (
        "Suspicious synthetic URL",
        "http://login-example.invalid/verify?account=123",
        0,
    ),
]

started = time.time()

for path in [DATA_PATH, PRODUCTION_MODEL_PATH]:
    if not path.exists():
        raise FileNotFoundError(path)

print("Loading URL dataset...")
df = pd.read_csv(DATA_PATH, usecols=["URL", "label"])

df = df.dropna(subset=["URL", "label"]).copy()
df["URL"] = df["URL"].astype(str).str.strip()
df = df[df["URL"] != ""]
df["label"] = pd.to_numeric(
    df["label"], errors="raise"
).astype(int)

if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected dataset labels.")

df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)

# Keep only HTTP(S) URLs with a usable hostname.
parsed = df["URL"].map(parse_url)
df["hostname"] = parsed.map(
    lambda item: item.get("hostname", "")
    if isinstance(item, dict) else ""
)

df = df[df["hostname"].astype(str).str.len() > 0].copy()
df = df.reset_index(drop=True)

print(f"Unique usable URLs: {len(df):,}")
print("Labels: 0=phishing, 1=legitimate")

# Group by hostname to reduce leakage across train and test.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(df["URL"], df["label"], groups=df["hostname"])
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

train_hosts = set(train_df["hostname"])
test_hosts = set(test_df["hostname"])
shared_hosts = train_hosts & test_hosts

if shared_hosts:
    raise RuntimeError(
        f"Hostname leakage detected: {len(shared_hosts)} shared hostnames."
    )

print(f"Training URLs: {len(train_df):,}")
print(f"Testing URLs: {len(test_df):,}")
print(f"Training hostnames: {len(train_hosts):,}")
print(f"Testing hostnames: {len(test_hosts):,}")
print(f"Shared hostnames: {len(shared_hosts)}")

# URLFeatureExtractor produces a structured feature matrix.
# TfidfVectorizer operates directly on the URL strings.
features = ColumnTransformer(
    transformers=[
        (
            "url_chars",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                max_features=300_000,
                sublinear_tf=True,
                dtype=np.float32,
            ),
            "URL",
        ),
        (
            "url_structure",
            Pipeline([
                ("extract", URLFeatureExtractor()),
                ("scale", StandardScaler()),
            ]),
            ["URL"],
        ),
    ],
    remainder="drop",
)

candidate = Pipeline([
    ("features", features),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

print("\nTraining experimental candidate...")
candidate.fit(
    train_df[["URL"]],
    train_df["label"],
)

print("Evaluating grouped holdout...")
test_predictions = candidate.predict(test_df[["URL"]])
test_probabilities = candidate.predict_proba(test_df[["URL"]])

classes = list(candidate.classes_)
if 0 not in classes:
    raise ValueError(f"Phishing class 0 not found: {classes}")

phishing_index = classes.index(0)
test_phishing_scores = test_probabilities[:, phishing_index]

# Convert label 0 (phishing) to positive class 1 for metrics.
y_true = (test_df["label"].to_numpy() == 0).astype(int)
y_pred = (test_predictions == 0).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

grouped_metrics = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "phishing_precision": float(
        precision_score(y_true, y_pred, zero_division=0)
    ),
    "phishing_recall": float(
        recall_score(y_true, y_pred, zero_division=0)
    ),
    "phishing_f1": float(
        f1_score(y_true, y_pred, zero_division=0)
    ),
    "legitimate_recall": float(
        recall_score(1 - y_true, 1 - y_pred, zero_division=0)
    ),
    "confusion_matrix": {
        "true_legitimate": int(tn),
        "false_phishing_alerts": int(fp),
        "missed_phishing": int(fn),
        "detected_phishing": int(tp),
    },
    "evaluation_rows": int(len(test_df)),
    "train_hostnames": int(len(train_hosts)),
    "test_hostnames": int(len(test_hosts)),
    "shared_hostnames": int(len(shared_hosts)),
}

print("\nGROUPED HOLDOUT METRICS")
print(json.dumps(grouped_metrics, indent=2))

print("\nEvaluating targeted URLs...")
production = joblib.load(PRODUCTION_MODEL_PATH)

targeted_results = []

for name, url, expected_label in TARGETED_TESTS:
    candidate_score = float(
        candidate.predict_proba(pd.DataFrame({"URL": [url]}))[
            0, phishing_index
        ]
    )
    candidate_label = (
        0 if candidate_score >= 0.50 else 1
    )

    production_score = float(
        production.predict_proba([url])[
            0, list(production.classes_).index(0)
        ]
    )
    production_label = int(production.predict([url])[0])

    result = {
        "name": name,
        "url": url,
        "expected_label": expected_label,
        "candidate_label": candidate_label,
        "candidate_phishing_score": candidate_score,
        "candidate_correct": candidate_label == expected_label,
        "production_label": production_label,
        "production_phishing_score": production_score,
        "production_correct": production_label == expected_label,
    }

    targeted_results.append(result)

    print(
        f"{name}: expected={expected_label}, "
        f"candidate={candidate_label} "
        f"(score={candidate_score:.4f}), "
        f"production={production_label} "
        f"(score={production_score:.4f})"
    )

candidate_correct = sum(
    item["candidate_correct"] for item in targeted_results
)
production_correct = sum(
    item["production_correct"] for item in targeted_results
)

# Save candidate to a separate file. Never overwrite production.
CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(candidate, CANDIDATE_PATH)

report = {
    "candidate_path": str(CANDIDATE_PATH),
    "production_model_path": str(PRODUCTION_MODEL_PATH),
    "label_mapping": {
        "0": "PHISHING",
        "1": "LEGITIMATE",
    },
    "threshold": 0.50,
    "score_is_calibrated_probability": False,
    "grouped_holdout_metrics": grouped_metrics,
    "targeted_diagnostic": {
        "number_of_cases": len(targeted_results),
        "candidate_correct": int(candidate_correct),
        "candidate_accuracy": float(
            candidate_correct / len(targeted_results)
        ),
        "production_correct": int(production_correct),
        "production_accuracy": float(
            production_correct / len(targeted_results)
        ),
        "cases": targeted_results,
    },
    "limitations": [
        "Dataset labels and collection biases may remain.",
        "The targeted set is small and hand-picked.",
        "A hostname-grouped split does not prove real-world performance.",
        "The phishing score is not a calibrated probability.",
        "The candidate has not been promoted to production.",
    ],
    "elapsed_seconds": round(time.time() - started, 2),
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\nCandidate saved to:", CANDIDATE_PATH)
print("Report saved to:", REPORT_PATH)
print("\nProduction model and API settings remain unchanged.")


