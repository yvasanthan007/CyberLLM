from pathlib import Path
from urllib.parse import urlsplit
import json
import time

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"

MODEL_OUT = ROOT / "models" / "phishing_url_candidate.joblib"
REPORT_OUT = ROOT / "artifacts" / "reports" / "phishing" / "candidate_evaluation.json"


def get_hostname(value):
    try:
        parsed = urlsplit(str(value).strip())

        if parsed.scheme.lower() not in {"http", "https"}:
            return None

        if not parsed.hostname:
            return None

        return parsed.hostname.lower().rstrip(".")
    except (ValueError, TypeError):
        return None


def main():
    started = time.time()

    print("Loading URL and label columns...")

    df = pd.read_csv(
        DATASET,
        usecols=["URL", "label"],
    )

    df = df.dropna(subset=["URL", "label"]).copy()
    df["URL"] = df["URL"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # This dataset uses 0 = phishing and 1 = legitimate.
    unexpected_labels = set(df["label"].unique()) - {0, 1}

    if unexpected_labels:
        raise ValueError(f"Unexpected labels: {unexpected_labels}")

    before = len(df)

    # Keep one occurrence of each exact URL.
    df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)

    print("Rows before deduplication:", before)
    print("Rows after deduplication:", len(df))
    print("Labels:", df["label"].value_counts().sort_index().to_dict())

    print("\nExtracting hostnames for group-aware splitting...")

    df["hostname"] = df["URL"].map(get_hostname)

    invalid = int(df["hostname"].isna().sum())

    print("Rows without a valid HTTP(S) hostname:", invalid)

    df = df.dropna(subset=["hostname"]).reset_index(drop=True)

    if df["label"].nunique() != 2:
        raise ValueError("Both classes must be present.")

    # Group by hostname so URLs from the same hostname cannot cross
    # the training/test boundary.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            df["URL"],
            df["label"],
            groups=df["hostname"],
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_hosts = set(train_df["hostname"])
    test_hosts = set(test_df["hostname"])
    shared_hosts = train_hosts.intersection(test_hosts)

    if shared_hosts:
        raise RuntimeError("Hostname leakage detected in split.")

    print("\n===== SPLIT SUMMARY =====")
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("Training hostnames:", len(train_hosts))
    print("Testing hostnames:", len(test_hosts))
    print("Shared hostnames:", len(shared_hosts))

    print("\nTraining labels:")
    print(train_df["label"].value_counts().sort_index().to_dict())

    print("\nTesting labels:")
    print(test_df["label"].value_counts().sort_index().to_dict())

    # Candidate uses only the URL string.
    candidate = Pipeline([
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

    print("\nTraining experimental candidate...")
    candidate.fit(train_df["URL"], train_df["label"])

    print("Evaluating on unseen hostnames...")

    y_true = test_df["label"]
    y_pred = candidate.predict(test_df["URL"])

    accuracy = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n===== CANDIDATE RESULTS =====")
    print("Accuracy:", round(accuracy, 6))
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["PHISHING", "LEGITIMATE"],
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)

    # Save to a separate filename; never overwrite the production model.
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, MODEL_OUT)

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "model_type": "experimental_url_only_tfidf_logistic_regression",
        "production_model_modified": False,
        "dataset": str(DATASET),
        "split_method": "GroupShuffleSplit by hostname",
        "random_state": 42,
        "test_size": 0.20,
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
        "training_hostnames": int(len(train_hosts)),
        "testing_hostnames": int(len(test_hosts)),
        "shared_hostnames": int(len(shared_hosts)),
        "invalid_hostname_rows_removed": invalid,
        "label_mapping": {
            "0": "PHISHING",
            "1": "LEGITIMATE",
        },
        "accuracy": accuracy,
        "per_class": {
            "PHISHING": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "LEGITIMATE": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "confusion_matrix_labels_0_then_1": matrix.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["PHISHING", "LEGITIMATE"],
            zero_division=0,
            output_dict=True,
        ),
        "limitations": [
            "Evaluation uses a single hostname-grouped split.",
            "Dataset collection and labelling biases may remain.",
            "Unseen hostnames do not represent all real-world URLs.",
            "Scores are not calibrated probabilities.",
            "This is an experimental candidate, not a production replacement.",
        ],
        "elapsed_seconds": round(time.time() - started, 2),
        "candidate_model_path": str(MODEL_OUT),
    }

    REPORT_OUT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n===== SAVED ARTIFACTS =====")
    print("Candidate model:", MODEL_OUT)
    print("Evaluation report:", REPORT_OUT)
    print("Production model was not modified.")
    print("Elapsed seconds:", report["elapsed_seconds"])


if __name__ == "__main__":
    main()
