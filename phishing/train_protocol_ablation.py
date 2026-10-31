from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import hashlib
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
PRODUCTION_MODEL = ROOT / "models" / "phishing_url_model.joblib"

OUTPUT_DIR = ROOT / "models"
REPORT_PATH = (
    ROOT / "artifacts" / "reports" / "phishing"
    / "protocol_ablation_evaluation.json"
)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def parse_url(value):
    try:
        parsed = urlsplit(str(value).strip())

        if parsed.scheme.lower() not in {"http", "https"}:
            return None

        if not parsed.hostname:
            return None

        return parsed

    except (ValueError, TypeError):
        return None


def get_hostname(value):
    parsed = parse_url(value)

    if parsed is None:
        return None

    return parsed.hostname.lower().rstrip(".")


def remove_protocol(value):
    """
    Remove only the HTTP(S) scheme.
    Preserve hostname, path, query, fragment and port.
    """
    parsed = parse_url(value)

    if parsed is None:
        return str(value).strip()

    return urlunsplit((
        "",
        parsed.netloc,
        parsed.path,
        parsed.query,
        parsed.fragment,
    )).lstrip("/")


def make_model():
    return Pipeline([
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


def evaluate(model, X_test, y_test, name):
    predictions = model.predict(X_test)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    report = {
        "name": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
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
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["PHISHING", "LEGITIMATE"],
            zero_division=0,
            output_dict=True,
        ),
    }

    print(f"\n===== {name} =====")
    print("Accuracy:", round(report["accuracy"], 6))
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["PHISHING", "LEGITIMATE"],
            zero_division=0,
            digits=4,
        )
    )
    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)

    return report


def main():
    started = time.time()

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    if not PRODUCTION_MODEL.exists():
        raise FileNotFoundError(
            f"Production model not found: {PRODUCTION_MODEL}"
        )

    production_hash_before = sha256_file(PRODUCTION_MODEL)

    print("Production model SHA-256 before experiment:")
    print(production_hash_before)

    print("\nLoading dataset...")
    df = pd.read_csv(DATASET, usecols=["URL", "label"])

    df = df.dropna(subset=["URL", "label"]).copy()
    df["URL"] = df["URL"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected labels; expected 0 and 1.")

    df = df[df["URL"] != ""].copy()
    df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)

    df["hostname"] = df["URL"].map(get_hostname)
    invalid_count = int(df["hostname"].isna().sum())
    df = df.dropna(subset=["hostname"]).reset_index(drop=True)

    if df["label"].nunique() != 2:
        raise ValueError("Both label classes must be present.")

    print("Usable unique URLs:", len(df))
    print("Invalid URLs removed:", invalid_count)
    print("Label counts:", df["label"].value_counts().sort_index().to_dict())

    # Use one shared hostname-grouped split for both experiments.
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

    shared_hosts = (
        set(train_df["hostname"]) & set(test_df["hostname"])
    )

    if shared_hosts:
        raise RuntimeError("Hostname leakage detected.")

    if (
        train_df["label"].nunique() != 2
        or test_df["label"].nunique() != 2
    ):
        raise RuntimeError(
            "Training or testing split lacks one class. "
            "Do not interpret the experiment; use a validated split."
        )

    print("\n===== SHARED SPLIT =====")
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("Shared hostnames:", len(shared_hosts))
    print("Training labels:", train_df["label"].value_counts().to_dict())
    print("Testing labels:", test_df["label"].value_counts().to_dict())

    y_train = train_df["label"]
    y_test = test_df["label"]

    # Experiment A: original URL strings.
    X_train_original = train_df["URL"]
    X_test_original = test_df["URL"]

    # Experiment B: remove only HTTP(S) protocol.
    X_train_stripped = train_df["URL"].map(remove_protocol)
    X_test_stripped = test_df["URL"].map(remove_protocol)

    print("\nTraining baseline model...")
    baseline = make_model()
    baseline.fit(X_train_original, y_train)

    print("\nTraining protocol-ablated model...")
    ablated = make_model()
    ablated.fit(X_train_stripped, y_train)

    baseline_report = evaluate(
        baseline,
        X_test_original,
        y_test,
        "BASELINE: ORIGINAL URL",
    )

    ablated_report = evaluate(
        ablated,
        X_test_stripped,
        y_test,
        "PROTOCOL ABLATION: SCHEME REMOVED",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    baseline_path = OUTPUT_DIR / "phishing_url_protocol_baseline_candidate.joblib"
    ablated_path = OUTPUT_DIR / "phishing_url_no_protocol_candidate.joblib"

    # Both filenames are experimental; production model is never overwritten.
    joblib.dump(baseline, baseline_path)
    joblib.dump(ablated, ablated_path)

    production_hash_after = sha256_file(PRODUCTION_MODEL)

    report = {
        "experiment": "protocol_ablation",
        "dataset": str(DATASET),
        "label_mapping": {
            "0": "PHISHING",
            "1": "LEGITIMATE",
        },
        "split_method": "GroupShuffleSplit by hostname",
        "random_state": 42,
        "test_size": 0.20,
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
        "training_hostnames": int(train_df["hostname"].nunique()),
        "testing_hostnames": int(test_df["hostname"].nunique()),
        "shared_hostnames": int(len(shared_hosts)),
        "invalid_rows_removed": invalid_count,
        "baseline": baseline_report,
        "protocol_ablation": ablated_report,
        "production_model": {
            "path": str(PRODUCTION_MODEL),
            "sha256_before": production_hash_before,
            "sha256_after": production_hash_after,
            "unchanged": production_hash_before == production_hash_after,
        },
        "experimental_models": {
            "baseline": str(baseline_path),
            "protocol_ablation": str(ablated_path),
        },
        "limitations": [
            "A single hostname-grouped split is not proof of real-world performance.",
            "The dataset contains a strong protocol-label association.",
            "Removing the scheme does not remove all dataset bias.",
            "This experiment does not validate performance on independently verified URLs.",
            "Candidate models are not production replacements.",
            "Model scores are not calibrated probabilities.",
        ],
        "elapsed_seconds": round(time.time() - started, 2),
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n===== SAFETY CHECK =====")
    print("Production model unchanged:", report["production_model"]["unchanged"])

    if not report["production_model"]["unchanged"]:
        raise RuntimeError(
            "Production model hash changed unexpectedly. Investigate immediately."
        )

    print("\n===== SAVED ARTIFACTS =====")
    print("Baseline candidate:", baseline_path)
    print("Protocol-ablated candidate:", ablated_path)
    print("Evaluation report:", REPORT_PATH)
    print("Elapsed seconds:", report["elapsed_seconds"])


if __name__ == "__main__":
    main()
