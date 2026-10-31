from pathlib import Path
from urllib.parse import urlsplit
from collections import Counter
import ipaddress
import json
import re
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"

MODEL_OUT = ROOT / "models" / "phishing_url_hostname_candidate.joblib"
REPORT_OUT = (
    ROOT / "artifacts" / "reports" / "phishing"
    / "hostname_candidate_evaluation.json"
)


def parse_url(value):
    value = str(value).strip()

    try:
        parsed = urlsplit(value)
        hostname = (parsed.hostname or "").lower().rstrip(".")
        scheme = parsed.scheme.lower()

        if scheme not in {"http", "https"} or not hostname:
            return None

        try:
            ipaddress.ip_address(hostname)
            is_ip = 1
        except ValueError:
            is_ip = 0

        return {
            "url": value,
            "hostname": hostname,
            "scheme": scheme,
            "path": parsed.path or "",
            "query": parsed.query or "",
            "fragment": parsed.fragment or "",
            "is_ip": is_ip,
        }
    except (ValueError, TypeError):
        return None


def hostname_features(url):
    parsed = parse_url(url)

    if parsed is None:
        return {
            "invalid_url": 1,
            "hostname_length": 0,
            "hostname_dot_count": 0,
            "subdomain_count": 0,
            "is_ip": 0,
        }

    host = parsed["hostname"]
    labels = host.split(".")
    path = parsed["path"]
    query = parsed["query"]
    url_lower = parsed["url"].lower()

    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0

    # Heuristic features only: these do not establish maliciousness.
    return {
        "invalid_url": 0,
        "hostname_length": len(host),
        "hostname_dot_count": host.count("."),
        "subdomain_count": max(0, len(labels) - 2),
        "is_ip": is_ip,
        "uses_https": int(parsed["scheme"] == "https"),
        "uses_http": int(parsed["scheme"] == "http"),
        "path_length": len(path),
        "query_length": len(query),
        "url_length": len(parsed["url"]),
        "hostname_hyphen_count": host.count("-"),
        "hostname_digit_count": sum(c.isdigit() for c in host),
        "hostname_underscore_count": host.count("_"),
        "hostname_has_punycode": int("xn--" in host),
        "hostname_has_brand_word": int(
            any(
                word in host
                for word in (
                    "paypal", "microsoft", "google", "apple",
                    "amazon", "facebook", "youtube", "netflix",
                )
            )
        ),
        "host_has_login_word": int(
            any(word in host for word in ("login", "signin", "verify", "secure"))
        ),
        "url_has_at": int("@" in parsed["url"]),
        "url_has_encoded_chars": int("%" in parsed["url"]),
        "url_has_double_slash_path": int("//" in path),
        "url_has_password_word": int(
            any(word in url_lower for word in ("password", "passwd"))
        ),
        "url_has_redirect_word": int(
            any(word in url_lower for word in ("redirect", "return=", "continue="))
        ),
        "url_has_ip_text": int(
            bool(re.search(r"(?:\d{1,3}\.){3}\d{1,3}", host))
        ),
        "tld_length": len(labels[-1]) if labels else 0,
    }


class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """Convert URL strings into numeric hostname/URL structure features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        records = [hostname_features(value) for value in X]
        return pd.DataFrame(records).fillna(0).to_numpy(dtype=np.float64)


def get_hostname(value):
    parsed = parse_url(value)
    return parsed["hostname"] if parsed else None


def main():
    started = time.time()

    print("Loading URL and label columns...")

    df = pd.read_csv(DATASET, usecols=["URL", "label"])
    df = df.dropna(subset=["URL", "label"]).copy()
    df["URL"] = df["URL"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    if set(df["label"].unique()) - {0, 1}:
        raise ValueError("Unexpected labels; expected 0=phishing, 1=legitimate.")

    before = len(df)
    df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)
    df["hostname"] = df["URL"].map(get_hostname)

    invalid = int(df["hostname"].isna().sum())
    df = df.dropna(subset=["hostname"]).reset_index(drop=True)

    print("Rows before deduplication:", before)
    print("Rows after deduplication:", len(df))
    print("Rows without valid HTTP(S) hostname removed:", invalid)
    print("Labels:", df["label"].value_counts().sort_index().to_dict())

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

    shared = set(train_df["hostname"]) & set(test_df["hostname"])

    if shared:
        raise RuntimeError("Hostname leakage detected.")

    print("\n===== SPLIT SUMMARY =====")
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("Training hostnames:", train_df["hostname"].nunique())
    print("Testing hostnames:", test_df["hostname"].nunique())
    print("Shared hostnames:", len(shared))

    # Both feature branches receive the raw URL string.
    features = ColumnTransformer(
        transformers=[
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=300_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
                0,
            ),
            (
                "structure",
                Pipeline([
                    ("extract", URLFeatureExtractor()),
                    ("scale", StandardScaler()),
                ]),
                0,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # ColumnTransformer expects a 2D input. The input is a one-column
    # DataFrame so both branches can access the URL string.
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

    X_train = train_df[["URL"]]
    X_test = test_df[["URL"]]
    y_train = train_df["label"]
    y_test = test_df["label"]

    print("\nTraining hostname-aware candidate...")
    candidate.fit(X_train, y_train)

    print("Evaluating on unseen hostnames...")
    predictions = candidate.predict(X_test)

    accuracy = float(accuracy_score(y_test, predictions))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    print("\n===== HOSTNAME-AWARE RESULTS =====")
    print("Accuracy:", round(accuracy, 6))
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["PHISHING", "LEGITIMATE"],
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, MODEL_OUT)

    report = {
        "model_type": "hostname_aware_char_tfidf_logistic_regression",
        "production_model_modified": False,
        "dataset": str(DATASET),
        "split_method": "GroupShuffleSplit by hostname",
        "random_state": 42,
        "test_size": 0.20,
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
        "training_hostnames": int(train_df["hostname"].nunique()),
        "testing_hostnames": int(test_df["hostname"].nunique()),
        "shared_hostnames": int(len(shared)),
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
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["PHISHING", "LEGITIMATE"],
            zero_division=0,
            output_dict=True,
        ),
        "limitations": [
            "A single hostname-grouped split does not establish real-world performance.",
            "The dataset may contain collection and labelling biases.",
            "Hostname and keyword features are heuristics, not proof of maliciousness.",
            "Scores are not calibrated probabilities.",
            "This is an experimental candidate, not a production replacement.",
        ],
        "elapsed_seconds": round(time.time() - started, 2),
        "candidate_model_path": str(MODEL_OUT),
    }

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n===== SAVED ARTIFACTS =====")
    print("Candidate model:", MODEL_OUT)
    print("Evaluation report:", REPORT_OUT)
    print("Production model was not modified.")
    print("Elapsed seconds:", report["elapsed_seconds"])


if __name__ == "__main__":
    main()
