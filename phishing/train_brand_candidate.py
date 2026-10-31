from pathlib import Path
import hashlib
import json
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

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

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phishing.url_feature_extractor import URLFeatureExtractor, parse_url


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

DATA_PATH = ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"

PRODUCTION_PATH = ROOT / "models" / "phishing_url_model.joblib"

PREVIOUS_CANDIDATE_PATH = (
    ROOT / "models" / "phishing_url_improved_candidate.joblib"
)

CANDIDATE_PATH = ROOT / "models" / "phishing_url_brand_candidate.joblib"

REPORT_PATH = (
    ROOT
    / "artifacts"
    / "reports"
    / "phishing"
    / "brand_candidate_evaluation.json"
)


# ---------------------------------------------------------
# LABEL MAPPING
# ---------------------------------------------------------
# Dataset/model label mapping:
# 0 = PHISHING
# 1 = LEGITIMATE
#
# Do not change this mapping without verifying the dataset
# and the production model's training code.


PHISHING_LABEL = 0
LEGITIMATE_LABEL = 1
DEFAULT_THRESHOLD = 0.50


# ---------------------------------------------------------
# TARGETED DIAGNOSTIC URLS
# ---------------------------------------------------------
# These are manually selected diagnostic examples, not a
# representative real-world test dataset.
#
# Expected labels:
# 0 = phishing
# 1 = legitimate


TARGETED_TESTS = [
    ("Google homepage", "https://www.google.com/", 1),
    ("YouTube homepage", "https://www.youtube.com/", 1),
    ("YouTube watch URL", "https://www.youtube.com/watch?v=example", 1),
    ("Wikipedia homepage", "https://en.wikipedia.org/", 1),
    ("Wikipedia article", "https://en.wikipedia.org/wiki/Phishing", 1),
    ("Microsoft homepage", "https://www.microsoft.com/", 1),
    ("Microsoft login", "https://login.microsoft.com/", 1),
    ("Office homepage", "https://www.office.com/", 1),
    ("Reserved example domain", "https://example.com/", 1),
    ("Reserved example login path", "https://example.com/login", 1),
    ("GitHub homepage", "https://github.com/", 1),
    (
        "Wikipedia security article",
        "https://en.wikipedia.org/wiki/Computer_security",
        1,
    ),
    (
        "PayPal lookalike",
        "http://paypal.com.security-check.example/verify",
        0,
    ),
    (
        "Microsoft lookalike",
        "http://microsoft.com.security-check.example/login",
        0,
    ),
    (
        "YouTube lookalike",
        "http://youtube.com.account-check.example/login",
        0,
    ),
    (
        "Wikipedia lookalike",
        "http://wikipedia.org.security-check.example/verify",
        0,
    ),
    (
        "Synthetic suspicious URL",
        "http://login-example.invalid/verify?account=123",
        0,
    ),
    (
        "Synthetic account URL",
        "http://secure-login-example.invalid/account",
        0,
    ),
]


# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------


def sha256_file(path):
    """Return a file's SHA-256 hash."""

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_phishing_score(model, input_data):
    """
    Return the model's score for class 0 (phishing).

    This score is not assumed to be a calibrated probability.
    """

    classes = list(model.classes_)

    if PHISHING_LABEL not in classes:
        raise ValueError(
            f"Model classes do not contain phishing label "
            f"{PHISHING_LABEL}: {classes}"
        )

    phishing_index = classes.index(PHISHING_LABEL)

    probabilities = model.predict_proba(input_data)

    return float(probabilities[0, phishing_index])


def score_to_label(score, threshold=DEFAULT_THRESHOLD):
    """
    Convert a phishing score into a binary model label.

    This uses the explicitly configured experimental threshold.
    """

    if score >= threshold:
        return PHISHING_LABEL

    return LEGITIMATE_LABEL


def model_feature_compatibility(model, input_data):
    """
    Test whether a previously saved model can process input_data.

    This catches old candidate models whose scaler expects a
    different number of numeric features.
    """

    try:
        model.predict_proba(input_data)
        return True, None

    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        return False, str(exc)


def calculate_binary_metrics(y_true, y_pred):
    """
    Calculate metrics with phishing treated as the positive class.

    y_true and y_pred use:
        0 = legitimate
        1 = phishing

    This internal convention makes the confusion matrix easier
    to read independently of the original dataset labels.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
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
        "confusion_matrix": {
            "true_legitimate": int(tn),
            "false_phishing_alerts": int(fp),
            "missed_phishing": int(fn),
            "detected_phishing": int(tp),
        },
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------


def main():
    started = time.time()

    print("=" * 65)
    print("CyberLLM — Experimental Brand-Feature Candidate")
    print("=" * 65)

    # Verify required files exist.

    for path in (DATA_PATH, PRODUCTION_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file was not found: {path}"
            )

    # Record production model identity before training.

    production_hash_before = sha256_file(PRODUCTION_PATH)
    production_size_before = PRODUCTION_PATH.stat().st_size

    print("\nLoading dataset...")

    df = pd.read_csv(
        DATA_PATH,
        usecols=["URL", "label"],
    )

    df = df.dropna(
        subset=["URL", "label"]
    ).copy()

    df["URL"] = df["URL"].astype(str).str.strip()

    df = df[df["URL"] != ""]

    df["label"] = pd.to_numeric(
        df["label"],
        errors="raise",
    ).astype(int)

    observed_labels = set(df["label"].unique())

    if observed_labels != {
        PHISHING_LABEL,
        LEGITIMATE_LABEL,
    }:
        raise ValueError(
            "Unexpected label values. "
            f"Expected {{0, 1}}, got {observed_labels}"
        )

    # Remove duplicate URLs before splitting.

    df = df.drop_duplicates(
        subset=["URL"]
    ).reset_index(drop=True)

    # Parse hostnames for group-based splitting.

    parsed = df["URL"].map(parse_url)

    df["hostname"] = parsed.map(
        lambda item: (
            item["hostname"]
            if isinstance(item, dict)
            else ""
        )
    )

    df = df[
        df["hostname"].astype(str).str.len() > 0
    ].reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            "No valid URLs with hostnames were found."
        )

    print(f"Usable unique URLs: {len(df):,}")
    print("Label mapping: 0=phishing, 1=legitimate")

    print("\nDataset label counts:")

    print(
        df["label"]
        .value_counts()
        .sort_index()
        .rename(index={
            0: "PHISHING (0)",
            1: "LEGITIMATE (1)",
        })
        .to_string()
    )

    # -----------------------------------------------------
    # HOSTNAME-GROUPED SPLIT
    # -----------------------------------------------------

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
        set(train_df["hostname"])
        & set(test_df["hostname"])
    )

    if shared_hosts:
        raise RuntimeError(
            "Hostname leakage detected: "
            f"{len(shared_hosts)} hostnames are shared."
        )

    print(f"\nTraining rows: {len(train_df):,}")
    print(f"Testing rows: {len(test_df):,}")
    print(f"Training hostnames: {train_df['hostname'].nunique():,}")
    print(f"Testing hostnames: {test_df['hostname'].nunique():,}")
    print(f"Shared hostnames: {len(shared_hosts)}")

    # -----------------------------------------------------
    # FEATURE PIPELINE
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    print(
        "\nTraining experimental candidate with "
        "the current URL feature extractor..."
    )

    candidate.fit(
        train_df[["URL"]],
        train_df["label"],
    )

    classes = list(candidate.classes_)

    if PHISHING_LABEL not in classes:
        raise RuntimeError(
            f"Trained model has unexpected classes: {classes}"
        )

    phishing_index = classes.index(PHISHING_LABEL)

    # -----------------------------------------------------
    # GROUPED HOLDOUT EVALUATION
    # -----------------------------------------------------

    print("\nEvaluating grouped holdout...")

    predictions = candidate.predict(
        test_df[["URL"]]
    )

    probabilities = candidate.predict_proba(
        test_df[["URL"]]
    )

    scores = probabilities[:, phishing_index]

    # Original labels:
    #   0 = phishing
    #   1 = legitimate
    #
    # Evaluation convention:
    #   1 = phishing (positive)
    #   0 = legitimate (negative)

    y_true = (
        test_df["label"].to_numpy() == PHISHING_LABEL
    ).astype(int)

    y_pred = (
        predictions == PHISHING_LABEL
    ).astype(int)

    metrics = calculate_binary_metrics(
        y_true,
        y_pred,
    )

    metrics.update({
        "test_rows": int(len(test_df)),
        "train_hostnames": int(
            train_df["hostname"].nunique()
        ),
        "test_hostnames": int(
            test_df["hostname"].nunique()
        ),
        "shared_hostnames": int(len(shared_hosts)),
        "threshold": DEFAULT_THRESHOLD,
        "score_is_calibrated_probability": False,
    })

    print("\nGROUPED HOLDOUT METRICS")

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    # -----------------------------------------------------
    # LOAD PRODUCTION AND OPTIONAL OLD CANDIDATE
    # -----------------------------------------------------

    print("\nLoading production model...")

    production = joblib.load(
        PRODUCTION_PATH
    )

    previous_candidate = None
    previous_candidate_status = {
        "available": PREVIOUS_CANDIDATE_PATH.exists(),
        "compatible": False,
        "reason": None,
    }

    if PREVIOUS_CANDIDATE_PATH.exists():
        try:
            print(
                "Checking compatibility of previous candidate..."
            )

            previous_candidate = joblib.load(
                PREVIOUS_CANDIDATE_PATH
            )

            compatible, reason = model_feature_compatibility(
                previous_candidate,
                pd.DataFrame({
                    "URL": [
                        "https://www.google.com/"
                    ]
                }),
            )

            previous_candidate_status["compatible"] = compatible
            previous_candidate_status["reason"] = reason

            if compatible:
                print(
                    "Previous candidate is compatible; "
                    "it will be included in diagnostics."
                )
            else:
                print(
                    "Previous candidate is incompatible and "
                    "will be skipped:"
                )
                print(reason)
                previous_candidate = None

        except Exception as exc:
            previous_candidate_status["compatible"] = False
            previous_candidate_status["reason"] = str(exc)
            previous_candidate = None

            print(
                "Could not load the previous candidate. "
                "Continuing without it."
            )
            print(str(exc))

    # -----------------------------------------------------
    # TARGETED DIAGNOSTICS
    # -----------------------------------------------------

    diagnostics = []

    print("\nTARGETED DIAGNOSTICS")
    print("-" * 65)

    for name, url, expected in TARGETED_TESTS:
        row = pd.DataFrame({
            "URL": [url]
        })

        # New candidate.

        candidate_score = get_phishing_score(
            candidate,
            row,
        )

        candidate_label = score_to_label(
            candidate_score
        )

        # Production model uses its existing prediction method.

        production_label = int(
            production.predict([url])[0]
        )

        production_score = get_phishing_score(
            production,
            [url],
        )

        item = {
            "name": name,
            "url": url,
            "expected_label": int(expected),

            "candidate_label": int(candidate_label),
            "candidate_phishing_score": candidate_score,
            "candidate_correct": (
                candidate_label == expected
            ),

            "production_label": production_label,
            "production_phishing_score": production_score,
            "production_correct": (
                production_label == expected
            ),
        }

        # Previous candidate is optional.

        if previous_candidate is not None:
            try:
                previous_score = get_phishing_score(
                    previous_candidate,
                    row,
                )

                previous_label = score_to_label(
                    previous_score
                )

                item.update({
                    "previous_candidate_label": int(
                        previous_label
                    ),
                    "previous_candidate_phishing_score": (
                        previous_score
                    ),
                    "previous_candidate_correct": (
                        previous_label == expected
                    ),
                })

            except Exception as exc:
                item[
                    "previous_candidate_comparison_skipped"
                ] = str(exc)

        diagnostics.append(item)

        expected_text = (
            "LEGITIMATE"
            if expected == LEGITIMATE_LABEL
            else "PHISHING"
        )

        candidate_text = (
            "PHISHING"
            if candidate_label == PHISHING_LABEL
            else "LEGITIMATE"
        )

        production_text = (
            "PHISHING"
            if production_label == PHISHING_LABEL
            else "LEGITIMATE"
        )

        print(
            f"{name}: expected={expected_text}, "
            f"candidate={candidate_text} "
            f"(score={candidate_score:.4f}), "
            f"production={production_text} "
            f"(score={production_score:.4f})"
        )

    # -----------------------------------------------------
    # TARGETED SUMMARY
    # -----------------------------------------------------

    case_count = len(diagnostics)

    candidate_correct = sum(
        item["candidate_correct"]
        for item in diagnostics
    )

    production_correct = sum(
        item["production_correct"]
        for item in diagnostics
    )

    targeted_summary = {
        "cases": case_count,

        "candidate_correct": int(candidate_correct),
        "candidate_accuracy": float(
            candidate_correct / case_count
        ),

        "production_correct": int(production_correct),
        "production_accuracy": float(
            production_correct / case_count
        ),

        "previous_candidate_status": (
            previous_candidate_status
        ),

        "results": diagnostics,
    }

    if previous_candidate is not None:
        previous_correct = sum(
            bool(
                item.get(
                    "previous_candidate_correct",
                    False,
                )
            )
            for item in diagnostics
        )

        targeted_summary[
            "previous_candidate_correct"
        ] = int(previous_correct)

        targeted_summary[
            "previous_candidate_accuracy"
        ] = float(previous_correct / case_count)

    print("\nTARGETED SUMMARY")

    print(
        json.dumps(
            {
                key: value
                for key, value in targeted_summary.items()
                if key != "results"
            },
            indent=2,
        )
    )

    # -----------------------------------------------------
    # SAVE NEW CANDIDATE ONLY
    # -----------------------------------------------------

    CANDIDATE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nSaving experimental candidate...")

    # This is a new filename, not the production model path.
    joblib.dump(
        candidate,
        CANDIDATE_PATH,
    )

    # Verify the production artifact has not changed.

    production_hash_after = sha256_file(
        PRODUCTION_PATH
    )

    production_size_after = (
        PRODUCTION_PATH.stat().st_size
    )

    production_unchanged = (
        production_hash_before == production_hash_after
        and production_size_before == production_size_after
    )

    if not production_unchanged:
        raise RuntimeError(
            "Production model file changed during this script. "
            "Stop and investigate before deploying anything."
        )

    # -----------------------------------------------------
    # REPORT
    # -----------------------------------------------------

    report = {
        "candidate_path": str(CANDIDATE_PATH),
        "production_path": str(PRODUCTION_PATH),

        "label_mapping": {
            "0": "PHISHING",
            "1": "LEGITIMATE",
        },

        "threshold": DEFAULT_THRESHOLD,
        "score_is_calibrated_probability": False,

        "candidate_sha256": sha256_file(
            CANDIDATE_PATH
        ),

        "production_model": {
            "sha256_before": production_hash_before,
            "sha256_after": production_hash_after,
            "size_bytes_before": production_size_before,
            "size_bytes_after": production_size_after,
            "unchanged": production_unchanged,
        },

        "grouped_holdout_metrics": metrics,

        "targeted_diagnostic": targeted_summary,

        "limitations": [
            "The dataset may contain collection and label biases.",
            "The targeted diagnostic set is small and hand-selected.",
            "Synthetic phishing examples do not represent all real attacks.",
            "Hostname-grouped holdout performance does not guarantee real-world performance.",
            "The score is not a calibrated probability.",
            "The brand-feature extractor is a heuristic and is not a full public-suffix-list implementation.",
            "The candidate has not been promoted to production.",
        ],

        "elapsed_seconds": round(
            time.time() - started,
            2,
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # FINAL STATUS
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("EXPERIMENT FINISHED")
    print("=" * 65)

    print("Candidate saved:")
    print(CANDIDATE_PATH)

    print("\nEvaluation report saved:")
    print(REPORT_PATH)

    print("\nProduction model unchanged:", production_unchanged)

    print(
        "Candidate targeted accuracy: "
        f"{candidate_correct}/{case_count} "
        f"({candidate_correct / case_count:.1%})"
    )

    print(
        "Production targeted accuracy: "
        f"{production_correct}/{case_count} "
        f"({production_correct / case_count:.1%})"
    )

    print(
        "\nIMPORTANT: The candidate is experimental. "
        "It has not been promoted to production."
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()