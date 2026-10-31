from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "phishing_diagnostic_urls.csv"
MODEL_PATH = ROOT / "models" / "phishing_url_model.joblib"
REPORT_PATH = (
    ROOT / "artifacts" / "reports" / "phishing"
    / "false_positive_feature_audit.json"
)

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

X = df["URL"].astype(str)
y_true = df["label"].astype(int)
y_pred = model.predict(X).astype(int)

probabilities = model.predict_proba(X)
classes = list(model.classes_)
phishing_index = classes.index(0)
phishing_scores = probabilities[:, phishing_index]

results = df.copy()
results["predicted_label"] = y_pred
results["phishing_score"] = phishing_scores

false_positives = results[
    (y_true == 1) & (results["predicted_label"] == 0)
].copy()

print("=== FALSE-POSITIVE REVIEW ===")
print(f"False positives: {len(false_positives)}")

report = {
    "model_path": str(MODEL_PATH),
    "false_positive_count": int(len(false_positives)),
    "cases": [],
    "warning": (
        "Feature contributions are model associations, not proof "
        "that a URL is malicious. Scores are not calibrated probabilities."
    ),
}

vectorizer = model.named_steps.get("tfidf")
classifier = model.named_steps.get("classifier")

if vectorizer is None or classifier is None:
    raise TypeError(
        "Expected pipeline steps 'tfidf' and 'classifier' were not found."
    )

feature_names = vectorizer.get_feature_names_out()
weights = classifier.coef_[0]

for row in false_positives.itertuples(index=False):
    url = str(row.URL)
    score = float(row.phishing_score)

    print(f"\nURL: {url}")
    print(f"Phishing score: {score:.6f}")

    transformed = vectorizer.transform([url])

    # Sparse-matrix indices are column indices; use them to select
    # the corresponding model weights and feature names.
    active_indices = transformed.indices
    feature_values = transformed.data

    contributions = [
        (
            str(feature_names[int(index)]),
            float(value) * float(weights[int(index)]),
        )
        for index, value in zip(active_indices, feature_values)
    ]

    contributions.sort(
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    top_features = []
    for feature, contribution in contributions[:12]:
        direction = (
            "toward phishing"
            if contribution > 0
            else "toward legitimate"
            if contribution < 0
            else "neutral"
        )

        top_features.append({
            "feature": feature,
            "contribution": contribution,
            "direction": direction,
        })

        print(
            f"  {feature!r}: {contribution:+.6f} "
            f"({direction})"
        )

    report["cases"].append({
        "url": url,
        "expected_class": "legitimate",
        "predicted_class": "phishing",
        "phishing_score": score,
        "top_feature_contributions": top_features,
    })

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print(f"\nReport saved to: {REPORT_PATH}")
print("Production model was loaded for evaluation only; it was not modified.")
