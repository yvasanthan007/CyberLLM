from pathlib import Path
import json

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "phishing_url_model.joblib"
OUT_PATH = (
    ROOT / "artifacts" / "reports" / "phishing"
    / "threshold_diagnostic.json"
)

TESTS = [
    ("Google homepage", "https://www.google.com/", 1),
    ("YouTube homepage", "https://www.youtube.com/", 1),
    ("YouTube watch URL", "https://www.youtube.com/watch?v=example", 1),
    ("Wikipedia homepage", "https://en.wikipedia.org/", 1),
    ("Wikipedia article", "https://en.wikipedia.org/wiki/Phishing", 1),
    ("Microsoft homepage", "https://www.microsoft.com/", 1),
    ("Microsoft login", "https://login.microsoft.com/", 1),
    ("Office homepage", "https://www.office.com/", 1),
    ("Example reserved domain", "https://example.com/", 1),
    ("Example login path", "https://example.com/login", 1),
    ("PayPal lookalike", "http://paypal.com.security-check.example/", 0),
    ("Suspicious test URL", "http://login-example.invalid/verify?account=123", 0),
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

model = joblib.load(MODEL_PATH)

urls = [url for _, url, _ in TESTS]
expected = [label for _, _, label in TESTS]

scores = model.predict_proba(urls)
classes = list(model.classes_)

if 0 not in classes:
    raise ValueError(f"Phishing class 0 not found. Classes: {classes}")

phishing_index = classes.index(0)
phishing_scores = scores[:, phishing_index]

print("PRODUCTION MODEL THRESHOLD EXPERIMENT")
print("Class 0 = phishing; class 1 = legitimate")
print("Scores are not calibrated probabilities.")
print("No model or API settings will be changed.\n")

rows = []

for (name, url, label), score in zip(TESTS, phishing_scores):
    rows.append({
        "name": name,
        "url": url,
        "expected_label": label,
        "phishing_score": float(score),
    })

print("Individual scores:")
for row in rows:
    print(
        f"{row['name']}: "
        f"score={row['phishing_score']:.6f}, "
        f"expected={'PHISHING' if row['expected_label'] == 0 else 'LEGITIMATE'}"
    )

summaries = []

for threshold in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
    predictions = [
        0 if score >= threshold else 1
        for score in phishing_scores
    ]

    false_positives = sum(
        expected[i] == 1 and predictions[i] == 0
        for i in range(len(TESTS))
    )

    false_negatives = sum(
        expected[i] == 0 and predictions[i] == 1
        for i in range(len(TESTS))
    )

    correct = sum(
        predictions[i] == expected[i]
        for i in range(len(TESTS))
    )

    summaries.append({
        "threshold": threshold,
        "correct": correct,
        "diagnostic_cases": len(TESTS),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "diagnostic_match_rate": correct / len(TESTS),
    })

print("\nTHRESHOLD COMPARISON")
print("Threshold | Correct | False positives | False negatives")

for item in summaries:
    print(
        f"{item['threshold']:9.2f} | "
        f"{item['correct']:7d} | "
        f"{item['false_positives']:15d} | "
        f"{item['false_negatives']:15d}"
    )

report = {
    "model_path": str(MODEL_PATH),
    "class_mapping": {"0": "PHISHING", "1": "LEGITIMATE"},
    "score_is_calibrated_probability": False,
    "thresholds": summaries,
    "cases": rows,
    "limitations": [
        "Only 12 hand-picked diagnostic URLs were evaluated.",
        "Synthetic suspicious URLs are not live threat intelligence.",
        "These results are not a statistically representative benchmark.",
        "No model, API threshold, or production setting was changed.",
    ],
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(f"\nReport saved to: {OUT_PATH}")
print("Production model remains unchanged.")
