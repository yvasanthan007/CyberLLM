from pathlib import Path
import sys
import json

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Register the custom feature extractor before loading the hostname candidate.
from phishing.url_feature_extractor import URLFeatureExtractor

MODELS = {
    "production": ROOT / "models" / "phishing_url_model.joblib",
    "character_candidate": ROOT / "models" / "phishing_url_candidate.joblib",
    "hostname_candidate": ROOT / "models" / "phishing_url_hostname_candidate.joblib",
}

OUT = (
    ROOT / "artifacts" / "reports" / "phishing"
    / "model_diagnostic_comparison.json"
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

results = []

for model_name, model_path in MODELS.items():
    print(f"\nLoading {model_name}: {model_path}")

    if not model_path.exists():
        print("  SKIPPED: model file not found")
        continue

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        print(f"  SKIPPED: model could not load: {type(exc).__name__}: {exc}")
        continue

    # Both candidate models use the pipeline input format shown below.
    # The production model is trained on raw URL strings.
    is_hostname = model_name == "hostname_candidate"

    correct = 0
    false_positives = 0
    false_negatives = 0
    rows = []

    for name, url, expected in TESTS:
        try:
            if is_hostname:
                X = pd.DataFrame({"URL": [url]})
            else:
                X = [url]

            prediction = int(model.predict(X)[0])
            classes = list(model.classes_) if hasattr(model, "classes_") else None

            if classes is None and hasattr(model, "named_steps"):
                classifier = model.named_steps.get("classifier")
                if classifier is None:
                    classifier = model.named_steps.get("model")
                if classifier is not None:
                    classes = list(classifier.classes_)

            score = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)[0]
                if classes is not None and 0 in classes:
                    score = float(probabilities[classes.index(0)])

            matched = prediction == expected
            correct += int(matched)

            if expected == 1 and prediction == 0:
                false_positives += 1
            if expected == 0 and prediction == 1:
                false_negatives += 1

            row = {
                "name": name,
                "url": url,
                "expected_label": expected,
                "predicted_label": prediction,
                "match": matched,
                "phishing_score_uncalibrated": score,
            }
            rows.append(row)

            verdict = "PHISHING" if prediction == 0 else "LEGITIMATE"
            print(f"  {name}: {verdict} | {'PASS' if matched else 'MISMATCH'}")

        except Exception as exc:
            rows.append({
                "name": name,
                "url": url,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"  {name}: ERROR - {type(exc).__name__}: {exc}")

    summary = {
        "model": model_name,
        "model_path": str(model_path),
        "diagnostic_cases": len(TESTS),
        "correct": correct,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "diagnostic_match_rate": correct / len(TESTS),
        "cases": rows,
    }

    results.append(summary)

    print(
        f"  Summary: {correct}/{len(TESTS)} matches; "
        f"{false_positives} false positives; "
        f"{false_negatives} false negatives"
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

print("\nComparison report saved to:")
print(OUT)
print("\nNo model or dataset was modified.")
