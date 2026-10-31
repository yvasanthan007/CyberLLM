from pathlib import Path
import json

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "phishing_url_model.joblib"

OUT_PATH = (
    ROOT / "artifacts" / "reports" / "phishing"
    / "feature_contribution_audit.json"
)

TEST_URLS = [
    ("Wikipedia homepage", "https://en.wikipedia.org/"),
    ("Wikipedia article", "https://en.wikipedia.org/wiki/Phishing"),
    ("Microsoft login", "https://login.microsoft.com/"),
    ("Reserved example domain", "https://example.com/"),
    (
        "YouTube watch URL",
        "https://www.youtube.com/watch?v=example",
    ),
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

print("Loading production model...")
model = joblib.load(MODEL_PATH)

if not hasattr(model, "named_steps"):
    raise TypeError("Expected a scikit-learn Pipeline.")

vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

if not hasattr(classifier, "coef_"):
    raise TypeError("Expected a linear classifier with coef_.")

feature_names = vectorizer.get_feature_names_out()
coefficients = classifier.coef_[0]

classes = list(classifier.classes_)

if classes != [0, 1]:
    raise ValueError(
        f"Unexpected class order {classes}; inspect before proceeding."
    )

print("Class 0 = phishing")
print("Class 1 = legitimate")
print("Positive coefficients favor class 1.")
print("Negative coefficients favor class 0.")
print()

report = {
    "model_path": str(MODEL_PATH),
    "classes": [int(value) for value in classes],
    "interpretation": {
        "negative_coefficient": "Favors phishing class 0",
        "positive_coefficient": "Favors legitimate class 1",
        "warning": (
            "Individual feature contributions are model associations, "
            "not proof that a URL is malicious or safe."
        ),
    },
    "urls": [],
}

for name, url in TEST_URLS:
    transformed = vectorizer.transform([url])
    indices = transformed.indices
    values = transformed.data

    contributions = []

    for index, value in zip(indices, values):
        contribution = float(value * coefficients[index])

        contributions.append({
            "ngram": str(feature_names[index]),
            "tfidf_value": float(value),
            "contribution_to_class_1": contribution,
        })

    contributions.sort(
        key=lambda item: abs(item["contribution_to_class_1"]),
        reverse=True,
    )

    probabilities = model.predict_proba([url])[0]
    phishing_score = float(probabilities[classes.index(0)])
    prediction = int(model.predict([url])[0])

    top_features = contributions[:15]

    item = {
        "name": name,
        "url": url,
        "predicted_class": prediction,
        "predicted_label": (
            "PHISHING" if prediction == 0 else "LEGITIMATE"
        ),
        "phishing_score": phishing_score,
        "top_15_features_by_absolute_contribution": top_features,
    }

    report["urls"].append(item)

    print("=" * 75)
    print(name)
    print("URL:", url)
    print("Prediction:", item["predicted_label"])
    print(f"Phishing score: {phishing_score:.6f}")
    print("\nTop character n-gram contributions:")
    print("Negative contribution = favors phishing")
    print("Positive contribution = favors legitimate")

    for feature in top_features:
        print(
            f"{feature['ngram']!r:20} "
            f"{feature['contribution_to_class_1']:+.6f}"
        )

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\nFeature contribution report saved to:")
print(OUT_PATH)
print("\nProduction model and API settings remain unchanged.")

