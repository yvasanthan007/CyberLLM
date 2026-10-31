from pathlib import Path
import joblib

model_path = Path("models/phishing_url_model.joblib")
model = joblib.load(model_path)

print("Model type:", type(model).__name__)
print("Model steps:", getattr(model, "named_steps", {}).keys())

if not hasattr(model, "named_steps"):
    raise SystemExit(
        "Unexpected model structure. Do not modify the production model."
    )

tfidf = model.named_steps.get("tfidf")
classifier = model.named_steps.get("classifier")

if tfidf is None or classifier is None:
    raise SystemExit(
        "Expected tfidf and classifier steps were not found."
    )

features = tfidf.get_feature_names_out()
coefficients = classifier.coef_[0]

print("\nClassifier classes:", classifier.classes_)
print("Feature count:", len(features))
print("Coefficient count:", len(coefficients))

if len(features) != len(coefficients):
    raise SystemExit("Feature/coefficient count mismatch.")

# For sklearn LogisticRegression, coef_[0] corresponds to
# classes_[1] in the binary classification case.
# Your label mapping is 0=PHISHING, 1=LEGITIMATE.
print("\nTop features pushing toward LEGITIMATE:")
for index in coefficients.argsort()[-25:][::-1]:
    print(f"{features[index]!r}: {coefficients[index]:.6f}")

print("\nTop features pushing toward PHISHING:")
for index in coefficients.argsort()[:25]:
    print(f"{features[index]!r}: {coefficients[index]:.6f}")
