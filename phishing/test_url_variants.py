
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "phishing_url_model.joblib"

print("Starting URL variant diagnostic...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
classes = list(model.named_steps["classifier"].classes_)

urls = [
    "https://www.youtube.com/",
    "https://www.youtube.com/watch",
    "https://www.youtube.com/watch?v=HdI_aYzgJIg",
    "https://www.youtube.com/watch?v=example",
    "https://www.youtube.com/?v=HdI_aYzgJIg",
]

for url in urls:
    probabilities = model.predict_proba([url])[0]
    phishing_score = float(probabilities[classes.index(0)])
    prediction = int(model.predict([url])[0])

    print(f"\nURL: {url}")
    print(f"Prediction: {'PHISHING' if prediction == 0 else 'LEGITIMATE'}")
    print(f"Phishing score: {phishing_score:.1%}")

print("\nDiagnostic complete.")
