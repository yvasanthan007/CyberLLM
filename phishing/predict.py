
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "phishing_url_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run python phishing/train.py first."
    )

url = " ".join(sys.argv[1:]).strip()
if not url:
    url = input("Enter a URL to check: ").strip()

if not url:
    raise ValueError("A URL is required.")

model = joblib.load(MODEL_PATH)

prediction = int(model.predict([url])[0])
probabilities = model.predict_proba([url])[0]
classes = list(model.named_steps["classifier"].classes_)

phishing_probability = float(probabilities[classes.index(0)])

print("\nURL:", url)
print("Prediction:", "PHISHING" if prediction == 0 else "LEGITIMATE")
print(f"Model phishing score: {phishing_probability:.1%}")
print(
    "Note: this is a model score, not a calibrated probability "
    "or a guarantee of safety."
)
