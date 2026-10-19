
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "phishing_url_model.joblib"

model = joblib.load(MODEL_PATH)

urls = [
    "https://www.youtube.com/",
    "https://www.youtube.com/watch?v=HdI_aYzgJIg",
    "https://www.google.com/",
    "https://www.microsoft.com/",
    "https://www.wikipedia.org/",
    "http://paypal.com.security-check.example/",
]

classes = list(model.named_steps["classifier"].classes_)

print("Diagnostic predictions (not a security verdict):\n")

for url in urls:
    probabilities = model.predict_proba([url])[0]
    prediction = int(model.predict([url])[0])
    phishing_score = float(probabilities[classes.index(0)])

    print(f"URL: {url}")
    print(f"Predicted class: {prediction} "
          f"({'PHISHING' if prediction == 0 else 'LEGITIMATE'})")
    print(f"Phishing score: {phishing_score:.1%}\n")
