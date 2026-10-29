
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "phishing_url_model.joblib"

model = joblib.load(MODEL_PATH)
classes = list(model.named_steps["classifier"].classes_)

cases = [
    # Expected legitimate examples
    ("legitimate", "https://www.youtube.com/"),
    ("legitimate", "https://www.youtube.com/watch?v=HdI_aYzgJIg"),
    ("legitimate", "https://www.youtube.com/results?search_query=music"),
    ("legitimate", "https://www.google.com/search?q=weather"),
    ("legitimate", "https://en.wikipedia.org/wiki/Phishing"),

    # Suspicious-looking examples; these are not live-site checks
    ("suspicious-pattern", "http://paypal.com.security-check.example/"),
    ("suspicious-pattern", "http://login-example.invalid/verify?account=123"),
]

print("Evaluation only: predictions are not verified security verdicts.\n")

for expected, url in cases:
    probs = model.predict_proba([url])[0]
    score = float(probs[classes.index(0)])
    pred = int(model.predict([url])[0])

    print(f"Expected category: {expected}")
    print(f"URL: {url}")
    print(f"Model prediction: {'PHISHING' if pred == 0 else 'LEGITIMATE'}")
    print(f"Phishing score: {score:.1%}\n")
