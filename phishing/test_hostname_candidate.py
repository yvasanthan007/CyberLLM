from pathlib import Path
import sys
import joblib
import pandas as pd

# Make D:\CyberLLM importable when this script runs from the phishing folder.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phishing.url_feature_extractor import URLFeatureExtractor

MODEL_PATH = ROOT / "models" / "phishing_url_hostname_candidate.joblib"

print("Loading candidate model...")
model = joblib.load(MODEL_PATH)
print("Candidate model loaded successfully.")

tests = [
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

print("\n===== HOSTNAME-AWARE CANDIDATE TEST =====")
print("Expected labels: 0 = PHISHING, 1 = LEGITIMATE\n")

correct = 0
classifier = model.named_steps["classifier"]
classes = list(classifier.classes_)

for name, url, expected in tests:
    X = pd.DataFrame({"URL": [url]})

    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    phishing_score = float(probabilities[classes.index(0)])

    verdict = "PHISHING" if prediction == 0 else "LEGITIMATE"
    expected_name = "PHISHING" if expected == 0 else "LEGITIMATE"
    result = "PASS" if prediction == expected else "MISMATCH"

    correct += int(prediction == expected)

    print(f"{name}: {result}")
    print(f"  URL: {url}")
    print(f"  Predicted: {verdict}")
    print(f"  Expected:  {expected_name}")
    print(f"  Phishing score (uncalibrated): {phishing_score:.6f}\n")

print(f"Diagnostic matches: {correct}/{len(tests)}")
print("These hand-picked tests are not a real-world accuracy estimate.")
