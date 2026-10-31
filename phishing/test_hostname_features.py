from pathlib import Path
import sys

# Add the CyberLLM project root to Python's import path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phishing.url_feature_extractor import (
    parse_url,
    hostname_features,
)

urls = [
    "https://www.youtube.com/",
    "https://www.youtube.com/watch?v=example",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://www.microsoft.com/",
    "https://login.microsoft.com/",
    "https://example.com/",
    "http://paypal.com.security-check.example/verify",
    "http://microsoft.com.security-check.example/login",
]

for url in urls:
    parsed = parse_url(url)
    features = hostname_features(url)

    print("\n" + "=" * 75)
    print("URL:", url)
    print("Parsed hostname:", parsed["hostname"] if parsed else None)
    print("Feature values:")

    for name, value in features.items():
        print(f"  {name}: {value}")
