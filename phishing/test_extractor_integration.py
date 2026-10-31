from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phishing.url_feature_extractor import URLFeatureExtractor

urls = [
    "https://www.youtube.com/",
    "https://login.microsoft.com/",
    "http://paypal.com.security-check.example/verify",
    "http://microsoft.com.security-check.example/login",
]

X = pd.DataFrame({"URL": urls})
extractor = URLFeatureExtractor()
features = extractor.fit_transform(X)

print("Input rows:", len(X))
print("Output shape:", features.shape)
print("All features finite:", bool(np.isfinite(features).all()))
print("Feature count:", features.shape[1])

if len(features) != len(urls):
    raise SystemExit("FAIL: incorrect number of output rows")

if not np.isfinite(features).all():
    raise SystemExit("FAIL: non-finite feature values")

print("PASS: extractor integration test")
