from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phishing.url_feature_extractor import hostname_features

tests = {
    "Legitimate YouTube": "https://www.youtube.com/",
    "Legitimate Microsoft login": "https://login.microsoft.com/",
    "PayPal hostname lookalike": "http://paypal.com.security-check.example/verify",
    "Microsoft hostname lookalike": "http://microsoft.com.security-check.example/login",
}

for name, url in tests.items():
    f = hostname_features(url)
    print(f"\n{name}")
    print("  hostname_has_brand_word:",
          f["hostname_has_brand_word"])
    print("  hostname_has_brand_as_subdomain:",
          f["hostname_has_brand_as_subdomain"])
    print("  hostname_has_suspicious_brand_domain_pattern:",
          f["hostname_has_suspicious_brand_domain_pattern"])
