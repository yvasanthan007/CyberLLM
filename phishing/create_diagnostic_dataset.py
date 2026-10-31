from pathlib import Path
import pandas as pd

output_path = (
    Path.cwd()
    / "data"
    / "processed"
    / "phishing_diagnostic_urls.csv"
)

rows = [
    # Legitimate diagnostic examples
    ("https://www.google.com/", 1, "legitimate", "homepage"),
    ("https://www.youtube.com/", 1, "legitimate", "homepage"),
    ("https://www.wikipedia.org/", 1, "legitimate", "homepage"),
    ("https://en.wikipedia.org/wiki/Phishing", 1, "legitimate", "article"),
    ("https://www.microsoft.com/", 1, "legitimate", "homepage"),
    ("https://www.office.com/", 1, "legitimate", "homepage"),
    ("https://www.github.com/", 1, "legitimate", "homepage"),
    ("https://example.com/", 1, "legitimate", "reserved_example_domain"),
    ("https://example.com/login", 1, "legitimate", "reserved_example_domain"),
    ("https://www.youtube.com/watch?v=example", 1, "legitimate", "video_path"),
    ("https://login.microsoft.com/", 1, "legitimate", "login_host"),
    ("https://www.wikipedia.org/wiki/Computer_security", 1, "legitimate", "article"),

    # Suspicious diagnostic examples
    ("http://paypal.com.security-check.example/verify", 0, "phishing", "lookalike_host"),
    ("http://login-example.invalid/verify?account=123", 0, "phishing", "synthetic_invalid_domain"),
    ("http://microsoft.com.security-check.example/login", 0, "phishing", "lookalike_host"),
    ("http://youtube.com.account-check.example/login", 0, "phishing", "lookalike_host"),
    ("http://wikipedia.org.security-check.example/verify", 0, "phishing", "lookalike_host"),
    ("http://secure-login-example.invalid/account", 0, "phishing", "synthetic_invalid_domain"),
]

df = pd.DataFrame(
    rows,
    columns=["URL", "label", "expected_class", "test_category"],
)

output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"Created {len(df)} diagnostic URLs.")
print("\nLabel counts:")
print(df["expected_class"].value_counts().to_string())
print(f"\nSaved to: {output_path}")
print("\nNo existing dataset or model was modified.")
