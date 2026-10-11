# features.py — CyberLLM phishing link analysis
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Step 1: Define URL feature extraction (simple lexical features)
def extract_url_features(url: str) -> list[float]:
    features = []
    features.append(len(url))  # URL length
    features.append(url.count('.'))  # subdomain depth
    features.append(url.count('@'))  # presence of '@'
    features.append(url.count('-'))  # suspicious characters
    features.append(url.startswith('https'))  # HTTPS flag
    features.append(sum(c.isdigit() for c in url))  # number of digits
    features.append(any(word in url.lower() for word in ['login', 'verify', 'update', 'secure']))  # phishing keywords
    features.append(url.count('/'))  # number of path segments
    return features


# Step 2: Example dataset (replace with your real data)
data = {
    "url": [
        "https://secure-login.com",
        "http://malicious-site.ru",
        "https://paypal.com",
        "http://phishingsite.xyz",
        "https://google.com"
    ],
    "label": [0, 1, 0, 1, 0]  # 0 = safe, 1 = malicious
}
# Step 2: Load dataset from CSV file
df = pd.read_csv("phishing_dataset.csv")


df = pd.DataFrame(data)

# Step 3: Convert URLs to feature vectors
X = df["url"].apply(lambda u: extract_url_features(u)).tolist()
X = pd.DataFrame(X)
y = df["label"]

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Predict new URL
new_url = "http://update-login-secure.ru"
new_features = pd.DataFrame([extract_url_features(new_url)])
prediction = model.predict(new_features)[0]

if prediction == 1:
    print(f"⚠️ The URL '{new_url}' is potentially malicious.")
else:
    print(f"✅ The URL '{new_url}' appears safe.")
