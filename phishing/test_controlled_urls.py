from pathlib import Path
import json
import os

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("CYBERLLM_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "CYBERLLM_API_KEY was not found. "
        "Check the root .env file without displaying its contents."
    )

BASE_URL = "http://127.0.0.1:8765"

URLS = [
    "http://example.com/",
    "https://example.com/",
    "https://example.com/login",
    "https://login.example.com/",
    "https://example.com/verify",
    "http://www.wikipedia.org/",
    "https://www.wikipedia.org/",
    "https://www.youtube.com/",
    "https://www.youtube.com/watch?v=example",
]

headers = {"X-API-Key": API_KEY}

with httpx.Client(
    base_url=BASE_URL,
    headers=headers,
    timeout=30,
) as client:

    # Verify connectivity and authentication first.
    health = client.get("/health")
    print("Health status:", health.status_code)

    if health.is_error:
        print(health.text)
        raise SystemExit("CyberLLM health check failed.")

    for url in URLS:
        print("\n" + "=" * 75)
        print("URL:", url)

        try:
            response = client.post(
                "/phishing/predict",
                json={"url": url},
            )

            print("HTTP status:", response.status_code)

            if response.is_error:
                print("API error:", response.text)
                continue

            result = response.json()
            print(json.dumps(result, indent=2))

        except httpx.RequestError as exc:
            print("Request failed:", str(exc))
