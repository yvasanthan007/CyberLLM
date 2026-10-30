"""Live API smoke test: hits every route on a running CyberLLM server.

Usage: python scripts/test_api_live.py  (server must be on 127.0.0.1:8765)
Reads CYBERLLM_API_KEY from .env; never prints it.
"""
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
BASE = os.getenv("CYBERLLM_BASE_URL", "http://127.0.0.1:8765")
KEY = os.getenv("CYBERLLM_API_KEY", "")
HEADERS = {"X-API-Key": KEY} if KEY else {}

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


with httpx.Client(base_url=BASE, timeout=90) as c:
    h = c.get("/health").json()
    check("GET /health", h.get("status") == "ok", str(h))

    r = c.post("/chat", json={"message": "hi"})
    check("POST /chat without key -> 401", r.status_code == 401, str(r.status_code))

    r = c.post("/chat", headers=HEADERS, json={"message": "What is SQL injection?"})
    body = r.json()
    check("POST /chat LLM/RAG", r.status_code == 200 and body.get("llm_generated") is True,
          f"source={body.get('source')}")

    r = c.post("/phishing/predict", headers=HEADERS,
               json={"url": "http://secure-login-verify-bank.example.top/a"})
    p = r.json()
    check("POST /phishing/predict", p.get("prediction") == "PHISHING", str(p.get("prediction")))

    r = c.post("/knowledge/search", headers=HEADERS, json={"query": "ransomware", "top_k": 3})
    check("POST /knowledge/search", r.status_code == 200 and r.json()["count"] > 0)

    r = c.get("/capabilities")
    check("GET /capabilities", r.status_code == 200)

    r = c.get("/models", headers=HEADERS)
    check("GET /models", r.status_code == 200 and len(r.json()["models"]) >= 2)

    r = c.post("/chat", headers=HEADERS, json={"message": ""})
    check("POST /chat empty -> 422", r.status_code == 422, str(r.status_code))

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("All live API checks passed.")
