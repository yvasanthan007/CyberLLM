from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit
import os
import secrets

import joblib
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from cyberllm.retrieval.lexical import load_documents, search_documents

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODEL_PATH = ROOT / "models" / "phishing_url_model.joblib"
KB_PATH = ROOT / "artifacts" / "vectorstore" / "cyberllm_kb.jsonl"

app = FastAPI(
    title="CyberLLM Security API",
    description="Defensive cybersecurity demo: phishing assessment and knowledge search",
    version="0.1.0",
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Depends(api_key_header)):
    expected_key = os.getenv("CYBERLLM_API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured",
        )

    if not api_key or not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )

    return True


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Phishing model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_kb():
    if not KB_PATH.is_file():
        raise FileNotFoundError(f"Knowledge base not found: {KB_PATH}")
    return load_documents(KB_PATH)


class URLRequest(BaseModel):
    url: str = Field(min_length=8, max_length=4096)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


@app.get("/health")
def health():
    status = {
        "status": "ok",
        "phishing_model": "unavailable",
        "knowledge_base": "unavailable",
    }

    try:
        load_model()
        status["phishing_model"] = "loaded"
    except Exception:
        status["status"] = "degraded"

    try:
        status["knowledge_base"] = f"loaded ({len(load_kb())} documents)"
    except Exception:
        status["status"] = "degraded"

    return status


@app.post("/phishing/predict", dependencies=[Depends(require_api_key)])
def predict_phishing(request: URLRequest):
    url = request.url.strip()

    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.hostname
            or any(char.isspace() for char in url)
        ):
            raise ValueError("Invalid URL")
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail="Provide a valid HTTP or HTTPS URL",
        )

    try:
        model = load_model()
        prediction = int(model.predict([url])[0])
        probabilities = model.predict_proba([url])[0]
        classes = list(model.named_steps["classifier"].classes_)
        score = float(probabilities[classes.index(0)])
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Phishing model is unavailable",
        )

    return {
        "url": url,
        "prediction": "PHISHING" if prediction == 0 else "LEGITIMATE",
        "phishing_score": round(score, 4),
        "score_is_calibrated_probability": False,
        "disclaimer": (
            "This is a model assessment, not a guarantee of safety. "
            "False positives and false negatives are possible."
        ),
    }


@app.post("/knowledge/search", dependencies=[Depends(require_api_key)])
def knowledge_search(request: SearchRequest):
    try:
        documents = load_kb()
        results = search_documents(
            request.query,
            documents,
            top_k=request.top_k,
        )
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is unavailable",
        )

    return {
        "query": request.query,
        "count": len(results),
        "results": results,
        "note": "Results are lexical matches, not generated LLM answers.",
    }
