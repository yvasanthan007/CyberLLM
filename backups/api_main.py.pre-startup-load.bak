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
from cyberllm.conversation.engine import get_engine, ENGINE_ID, ENGINE_DESCRIPTION
from cyberllm.orchestrator import get_inference
from cyberllm.llm.adapter import get_adapter

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODEL_PATH = ROOT / "models" / "phishing_url_model.joblib"
KB_PATH = ROOT / "artifacts" / "vectorstore" / "cyberllm_kb.jsonl"

app = FastAPI(
    title="CyberLLM Security API",
    description="Defensive cybersecurity API: conversational assistant, phishing assessment, and knowledge search",
    version="0.2.0",
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


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    model: str
    model_description: str
    source: str
    confidence: float
    status: str = "ok"


class URLRequest(BaseModel):
    url: str = Field(min_length=8, max_length=4096)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


@app.on_event("startup")
def _load_llm_once():
    """Load the local LLM exactly once at server startup (best-effort).

    Failure never crashes the API; /health and /chat report the real status.
    """
    try:
        get_adapter().ensure_loaded()
    except Exception:
        pass


@app.get("/health")
def health():
    status = {
        "status": "ok",
        "phishing_model": "unavailable",
        "knowledge_base": "unavailable",
        "local_llm": "disabled_or_not_loaded",
    }
    try:
        adapter = get_adapter()
        info = adapter.status()
        if info["loaded"]:
            status["local_llm"] = f"loaded ({info['model_id']} on {info['device']})"
        elif not info["enabled"]:
            status["local_llm"] = "disabled"
        elif info["load_error"]:
            status["local_llm"] = f"error: {info['load_error']}"
    except Exception:
        pass

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


# ------------------------------------------------------------------
# Conversational endpoints (CyberLLM chat)
# ------------------------------------------------------------------


@app.get("/capabilities")
def capabilities():
    """Public capability summary for integrators (no auth required)."""
    try:
        engine = get_engine()
        conversation_loaded = len(engine.records) > 0
    except Exception:
        conversation_loaded = False

    return {
        "service": "CyberLLM",
        "chat": conversation_loaded,
        "chat_model": ENGINE_ID if conversation_loaded else None,
        "chat_description": ENGINE_DESCRIPTION,
        "local_llm": get_adapter().status(),
        "phishing_predict": MODEL_PATH.is_file(),
        "knowledge_search": KB_PATH.is_file(),
        "auth_required": True,
        "auth_header": "X-API-Key",
        "endpoints": [
            "GET /health",
            "GET /capabilities",
            "GET /models",
            "POST /chat",
            "POST /phishing/predict",
            "POST /knowledge/search",
        ],
    }


@app.get("/models", dependencies=[Depends(require_api_key)])
def list_models():
    inf = get_inference().capabilities()
    llm_cap = inf["llm"]
    models = [
        {
            "id": ENGINE_ID,
            "kind": "conversational",
            "description": ENGINE_DESCRIPTION,
            "trained_from_scratch": False,
            "is_pretrained_llm": False,
            "approach": "TF-IDF retrieval over curated conversational dataset + BM25 fallback over knowledge base",
        },
        {
            "id": "phishing_url_tfidf_logreg",
            "kind": "classification",
            "description": "TF-IDF char n-gram + LogisticRegression URL classifier",
            "artifact": "models/phishing_url_model.joblib",
        },
    ]
    if llm_cap["available"]:
        models.insert(0, {
            "id": llm_cap["model_id"],
            "kind": "conversational",
            "description": "Local pretrained instruction-tuned LLM (inference adapter)",
            "trained_from_scratch": False,
            "is_pretrained_llm": True,
            "fine_tuned_adapter": llm_cap.get("fine_tuned_adapter"),
            "approach": "Transformers generation, optionally RAG-augmented from the local KB",
        })
    return {"models": models}


@app.post("/chat", dependencies=[Depends(require_api_key)])
def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be empty")

    try:
        answer, metadata = get_inference().respond(message)
    except Exception:
        raise HTTPException(status_code=503, detail="Chat engine is unavailable")

    source = metadata.get("source", "unknown")
    is_llm = source in ("llm", "llm+rag")
    return {
        "answer": answer,
        "model": metadata.get("model_id", ENGINE_ID),
        "model_description": (
            metadata.get("note") or
            f"Local LLM generation" if is_llm else ENGINE_DESCRIPTION),
        "source": metadata.get("source", "unknown"),
        "confidence": metadata.get("confidence", 0.0),
        "matched_question": metadata.get("matched_question"),
        "category": metadata.get("category"),
        "related_questions": metadata.get("related_questions", []),
        "refused": metadata.get("refused", False),
        "llm_generated": is_llm,
        "retrieved_sources": metadata.get("retrieved_sources", []),
        "status": "ok",
        "note": (
            "Answer generated by the local LLM." if is_llm
            else "Responses come from the actual loaded retrieval engine, not hardcoded strings."),
    }
