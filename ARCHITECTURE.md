# CyberLLM Architecture

Updated: 2026-10-30

## Components

```
Zyra desktop assistant (external repo, not in this workspace)
   │  HTTP + X-API-Key  (examples/zyra_cyberllm_client.py)
   ▼
FastAPI  (api/main.py, 127.0.0.1:8765)
   ├── GET  /health            public status
   ├── GET  /capabilities      public capability summary
   ├── GET  /models            model inventory (auth)
   ├── POST /chat              unified inference (auth)
   ├── POST /phishing/predict  TF-IDF+LogReg URL classifier (auth)
   └── POST /knowledge/search  BM25 lexical KB search (auth)
   ▼
UnifiedInference (cyberllm/orchestrator.py)
   1. safety regex gate
   2. retrieval direct hit? -> curated-dataset answer (labeled
      "conversation_dataset", NOT claimed as LLM)
   3. local LLM available? -> Qwen2.5-0.5B-Instruct generation,
      RAG context from KB (source "llm" / "llm+rag")
   4. fallback -> retrieval engine, honestly labeled
   ▼
cyberllm/llm/adapter.py     LocalLLMAdapter — loads Qwen/Qwen2.5-0.5B-Instruct
                            once, thread-safe generate(), safety refusal,
                            real status reporting (never fakes a load)
cyberllm/rag/pipeline.py    BM25 retrieval over 2,427-doc KB, min-score gating,
                            source-attributed context, char budgets
cyberllm/conversation/      TF-IDF + BM25 retrieval engine (original, preserved)
cyberllm/retrieval/         lexical search over artifacts/vectorstore/cyberllm_kb.jsonl
models/phishing_url_model.joblib   original phishing classifier (preserved)
```

## Key properties

- Every answer carries a `source` label: `llm`, `llm+rag`,
  `conversation_dataset`, `knowledge_base`, `safety_filter`, or `fallback`.
  Retrieval results are never passed off as LLM generation.
- The local LLM is loaded once at API startup (`@app.on_event("startup")`),
  not per request.
- Auth: `X-API-Key` header compared with constant-time `secrets.compare_digest`
  against `CYBERLLM_API_KEY` from `.env`. The key never appears in responses.
- KB index is reused as-is; nothing is rebuilt at runtime.

## Hardware reality (verified 2026-10-30)

- RTX 3050 6 GB present, but torch `2.14.1+cpu` -> CUDA unavailable.
- LLM inference runs on CPU (~10 tok/s for 0.5B fp32) — acceptable for chat.
- Fine-tuning requires a CUDA torch build (needs Python 3.11/3.12 venv;
  Python 3.14 has no CUDA wheels yet). `scripts/train_lora.py` gates on this
  honestly and exits with code 2 when blocked.
