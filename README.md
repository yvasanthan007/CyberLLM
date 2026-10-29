# CyberLLM — Conversational Cybersecurity Assistant

CyberLLM answers cybersecurity questions conversationally, assesses URLs
for phishing, and searches a local cybersecurity knowledge base — all
locally, with no external LLM dependency.

## Quick Start (one command pipeline)

```powershell
cd D:\CyberLLM
.\.venv\Scripts\Activate.ps1

# Setup + data + evaluation (safe, non-training, ~10 seconds)
.\.venv\Scripts\python.exe scripts\train_conversations.py

# Evaluate the conversational engine
.\.venv\Scripts\python.exe scripts\evaluate_conversations.py

# Evaluate the phishing classifier (separate pipeline)
.\.venv\Scripts\python.exe scripts\evaluate_phishing.py

# Start the API
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8765
```

Then test a conversational request (in another terminal):

```powershell
$key = (Get-Content .env | Select-String 'CYBERLLM_API_KEY=(.*)').Matches.Groups[1].Value
$h = @{ 'X-API-Key' = $key; 'Content-Type'='application/json' }
Invoke-RestMethod -Uri http://127.0.0.1:8765/chat -Method Post -Headers $h `
  -Body '{"message":"What is ethical hacking?"}'
```

Expected answer begins: *"Ethical hacking is the authorized process of
identifying and assessing security vulnerabilities..."*

## Endpoints

| Endpoint | Auth | Purpose |
|---|---|---|
| GET  /health            | no  | Service status |
| GET  /capabilities      | no  | Capability summary for integrators |
| GET  /models            | yes | Model registry with honest descriptions |
| POST /chat              | yes | Conversational cybersecurity Q&A |
| POST /phishing/predict  | yes | URL phishing classifier |
| POST /knowledge/search  | yes | Lexical KB retrieval |

Auth header: `X-API-Key: <CYBERLLM_API_KEY from .env>`

See `API_DOCUMENTATION.md` for full request/response schemas.

## Project Layout

```
api/main.py                     FastAPI backend
cyberllm/conversation/          Conversational retrieval engine
cyberllm/retrieval/lexical.py   BM25-style KB search
data/conversation/              Conversational dataset (synthetic)
data/raw/                       Original external datasets (untouched)
dataset/                        CVE/MITRE/Sigma instruction data (untouched)
models/                         Trained artifacts (phishing classifier)
artifacts/reports/              Training and evaluation reports
scripts/                        Automated pipeline scripts
phishing/                       Original phishing pipeline (untouched)
tests/                          pytest suite (run: python -m pytest -q)
```

## Honest Capability Statement

The conversational engine is **retrieval-based (RAG hybrid)**, not a
pretrained LLM. See `MODEL_CARD.md` and `TRAINING_REPORT.md` for why, and
what would change if a CUDA-enabled PyTorch were installed.

## Key Documents

- `AUDIT_REPORT.md` — initial project audit
- `DATASET_REPORT.md` — conversational dataset provenance and stats
- `TRAINING_REPORT.md` — pipeline runs and evaluation results
- `MODEL_CARD.md` — model description and limitations
- `API_DOCUMENTATION.md` — full endpoint reference
- `ZYRA_INTEGRATION.md` — how to connect Zyra desktop assistant
