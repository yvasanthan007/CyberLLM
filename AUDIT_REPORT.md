# CyberLLM Audit Report

Date: 2026-10-29
Auditor: automated audit (agent)

## Summary

CyberLLM is a defensive cybersecurity assistant project. It currently has a
working phishing URL classifier, a lexical cybersecurity knowledge base, a
FastAPI backend, and an existing instruction dataset built from CVE, MITRE
ATT&CK, and Sigma sources. It does NOT yet have:

- A conversational (question/answer) cybersecurity dataset.
- A trained conversational language model.
- A `/chat` endpoint.
- Automated conversational training and evaluation pipelines.
- Zyra integration.

## Hardware / Environment

- OS: Windows 11 x64
- CPU: 12 logical cores
- RAM: 16 GB
- GPU: NVIDIA GeForce RTX 3050 6 GB Laptop GPU (CUDA supported)
- Python: 3.14.2 (project `.venv`)
- Installed: torch 2.14.1+**cpu** (CUDA disabled in this build),
  transformers 5.19.0, peft 0.21.2, trl 1.15.0, datasets 5.1.0

Key constraint: the installed PyTorch is CPU-only even though an RTX 3050 is
present. Full LLM fine-tuning (LoRA) on CPU with 16 GB RAM is not practical in
reasonable time. The project therefore uses a **retrieval-augmented
generation (RAG) + semantic-router hybrid** approach with a small, locally
runnable generative backbone, rather than blind large-model downloads.

## Existing Components (must be preserved)

### Data
- `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv` — phishing URL dataset
- `data/raw/mitre_attack/enterprise-attack.json` — MITRE ATT&CK
- `data/raw/unsw_nb15/` — UNSW-NB15 network intrusion dataset
- `data/raw/cisa_kev/known_exploited_vulnerabilities.json` — CISA KEV
- `data/raw/synthetic/nids_synthetic_fixture.csv` — synthetic NIDS fixture
- `dataset/raw/cve/*.json` — CVE records (5 batches)
- `dataset/processed/*.jsonl` — cleaned CVE/MITRE/Sigma
- `dataset/instruction/*.jsonl` — instruction-formatted CVE/MITRE/Sigma
- `dataset/final/cyberllm_train.jsonl` — 62,095 instruction records
- `artifacts/vectorstore/cyberllm_kb.jsonl` — 2,427-doc knowledge base

### Models / artifacts
- `models/phishing_url_model.joblib` — TF-IDF + LogisticRegression
- `artifacts/models/nids/lightgbm/` and `artifacts/models/nids/random_forest/`
  — UNSW-NB15 NIDS models with model cards

### Code
- `api/main.py` — FastAPI: `/health`, `/phishing/predict`,
  `/knowledge/search`, API-key auth via `X-API-Key` and `CYBERLLM_API_KEY`
- `cyberllm/retrieval/lexical.py` — BM25-style lexical search
- `cyberllm/datasets/` — dataset acquisition/validation/manifest tooling
- `phishing/train.py`, `phishing/predict.py`, and related scripts
- `configs/config.py` + empty YAML placeholders
- `tests/` — pytest suite

### Configuration
- `.env` — contains `CYBERLLM_API_KEY` (secret, not to be exposed)
- `.gitignore` present; project under git

## Risks / Gaps Identified

1. Torch is CPU-only; CUDA fine-tuning unusable without reinstall.
2. No conversational dataset (existing instruction data is CVE/MITRE/Sigma
   oriented, not Q&A).
3. No `/chat`, `/capabilities`, or `/models` endpoints.
4. No model card, dataset report, or training report.
5. `configs/model_config.yaml` and `training_config.yaml` are empty.
6. No Zyra integration documentation or client.
7. Existing instruction dataset is useful context but not conversational.

## Constraints Honored Going Forward

- Never delete or overwrite existing datasets/models.
- Keep phishing data separate from conversational data.
- Back up before modifying tracked files.
- Use existing `.venv` and installed packages.
- Respect model licenses; prefer open-weight models.
