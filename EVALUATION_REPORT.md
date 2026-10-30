# CyberLLM Evaluation Report

Run date: 2026-10-30

## 1. Regression tests (pytest)

Command: `python -m pytest tests phishing -q`

**Result: 17 passed** (warnings only: Starlette testclient deprecation, FastAPI
`on_event` deprecation). No failures.

## 2. Local model loading + generation test

Script: `scripts/check_llm.py`

- Model: **Qwen/Qwen2.5-0.5B-Instruct** (Apache-2.0), loaded from the local
  HF cache (`HF_HOME=D:\HuggingFace`), snapshot `7ae55760...`.
- Loaded on **CPU** (torch `2.14.1+cpu`; RTX 3050 GPU present but unusable).
- Generation test ("What is phishing and how can I protect myself?"):
  coherent, on-topic 120-token answer in **11.67 s** (~10 tok/s CPU).

## 3. Conversational retrieval engine (existing baseline)

From `artifacts/reports/conversation/evaluation_report.json` (re-ran
`scripts/evaluate_conversations.py` on 2026-10-30):

| Metric | Result | Threshold | Verdict |
|---|---|---|---|
| Held-out accuracy (8 test questions) | 1.00 | >= 0.70 | PASS |
| Safety refusal (3 malicious prompts) | 1.00 | 1.00 | PASS |
| Honest fallback (2 out-of-scope) | 1.00 | 1.00 | PASS |

Note: the held-out set is 8 synthetic questions; a 100% score there does **not**
prove general conversational intelligence.

## 4. Local LLM via unified inference (live API, new path)

Live server tests on 127.0.0.1:8765 (real HTTP, real key):

| Request | Result |
|---|---|
| "What is two-factor authentication?" | `source=conversation_dataset` (TF-IDF direct hit — correct routing) |
| "What is SQL injection?" | `source=llm+rag`, `llm_generated=true`, KB sources attached |
| "Explain password salting briefly" | full LLM answer generated (~35 s incl. load) |
| Missing/invalid API key on POST /chat | HTTP 401 |
| GET /health | `local_llm: loaded (Qwen/Qwen2.5-0.5B-Instruct on cpu)` |

Routing note: exact matches against the curated dataset are still answered by
the retrieval engine (faster, deterministic) and labeled honestly; open-ended
questions go to the LLM with RAG context.

## 5. Phishing classifier (separate model, separate data)

From `artifacts/reports/phishing/phishing_evaluation.json` (re-run 2026-10-30):

| Metric | Value |
|---|---|
| Accuracy | 0.9961 |
| F1 (phishing class) | 0.9955 |
| Test size | 47,074 URLs |

Live spot-check: `http://secure-login-verify-bank.example.top/account` ->
`PHISHING`, score 1.0.

## 6. Fine-tuning status

`python scripts/train_lora.py` exits with code 2 and reports:

> Blocker: torch build '2.14.1+cpu' is CPU-only; the GPU cannot be used.

**Nothing was trained.** No adapter weights exist under `artifacts/lora/`.
Every API response remains honestly labeled (see section 4).
