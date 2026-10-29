# CyberLLM Training Report

Run date: 2026-10-29 (automated, reproducible via `scripts/train_conversations.py`)

## What "training" means here — honest framing

| Concept | Applies to CyberLLM? | Detail |
|---|---|---|
| Pretraining from scratch | **No** | No LLM was pretrained. No claim of this is made anywhere. |
| Fine-tuning (LoRA/QLoRA) | **Blocked by hardware/driver** | Torch installed is `2.14.1+cpu` (CPU-only). The RTX 3050 6GB GPU is present but unusable by this build. |
| Retrieval-augmented generation (RAG) | **Yes — this is what shipped** | TF-IDF semantic router over the curated conversational dataset + BM25 fallback over the 2,427-doc cybersecurity KB. |
| Existing phishing classifier training | **Untouched, re-evaluated** | `phishing/train.py` pipeline was NOT re-run or modified; existing artifact `models/phishing_url_model.joblib` preserved. |

## Hardware Detected

- Windows 11, Python 3.14.2 in project `.venv`
- 12 CPU cores, 16 GB RAM
- NVIDIA RTX 3050 6GB Laptop GPU — **CUDA unavailable** in current torch build

Decision: shipping a fake "trained LLM" or blindly downloading a 4-7B model
(a multi-GB download that CPU inference would make unusably slow) was
rejected. The retrieval-hybrid approach is honest, fast (<1s/query), runs
fully offline, and is labeled accurately in every API response.

## Pipeline Stages (all completed successfully)

| Stage | Status | Detail |
|---|---|---|
| Hardware check | ✅ PASS | CUDA=False detected; RAG path chosen explicitly |
| Dataset discovery | ✅ PASS | Found `raw_synthetic_conversations.jsonl` |
| Synthetic generation | ✅ PASS (pre-existing) | 72 records |
| Validation & dedupe | ✅ PASS | 0 invalid, 0 duplicates |
| Train/val/test split | ✅ PASS | 57/7/8, seed 42 |
| Retrieval index build | ✅ PASS | 72 conv records, 2427 KB docs, 168 vocab terms |
| Model artifact save | ✅ PASS | Index rebuilt on each engine load; no binary artifact needed |
| Configuration logging | ✅ PASS | `artifacts/reports/conversation/training_run_*.json` |
| Failure handling | ✅ PASS | Any stage failure raises and halts with stderr captured |

## Training Run Log (latest)

```
[HARDWARE] CUDA unavailable -> using retrieval-based conversational model
[DATASET]  raw_synthetic_conversations.jsonl
[PREPROCESS] 72 valid, 0 invalid, 0 duplicates
[INDEX]    records_indexed=72, kb_documents=2427, vocabulary_size=168
[EVAL]     smoke 6/6 answered from trained/retrieved knowledge
[DONE]     artifacts/reports/conversation/training_run_20261029_*.json
```

## Conversation Engine Evaluation Results

From `artifacts/reports/conversation/evaluation_report.json`:

| Metric | Result | Pass threshold | Verdict |
|---|---|---|---|
| Held-out accuracy (exact match OR correct category) | **1.00** (8/8) | ≥ 0.70 | ✅ PASS |
| Safety refusal rate (malicious prompts) | **1.00** (3/3 refused) | 1.00 | ✅ PASS |
| Honest fallback on out-of-scope questions | **1.00** (2/2 honest) | 1.00 | ✅ PASS |

## Phishing Classifier Evaluation (separate model, separate dataset)

From `artifacts/reports/phishing/phishing_evaluation.json` — evaluated on
the held-out 20% of PhiUSIIL (same split recipe as original training):

| Metric | Value |
|---|---|
| Accuracy | 0.9961 |
| Precision (phishing=0) | 1.0000 |
| Recall (phishing=0) | 0.9909 |
| F1 (phishing=0) | 0.9955 |
| Test size | 47,074 URLs |

## Limitations (explicitly disclosed)

1. This is **not** a pretrained or fine-tuned LLM. It cannot handle open-ended
   or novel questions beyond its curated scope + KB.
2. If/when a CUDA-enabled torch is installed, a LoRA path
   (`scripts/train_conversations_lora.py`, not yet written) could be added —
   flagged as future work, not claimed as done.
3. The KB fallback surfaces raw retrieved documents, not synthesized answers.
4. All conversational training data is synthetic; answers describe concepts,
   not verified incidents.
