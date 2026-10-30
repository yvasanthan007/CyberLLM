# CyberLLM Datasets

Updated: 2026-10-30

Conversation data and phishing classification data are kept **strictly
separate** (different directories, different schemas, different purposes).

## 1. Conversational data (synthetic, labeled as synthetic)

| File | Records | Purpose |
|---|---|---|
| `data/conversation/raw_synthetic_conversations.jsonl` | 72 | original raw synthetic Q&A (preserved, untouched) |
| `data/conversation/processed/conversations_clean.jsonl` | 72 | validated + deduplicated |
| `data/conversation/splits/train.jsonl` | 57 | retrieval-index / SFT training |
| `data/conversation/splits/validation.jsonl` | 7 | validation |
| `data/conversation/splits/test.jsonl` | 8 | held-out test (never trained on) |

Schema: `{"instruction", "input", "output", "category", "source"}` with
`source` always recording the originating synthetic file. Split seed: 42.
Categories cover greetings, ethical hacking, networking, phishing/malware,
authentication/passwords, cryptography, web security, Linux tools, incident
response, and privacy/risk.

Topic-specific instruction corpora (kept separate from conversations, used for
the KB, NOT converted into chat pairs):

- `dataset/instruction/cve_instruction.jsonl` (from CVE raw batches)
- `dataset/instruction/mitre_instruction.jsonl` (from MITRE ATT&CK)
- `dataset/instruction/sigma_instruction.jsonl` (from Sigma rules)
- `dataset/final/cyberllm_train.jsonl` (merged instruction corpus)

## 2. Phishing data (classification, never conversationalized)

| File | Size | Purpose |
|---|---|---|
| `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv` | 56.8 MB | original phishing URL dataset (preserved) |
| `models/phishing_url_model.joblib` | 10 MB | trained TF-IDF char-ngram + LogReg classifier (preserved) |

The phishing dataset was NOT converted into conversational examples — no such
validated transformation exists in this project.

## 3. Knowledge base (RAG)

| File | Documents |
|---|---|
| `artifacts/vectorstore/cyberllm_kb.jsonl` (+ `.npy`, `.meta.json`) | 2,427 |

Built from MITRE ATT&CK, CVE, and Sigma sources; reused as-is by BM25 search
and the RAG pipeline. Never rebuilt during this run.

## Regeneration

```powershell
.\CyberLLM.ps1 dataset        # or: python scripts/preprocess_conversations.py
```

Validation, deduplication, seed-fixed splitting, and leakage prevention
(train/test disjoint by construction from the dedup pass) run inside the
preprocess script; results in `data/conversation/processed/preprocess_report.json`.

## Pipeline validation notes
The recorded split is 57/7/8 (train/validation/test), seed 42. The held-out
questions were not used in the LoRA training script, which reads only
`data/conversation/splits/train.jsonl`. Existing 72 records are retained; no
new generated synthetic pairs were added during this run. The current
preprocessing uses normalized deduplication before splitting; it does not
perform semantic near-duplicate clustering, so paraphrase leakage is not
ruled out.
