# Model Card — CyberLLM Conversational Assistant

**Model ID:** `cyberllm-rag-hybrid-v1`  +  local LLM **`Qwen/Qwen2.5-0.5B-Instruct`**
**Version:** 1.1
**Date:** 2026-10-30 (updated)
**License:** MIT (project code); Qwen2.5-0.5B-Instruct is Apache-2.0
**Developer:** CyberLLM project

## Model Type

**Hybrid: local pretrained LLM + retrieval.** Since 2026-10-30 the stack
includes the pretrained (NOT fine-tuned) Qwen2.5-0.5B-Instruct model running
locally on CPU via the Transformers inference adapter
(`cyberllm/llm/adapter.py`), augmented with BM25 RAG context from the
2,427-document KB. The original retrieval engine remains the fast path for
curated-dataset direct hits and the honest fallback when the LLM is
unavailable. Two cooperating retrievers:

1. **Primary:** TF-IDF cosine similarity router over 72 curated
   cybersecurity Q&A pairs (`data/conversation/processed/conversations_clean.jsonl`).
2. **Fallback:** BM25-style lexical ranker over the existing 2,427-document
   cybersecurity knowledge base (`artifacts/vectorstore/cyberllm_kb.jsonl`,
   built previously from MITRE ATT&CK, CVE, and Sigma sources).
3. **Safety layer:** regex-based refusal for requests to create malware or
   perform unauthorized intrusion.

## Intended Use

- Answering clear cybersecurity educational/defensive questions.
- Integration into desktop assistants (e.g., Zyra) via the `/chat` endpoint.
- Not for: offensive guidance, generating exploits, real-time threat intel,
  legal advice, or open-domain conversation.

## Out-of-Scope / Known Failures

- Questions outside cybersecurity → honest "I don't have a confident answer" fallback.
- Requests for offensive tooling → refused.
- Ambiguous or multi-part questions → may match partially; answer metadata
  (`confidence`, `matched_question`) lets callers decide whether to show or re-ask.

## Training Data

Entirely synthetic, authored for this project (see `DATASET_REPORT.md`).
No web scraping. No external citations embedded. 72 records / 10 categories.

## Performance

| Metric | Value |
|---|---|
| Held-out accuracy (8 records) | 1.00 |
| Safety refusal rate | 1.00 |
| Honest fallback on unsupported probes | 1.00 |
| Typical response latency | <100 ms (no GPU required) |

## Hardware Requirements

None beyond CPU — runs comfortably within 16 GB RAM and 12 CPU cores.
No CUDA dependency. No external model downloads.

## Alternatives Considered

| Option | Rejected because |
|---|---|
| Download and LoRA-tune a 1–7B open-weight model | Installed torch is CPU-only; RTX 3050 6GB insufficient for QLoRA on ≥3B params; multi-GB download unnecessary for curated-scope Q&A. |
| Use OpenAI/Anthropic hosted API | Requires credentials, network, cost; violates offline/no-secrets constraint. |
| Train from scratch | Not meaningful for 72 examples; would not generalize. |

## Ethical Considerations

- All responses labeled with `source` and `confidence` — callers can tell
  curated matches from KB retrievals from fallbacks.
- Refusals are explicit and explained, not silent.
- No claim of being a pretrained LLM is made anywhere in the codebase, API,
  or docs.

## How to Cite / Attribute

If you integrate this into Zyra or another assistant, describe it as:
"CyberLLM retrieval-based cybersecurity Q&A (synthetic curated dataset +
BM25 knowledge-base fallback)". Do not describe it as a language model.
