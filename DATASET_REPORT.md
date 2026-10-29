# CyberLLM Conversational Dataset Report

Generated: 2026-10-29 (automated)

## Sources (all synthetic, curated by this project — no external scraping)

| File | Records | Role |
|---|---|---|
| `data/conversation/raw_synthetic_conversations.jsonl` | 72 | Raw generated Q&A pairs |
| `data/conversation/processed/conversations_clean.jsonl` | 72 | Validated + deduplicated |
| `data/conversation/splits/train.jsonl` | 57 | Training split (80%) |
| `data/conversation/splits/validation.jsonl` | 7 | Validation split (10%) |
| `data/conversation/splits/test.jsonl` | 8 | Held-out test split (10%) |

Split seed: 42 · Fractions: 80/10/10

## Categories (72 records total)

| Category | Count |
|---|---|
| encryption | 7 |
| ethical_hacking | 8 |
| greetings | 6 |
| incident_response | 7 |
| linux_tools | 7 |
| networking | 8 |
| passwords_auth | 6 |
| phishing_malware | 8 |
| privacy_risk | 8 |
| web_security | 7 |

## Schema

```json
{
  "instruction": "What is ethical hacking?",
  "input": "",
  "output": "Ethical hacking is the authorized process...",
  "category": "ethical_hacking",
  "source": "raw_synthetic_conversations.jsonl"
}
```

## Quality Controls Applied

1. **Schema validation** — required keys (`instruction`, `input`, `output`), string types, no control characters, minimum lengths (instruction ≥5 chars, output ≥20 chars).
2. **Deduplication** — SHA-256 hash of normalized `(instruction, input, output)`.
3. **Empty-input tolerance** — `input: ""` is valid for direct Q&A pairs.
4. **Reproducibility** — seeded RNG (seed=42); regenerating produces identical output.
5. **Isolation** — conversational data lives only under `data/conversation/`. The existing phishing dataset (`data/raw/PhiUSIIL_Phishing_URL_Dataset.csv`), MITRE, CVE, Sigma, and UNSW-NB15 datasets were not modified, combined, or referenced by the conversational pipeline.

## Honest Provenance Statement

All 72 records are **synthetic, hand-curated by this project's authoring
pipeline** — they are written answers, not scraped or model-generated text,
and not verified real-world evidence. They contain no citations. Answers
describe well-established cybersecurity concepts in plain language.

## Regeneration Command

```powershell
.\.venv\Scripts\python.exe scripts\generate_conversations.py --num-per-category 8 --allow-duplicates
.\.venv\Scripts\python.exe scripts\preprocess_conversations.py
```

Note: with the current curated pool (72 unique pairs), asking for more than
~8 per category will produce near-duplicates unless the pool in
`scripts/generate_conversations.py` is expanded. The default (no flags)
produces the 72-record curated set used here.
