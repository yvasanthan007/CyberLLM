# CyberLLM - Project Status

> Living document, updated whenever a phase completes or a result is verified.
> Branch: `feature/cyberllm-platform` (branched from `LLM_build` / `main`).
> Existing work was preserved; no history was rewritten.

## Environment (verified)

| Item | Value |
|---|---|
| Python | 3.12.7 (3.14 also present) |
| OS / Shell | Windows, PowerShell 5.1 |
| CPU / RAM | 12 logical / 15.7 GB |
| GPU | RTX 3050 Laptop, 6 GB (not required) |
| Disk free | 79 GB (D:) |
| Key packages | FastAPI 0.133, ChromaDB 1.5.9, scikit-learn 1.9, torch 2.13, transformers 5.14, peft, trl, typer, rich, pytest 9.1, streamlit 1.64, lightgbm |
| Network | NVD API, CISA KEV, MITRE CTI, Ollama (127.0.0.1:11434) - all reachable |

## Phase status

| Phase | Scope | Status |
|---|---|---|
| 1 | Inspect repo + environment, checklist | **Done** |
| 2 | Package, config, logging, CLI | **Done** |
| 3 | Dataset registry, acquisition, validation | **Done** |
| 4 | CVE/NVD, CISA KEV, MITRE ATT&CK ingestion | **Done** (NVD needs a key) |
| 5 | RAG pipeline + chat | **Done** |
| 6 | Network intrusion detection | **Done** (trained on synthetic) |
| 7 | Phishing + malware pipeline | **Done** (malware model blocked) |
| 8 | Optional LoRA fine-tuning | **Not started** |
| 9 | FastAPI + Streamlit | **Done** |
| 10 | Security controls, tests, CI, docs | **Done** |
| 11 | End-to-end verification | **Done** |
| 12 | Final report | **Done** |

## Verified results

### Commands actually executed

```
python -m pytest tests -q          ->  79 passed, 1 skipped
python scripts/static_check.py     ->  0 issue(s)
python -m cyberllm.cli health      ->  exit 0, all capabilities reported
```

The skipped test is conditional on a trained malware model not being present.

### Datasets

| Dataset | Size | Records | Status |
|---|---|---|---|
| `cisa_kev` | 1.68 MB | 1,730 | **acquired + validated** (SHA-256 recorded) |
| `mitre_attack` | 45.7 MB | 26,085 STIX / 697 techniques | **acquired + validated** |
| `nvd_cve` | - | - | endpoint verified; blocked on API key |
| UNSW-NB15, CIC-IDS2017/2018, CIC-MalMem-2022, phishing URLs | - | - | **manual acquisition required** - steps in `docs/datasets.md` |

### Retrieval
- Knowledge base: **2,427 chunks** (1,730 KEV + 697 ATT&CK), persisted.
- Exact identifier index resolves `CVE-2021-44228` and `T1059.001` at rank 1.
- Pure vector search was **measured as insufficient** - the correct Log4Shell
  record ranked 66/2427 from hash collisions at 384 dimensions. Fixed with an
  exact index; the measurement is the reason.

### Chat (both modes verified)
- **Demo mode**: 4 source excerpts + 5 numbered citations with URLs and
  retrieval scores; every response labelled `is_demo`.
- **Ollama / llama3**: real generation, accurate cited answer, 33.8 s on CPU.

### Intrusion detection
Trained on a **synthetic** fixture (lab data not yet acquired):

| Model | Accuracy | F1 weighted | F1 macro | Recall macro |
|---|---|---|---|---|
| RandomForest | 0.9985 | 0.9985 | 0.9985 | 0.9985 |
| LightGBM | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

> **These are not field-accuracy numbers.** They validate the pipeline on
> separable synthetic data. The API, CLI and UI all flag this.

### Live service
- API on `:8010` returned `health=ok` with 2,427 chunks over real HTTP.
- Streamlit console returned HTTP 200; the phishing page analysed a URL end to
  end (score 0.278 vs threshold 0.50, with risk signals).
- Port 8000 is reserved on this machine (WinError 10013); use `--port 8010`.


## Defects found and fixed during the build

Each was found by a test or by driving the running system, not by review.

1. **Credential leak in the log redaction filter** (security). A bare
   `except Exception: return True` let malformed records reach the handler;
   `logging.Handler.handleError` then dumped the raw record - including the
   unredacted credential - to stderr. Reproduced, fixed with explicit
   `(TypeError, ValueError, KeyError, IndexError)` handling, a drop path for
   unexpected errors, a non-recursive failure counter and a hardened JSON
   formatter. 27 tests.
2. **Label/feature misalignment** (correctness). `prepare_flow_data`
   de-duplicated features but sliced labels positionally, so most rows trained
   against the wrong target: macro-F1 0.25 instead of 0.9985. Fixed by carrying
   labels through the index mask; a regression test pins the invariant.
3. **Non-deterministic column mapping**. `COLUMN_ALIASES` used sets, so
   `time_to_live` bound to `sttl` or `dttl` at random. Now ordered tuples with
   a determinism test.
4. **False-positive Ollama availability**. `available()` only probed the
   daemon, not the model, so a missing model surfaced later as a 404.
5. **Unusable Streamlit pages**. `if not url or not st.button(...)`
   short-circuits, so the Analyse and Look up buttons never rendered until text
   was typed. Found by driving the real page headlessly.
6. **Invalid Streamlit icons**. 1.64 requires `:material/<name>:` syntax.
7. **Missing imports found only at runtime** (`numpy`, `validate_flow_record`).
   `static_check.py` now detects undefined names across functions *and*
   methods, and flags backslash continuations.

## Blockers and outstanding work

**Blocked on external access**

- `nvd_cve`: NVD rate-limits to ~5 req/30 s without a key. A free key
  (`NVD_API_KEY`) raises this to ~50/30 s. Code path implemented and verified
  against the live endpoint.
- UNSW-NB15, CIC-IDS2017, CSE-CIC-IDS2018, CIC-MalMem-2022 and the phishing URL
  corpus all require accepting a publisher's terms. CyberLLM does not bypass
  access controls. Exact steps are in `docs/datasets.md` and
  `datasets list --name <id>`.

**Consequences**

- No real-data intrusion/phishing/malware benchmark exists yet. Synthetic
  fixtures are used for tests and labelled everywhere.
- No malware classifier is trained, so `/malware/analyse` returns 503 with setup
  instructions by design.

**Not started**

- Phase 8 (LoRA/QLoRA fine-tuning). The corpus exists at
  `dataset/final/cyberllm_train.jsonl` and `peft`/`trl` are installed, but no
  fine-tuning script has been written and no model has been trained.
  Fine-tuning is explicitly **not** required for the application to function.

**Known quality limits**

- Default embeddings are lexical (hashing), so free-text recall is
  keyword-driven. Identifiers are handled exactly. Enabling
  `sentence-transformers` improves semantic recall.
- Two small test expectations were corrected during the build (a 400 ms vs
  200 ms sleep in a timing test, and a train/test split size assertion);
  neither indicated a product defect.

## Next actions

1. Obtain a free NVD API key and index CVEs.
2. Acquire UNSW-NB15 (smallest flow dataset, 250 MB) and retrain NIDS on real
   data; compare against the synthetic baseline.
3. Acquire a licensed phishing URL CSV and train the classifier, replacing the
   heuristic.
4. Acquire CIC-MalMem-2022 features and train the malware model.
5. Implement Phase 8 LoRA fine-tuning and a documented before/after evaluation.
6. Consider `sentence-transformers` embeddings for semantic retrieval.

