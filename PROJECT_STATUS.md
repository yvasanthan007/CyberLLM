# CyberLLM - Project Status

> Living document. Updated at the end of every implementation phase.
> Branch: `feature/cyberllm-platform` (branched from `LLM_build` / `main`).

## Environment (verified)

| Item | Value |
|---|---|
| Python | 3.12.7 (also 3.14 available) |
| OS / Shell | Windows, PowerShell 5.1 |
| CPU | 12 logical processors |
| RAM | 15.7 GB |
| GPU | NVIDIA RTX 3050 Laptop, 6 GB |
| Disk free | 79 GB (D:), 113 GB (C:) |
| Key packages | FastAPI 0.133, Uvicorn, ChromaDB 1.5.9, scikit-learn 1.9, torch 2.13, transformers 5.14, peft, trl, typer, pytest 9.1, streamlit 1.64 |
| Network | NVD API, CISA KEV, MITRE CTI, Ollama daemon on 127.0.0.1:11434 - all reachable |

## Phase status

| Phase | Scope | Status |
|---|---|---|
| 1 | Inspect repo + environment, checklist | **Done** |
| 2 | Package structure, config, logging, CLI | **In progress** (config/logging done, CLI pending) |
| 3 | Dataset registry, acquisition, validation | **Done** |
| 4 | CVE/NVD, CISA KEV, MITRE ATT&CK ingestion | **Done** |
| 5 | RAG pipeline + chat integration | **Done** |
| 6 | Network intrusion detection | Pending |
| 7 | Phishing + malware pipeline | Pending |
| 8 | Optional LoRA fine-tuning | Pending |
| 9 | FastAPI + Streamlit | Pending |
| 10 | Security controls, tests, CI, docs | Partial |
| 11 | End-to-end verification | Pending |
| 12 | Final report | Pending |

## Verified working (commands actually executed)

### Datasets acquired and validated
| Dataset | Size | Records | Status |
|---|---|---|---|
| `cisa_kev` | 1.68 MB | 1,730 vulnerabilities | downloaded + validated (SHA-256 recorded) |
| `mitre_attack` | 45.7 MB | 26,085 STIX objects / 697 active techniques | downloaded + validated |
| `nvd_cve` | - | - | API reachable; not yet pulled (see blockers) |
| `unsw_nb15`, `cic_ids2017`, `cse_cic_ids2018`, `cic_malmem_2022`, `phishing_urls` | - | - | **manual acquisition required** - exact steps recorded in `data/README.md` |

### Retrieval
- Knowledge base: **2,427 chunks** indexed (1,730 KEV + 697 ATT&CK).
- Exact identifier resolution: `CVE-2021-44228` and `T1059.001` both return the
  correct record at rank 1 (score 1.000) via an exact identifier index.
- Pure-vector search was measured as **insufficient**: the correct Log4Shell
  record ranked 66/2427 due to hash collisions at 384 dimensions. Fixed with
  `cyberllm/rag/identifiers.py` + an O(1) identifier index in the store.

### Chat (both modes tested)
- **Demo mode**: retrieval-backed template, clearly labelled, 4 excerpts + 5
  numbered citations with URLs and retrieval scores.
- **Ollama / llama3**: real generation, accurate cited answer, 33.8 s on CPU.

## Test results (actual)

```
python -m pytest tests -q          ->  27 passed
python scripts/static_check.py     ->  0 issue(s)
```

`scripts/static_check.py` is a custom AST linter that catches the defect
classes actually hit during this build: syntax errors from interrupted edits,
functions with no implementation, and broad `except` blocks that silently
swallow failures.

## Defects found and fixed during the build

1. **Credential leak in the log redaction filter** (security). A bare
   `except Exception: return True` let malformed records reach the handler;
   `logging.Handler.handleError` then dumped the raw record - including the
   unredacted credential - to stderr. Reproduced, fixed with explicit
   `(TypeError, ValueError, KeyError, IndexError)` handling, a drop path for
   unexpected errors, a non-recursive failure counter, and a hardened JSON
   formatter. Covered by 27 tests in `tests/test_logging_config.py`.
2. **False-positive Ollama availability**. `available()` only probed the
   daemon, not the model, so a missing model surfaced later as a 404. Now
   checks the installed model list and reports what is available.
3. **Prompt-injection notice inside content tags** polluted extracted
   excerpts. The notice now sits outside the content tags.
4. **Missing `KeyError` handling** in the log filter (found by a failing test).

## Blockers / outstanding

- `nvd_cve` not yet pulled: NVD rate-limits keyless requests to ~5 per 30 s. A
  free API key (`NVD_API_KEY`) raises this to 50/30 s. The code path is
  implemented and the endpoint verified reachable.
- The four large flow/malware/URL datasets require manual acceptance of their
  publishers' terms, so they cannot be downloaded by the tool. Exact steps are
  recorded; the preprocessing and training code paths accept them unchanged.
- No real-data intrusion/phishing/malware benchmark yet (blocked on the above).
  Synthetic fixtures are used for tests and are labelled as such.

## Next actions

1. `cyberllm/cli.py` (typer) exposing the documented commands.
2. NIDS preprocessing + training + evaluation (RF, LightGBM).
3. Phishing classifier and the safe malware feature pipeline.
4. FastAPI service and Streamlit console.
5. Remaining tests, CI workflow, and documentation.
