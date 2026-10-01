# Architecture

## Design principles

1. **Honesty over convenience.** A capability that is not configured says so,
   with the exact command to configure it. Demo mode is labelled everywhere it
   appears. Synthetic results are never ranked against real ones.
2. **Untrusted input is assumed.** Every external string passes a validator
   before it reaches business logic, and every limit is explicit.
3. **Provenance is a first-class field.** Documents, predictions and datasets
   all carry where they came from, so an answer traces back to a source.
4. **Reproducibility is recorded, not asserted.** Seeds, splits, data hashes
   and library versions are written into model cards.

## Layers

```
   terminal ──> CLI (typer) ──┐
   browser  ──> Streamlit ────┼──> FastAPI service ──┬──> RAG pipeline ──> vector store
                              │                      ├──> datasets (registry, manifest)
                              └──────────────────────┴──> detection / phishing / malware
                                                          │
                                                          └──> artifacts/ (models, reports)
```

The CLI and the Streamlit console are clients of the same service surface.
Neither holds security logic of its own, so both inherit the same validation,
limits and error typing.

## Data flow: a chat question

1. `ChatService.ask()` validates and sanitises the question
   (`security.guards`, bounded by `max_prompt_chars`).
2. `RAGPipeline.build_context()` searches in two tiers:
   - security identifiers (`CVE-…`, `T1059.001`, `CWE-…`) resolve through an
     **exact index** first;
   - remaining slots are filled by vector similarity, excluding duplicates.
3. Each retrieved chunk is scanned for prompt-injection patterns and wrapped
   in `<…_content>` delimiters, with the trust notice *outside* the tags so
   consumers extract only genuine source text.
4. The prompt is assembled with citation numbers and handed to the backend.
5. The response records `mode`, `is_demo`, citation metadata, weak-evidence
   status and any injection warnings.

## Key decisions

**Exact identifier index, not just vector search.** At 384 dimensions the
hashing embedder produced hash collisions severe enough that the correct
`CVE-2021-44228` record ranked 66th out of 2427. Identifiers are *keys*, not
fuzzy terms, so they get a dict lookup; vector search ranks everything else.
Measured, not assumed - see `cyberllm/rag/identifiers.py`.

**Two vector-store backends.** A self-contained NumPy store is the default: no
native wheels, one auditable directory, tests run with zero setup. ChromaDB is
available behind the same interface for larger corpora.

**Leakage control before splitting.** Flow preprocessing removes identifier
columns (`srcip`, `dstip`, ports) and de-duplicates *before*
`train_test_split`, carrying each row's own label through the filter. An
earlier version sliced labels positionally and trained on scrambled targets.

**Training, ingestion and the app are separate processes.** Datasets and
artefacts live on disk and are read by the API, so the service needs no write
access to data and a long training run cannot disturb a running server.

## Module map

| Module | Responsibility |
|---|---|
| `core/config.py` | Env-overridable settings with safe defaults |
| `core/errors.py` | Typed exceptions mapped to HTTP status codes |
| `core/logging_config.py` | Logging with credential redaction |
| `security/guards.py` | All untrusted-input validation |
| `datasets/` | Registry, acquisition, validation, manifest |
| `rag/` | Chunking, embeddings, store, retrieval, loaders |
| `llm/` | Provider-independent interface (4 backends) |
| `detection/` | Flow preprocessing, training, inference |
| `phishing/`, `malware/` | Offline feature classification |
| `service/chat.py` | Answer contract: citations, mode, honesty |
| `api/` | HTTP boundary, validation, error envelope |
| `app/ui.py` | Thin client; holds no security logic |

## Error handling

Domain errors carry a `status_code`, a `code` and a `setup_hint`. The API maps
them onto semantic HTTP responses; the CLI and console render the same hint.
Unexpected errors return a correlation id and never a traceback.

## Entry points

| Command | Starts |
|---|---|
| `python -m cyberllm.cli` | Interactive CLI |
| `python -m cyberllm.cli app` | API + console |
| `uvicorn cyberllm.api.app:app` | API only |
| `streamlit run cyberllm/app/ui.py` | Console only |

See also: [security.md](security.md), [datasets.md](datasets.md),
[development.md](development.md), [coding_standards.md](coding_standards.md).

