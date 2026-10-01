# Development

## Setup

```bash
python -m pip install -r requirements.txt
python -m cyberllm.cli health
```

Verified on Python 3.12.7 (Windows, 12 vCPU, 15.7 GB RAM). CPU-only works;
no GPU is required for any default path.

## Tests

```bash
python -m pytest tests -q            # whole suite
python -m pytest tests/test_detection.py -v
```

Tests use synthetic fixtures only, so they are deterministic and need no
downloads. The pre-existing `tests/test_*.py` files from the dataset pipeline
are standalone scripts rather than pytest modules - run them directly.

Real-data benchmarks are kept separate from the test suite. Never let a
synthetic metric be reported as field accuracy.

## Static analysis

```bash
python scripts/static_check.py
```

A purpose-built AST linter. It checks the defect classes that actually
occurred while building this project:

| Code | Catches |
|---|---|
| `syntax_error` | Unparseable modules (e.g. an interrupted edit) |
| `stub_body` | Functions with no implementation |
| `silent_except` | Broad `except` blocks that swallow failures |
| `undefined_name` | Names used but never imported - the class of bug that compiles cleanly and fails mid-run |
| `no_imports` | Dead modules |
| `line_continuation` | Backslash continuations, which hide broken conditionals |

It walks methods as well as functions, resolves closures, and is deliberately
conservative: over-approximating local names only suppresses findings, so it
does not produce false alarms.

## Configuration precedence

Environment > `configs/cyberllm.yaml` > dataclass defaults. `get_settings()`
is cached; `reload_settings()` clears it (used by tests).

## Troubleshooting

**Port 8000 unavailable on Windows** - reserved by Hyper-V/WSL in some setups.
Use another port: `python -m cyberllm.cli app --port 8010`, and point the
console at it with `set CYBERLLM_API_URL=http://127.0.0.1:8010`.

**"No Ollama daemon"** - start with `ollama serve`, then `ollama list` to
confirm the model is actually pulled. The backend checks the installed model
list, not just that the daemon responds.

**NVD returns HTTP 429** - rate limited. Set `NVD_API_KEY`, or lower
`CYBERLLM_NVD_MAX_RECORDS` and be patient between keyless calls.

**Retrieval returns nothing** - the knowledge base is empty. Run
`python -m cyberllm.cli rag ingest` after downloading datasets; check
`rag status`.

**Model trained on synthetic data** - no real dataset was present so a fixture
was generated. Supply `--data <real.csv>`; the SYNTHETIC banner is deliberate.

**A model bundle will not load** - usually a library-version mismatch between
the training and serving environments. Retrain; the error says so.

**Streamlit shows an icon exception** - Streamlit >= 1.64 requires
`:material/<snake_case_name>:` syntax.

## Adding a feature

1. Put the logic in a domain module (`rag/`, `detection/`, …), not in the API
   or UI layer. The console is a thin client and must stay that way.
2. Add a typed exception in `core/errors.py` with a `setup_hint`, so an
   unconfigured capability teaches rather than fails.
3. Add tests covering the success path, invalid input and the unconfigured path.
4. Run `python scripts/static_check.py` and `python -m pytest tests -q`.
5. Update `PROJECT_STATUS.md`.
