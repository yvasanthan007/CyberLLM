# CyberLLM

**AI-powered cybersecurity platform for defensive security research.**

CyberLLM combines retrieval-augmented generation over authoritative threat
intelligence with classical ML detection for network flows, phishing URLs and
malware memory features. It ships with a FastAPI service, a Streamlit operator
console and a CLI.

> **Classifier outputs are probabilistic analyst aids, not detection
> guarantees.** Verify against the cited source before acting. See
> [Security and limitations](docs/SECURITY.md).

---

## Quick start (demo mode, no API key, no datasets)

```bash
python -m pip install -r requirements.txt
python -m cyberllm.cli health          # what is configured / ready / missing
python -m cyberllm.cli datasets download --all
python -m cyberllm.cli rag ingest
python -m cyberllm.cli app            # API on :8000, console on :8501
```

The application starts in a clearly labelled **demo mode** when no language
model is reachable. Demo mode performs no inference: it restates retrieved
source records and labels every answer `is_demo=true`. It is never presented
as model output.

To enable a real model, start Ollama and point CyberLLM at it:

```bash
ollama serve
ollama pull qwen2.5:3b
set CYBERLLM_PROVIDER=ollama
set CYBERLLM_MODEL_OLLAMA=qwen2.5:3b
python -m cyberllm.cli chat "What is CVE-2021-44228?"
```

---

## What it does

| Capability | Status | Notes |
|---|---|---|
| Threat-intel chat with citations | Working | CISA KEV + MITRE ATT&CK indexed; demo and Ollama modes both verified |
| CVE / vulnerability lookup | Working | Exact-identifier index; served offline from the local store |
| Network intrusion detection | Working | RF + LightGBM, macro-F1 ranked, leakage-controlled splits |
| Phishing URL analysis | Working | Fully offline; never fetches the submitted URL |
| Malware feature analysis | Pipeline ready | **No model trained** - needs CIC-MalMem-2022 (manual acquisition) |
| Dataset inventory | Working | Status reconciled against real files on disk |

---

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Module layout, data flow, design decisions |
| [docs/security.md](docs/security.md) | Threat model, controls, limitations, privacy |
| [docs/datasets.md](docs/datasets.md) | Every dataset, licence, acquisition steps |
| [docs/development.md](docs/development.md) | Setup, tests, linting, troubleshooting |
| [docs/coding_standards.md](docs/coding_standards.md) | Shared style rules for both developers |
| [data/README.md](data/README.md) | Dataset directory layout and manifest semantics |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Phase status, verified results, blockers |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Branching, ownership split, review rules |

---

## Commands

```bash
python -m cyberllm.cli datasets list                 # registry + real status
python -m cyberllm.cli datasets download --all       # automated sources only
python -m cyberllm.cli datasets validate --name X
python -m cyberllm.cli datasets preprocess           # threat-intel -> index
python -m cyberllm.cli models train --task nids --data <flows.csv>
python -m cyberllm.cli models evaluate --task nids
python -m cyberllm.cli rag ingest [--rebuild]
python -m cyberllm.cli rag search "CVE-2021-44228"
python -m cyberllm.cli chat "How do I detect lateral movement?"
python -m cyberllm.cli health
python -m cyberllm.cli app [--api | --ui]
```

Add `--json` to any command for machine-readable output.

---

## Configuration

All configuration is environment-driven and optional. Copy `.env.example` to
`.env` and edit. No secret is ever committed; the log filter redacts
credential-shaped strings as a second line of defence.

| Variable | Default | Purpose |
|---|---|---|
| `CYBERLLM_PROVIDER` | `auto` | `auto` / `transformers` / `ollama` / `openai` / `demo` |
| `CYBERLLM_MODEL` | `sshleifer/tiny-gpt2` | Local transformers model id |
| `CYBERLLM_DEMO_MODE` | `true` | Allow startup with no reachable model |
| `CYBERLLM_API_KEY` | unset | Shared secret; **required** for non-localhost binding |
| `NVD_API_KEY` | unset | Raises the NVD rate limit from ~5 to ~50 req/30 s |
| `CYBERLLM_MAX_UPLOAD_BYTES` | 5 MiB | Upload ceiling |

---

## Project layout

```
cyberllm/
  core/        config, logging with redaction, typed errors
  security/    input validation, prompt-injection defence, sandboxing
  datasets/    registry, acquisition, validation, manifest
  rag/         chunking, embeddings, vector store, retrieval, loaders
  llm/         provider-independent interface + demo backend
  detection/   flow preprocessing, training, inference
  phishing/    offline URL features and classifier
  malware/     offline feature classification
  service/     chat orchestration
  api/         FastAPI app and schemas
  app/         Streamlit console
  cli.py       command-line interface
data/          dataset manifest and documentation (data itself is gitignored)
artifacts/     models, vector store, reports (gitignored)
```

---

## Safety posture

- **No malware is ever executed.** Only pre-extracted numeric features are
  accepted; there is no sandbox, loader or emulator in the codebase.
- **Submitted URLs are never fetched.** Phishing analysis is purely lexical -
  no DNS, HTTP, WHOIS or third-party reputation lookup.
- **Retrieved text is data, never instructions.** Documents are scanned for
  prompt injection and wrapped in delimiters before reaching a model.
- **The LLM has no tools.** It cannot execute commands, scan networks or
  modify systems.
- **Bind to localhost only.** The service has no TLS and authentication is
  opt-in; the CLI refuses a non-localhost bind without `CYBERLLM_API_KEY`.

---

## Licence

Code is MIT licensed (see `LICENSE`). Datasets retain their original licences
and attribution requirements - see [docs/datasets.md](docs/datasets.md).

