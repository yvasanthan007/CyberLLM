# CyberLLM Test Report
Run date: 2026-10-30. Detailed metrics: `EVALUATION_REPORT.md`.

## Results
| Test group | Result |
|---|---|
| Existing regression suite (`python -m pytest tests phishing -q`) | **17 passed**, 3 deprecation warnings |
| Local model loading + CPU generation | **PASS** — Qwen2.5-0.5B loaded; on-topic answer generated |
| Live API checks (`python scripts/test_api_live.py`) | **8/8 PASS** |
| Conversation held-out eval (8 test items) | 8/8 by the existing retrieval engine; limited synthetic test set |
| Phishing regression eval (47,074 held-out URLs) | Accuracy 0.9961, phishing F1 0.9955 |
| Fine-tuning smoke run | **BLOCKED honestly** — CPU-only torch, no CUDA build; nothing trained |
| Zyra client | **PASS** — tested live: greeting + substantive question |
| Python syntax (`py_compile`) | **PASS** — adapter, API, and test scripts |
| PowerShell script parser | **PASS** — `CyberLLM.ps1` |

## Live API checks
`GET /health`, missing-auth `POST /chat` -> 401, LLM/RAG chat,
phishing prediction, knowledge search, `/capabilities`, `/models`, and empty
chat -> 422 all passed on `127.0.0.1:8765`.

## Asset integrity
Compared the pre-run repository snapshot with current files:

- Phishing dataset `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv`:
  SHA-256 `A236549CD369CD80BD478FF8E1779CBF44C58D5C3F79F7A51A1ADBED7D06D1C6`
- Phishing model `models/phishing_url_model.joblib`:
  SHA-256 `FF98F78E0E52C22A1888D2859820B0A9BC524B2E72B3AC8DBC79C9F832845342`
- KB: 2,427 documents. Original data, classifier, API-key auth, and routes
  were preserved. Modified adapter and API files were backed up under `backups/`.

## Limitations
- CPU generation is slower than GPU; a representative 120-token response took
 11.67 s. Keep generation token limits conservative.
- Fine-tuning was not performed (PyTorch is `2.14.1+cpu`).
- Zyra's source repository is not available in this workspace; only the
  reusable client/integration guide was tested.
- Conversation held-out metrics describe 8 synthetic items only; not evidence
  of broad conversational intelligence.
