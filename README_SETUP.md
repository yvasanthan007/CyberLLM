# CyberLLM Setup & Operation Guide
Updated: 2026-10-30. Windows PowerShell, project root `D:\CyberLLM`.

## Verified baseline
- Python 3.14.2 in `.venv`; torch `2.14.1+cpu`; CUDA unavailable to torch.
- RTX 3050 Laptop GPU, 6 GB VRAM present; 16 GB RAM.
- Base model `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0) downloaded in the
  existing HF cache at `D:\HuggingFace`; CPU loading and generation verified.
- Fine-tuning is gated and will not run on this CPU-only environment.

## Commands
```powershell
Set-Location D:\CyberLLM
.\.venv\Scripts\Activate.ps1
```

1. Hardware and dependency check:
   ```powershell
   .\CyberLLM.ps1 status
   ```
2. Validate and prepare conversational splits (separate from phishing data):
   ```powershell
   .\CyberLLM.ps1 dataset
   ```
3. Download/load selected model (safe to repeat; Hugging Face cache resumes):
   ```powershell
   .\CyberLLM.ps1 download-model
   ```
4. Fine-tuning (currently reports blocker, exits nonzero; no false success):
   ```powershell
   .\CyberLLM.ps1 train --smoke
   .\CyberLLM.ps1 train --full
   ```
   For CUDA training, use a separate Python 3.11/3.12 environment with a
   matching CUDA torch build; do not replace the API environment blindly.
5. Evaluation (conversation + phishing + model load/generation):
   ```powershell
   .\CyberLLM.ps1 evaluate
   ```
6. Start API (model loads once at server startup):
   ```powershell
   .\CyberLLM.ps1 serve
   ```
   Default: `http://127.0.0.1:8765`. Override with Uvicorn `--host`/`--port`.
7. Live API checks (in another activated terminal):
   ```powershell
   .\CyberLLM.ps1 test-api
   ```
8. Zyra: configure its **backend/main-process** environment with
   `CYBERLLM_BASE_URL=http://127.0.0.1:8765` and its API secret as
   `CYBERLLM_API_KEY`, copy `examples/zyra_cyberllm_client.py` into that layer,
   then launch Zyra. Full instructions in `ZYRA_INTEGRATION.md`.

## Configuration
- Real API key remains in the existing `.env` and is not documented here.
- `.env.example` contains safe placeholders and LLM settings.
- Model settings: `CYBERLLM_LLM_ENABLED`, `CYBERLLM_LLM_MODEL`,
  `CYBERLLM_LLM_MAX_NEW_TOKENS`, `CYBERLLM_LLM_TIMEOUT_S`,
  `CYBERLLM_LLM_MAX_CONTEXT_CHARS`.
- Default HF cache: `D:\HuggingFace` (override `HF_HOME`).

## Artifacts and reports
- Original conversation splits: `data/conversation/splits/` (57/7/8).
- Phishing classifier: `models/phishing_url_model.joblib` (preserved).
- Existing KB: `artifacts/vectorstore/cyberllm_kb.jsonl` (2,427 docs; reused).
- No fine-tuning adapters exist; no fine-tuning completed.
- Test/eval results: `TEST_REPORT.md`, `EVALUATION_REPORT.md`.
