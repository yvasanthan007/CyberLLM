# Zyra Integration Guide

Updated 2026-10-30. **Note:** the Zyra desktop repository is NOT present in
this workspace (searched `D:\` and the user profile). Integration is fully
documented and the tested client is ready to copy; see "Status" below.

## Status

- Tested client: `examples/zyra_cyberllm_client.py` — verified live on
  2026-10-30 against a running CyberLLM server:
  - `Hi Zyra` → local greeting (no API call). Update its greeting string to
    "Welcome, how can I help you?" if you prefer that wording.
  - Substantive questions → `/chat`; the LLM answer is only spoken/displayed
    after a successful HTTP 200 with a valid JSON body. Failed requests yield
    a clear error message, never a fabricated answer.
- Copy `examples/zyra_cyberllm_client.py` into Zyra's **backend/main-process**
  layer (not the frontend). Set `CYBERLLM_BASE_URL` and `CYBERLLM_API_KEY` in
  Zyra's process environment or its own secret store.

## Architecture

```
┌──────────────┐        HTTP POST /chat         ┌──────────────┐
│    Zyra      │ ─────────────────────────────► │   CyberLLM   │
│  (desktop)   │ ◄───────────────────────────── │  (FastAPI)   │
└──────────────┘        JSON response           └──────────────┘
       │                                              │
       └─► display text                               └─► /chat (retrieval engine)
       └─► optional TTS                                   /phishing/predict
                                                          /knowledge/search
```

## 1. Start CyberLLM (backend)

```powershell
Set-Location D:\CyberLLM
.\.venv\Scripts\Activate.ps1
.\CyberLLM.ps1 serve
```

The local model loads once at API startup (first start after download takes
longer). Verify using `http://127.0.0.1:8765/health`; successful status
contains `loaded (Qwen/Qwen2.5-0.5B-Instruct on cpu)` on the verified hardware.

Keep this running in a separate terminal (or install as a service).

## 2. Configure Zyra

Zyra should store two settings — **never hardcode the API key in frontend
source code**. Use an environment variable, a config file outside version
control, or the OS keychain:

| Setting | Value |
|---|---|
| `CYBERLLM_BASE_URL` | `http://127.0.0.1:8765` (or LAN IP if remote) |
| `CYBERLLM_API_KEY` | copied from CyberLLM's `.env` file |

Example `.env` for Zyra (do NOT commit this file):

```
CYBERLLM_BASE_URL=http://127.0.0.1:8765
CYBERLLM_API_KEY=your-key-here
```

## 3. Python API Client (drop-in for Zyra)

Save as `zyra_cyberllm_client.py` inside Zyra's codebase:

```python
"""Zyra client for the CyberLLM cybersecurity API."""

import os
from typing import Optional
import httpx


class CyberLLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.base_url = (base_url or os.getenv("CYBERLLM_BASE_URL", "http://127.0.0.1:8765")).rstrip("/")
        self.api_key = api_key or os.getenv("CYBERLLM_API_KEY", "")
        self.timeout = timeout

    def _headers(self):
        if not self.api_key:
            raise RuntimeError("CYBERLLM_API_KEY is not configured")
        return {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    def health(self) -> dict:
        response = httpx.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def capabilities(self) -> dict:
        response = httpx.get(f"{self.base_url}/capabilities", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def chat(self, message: str) -> dict:
        """Send a cybersecurity question; returns dict with 'answer', 'confidence', 'source'."""
        response = httpx.post(
            f"{self.base_url}/chat",
            headers=self._headers(),
            json={"message": message},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def check_phishing_url(self, url: str) -> dict:
        response = httpx.post(
            f"{self.base_url}/phishing/predict",
            headers=self._headers(),
            json={"url": url},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
```

## 4. Wiring into Zyra's message loop

When Zyra detects a cybersecurity-related intent (or as a fallback when its
main LLM is unsure about a security topic):

```python
from zyra_cyberllm_client import CyberLLMClient

client = CyberLLMClient()

def handle_user_message(user_text: str, tts_callback=None):
    try:
        result = client.chat(user_text)
    except httpx.ConnectError:
        return "CyberLLM service is not running."
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            return "CyberLLM API key is invalid."
        return f"CyberLLM error: {exc.response.status_code}"

    answer = result["answer"]

    # Optional: suppress low-confidence answers rather than reading garbage
    if result.get("confidence", 0) < 0.2 and result.get("source") == "fallback":
        return "I couldn't find a reliable cybersecurity answer for that."

    # Display
    print(f"CyberLLM: {answer}")

    # Optional: read aloud through Zyra's existing TTS
    if tts_callback:
        tts_callback(answer)

    return answer
```

## 5. Security Best Practices for Zyra

- **Never** put the API key in Zyra's frontend/UI code, JS bundles, or repo.
- Keep CyberLLM bound to `127.0.0.1` unless Zyra runs on another machine;
  if remote, use HTTPS (reverse proxy with TLS) and a strong key.
- Rotate `CYBERLLM_API_KEY` periodically; both sides read it from env.
- CyberLLM already refuses malicious requests (`refused: true`); Zyra can
  surface that gracefully rather than hiding it.

## 6. Verifying the Integration

Once CyberLLM is running, test from Zyra's environment:

```powershell
python -c "from zyra_cyberllm_client import CyberLLMClient; print(CyberLLMClient().chat('What is ethical hacking?')['answer'])"
```

Expected output starts with: *"Ethical hacking is the authorized process of
identifying and assessing security vulnerabilities..."*

## 7. TTS

Zyra should pass `answer` (not the whole JSON) to its TTS engine. The text
is plain English, ~1–3 sentences, so it reads naturally. Skip TTS when
`refused: true` or `confidence < 0.2` — those responses are meant for
display/logging rather than speech.
