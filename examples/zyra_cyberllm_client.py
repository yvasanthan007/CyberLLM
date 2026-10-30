# CyberLLM API client example for Zyra integration.
# Keeps the real API key in an environment variable / Zyra's own secret store.
# Copy this file into Zyra's source tree and adapt imports as needed.

import os
import re
from typing import Optional

import httpx

# Greeting pattern: when the user just says hi, Zyra greets locally and
# skips the API call entirely (see is_greeting()).
_GREETING_RE = re.compile(r"^\s*(hi+|hello+|hey+|good (morning|evening|afternoon))\b[!,. ]*$", re.I)


def is_greeting(text: str) -> bool:
    return bool(_GREETING_RE.match(text or ""))


class CyberLLMClient:
    """Minimal HTTP client for CyberLLM from Zyra or other desktop assistants."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.base_url = (
            base_url
            or os.getenv("CYBERLLM_BASE_URL", "http://127.0.0.1:8765")
        ).rstrip("/")
        self.api_key = api_key or os.getenv("CYBERLLM_API_KEY", "")
        self.timeout = timeout

    def _headers(self):
        if not self.api_key:
            raise RuntimeError(
                "CYBERLLM_API_KEY is not configured in the environment"
            )
        return {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    def health(self) -> dict:
        response = httpx.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def capabilities(self) -> dict:
        response = httpx.get(f"{self.base_url}/capabilities", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def models(self) -> dict:
        response = httpx.get(
            f"{self.base_url}/models", headers=self._headers(), timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def chat(self, message: str) -> dict:
        """Ask CyberLLM a cybersecurity question. Returns dict with 'answer'.

        Raises httpx errors on failure so the caller can decide what to
        say; this module never invents an answer for a failed request.
        """
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

    def search_knowledge(self, query: str, top_k: int = 3) -> dict:
        response = httpx.post(
            f"{self.base_url}/knowledge/search",
            headers=self._headers(),
            json={"query": query, "top_k": top_k},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    import json

    client = CyberLLMClient()
    user_text = input("You: ")
    if is_greeting(user_text):
        print("Zyra: Welcome, how can I help you?")
    else:
        try:
            result = client.chat(user_text)
            print(f"Zyra: {result['answer']}")
        except httpx.ConnectError:
            print("Zyra: CyberLLM service is not running.")
        except httpx.HTTPStatusError as exc:
            print(f"Zyra: CyberLLM error {exc.response.status_code}")
