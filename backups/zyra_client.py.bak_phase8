# CyberLLM API client example for Zyra integration.
# Keeps the real API key in an environment variable / Zyra's own secret store.
# Copy this file into Zyra's source tree and adapt imports as needed.

import os
from typing import Optional

import httpx


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
        """Ask CyberLLM a cybersecurity question. Returns dict with 'answer'."""
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
    print(json.dumps(client.chat("What is ethical hacking?"), indent=2))
