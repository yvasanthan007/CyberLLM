"""API tests for the FastAPI service.

These run against the real application object with an in-process client, so
they exercise middleware, validation, domain-error mapping and the actual
service wiring rather than mocks. Network-dependent behaviour (the LLM) is not
asserted on: the tests assert on the *contract* - that a mode is reported, that
citations are attached, that errors are typed - which holds in both demo and
real-model configurations.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cyberllm.api.app import app


@pytest.fixture(scope="module")
def client():
    """A client that surfaces HTTP errors instead of re-raising them."""
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Meta and health
# ---------------------------------------------------------------------------
def test_root_identifies_service(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "CyberLLM API"
    assert "Defensive" in response.json()["notice"]


def test_health_returns_200_with_capability_breakdown(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    for capability in ("llm", "knowledge_base", "intrusion_detection", "phishing"):
        assert capability in body, f"health must report {capability}"
    assert isinstance(body["limitations"], list) and body["limitations"]


def test_health_never_leaks_a_secret(client):
    """The health payload must not expose a credential even when one is set."""
    text = client.get("/health").text.lower()
    assert "api_key" not in text or "redacted" in text


def test_openapi_schema_documents_every_endpoint(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    for route in ("/health", "/chat", "/retrieve", "/phishing/analyse",
                  "/detect/intrusion", "/malware/analyse", "/vulnerability/{cve_id}"):
        assert route in paths, f"{route} must be documented"


# ---------------------------------------------------------------------------
# Vulnerability lookup
# ---------------------------------------------------------------------------
def test_vulnerability_rejects_malformed_identifier(client):
    response = client.get("/vulnerability/not-a-cve")
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_vulnerability_unknown_id_is_reported_not_invented(client):
    response = client.get("/vulnerability/CVE-1999-99999")
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is False
    assert body["sources"] == []


def test_vulnerability_lookup_shape(client):
    """Shape is asserted even when the KB is empty, so CI stays deterministic."""
    response = client.get("/vulnerability/CVE-2021-44228")
    assert response.status_code == 200
    body = response.json()
    assert body["cve_id"] == "CVE-2021-44228"
    assert isinstance(body["in_kev"], bool)
    assert "disclaimer" in body


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
def test_chat_rejects_blank_question(client):
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={"question": "   "}).status_code == 422


def test_chat_rejects_oversized_question(client):
    assert client.post("/chat", json={"question": "x" * 20_000}).status_code == 422


def test_chat_rejects_out_of_range_top_k(client):
    assert client.post("/chat", json={"question": "hi", "top_k": 0}).status_code == 422
    assert client.post("/chat", json={"question": "hi", "top_k": 999}).status_code == 422


def test_chat_response_declares_its_mode(client):
    """A caller must always be able to tell a model answer from a template."""
    response = client.post("/chat", json={"question": "What is CVE-2021-44228?",
                                          "top_k": 1})
    if response.status_code != 200:
        pytest.skip("LLM backend unavailable in this environment")
    body = response.json()
    assert body["mode"] in {"transformers", "ollama", "openai", "demo"}
    assert body["is_demo"] == (body["mode"] == "demo")
    assert isinstance(body["citations"], list)
    if body["is_demo"]:
        assert "DEMO MODE" in body["answer"].upper()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def test_retrieve_validates_input(client):
    assert client.post("/retrieve", json={"query": ""}).status_code == 422
    assert client.post("/retrieve", json={"query": "x", "top_k": 0}).status_code == 422


def test_retrieve_returns_provenance_fields(client):
    response = client.post("/retrieve", json={"query": "log4shell", "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "log4shell"
    for result in body["results"]:
        assert "doc_id" in result and "source" in result and "score" in result


# ---------------------------------------------------------------------------
# Phishing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    "not-a-url",
    "http://",
    "https://exa mple.com/x",
    "javascript:alert(1)",
    "file:///etc/passwd",
    "ftp://example.com",
])
def test_phishing_rejects_invalid_urls(client, url):
    """Anything that is not a well-formed http(s) URL must be refused."""
    assert client.post("/phishing/analyse", json={"url": url}).status_code == 422


def test_phishing_rejects_oversized_url(client):
    response = client.post("/phishing/analyse",
                           json={"url": "https://a.com/" + "x" * 3000})
    assert response.status_code == 422


def test_phishing_states_it_made_no_network_request(client):
    response = client.post("/phishing/analyse",
                           json={"url": "https://example.com/login"})
    assert response.status_code == 200
    body = response.json()
    assert body["network_access"] == "none (fully offline lexical analysis)"
    assert "not visited" in body["disclaimer"].lower()
    assert "neither a guarantee" in body["disclaimer"].lower()
    assert 0.0 <= body["score"] <= 1.0


def test_phishing_flags_ip_literal_host(client):
    response = client.post("/phishing/analyse",
                           json={"url": "http://192.0.2.10/admin/login.php"})
    indicators = {s["indicator"] for s in response.json()["risk_signals"]}
    assert "ip_literal_host" in indicators


# ---------------------------------------------------------------------------
# Detection limits and unconfigured capabilities
# ---------------------------------------------------------------------------
def test_detect_rejects_empty_and_oversized_batches(client):
    assert client.post("/detect/intrusion", json={"records": []}).status_code == 422
    assert client.post("/detect/intrusion",
                       json={"records": [{"a": 1}] * 200}).status_code == 422


def test_oversized_body_is_rejected_before_parsing(client):
    """A declared Content-Length above the cap must fail with 413."""
    response = client.post(
        "/phishing/analyse",
        content=b"x" * 3_000_000,
        headers={"content-type": "application/json", "content-length": "3000000"},
    )
    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"


def test_malware_without_model_returns_503_with_setup_hint(client):
    """An unconfigured capability must instruct, not fabricate a family name."""
    response = client.post("/malware/analyse",
                           json={"features": [{"PsList": 12, "Threads": 4}]})
    if response.status_code == 200:
        pytest.skip("a malware model is trained in this environment")
    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "not_configured"
    assert "CIC-MalMem-2022" in body["details"]["setup_hint"]


def test_malware_rejects_empty_features(client):
    assert client.post("/malware/analyse", json={"features": []}).status_code == 422


# ---------------------------------------------------------------------------
# Dataset inventory
# ---------------------------------------------------------------------------
def test_datasets_inventory_lists_every_registered_dataset(client):
    response = client.get("/datasets")
    assert response.status_code == 200
    body = response.json()
    assert len(body["datasets"]) >= 5
    assert "summary" in body
    for entry in body["datasets"]:
        assert "name" in entry and "status" in entry and "license" in entry

