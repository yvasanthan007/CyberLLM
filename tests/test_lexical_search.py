import pytest

from cyberllm.retrieval.lexical import search_documents, tokenize


@pytest.fixture
def documents():
    return [
        {
            "title": "Cisco SD-WAN vulnerability",
            "text": "Cisco SD-WAN Manager has a security vulnerability. Apply the vendor patch.",
            "source": "cisa_kev",
            "url": "https://example.com/cisco",
            "doc_id": "cve-1",
        },
        {
            "title": "SSH hardening guide",
            "text": "Disable password authentication and use strong SSH keys.",
            "source": "security_guide",
            "url": "https://example.com/ssh",
            "doc_id": "guide-1",
        },
    ]


def test_tokenize_lowercases_text():
    assert tokenize("CVE-2026 Cisco") == ["cve-2026", "cisco"]


def test_relevant_document_ranks_first(documents):
    results = search_documents("Cisco SD-WAN vulnerability", documents)
    assert results
    assert results[0]["doc_id"] == "cve-1"


def test_result_contains_source_attribution(documents):
    result = search_documents("Cisco", documents)[0]
    assert result["source"] == "cisa_kev"
    assert result["url"] == "https://example.com/cisco"


def test_no_match_returns_empty_list(documents):
    assert search_documents("quantum submarine", documents) == []


def test_empty_query_returns_empty_list(documents):
    assert search_documents("   ", documents) == []


def test_invalid_top_k_is_rejected(documents):
    with pytest.raises(ValueError):
        search_documents("Cisco", documents, top_k=0)
