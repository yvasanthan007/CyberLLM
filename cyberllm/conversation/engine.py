"""
CyberLLM conversational engine.

Hardware reality: the installed PyTorch is CPU-only and the RTX 3050 6GB
GPU cannot host a full fine-tuned LLM reliably within this project's
constraints. Rather than shipping a fake "trained model", this engine
provides an honest hybrid conversational capability:

1. Semantic-router style retrieval over the curated conversational
   dataset (data/conversation/processed) - TF-IDF cosine similarity.
2. Fallback lexical retrieval over the 2,427-document cybersecurity
   knowledge base (artifacts/vectorstore/cyberllm_kb.jsonl).
3. A safety/quality filter layer (refusals for clearly harmful asks).

This is RAG, not a pretrained LLM. It is labeled as such in API
responses so callers (like Zyra) know what they are talking to.

A separate optional path exists for LoRA/QLoRA fine-tuning once a CUDA
build of PyTorch is installed (see scripts/train_conversations.py).
"""

import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONVERSATION_DATA = ROOT / "data" / "conversation" / "processed" / "conversations_clean.jsonl"
KB_PATH = ROOT / "artifacts" / "vectorstore" / "cyberllm_kb.jsonl"

ENGINE_ID = "cyberllm-rag-hybrid-v1"
ENGINE_DESCRIPTION = (
    "Hybrid retrieval: TF-IDF over curated conversational cybersecurity "
    "pairs + BM25 lexical fallback over a 2,427-doc cybersecurity KB. "
    "Not a pretrained LLM; no generative weights."
)

# Words that indicate a request we should decline
HARMFUL_PATTERNS = re.compile(
    r"\b(write|create|build|make|give|generate|develop|code|script)\b.*\b(malware|virus|ransomware|keylogger|rootkit|botnet|exploit payload)\b"
    r"|\b(hack into|break into|crack password of|steal|ddos|dox)\b",
    re.IGNORECASE,
)

REFUSAL_TEXT = (
    "I can't help with that. I'm designed for defensive and educational "
    "cybersecurity questions only — things like explaining how attacks work "
    "conceptually, describing defenses, or incident response. If you're "
    "working on authorized security testing, describe your authorized scope "
    "and I can suggest defensive or assessment approaches."
)


def _tokenize(text):
    return re.findall(r"[a-z0-9]+(?:[-_'][a-z0-9]+)*", str(text).lower())


@lru_cache(maxsize=1)
def load_conversation_data():
    records = []
    if CONVERSATION_DATA.is_file():
        with open(CONVERSATION_DATA, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


@lru_cache(maxsize=1)
def load_kb_documents():
    documents = []
    if KB_PATH.is_file():
        with open(KB_PATH, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("text"):
                    documents.append(record)
    return documents


class ConversationEngine:
    """Retrieval-based conversational cybersecurity assistant."""

    def __init__(self):
        self.records = load_conversation_data()
        self.kb_documents = load_kb_documents()
        self._build_vectors()

    def _build_vectors(self):
        from collections import Counter
        import math

        # Build IDF over conversation corpus
        self.record_tokens = []
        document_frequency = Counter()
        for record in self.records:
            terms = _tokenize(record["instruction"]) + _tokenize(record.get("category", ""))
            self.record_tokens.append(Counter(terms))
            document_frequency.update(set(terms))

        n_docs = len(self.records) or 1
        self.idf = {
            term: math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in document_frequency.items()
        }
        self.record_vectors = [
            self._vectorize(counter) for counter in self.record_tokens
        ]

    def _vectorize(self, counter):
        import math
        vector = {}
        norm = 0.0
        for term, freq in counter.items():
            weight = freq * self.idf.get(term, 1.0)
            vector[term] = weight
            norm += weight * weight
        norm = math.sqrt(norm) or 1.0
        return {term: w / norm for term, w in vector.items()}

    def _query_vector(self, text):
        from collections import Counter
        return self._vectorize(Counter(_tokenize(text)))

    @staticmethod
    def _cosine(a, b):
        score = 0.0
        if len(a) > len(b):
            a, b = b, a
        for term, weight in a.items():
            score += weight * b.get(term, 0.0)
        return score

    def _search_conversations(self, message, top_k=3):
        query = self._query_vector(message)
        scored = []
        for index, vector in enumerate(self.record_vectors):
            score = self._cosine(query, vector)
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {"score": score, "record": self.records[index]}
            for score, index in scored[:top_k]
        ]

    def _search_kb(self, message, top_k=2):
        try:
            from cyberllm.retrieval.lexical import load_documents, search_documents
            documents = self.kb_documents
            if not documents:
                return []
            results = search_documents(message, documents, top_k=top_k)
            return results
        except Exception:
            return []

    def is_harmful(self, message):
        return bool(HARMFUL_PATTERNS.search(message))

    def respond(self, message):
        """Return (answer, metadata) for a user message."""
        message = (message or "").strip()
        if not message:
            return (
                "Please ask a cybersecurity question — for example, "
                "'What is ethical hacking?' or 'How does phishing work?'",
                {"source": "fallback", "confidence": 0.0},
            )

        if self.is_harmful(message):
            return REFUSAL_TEXT, {
                "source": "safety_filter",
                "confidence": 1.0,
                "refused": True,
            }

        matches = self._search_conversations(message)
        if matches and matches[0]["score"] >= 0.30:
            # Guard: cosine of 0.30 on tiny questions like "who won?" can
            # spuriously match greetings. Require the match to look
            # topically related (share a meaningful content term) OR have
            # a strong score.
            query_terms = set(_tokenize(message))
            matched_terms = set(_tokenize(matches[0]["record"]["instruction"]))
            overlap = query_terms & matched_terms - {"what", "is", "the", "a", "of", "who", "how", "do", "i", "you", "are", "can", "my", "in", "to", "for", "and"}
            strong = matches[0]["score"] >= 0.55 and len(overlap) >= 2
            weak_ok = matches[0]["score"] >= 0.30 and len(overlap) >= 1
            if not (strong or weak_ok):
                matches = []
        if matches and matches[0]["score"] >= 0.30:
            best = matches[0]["record"]
            answer = best["output"]
            related = [
                m["record"]["instruction"] for m in matches[1:3]
                if m["score"] >= 0.15
            ]
            metadata = {
                "source": "conversation_dataset",
                "confidence": round(min(matches[0]["score"] / 0.8, 1.0), 3),
                "matched_question": best["instruction"],
                "category": best.get("category", "general"),
                "related_questions": related,
            }
            return answer, metadata

        kb_results = self._search_kb(message, top_k=2)
        if kb_results and kb_results[0]["score"] > 1.5:
            top = kb_results[0]
            snippet = top["text"][:600].strip()
            answer = (
                f"Here's what I found in the cybersecurity knowledge base:\n\n"
                f"{snippet}"
            )
            if top.get("source"):
                answer += f"\n\n(Source: {top['source']})"
            metadata = {
                "source": "knowledge_base",
                "confidence": 0.6,
                "retrieved": {
                    "title": top.get("title", ""),
                    "doc_id": top.get("doc_id", ""),
                    "score": round(top["score"], 3),
                },
            }
            return answer, metadata

        return (
            "I don't have a confident answer for that yet. My curated "
            "knowledge covers ethical hacking, networking, phishing, "
            "malware, passwords, encryption, web security, Linux security "
            "tools, incident response, and privacy/risk. Try asking about "
            "one of those, or rephrase your question.",
            {"source": "fallback", "confidence": 0.0},
        )


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = ConversationEngine()
    return _engine
