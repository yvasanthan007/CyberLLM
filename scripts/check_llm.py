"""Load the local LLM once and generate a test answer (Phase 9 verification)."""
import json
import sys

sys.path.insert(0, r"D:\CyberLLM")
from cyberllm.llm.adapter import get_adapter

adapter = get_adapter()
try:
    adapter.load()
except Exception as exc:
    print(json.dumps({"loaded": False, "error": str(exc)}))
    sys.exit(1)

info = adapter.status()
result = adapter.generate("What is phishing and how can I protect myself?", max_new_tokens=120)
print(json.dumps({
    "status": info,
    "test_answer": result["answer"][:400],
    "elapsed_s": result["elapsed_s"],
    "device": result["device"],
}, indent=2))
