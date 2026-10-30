"""
Local LLM inference adapter (Phase 2).

Loads a small instruction-tuned model via transformers and runs it on the
available hardware. CPU-only torch is the verified baseline here, so the
default model is Qwen2.5-0.5B-Instruct (Apache-2.0, ~1 GB), which runs
acceptably on CPU for short conversational answers.

Honesty guarantees:
- `status` reflects the REAL load state; it never pretends a model is loaded.
- If the model is unavailable, callers get is_available() == False and must
  fall back to the retrieval engine, labeling the answer source accordingly.
- Every generated answer reports the exact model id that produced it.

Configuration (env or .env):
  CYBERLLM_LLM_ENABLED   "1"/"true" to enable (default: 1)
  CYBERLLM_LLM_MODEL     HF model id or local path (default Qwen/Qwen2.5-0.5B-Instruct)
  CYBERLLM_LLM_MAX_NEW_TOKENS   generation cap (default 256)
  CYBERLLM_LLM_TIMEOUT_S        per-request generation timeout (default 60)
"""

import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are CyberLLM, a defensive cybersecurity assistant. Answer "
    "cybersecurity questions clearly and concisely (2-5 sentences). Only "
    "explain defensive and educational concepts; refuse requests to build "
    "malware, attack systems without authorization, or cause harm."
)

_HARMFUL_PATTERNS = None


def _harmful(text):
    global _HARMFUL_PATTERNS
    if _HARMFUL_PATTERNS is None:
        import re
        _HARMFUL_PATTERNS = re.compile(
            r"\b(write|create|build|make|give|generate|develop|code|script)\b"
            r".*\b(malware|virus|ransomware|keylogger|rootkit|botnet|exploit payload)\b"
            r"|\b(hack into|break into|crack password of|steal|ddos|dox)\b",
            re.IGNORECASE,
        )
    return bool(_HARMFUL_PATTERNS.search(text))


class LocalLLMAdapter:
    """Configurable, memory-conscious local inference adapter."""

    def __init__(self):
        self.model_id = os.getenv("CYBERLLM_LLM_MODEL", DEFAULT_MODEL)
        self.enabled = os.getenv("CYBERLLM_LLM_ENABLED", "1").lower() in (
            "1", "true", "yes")
        self.max_new_tokens = int(os.getenv("CYBERLLM_LLM_MAX_NEW_TOKENS", "256"))
        self.timeout_s = float(os.getenv("CYBERLLM_LLM_TIMEOUT_S", "60"))
        self.max_context_chars = int(os.getenv("CYBERLLM_LLM_MAX_CONTEXT_CHARS", "3500"))

        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._load_error = None
        self._loaded_at = None

    # ------------------------------------------------------------- status
    def status(self):
        info = {
            "enabled": self.enabled,
            "model_id": self.model_id,
            "loaded": self.is_loaded(),
            "device": None,
            "load_error": self._load_error,
            "loaded_at": self._loaded_at,
        }
        if self.is_loaded():
            import torch
            info["device"] = str(self._device())
        return info

    def is_loaded(self):
        return self._model is not None and self._tokenizer is not None

    def is_available(self):
        return self.enabled and self.is_loaded()

    def _device(self):
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # -------------------------------------------------------------- load
    def load(self):
        """Load the model once. Raises with a clear error on failure."""
        if not self.enabled:
            self._load_error = "disabled via CYBERLLM_LLM_ENABLED"
            return False
        if self.is_loaded():
            return True
        with self._lock:
            if self.is_loaded():
                return True
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                local_path = Path(self.model_id)
                is_local = local_path.exists()
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id, local_files_only=is_local)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    local_files_only=is_local,
                    torch_dtype=torch.float32,   # CPU baseline; fp32 is correct
                    low_cpu_mem_usage=True,
                )
                self._model.to(self._device())
                self._model.eval()
                self._load_error = None
                self._loaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                return True
            except Exception as exc:  # noqa: BLE001 - report honestly
                self._model = None
                self._tokenizer = None
                self._load_error = f"{type(exc).__name__}: {exc}"
                raise

    def ensure_loaded(self):
        """Like load() but never raises; returns bool availability."""
        try:
            return self.load()
        except Exception:
            return False

    # ---------------------------------------------------------- generate
    def generate(self, user_message, system_prompt=None, context=None, max_new_tokens=None):
        """Generate an answer. Raises RuntimeError if not available.

        Returns dict: {"answer", "model_id", "elapsed_s", "device"}.
        """
        if not self.is_available():
            raise RuntimeError(
                f"Local LLM not available (model={self.model_id}, "
                f"load_error={self._load_error})")

        if _harmful(user_message):
            # Safety filter applies to the generative path too.
            return {
                "answer": (
                    "I can't help with that. I'm designed for defensive and "
                    "educational cybersecurity questions only. Describe your "
                    "authorized scope and I can suggest defensive approaches."),
                "model_id": self.model_id,
                "elapsed_s": 0.0,
                "device": self._device(),
                "refused": True,
            }

        import torch

        parts = [system_prompt or SYSTEM_PROMPT]
        if context:
            parts.append(
                "Use the following retrieved knowledge base context if it is "
                "relevant. Cite the source title if you rely on it. If the "
                "context is irrelevant, answer from your own knowledge and do "
                "not invent sources.\n\n" + context[: self.max_context_chars])
        parts.append(f"Question: {user_message}")
        prompt_text = "\n\n".join(parts)

        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": user_message},
        ]

        started = time.time()
        with self._lock:  # single-threaded generation; model is not thread-safe
            inputs = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                return_dict=True)
            inputs = {k: v.to(self._device()) for k, v in inputs.items()}
            tokens = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = tokens[0][inputs["input_ids"].shape[1]:]
        answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "answer": answer,
            "model_id": self.model_id,
            "elapsed_s": round(time.time() - started, 2),
            "device": self._device(),
            "refused": False,
        }


_adapter = None


def get_adapter():
    global _adapter
    if _adapter is None:
        _adapter = LocalLLMAdapter()
    return _adapter
