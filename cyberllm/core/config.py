"""Application configuration.

Design goals
------------
* **No secrets in code.** Every credential is read from the environment
  (optionally seeded from a ``.env`` file that is git-ignored).
* **Safe defaults.** With zero configuration the platform starts in a clearly
  labelled *demo mode* rather than failing or pretending a model is running.
* **Testable.** :func:`get_settings` is cached but :func:`reload_settings`
  clears the cache so tests can build isolated settings objects.

Precedence (highest first): environment > YAML file (``configs/cyberllm.yaml``)
> dataclass defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

from .errors import ConfigurationError

try:  # PyYAML is optional; we degrade to env-only config without it.
    import yaml
except ImportError:  # pragma: no cover - exercised only in minimal installs
    yaml = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


ENV_PREFIX = "CYBERLLM_"

#: Default location of the repository root (parent of the ``cyberllm`` package).
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(ENV_PREFIX + name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigurationError(f"{ENV_PREFIX}{name} must be an integer") from exc
def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigurationError(f"{ENV_PREFIX}{name} must be a number") from exc


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem layout. ``root`` is configurable so tests can sandbox I/O."""

    root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "artifacts" / "models"
    reports_dir: Path = PROJECT_ROOT / "artifacts" / "reports"
    vector_dir: Path = PROJECT_ROOT / "artifacts" / "vectorstore"
    db_path: Path = PROJECT_ROOT / "artifacts" / "cyberllm.db"
    manifest_path: Path = PROJECT_ROOT / "data" / "dataset_manifest.csv"

    def ensure(self) -> "PathsConfig":
        """Create every managed directory. Safe to call repeatedly."""
        for directory in (
            self.data_dir, self.raw_dir, self.processed_dir,
            self.models_dir, self.reports_dir, self.vector_dir,
            self.manifest_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def relative(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(path)


@dataclass(frozen=True)
class LLMConfig:
    """Language-model provider configuration.

    ``provider="auto"`` probes for a local transformers model, then Ollama, then
    falls back to demo mode. Demo mode never pretends to be a real model - it
    returns a clearly labelled, retrieval-backed template response.
    """

    provider: str = "auto"           # auto | transformers | ollama | openai | demo
    model_name: str = "sshleifer/tiny-gpt2"
    ollama_model: str = "qwen2.5:3b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    openai_model: str = "gpt-4o-mini"
    openai_api_key_env: str = "OPENAI_API_KEY"
    hf_token_env: str = "HF_TOKEN"
    max_new_tokens: int = 512
    temperature: float = 0.2
    context_chars: int = 6000
    request_timeout: int = 120
    load_timeout: int = 600

    def api_key(self) -> str | None:
        """Resolve the API key from the environment. Never logged, never stored."""
        return os.environ.get(self.openai_api_key_env) or None




@dataclass(frozen=True)
class RAGConfig:
    """Retrieval-augmented generation configuration."""

    collection_name: str = "cyberllm_kb"
    embedding_backend: str = "hashing"   # hashing | sentence-transformers
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    chunk_size: int = 900
    chunk_overlap: int = 120
    top_k: int = 5
    max_context_docs: int = 6
    relevance_floor: float = 0.0
    #: Over-fetch multiplier used before identifier-based re-ranking.
    candidate_multiplier: int = 6


@dataclass(frozen=True)
class SecurityConfig:
    """Defensive limits applied to every untrusted input path."""

    max_upload_bytes: int = 5 * 1024 * 1024
    max_request_bytes: int = 2 * 1024 * 1024
    max_csv_rows: int = 200_000
    max_prompt_chars: int = 16_000
    max_chat_history: int = 50
    api_key: str | None = None      # optional shared secret for non-localhost use
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1")
    cors_origins: tuple[str, ...] = ("http://localhost:8501", "http://127.0.0.1:8501")
    history_retention_days: int = 30
    store_history: bool = True


@dataclass(frozen=True)
class AppConfig:
    """Top-level immutable settings object."""

    env: str = "development"
    debug: bool = False
    demo_mode: bool = True
    log_level: str = "INFO"
    log_format: str = "text"
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict. Secrets are redacted."""
        data = asdict(self)
        data["paths"] = {k: str(v) for k, v in asdict(self.paths).items()}
        data["security"]["api_key"] = "***redacted***" if self.security.api_key else None
        return data

    def redacted_summary(self) -> str:
        """One-line description used by the CLI and the health endpoint."""
        return (
            f"env={self.env} demo_mode={self.demo_mode} "
            f"llm.provider={self.llm.provider} model={self.llm.model_name} "
            f"embed={self.rag.embedding_backend}"
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read the optional YAML config file. Missing file -> empty mapping."""
    if yaml is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"Top level of {path} must be a mapping")
    return data


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge where ``override`` wins."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


_SECTION_CLASSES = {
    "paths": PathsConfig, "llm": LLMConfig,
    "rag": RAGConfig, "security": SecurityConfig,
}


def _build_section(section_cls: type, data: dict[str, Any], label: str) -> Any:
    """Instantiate a config section, rejecting unknown keys loudly."""
    valid = {f.name for f in fields(section_cls)}
    unknown = set(data) - valid
    if unknown:
        raise ConfigurationError(
            f"Unknown keys in [{label}]: {sorted(unknown)} (allowed: {sorted(valid)})"
        )
    return section_cls(**data)


def build_settings(
    config_file: Path | None = None,
    *,
    env_file: Path | None = None,
    load_env: bool = True,
) -> AppConfig:
    """Construct settings from file + environment.

    This is the single place configuration is assembled, which makes the
    precedence rules straightforward to unit test.
    """
    if load_env:
        load_dotenv(env_file or (PROJECT_ROOT / ".env"), override=False)

    cfg_path = Path(config_file) if config_file else (PROJECT_ROOT / "configs" / "cyberllm.yaml")
    file_cfg: dict[str, Any] = _load_yaml(cfg_path) if cfg_path.exists() else {}

    # Environment variables win over the file.
    env_overrides: dict[str, Any] = {}
    for env_name, section, key in (
        ("PROVIDER", "llm", "provider"),
        ("MODEL", "llm", "model_name"),
        ("EMBEDDING_BACKEND", "rag", "embedding_backend"),
    ):
        value = _env(env_name)
        if value:
            env_overrides.setdefault(section, {})[key] = value
    for env_name, key in (("LOG_LEVEL", "log_level"), ("DATA_DIR", "data_dir")):
        value = _env(env_name)
        if value:
            if key == "data_dir":
                env_overrides.setdefault("paths", {})[key] = value
            else:
                env_overrides[key] = value

    merged = _merge(file_cfg, env_overrides)
    merged["env"] = _env("ENV", merged.get("env", "development"))
    merged["debug"] = _env_bool("DEBUG", bool(merged.get("debug", False)))
    merged["demo_mode"] = _env_bool("DEMO_MODE", bool(merged.get("demo_mode", True)))
    merged["log_level"] = str(_env("LOG_LEVEL", merged.get("log_level", "INFO"))).upper()
    merged["log_format"] = merged.get("log_format", "text")
    merged["seed"] = _env_int("SEED", int(merged.get("seed", 42)))

    valid_top = {f.name for f in fields(AppConfig)}
    unknown = set(merged) - valid_top
    if unknown:
        raise ConfigurationError(
            f"Unknown configuration keys: {sorted(unknown)} (allowed: {sorted(valid_top)})"
        )

    for section_name, section_cls in _SECTION_CLASSES.items():
        if section_name in merged and isinstance(merged[section_name], dict):
            merged[section_name] = _build_section(
                section_cls, merged[section_name], section_name
            )

    if isinstance(merged.get("llm"), LLMConfig) and _env("MAX_NEW_TOKENS"):
        merged["llm"].max_new_tokens = _env_int(
            "MAX_NEW_TOKENS", merged["llm"].max_new_tokens
        )
    if isinstance(merged.get("rag"), RAGConfig) and _env("TOP_K"):
        merged["rag"].top_k = _env_int("TOP_K", merged["rag"].top_k)
    if isinstance(merged.get("security"), SecurityConfig):
        sec = merged["security"]
        object.__setattr__(sec, "max_upload_bytes", _env_int("MAX_UPLOAD_BYTES", sec.max_upload_bytes))
        object.__setattr__(sec, "api_key", os.environ.get("CYBERLLM_API_KEY") or sec.api_key)
        if isinstance(sec.cors_origins, list):
            object.__setattr__(sec, "cors_origins", tuple(sec.cors_origins))
        if isinstance(sec.allowed_hosts, list):
            object.__setattr__(sec, "allowed_hosts", tuple(sec.allowed_hosts))
        if _env_bool("NO_HISTORY", False):
            object.__setattr__(sec, "store_history", False)

    app = AppConfig(**merged)
    app.paths.ensure()
    return app


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    """Process-wide cached settings."""
    return build_settings()


def reload_settings(config_file: Path | None = None) -> AppConfig:
    """Clear the cache and rebuild - used by tests and the CLI."""
    get_settings.cache_clear()
    return build_settings(config_file)


__all__ = [
    "AppConfig", "LLMConfig", "PathsConfig", "RAGConfig", "SecurityConfig",
    "build_settings", "get_settings", "reload_settings",
    "PROJECT_ROOT", "PACKAGE_ROOT",
]

