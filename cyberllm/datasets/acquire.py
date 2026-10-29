"""HTTP acquisition with retries, timeouts, resume and integrity checks.

Design constraints
------------------
* **No fabricated URLs.** Downloads only ever use URLs declared in the
  registry (:mod:`cyberllm.datasets.registry`).
* **Resumable.** ``.part`` files plus ``Range`` headers let an interrupted
  multi-hundred-MB fetch continue instead of restarting.
* **Bounded.** Connect/read timeouts, a retry budget with exponential backoff,
  and a hard cap on bytes read protect against a hostile or misconfigured host.
* **Auditable.** Every attempt (success or failure) is returned as a
  :class:`DownloadResult` so the caller records it in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from ..core.errors import DatasetDownloadError
from ..core.logging_config import get_logger

logger = get_logger("datasets.acquire")

DEFAULT_TIMEOUT = 60          # seconds, per request
DEFAULT_RETRIES = 3
CHUNK_SIZE = 1 << 20          # 1 MiB


@dataclass
class DownloadResult:
    """Outcome of a single download attempt (possibly after retries)."""

    url: str
    path: Path | None = None
    ok: bool = False
    status: str = "pending"
    bytes_downloaded: int = 0
    sha256: str | None = None
    attempts: int = 0
    elapsed_seconds: float = 0.0
    http_status: int | None = None
    error: str | None = None
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    headers: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path) if self.path else None
        # Headers can carry cookies/tokens; keep only safe metadata.
        data["headers"] = {
            k: v for k, v in self.headers.items()
            if k.lower() in {"content-type", "content-length", "last-modified", "etag"}
        }
        return data


def sha256_file(path: Path, chunk: int = CHUNK_SIZE) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _requests_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    })
    return session


def download_file(
    url: str,
    dest: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    max_bytes: int = 4 * 1024 ** 3,
    headers: dict[str, str] | None = None,
    expected_sha256: str | None = None,
    progress: Callable[[int, int], None] | None = None,
    session: requests.Session | None = None,
) -> DownloadResult:
    """Download ``url`` to ``dest`` with resume and retry support.

    ``url`` must come from the registry - this function never invents an
    endpoint. Resumable via ``Range`` + a ``.part`` sibling file, bounded by
    ``timeout``/``retries``/``max_bytes``, and checksum-verified when the
    publisher supplies a digest.
    """
    if not url.lower().startswith(("http://", "https://")):
        raise DatasetDownloadError(f"Refusing non-HTTP(S) URL: {url}")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    result = DownloadResult(url=url, path=dest)
    started = time.monotonic()
    session = session or _requests_session("CyberLLM/0.1 (defensive security research)")

    for attempt in range(1, retries + 2):
        result.attempts = attempt
        try:
            resume_from = part.stat().st_size if part.exists() else 0
            if resume_from > max_bytes:
                part.unlink(missing_ok=True)
                resume_from = 0

            request_headers = dict(headers or {})
            mode = "wb"
            if resume_from:
                request_headers["Range"] = f"bytes={resume_from}-"
                mode = "ab"
                logger.info("resuming %s at %d bytes", dest.name, resume_from)

            with session.get(
                url, headers=request_headers, stream=True, timeout=timeout
            ) as response:
                result.http_status = response.status_code
                result.headers = dict(response.headers)

                if response.status_code == 416:  # range not satisfiable -> done
                    part.replace(dest)
                    result.ok, result.status = True, "downloaded"
                    break
                if response.status_code in (401, 403):
                    raise DatasetDownloadError(
                        f"Access denied (HTTP {response.status_code}) for {url}. "
                        "This dataset requires manual acquisition - run "
                        "`datasets list` for the exact steps.",
                        details={"http_status": response.status_code, "url": url},
                    )
                if response.status_code == 429:
                    raise DatasetDownloadError(
                        "Rate limited (HTTP 429) - reduce request frequency or "
                        "supply an API key.",
                        details={"http_status": 429},
                    )
                if response.status_code >= 400:
                    raise DatasetDownloadError(
                        f"HTTP {response.status_code} for {url}",
                        details={"http_status": response.status_code},
                    )

                # A server that ignores Range restarts the file.
                if resume_from and response.status_code != 206:
                    logger.warning("server ignored Range header; restarting download")
                    mode, resume_from = "wb", 0

                declared = response.headers.get("Content-Length")
                total = (int(declared) + resume_from) if (declared or "").isdigit() else 0
                if total > max_bytes:
                    raise DatasetDownloadError(
                        f"Remote file is {total:,} bytes which exceeds the "
                        f"{max_bytes:,} byte cap.",
                        details={"size_bytes": total, "max_bytes": max_bytes},
                    )

                written = resume_from
                with part.open(mode) as handle:
                    for block in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not block:
                            continue
                        written += len(block)
                        if written > max_bytes:
                            raise DatasetDownloadError(
                                f"Download exceeded the {max_bytes:,} byte cap."
                            )
                        handle.write(block)
                        if progress:
                            progress(written, total)
                part.replace(dest)

            digest = sha256_file(dest)
            if expected_sha256 and digest.lower() != expected_sha256.lower():
                dest.unlink(missing_ok=True)
                raise DatasetDownloadError(
                    "Checksum mismatch - the downloaded file was discarded.",
                    details={"expected": expected_sha256, "actual": digest},
                )

            result.ok = True
            result.status = "downloaded"
            result.bytes_downloaded = dest.stat().st_size
            result.sha256 = digest
            logger.info("downloaded %s (%s bytes)", dest.name, f"{result.bytes_downloaded:,}")
            break

        except (requests.RequestException, DatasetDownloadError, OSError) as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.ok = False
            result.status = "failed"
            logger.warning("attempt %d/%d failed for %s: %s", attempt, retries + 1, url, exc)
            if attempt <= retries:
                backoff = min(2 ** attempt, 30)
                logger.info("retrying in %ds", backoff)
                time.sleep(backoff)
            else:
                result.elapsed_seconds = time.monotonic() - started
                logger.error("giving up on %s after %d attempts", url, attempt)
                return result

    result.elapsed_seconds = time.monotonic() - started
    if result.ok and result.sha256 is None and result.path and result.path.exists():
        result.sha256 = sha256_file(result.path)
        result.bytes_downloaded = result.path.stat().st_size
    return result




def fetch_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    max_bytes: int = 512 * 1024 * 1024,
) -> tuple[Any, DownloadResult]:
    """Fetch a JSON document into memory with the same retry policy.

    Used for the NVD API where results are paginated and we do not want to
    stream a single response to disk.
    """
    result = DownloadResult(url=url, path=None)
    started = time.monotonic()
    session = _requests_session("CyberLLM/0.1 (defensive security research)")

    for attempt in range(1, retries + 2):
        result.attempts = attempt
        try:
            with session.get(
                url, params=params, headers=headers, stream=True, timeout=timeout
            ) as response:
                result.http_status = response.status_code
                if response.status_code in (401, 403):
                    raise DatasetDownloadError(
                        f"Access denied (HTTP {response.status_code}). Provide a valid "
                        "API key in the environment and retry.",
                        details={"http_status": response.status_code},
                    )
                if response.status_code == 404:
                    raise DatasetDownloadError(
                        "Resource not found (HTTP 404). Verify the endpoint URL.",
                        details={"http_status": 404},
                    )
                if response.status_code == 429:
                    raise DatasetDownloadError(
                        "Rate limited (HTTP 429). NVD allows ~5 requests/30s without an "
                        "API key and 50/30s with one.",
                        details={"http_status": 429},
                    )
                response.raise_for_status()

                buffer = bytearray()
                for block in response.iter_content(chunk_size=CHUNK_SIZE):
                    buffer.extend(block)
                    if len(buffer) > max_bytes:
                        raise DatasetDownloadError(
                            f"Response exceeded the {max_bytes:,} byte cap."
                        )

            payload = json.loads(buffer.decode("utf-8", errors="replace"))
            result.ok = True
            result.status = "downloaded"
            result.bytes_downloaded = len(buffer)
            result.sha256 = hashlib.sha256(bytes(buffer)).hexdigest()
            result.elapsed_seconds = time.monotonic() - started
            return payload, result

        except (requests.RequestException, DatasetDownloadError, ValueError) as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.status = "failed"
            logger.warning("fetch attempt %d/%d failed: %s", attempt, retries + 1, exc)
            if attempt <= retries:
                time.sleep(min(2 ** attempt, 30))
            else:
                result.elapsed_seconds = time.monotonic() - started
                raise DatasetDownloadError(
                    f"Failed to fetch {url}: {exc}",
                    details={"attempts": attempt, "error": str(exc)},
                ) from exc

    raise DatasetDownloadError(f"Failed to fetch {url}")  # pragma: no cover


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write rows as UTF-8 JSON Lines, creating parents as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Read a JSON Lines file, skipping malformed lines with a warning."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("skipping malformed line %d in %s: %s", lineno, path.name, exc)
            if limit and len(rows) >= limit:
                break
    return rows


__all__ = [
    "DownloadResult", "download_file", "fetch_json", "sha256_file",
    "write_jsonl", "read_jsonl",
]

