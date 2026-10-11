"""Dataset manifest: the on-disk record of what is actually present.

``data/dataset_manifest.csv`` is a first-class artefact, not a log. It records
for every registered dataset: source URL, retrieval date, licence, version,
size, checksum, acquisition status and preprocessing status. Both the CLI and
the API read it, so the inventory a user sees is always derived from real
filesystem state rather than optimism.

The CSV is intentionally human-readable and diff-able so it can live in Git
(sizes/checksums only - never the data itself).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..core.errors import DatasetValidationError
from ..core.logging_config import get_logger
from .registry import DATASETS, DatasetStatus

logger = get_logger("datasets.manifest")

MANIFEST_COLUMNS = [
    "name", "kind", "status", "access", "license", "version", "year",
    "source_url", "download_url", "retrieved_at", "file_path",
    "file_size_bytes", "file_size_mb", "sha256", "row_count", "column_count",
    "preprocess_status", "preprocess_path", "preprocessed_at", "error",
    "approx_size_mb", "priority", "notes",
]

_INT_FIELDS = {"file_size_bytes", "row_count", "column_count", "priority"}
_FLOAT_FIELDS = {"file_size_mb", "approx_size_mb"}


def _coerce_row(payload: dict[str, Any]) -> "ManifestRow":
    """Cast CSV strings into the types declared on :class:`ManifestRow`."""
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        if value in (None, ""):
            clean[key] = 0 if key in _INT_FIELDS | _FLOAT_FIELDS else ""
            continue
        if key in _INT_FIELDS:
            try:
                clean[key] = int(float(value))
            except (TypeError, ValueError):
                clean[key] = 0
        elif key in _FLOAT_FIELDS:
            try:
                clean[key] = float(value)
            except (TypeError, ValueError):
                clean[key] = 0.0
        else:
            clean[key] = value
    return ManifestRow(**clean)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ManifestRow:
    """One dataset's acquisition state."""

    name: str
    kind: str = ""
    status: str = DatasetStatus.REGISTERED.value
    access: str = "public"
    license: str = ""
    version: str = "unknown"
    year: str = ""
    source_url: str = ""
    download_url: str = ""
    retrieved_at: str = ""
    file_path: str = ""
    file_size_bytes: int = 0
    file_size_mb: float = 0.0
    sha256: str = ""
    row_count: int = 0
    column_count: int = 0
    preprocess_status: str = "not_started"
    preprocess_path: str = ""
    preprocessed_at: str = ""
    error: str = ""
    approx_size_mb: float = 0.0
    priority: int = 2
    notes: str = ""

    @classmethod
    def from_spec(cls, spec: Any) -> "ManifestRow":
        """Seed a row from the registry declaration (pre-acquisition state)."""
        return cls(
            name=spec.name,
            kind=spec.kind.value,
            status=(
                DatasetStatus.SYNTHETIC.value if spec.synthetic
                else DatasetStatus.MANUAL.value if spec.access == "manual"
                else DatasetStatus.DOWNLOADABLE.value
            ),
            access=spec.access,
            license=spec.license,
            version=spec.version,
            year=spec.year,
            source_url=spec.source_url,
            download_url=spec.download_url or "",
            approx_size_mb=spec.approx_size_mb,
            priority=spec.priority,
            notes=spec.notes[:500],
        )

    def as_row(self) -> dict[str, Any]:
        data = asdict(self)
        return {col: data.get(col, "") for col in MANIFEST_COLUMNS}

    @property
    def is_usable(self) -> bool:
        """Can downstream training/RAG run against this dataset right now?"""
        return self.status in {
            DatasetStatus.DOWNLOADED.value,
            DatasetStatus.VALIDATED.value,
            DatasetStatus.PREPROCESSED.value,
        }


class DatasetManifest:
    """Read/write access to ``dataset_manifest.csv``.

    Keyed by dataset name; unknown datasets encountered in the CSV are
    preserved on write so a teammate's rows are never silently lost.
    """

    def __init__(self, path: Path, rows: Iterable[ManifestRow] = ()) -> None:
        self.path = Path(path)
        self._rows: dict[str, ManifestRow] = {r.name: r for r in rows}

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        """Read the manifest, seeding declared-but-absent datasets from the registry."""
        manifest = cls(path)
        if manifest.path.exists():
            with manifest.path.open("r", encoding="utf-8", newline="") as handle:
                for raw in csv.DictReader(handle):
                    name = (raw.get("name") or "").strip()
                    if not name:
                        continue
                    known = set(MANIFEST_COLUMNS)
                    payload = {k: raw.get(k, "") for k in known}
                    manifest._rows[name] = _coerce_row(payload)
        for spec in DATASETS:
            manifest._rows.setdefault(spec.name, ManifestRow.from_spec(spec))
        return manifest

    @classmethod
    def initialise(cls, path: Path, *, force: bool = False) -> "DatasetManifest":
        """Create the manifest from the registry (idempotent unless ``force``)."""
        if Path(path).exists() and not force:
            return cls.load(path)
        manifest = cls(path, [ManifestRow.from_spec(s) for s in DATASETS])
        manifest.save()
        return manifest

    # ------------------------------------------------------------------
    def get(self, name: str) -> ManifestRow:
        return self._rows[name.strip().lower()]

    def all(self) -> list[ManifestRow]:
        return sorted(self._rows.values(), key=lambda r: (r.priority, r.name))

    def update(self, name: str, **fields: Any) -> ManifestRow:
        row = self.get(name)
        for key, value in fields.items():
            if not hasattr(row, key):
                raise DatasetValidationError(
                    f"Unknown manifest field '{key}' for dataset '{name}'"
                )
            setattr(row, key, value)
        return row

    def record_download(self, name: str, result: Any) -> ManifestRow:
        """Fold a :class:`~cyberllm.datasets.acquire.DownloadResult` into the row."""
        fields: dict[str, Any] = {
            "retrieved_at": result.retrieved_at or _now(),
            "file_size_bytes": result.bytes_downloaded,
            "file_size_mb": round(result.bytes_downloaded / 1_048_576, 3),
        }
        if result.ok:
            fields["status"] = DatasetStatus.DOWNLOADED.value
            fields["error"] = ""
            fields["sha256"] = result.sha256 or ""
            if result.path:
                fields["file_path"] = str(result.path)
        else:
            fields["status"] = DatasetStatus.UNAVAILABLE.value
            fields["error"] = (result.error or "download failed")[:500]
        return self.update(name, **fields)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file then swap, so an interrupted write cannot
        # corrupt the manifest.
        tmp = self.path.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for row in self.all():
                writer.writerow(row.as_row())
        tmp.replace(self.path)
        logger.info("manifest saved: %s (%d rows)", self.path.name, len(self._rows))
        return self.path

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, int]:
        """Count rows per status - used by the CLI summary and /health."""
        counts: dict[str, int] = {}
        for row in self.all():
            counts[row.status] = counts.get(row.status, 0) + 1
        return counts

    def usable(self, kind: str | None = None) -> list[ManifestRow]:
        rows = [r for r in self.all() if r.is_usable]
        if kind:
            rows = [r for r in rows if r.kind == kind]
        return rows


__all__ = ["DatasetManifest", "ManifestRow", "MANIFEST_COLUMNS"]

