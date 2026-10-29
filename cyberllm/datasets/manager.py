"""Orchestration layer tying acquisition, validation and the manifest together.

The CLI and API both call into :class:`DatasetManager` rather than
reimplementing the acquire -> validate -> record sequence. That keeps status
transitions in exactly one place, which is what makes the inventory honest.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..core.config import AppConfig, get_settings
from ..core.errors import DatasetDownloadError, DatasetNotFoundError
from ..core.logging_config import get_logger
from .acquire import DownloadResult, download_file, fetch_json, sha256_file
from .manifest import DatasetManifest, ManifestRow
from .registry import (
    DATASETS, DatasetKind, DatasetSpec, DatasetStatus, get_dataset, list_datasets,
)
from .validate import ValidationReport, find_dataset_files, validate_dataset

logger = get_logger("datasets.manager")

#: Files each automated dataset lands as.
_TARGET_FILENAMES = {
    "cisa_kev": "known_exploited_vulnerabilities.json",
    "mitre_attack": "enterprise-attack.json",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DatasetManager:
    """High-level dataset operations bound to one settings object."""

    def __init__(self, settings: AppConfig | None = None) -> None:
        self.settings = settings or get_settings()
        self.paths = self.settings.paths
        self.manifest = DatasetManifest.load(self.paths.manifest_path)

    # ------------------------------------------------------------------
    def dataset_dir(self, name: str) -> Path:
        return self.paths.raw_dir / get_dataset(name).name

    def processed_dir(self, name: str) -> Path:
        return self.paths.processed_dir / get_dataset(name).name

    # ------------------------------------------------------------------
    def inventory(self, *, kind: str | None = None) -> list[dict[str, Any]]:
        """Merged registry + manifest + on-disk reality for every dataset."""
        rows: list[dict[str, Any]] = []
        for spec in list_datasets(kind=DatasetKind(kind) if kind else None):
            row = self.manifest._rows.get(spec.name)  # noqa: SLF001 - internal by design
            files = find_dataset_files(self.dataset_dir(spec.name))
            actual_bytes = sum(f.stat().st_size for f in files)
            rows.append({
                "name": spec.name,
                "kind": spec.kind.value,
                "status": self._effective_status(spec, row, files),
                "access": spec.access,
                "license": spec.license,
                "version": spec.version,
                "year": spec.year,
                "source_url": spec.source_url,
                "download_url": spec.download_url,
                "approx_size_mb": spec.approx_size_mb,
                "actual_size_mb": round(actual_bytes / 1_048_576, 3),
                "file_count": len(files),
                "files": [str(f.relative_to(self.paths.root)) for f in files[:10]],
                "retrieved_at": row.retrieved_at if row else "",
                "sha256": (row.sha256 if row else "")[:16],
                "preprocess_status": row.preprocess_status if row else "not_started",
                "preprocess_path": row.preprocess_path if row else "",
                "error": row.error if row else "",
                "notes": spec.notes,
                "manual_steps": list(spec.manual_steps),
                "priority": spec.priority,
                "tags": list(spec.tags),
                "synthetic": spec.synthetic,
            })
        return rows

    def _effective_status(self, spec: DatasetSpec, row: ManifestRow | None,
                          files: list[Path]) -> str:
        """Reconcile the recorded status with what is actually on disk.

        A file that disappeared since the last run must not keep reporting
        ``downloaded`` - filesystem state wins over the recorded value.
        """
        if files:
            recorded = row.status if row else ""
            if recorded == DatasetStatus.PREPROCESSED.value:
                return DatasetStatus.PREPROCESSED.value
            if recorded == DatasetStatus.VALIDATED.value:
                return DatasetStatus.VALIDATED.value
            return DatasetStatus.DOWNLOADED.value
        if spec.access == "manual":
            return DatasetStatus.MANUAL.value
        if row and row.status == DatasetStatus.UNAVAILABLE.value and row.error:
            return DatasetStatus.UNAVAILABLE.value
        if spec.access == "public" and spec.download_url:
            return DatasetStatus.DOWNLOADABLE.value
        return DatasetStatus.REGISTERED.value

    # ------------------------------------------------------------------
    def download(self, name: str, *, force: bool = False,
                 timeout: int = 120, max_records: int | None = None) -> DownloadResult:
        """Download a dataset and record the outcome in the manifest.

        Datasets with ``access="manual"`` are never fetched automatically -
        the returned result explains exactly what to do instead.
        """
        spec = get_dataset(name)
        target_dir = self.dataset_dir(name)

        if spec.name == "unsw_nb15":
            return self._download_unsw_nb15(target_dir, force=force, timeout=timeout)

        if spec.access == "manual":
            logger.warning("%s requires manual acquisition", spec.name)
            return DownloadResult(
                url=spec.source_url, path=target_dir, ok=False,
                status=DatasetStatus.MANUAL.value,
                error="Dataset requires manual acquisition (registration/terms).",
            )

        if not spec.download_url:
            return DownloadResult(
                url="", path=target_dir, ok=False, status=DatasetStatus.REGISTERED.value,
                error="No automated download endpoint declared in the registry.",
            )

        filename = _TARGET_FILENAMES.get(spec.name, Path(spec.download_url).name or "data.bin")
        dest = target_dir / filename
        if dest.exists() and not force:
            logger.info("%s already present at %s (use force=True to refetch)", spec.name, dest)
            result = DownloadResult(
                url=spec.download_url, path=dest, ok=True,
                status=DatasetStatus.DOWNLOADED.value,
                bytes_downloaded=dest.stat().st_size,
            )
            result.sha256 = sha256_file(dest)
            self.manifest.record_download(spec.name, result)
            self.manifest.save()
            return result

        logger.info("downloading %s from %s", spec.name, spec.download_url)
        if spec.name == "nvd_cve":
            api_key = _nvd_api_key()
            result = self._download_nvd(
                target_dir, {"apiKey": api_key} if api_key else {}, max_records or 400
            )
        else:
            result = download_file(spec.download_url, dest, timeout=timeout, retries=3)

        self.manifest.record_download(spec.name, result)
        self.manifest.save()
        return result

    def download_all(self, *, kinds: Iterable[str] = ()) -> list[DownloadResult]:
        """Fetch every automatically-acquirable dataset, priority first."""
        names = [d.name for d in list_datasets()
                 if d.access != "manual" and (not kinds or d.kind.value in set(kinds))]
        return [self.download(n) for n in names]

    # ------------------------------------------------------------------
    def _download_unsw_nb15(self, target_dir: Path, *, force: bool,
                            timeout: int) -> DownloadResult:
        """Fetch UNSW-NB15 from the publisher's public folder share.

        UNSW-NB15 ships as three separate CSVs behind a SharePoint folder
        listing rather than a single file URL, so it gets its own route. The
        manifest carries a single ``sha256`` field, so a composite digest over
        the per-file digests is recorded there; the full per-file breakdown is
        written to ``PROVENANCE.json`` beside the data.

        Any failure raises a typed error carrying the manual procedure rather
        than returning a partial success that looks like a complete dataset.
        """
        from .unsw_nb15 import (
            SHARE_URL, download_unsw_nb15, manual_steps, provenance,
        )

        results = download_unsw_nb15(target_dir, group="core", force=force,
                                     timeout=timeout)
        failed = [r for r in results if not r.ok]
        if failed:
            raise DatasetDownloadError(
                f"{len(failed)} UNSW-NB15 file(s) could not be retrieved: "
                + ", ".join(Path(str(r.path)).name for r in failed if r.path),
                details={"setup_hint": " ".join(manual_steps())},
            )

        per_file = {
            Path(str(r.path)).name: {
                "bytes": r.bytes_downloaded, "sha256": r.sha256,
                "status": r.status,
            }
            for r in results if r.path
        }
        (target_dir / "PROVENANCE.json").write_text(
            json.dumps(provenance(per_file), indent=2), encoding="utf-8"
        )

        composite = hashlib.sha256(
            "".join(
                f"{name}:{info['sha256']}" for name, info in sorted(per_file.items())
            ).encode("utf-8")
        ).hexdigest()
        total = sum(int(info["bytes"]) for info in per_file.values())
        result = DownloadResult(
            url=SHARE_URL, path=target_dir, ok=True,
            status=DatasetStatus.DOWNLOADED.value, bytes_downloaded=total,
            sha256=composite, headers={"content-length": str(total)},
        )
        logger.info("UNSW-NB15: %d file(s), %.1f MB, composite sha256 %s",
                    len(per_file), total / 1_048_576, composite[:16])
        self.manifest.record_download("unsw_nb15", result)
        self.manifest.save()
        return result

    # ------------------------------------------------------------------
    def _download_nvd(self, target_dir: Path, headers: dict[str, str],
                      max_records: int) -> DownloadResult:
        """Pull a bounded sample of NVD CVE 2.0 records.

        NVD rate-limits aggressively without an API key (~5 req / 30 s), so
        the sample is deliberately small and configurable via
        ``CYBERLLM_NVD_MAX_RECORDS``.
        """
        import json
        import os

        limit = int(os.environ.get("CYBERLLM_NVD_MAX_RECORDS", str(max_records)))
        url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        params = {"resultsPerPage": min(limit, 2000)}

        payload, result = fetch_json(url, params=params, headers=headers or None)
        items = payload.get("vulnerabilities", [])
        dest = target_dir / "nvd_cve_sample.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=1), encoding="utf-8")

        result.path = dest
        result.ok = True
        result.status = DatasetStatus.DOWNLOADED.value
        result.bytes_downloaded = dest.stat().st_size
        result.sha256 = sha256_file(dest)
        logger.info("saved %d NVD records to %s", len(items), dest)
        return result


    def validate(self, name: str | None = None) -> dict[str, ValidationReport]:
        """Validate one dataset, or every dataset that has files on disk."""
        names = [name] if name else [
            s.name for s in list_datasets() if find_dataset_files(self.dataset_dir(s.name))
        ]
        reports: dict[str, ValidationReport] = {}
        for dataset_name in names:
            report = validate_dataset(
                dataset_name, root=self.dataset_dir(dataset_name),
                max_rows=self.settings.security.max_csv_rows,
            )
            reports[dataset_name] = report
            status = (DatasetStatus.VALIDATED.value if report.ok
                      else DatasetStatus.DOWNLOADED.value)
            self.manifest.update(
                dataset_name,
                status=status,
                file_size_bytes=report.file_size_bytes,
                file_size_mb=round(report.file_size_bytes / 1_048_576, 3),
                row_count=report.row_count,
                column_count=len(report.columns),
                error="; ".join(report.errors)[:500] if report.errors else "",
            )
        self.manifest.save()
        return reports

    def refresh(self) -> None:
        """Rewrite the manifest without touching the network."""
        self.manifest = DatasetManifest.load(self.paths.manifest_path)
        for spec in DATASETS:
            files = find_dataset_files(self.dataset_dir(spec.name))
            if not files:
                continue
            size = sum(f.stat().st_size for f in files)
            fields: dict[str, Any] = {
                "file_size_bytes": size,
                "file_size_mb": round(size / 1_048_576, 3),
                "file_path": str(files[0]),
            }
            recorded = self.manifest._rows[spec.name].status  # noqa: SLF001
            if recorded not in {DatasetStatus.VALIDATED.value,
                                DatasetStatus.PREPROCESSED.value}:
                fields["status"] = DatasetStatus.DOWNLOADED.value
            self.manifest.update(spec.name, **fields)
        self.manifest.save()

    def mark_preprocessed(self, name: str, *, path: Path, rows: int, columns: int) -> None:
        """Record that preprocessing produced artefacts for a dataset."""
        self.manifest.update(
            name,
            status=DatasetStatus.PREPROCESSED.value,
            preprocess_status=DatasetStatus.PREPROCESSED.value,
            preprocess_path=str(path),
            preprocessed_at=_now(),
            row_count=rows,
            column_count=columns,
        )
        self.manifest.save()

    def manual_instructions(self, name: str) -> list[str]:
        return list(get_dataset(name).manual_steps)

    def resolve(self, name: str) -> DatasetSpec:
        return get_dataset(name)


def _nvd_api_key() -> str | None:
    import os
    return os.environ.get("NVD_API_KEY") or os.environ.get("CYBERLLM_NVD_API_KEY")


__all__ = ["DatasetManager"]

