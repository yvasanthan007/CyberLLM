"""Dataset-specific preprocessing behind ``datasets preprocess``.

Threat-intel datasets (CISA KEV, NVD, MITRE ATT&CK) preprocess into the
retrieval store, because that is the only form in which they are useful to
the application. Tabular datasets (flows, URLs, malware features) preprocess
into a cleaning report, because they are consumed at training time by
``models train`` and producing a second on-disk copy would only create two
sources of truth.

Keeping the distinction explicit is deliberate: a reviewer can see exactly
which datasets have derived artefacts and which are read directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.config import get_settings
from ..core.errors import DatasetNotFoundError, ValidationError
from ..core.logging_config import get_logger
from .registry import DatasetKind, get_dataset, list_datasets
from .validate import find_dataset_files, validate_dataset

logger = get_logger("datasets.preprocess")


class DatasetPreprocessor:
    """Dispatch preprocessing to the right handler for each dataset kind."""

    def __init__(self, settings: Any = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    def run(self, name: str | None = None) -> dict[str, dict[str, Any]]:
        """Preprocess one dataset, or every dataset that has files."""
        names = [name] if name else [
            spec.name for spec in list_datasets()
            if find_dataset_files(self.settings.paths.raw_dir / spec.name)
        ]
        results: dict[str, dict[str, Any]] = {}
        for dataset in names:
            try:
                spec = get_dataset(dataset)
            except DatasetNotFoundError:
                results[dataset] = {
                    "ok": False, "dataset": dataset, "detail": "Unknown dataset.",
                    "hint": "Run `datasets list` to see valid ids.",
                }
                continue
            results[dataset] = self._dispatch(spec)
        return results

    def _dispatch(self, spec: Any) -> dict[str, Any]:
        if spec.kind == DatasetKind.THREAT_INTEL:
            return self._preprocess_threat_intel(spec)
        if spec.kind == DatasetKind.LLM_TRAINING:
            return self._preprocess_sft(spec)
        return self._preprocess_tabular(spec)

    def _source_file(self, spec: Any) -> Path | None:
        files = find_dataset_files(self.settings.paths.raw_dir / spec.name)
        return files[0] if files else None

    def _missing(self, spec: Any, what: str) -> dict[str, Any]:
        """Consistent 'nothing here yet' response with the exact next step."""
        if spec.access == "manual":
            hint = f"Manual acquisition required - see `datasets list --name {spec.name}`."
        else:
            hint = f"Download it first: `datasets download --name {spec.name}`."
        return {"ok": False, "dataset": spec.name, "detail": what, "hint": hint}

    # ------------------------------------------------------------------
    def _preprocess_threat_intel(self, spec: Any) -> dict[str, Any]:
        """Index a threat-intel dataset into the retrieval store."""
        from ..rag.pipeline import RAGPipeline

        path = self._source_file(spec)
        if path is None:
            return self._missing(spec, "No raw file present.")
        try:
            result = RAGPipeline(self.settings).ingest(spec.name, path)
        except (ValueError, FileNotFoundError) as exc:
            return {"ok": False, "dataset": spec.name, "detail": str(exc),
                    "hint": "Check the file matches the registered loader."}
        if result.errors:
            return {"ok": False, "dataset": spec.name,
                    "detail": "; ".join(result.errors), "hint": ""}
        return {
            "ok": True, "dataset": spec.name,
            "detail": (f"{result.documents_loaded} records -> "
                       f"{result.chunks_indexed} indexed chunks "
                       f"in {result.elapsed_seconds:.1f}s"),
            "injection_flagged": result.injection_flagged,
        }

    def _preprocess_sft(self, spec: Any) -> dict[str, Any]:
        """Validate the in-repo instruction corpus."""
        path = self._source_file(spec)
        if path is None:
            repo_copy = (self.settings.paths.root / "dataset" / "final"
                        / "cyberllm_train.jsonl")
            path = repo_copy if repo_copy.exists() else None
        if path is None:
            return {
                "ok": False, "dataset": spec.name,
                "detail": "No JSONL corpus found.",
                "hint": "Expected dataset/final/cyberllm_train.jsonl in the repo.",
            }
        report = validate_dataset(spec.name, root=path.parent)
        return {
            "ok": report.ok, "dataset": spec.name,
            "detail": f"{report.row_count} instruction/response pairs",
            "hint": "" if report.ok else "; ".join(report.errors[:2]),
        }

    def _preprocess_tabular(self, spec: Any) -> dict[str, Any]:
        """Report on a tabular dataset; training consumes it directly."""
        path = self._source_file(spec)
        if path is None:
            return self._missing(spec, "No CSV files present.")
        report = validate_dataset(spec.name, root=path.parent)
        if not report.ok:
            return {"ok": False, "dataset": spec.name,
                    "detail": "; ".join(report.errors[:2]), "hint": ""}
        if spec.kind == DatasetKind.NETWORK_IDS:
            summary = self._describe_flows(path)
            task = "nids"
        else:
            summary = f"{report.row_count:,} rows x {len(report.columns)} columns"
            task = "phishing" if spec.kind == DatasetKind.PHISHING else spec.name
        return {
            "ok": True, "dataset": spec.name, "detail": summary,
            "hint": f"Ready to train: `models train --task {task} --data {path}`",
        }

    @staticmethod
    def _describe_flows(path: Path) -> str:
        """Run the read-only schema inspection and format the summary.

        A malformed file must not abort preprocessing of the other datasets, so
        inspection errors are caught here and reported as a description rather
        than raised. The underlying exception is logged with its type.
        """
        from ..detection.preprocess import inspect_csv

        try:
            _, schema = inspect_csv(path, max_rows=50_000)
        except (ValidationError, OSError, ValueError) as exc:
            logger.warning("flow inspection failed for %s: %s: %s",
                           path.name, type(exc).__name__, exc)
            return f"could not inspect ({type(exc).__name__}): {exc}"
        return (f"{schema.rows:,} rows; label='{schema.label_column}'; "
                f"{len(schema.label_distribution)} classes; "
                f"{schema.duplicate_rows:,} duplicate rows")


__all__ = ["DatasetPreprocessor"]

