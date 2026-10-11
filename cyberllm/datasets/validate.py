"""Dataset validation: file signature, schema and content checks.

A download that returns HTTP 200 is *not* proof of a usable dataset - it may
be an HTML error page, a truncated archive, or a CSV with different columns
than expected. This module answers the question "can we actually train or
index this?" before any expensive work starts.

Checks performed
----------------
1. **Existence / size** - file present and non-trivially sized.
2. **Signature** - magic bytes / extension agreement (a ``.csv`` that is
   really HTML is rejected).
3. **Parsability** - JSON parses, CSV opens with a header.
4. **Schema** - required columns from the registry are present.
5. **Content sanity** - row count, label distribution, obvious corruption.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import DatasetValidationError
from ..core.logging_config import get_logger
from .registry import DatasetSpec

logger = get_logger("datasets.validate")

#: Leading bytes that identify common formats.
_MAGIC = {
    b"PK\x03\x04": "zip",
    b"\x1f\x8b": "gzip",
    b"BZh": "bzip2",
    b"\xfd7zXZ\x00": "xz",
    b"7z\xbc\xaf\x27\x1c": "7zip",
    b"Rar!": "rar",
}

_MAGIC_MIN_BYTES = 8
MIN_PLAUSIBLE_BYTES = 64


@dataclass
class ValidationReport:
    """Structured result of validating one dataset."""

    name: str
    ok: bool = False
    file_path: str = ""
    checks: list[dict[str, Any]] = field(default_factory=list)
    detected_format: str = ""
    file_size_bytes: int = 0
    row_count: int = 0
    columns: list[str] = field(default_factory=list)
    label_distribution: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add(self, check: str, passed: bool, detail: str) -> None:
        self.checks.append({"check": check, "passed": passed, "detail": detail})
        if not passed:
            self.errors.append(f"{check}: {detail}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "file_path": self.file_path,
            "detected_format": self.detected_format,
            "file_size_bytes": self.file_size_bytes,
            "row_count": self.row_count,
            "column_count": len(self.columns),
            "columns": self.columns[:50],
            "label_distribution": self.label_distribution,
            "checks": self.checks,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def sniff_format(path: Path) -> str:
    """Identify a file's real format from its extension and magic bytes."""
    suffix = path.suffix.lower().lstrip(".")
    with path.open("rb") as handle:
        head = handle.read(_MAGIC_MIN_BYTES)

    for magic, name in _MAGIC.items():
        if head.startswith(magic):
            return f"{name} (magic)" if suffix not in name else name

    if suffix in {"csv", "tsv", "txt"}:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                sample = handle.read(2048).lstrip()
        except OSError:
            return "unreadable"
        lowered = sample[:400].lower()
        if lowered.startswith(("<!doctype html", "<html", "<?xml")):
            return "html-error-page"
        if "<html" in lowered or "<body" in lowered:
            return "html-error-page"
        return "csv" if suffix == "csv" else ("tsv" if suffix == "tsv" else "text")

    if suffix in {"json", "jsonl"}:
        return suffix
    if suffix == "npy":
        return "numpy"
    if suffix == "parquet":
        return "parquet"
    return suffix or "unknown"


def _validate_json(path: Path, report: ValidationReport, spec: DatasetSpec) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        report.add("json_parsable", False, f"invalid JSON at line {exc.lineno}: {exc.msg}")
        return
    report.add("json_parsable", True, "parsed successfully")

    if isinstance(data, dict):
        # CISA KEV style: {"vulnerabilities": [...]}; NVD: {"vulnerabilities": [...]}
        rows = data.get("vulnerabilities") or data.get("objects") or []
        if isinstance(rows, list):
            report.row_count = len(rows)
            report.label_distribution = {"records": len(rows)}
            if rows and isinstance(rows[0], dict):
                report.columns = sorted(rows[0].keys())
        else:
            report.columns = sorted(data.keys())
    elif isinstance(data, list):
        report.row_count = len(data)
        report.label_distribution = {"records": len(data)}
        if data and isinstance(data[0], dict):
            report.columns = sorted(data[0].keys())

    _check_expected_columns(report, spec)
    if report.row_count == 0:
        report.add("non_empty", False, "container parsed but contained zero records")


def _check_expected_columns(report: ValidationReport, spec: DatasetSpec) -> None:
    if not spec.expected_columns:
        return
    lowered = {c.lower() for c in report.columns}
    missing = [c for c in spec.expected_columns if c.lower() not in lowered]
    if missing:
        report.add("expected_columns", False, f"missing field(s): {', '.join(missing)}")
    else:
        report.add("expected_columns", True,
                   f"all {len(spec.expected_columns)} expected fields present")


def _validate_csv(path: Path, report: ValidationReport, spec: DatasetSpec,
                  max_rows: int = 200_000) -> None:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                report.add("csv_header", False, "file has no header row")
                return
            report.add("csv_header", True, f"{len(header)} columns")
            report.columns = [c.strip() for c in header]
            count = 0
            for _ in reader:
                count += 1
                if count >= max_rows:
                    report.warn(f"row scan truncated at {max_rows:,} rows")
                    break
            report.row_count = count
    except (OSError, csv.Error) as exc:
        report.add("csv_readable", False, f"could not read CSV: {exc}")
        return
    report.add("csv_readable", True, f"{report.row_count:,} data rows")
    _check_expected_columns(report, spec)
    if report.row_count == 0:
        report.add("non_empty", False, "CSV contains a header but no data rows")


def _validate_jsonl(path: Path, report: ValidationReport, spec: DatasetSpec) -> None:
    total, first, bad = 0, None, 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            if first is None:
                try:
                    first = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
            if total >= 200_000:
                report.warn("row scan truncated at 200,000 lines")
                break
    report.row_count = total
    if first and isinstance(first, dict):
        report.columns = sorted(first.keys())
    if total and bad == total:
        report.add("jsonl_parsable", False, "no line parsed as JSON")
        return
    report.add("jsonl_parsable", True, f"{total:,} lines")
    _check_expected_columns(report, spec)



def find_dataset_files(directory: Path) -> list[Path]:
    """List candidate data files in a dataset directory (CSV/JSON/NPY)."""
    directory = Path(directory)
    if not directory.exists():
        return []
    out: list[Path] = []
    for pattern in ("*.csv", "*.json", "*.jsonl", "*.npy", "*.parquet", "*.tsv"):
        out.extend(sorted(directory.rglob(pattern)))
    return [p for p in out if p.is_file() and not p.name.endswith(".part")]


def validate_file(path: Path, spec: DatasetSpec, *, max_rows: int = 200_000) -> ValidationReport:
    """Validate a single file against a dataset specification."""
    path = Path(path)
    report = ValidationReport(name=spec.name, file_path=str(path))

    if not path.exists():
        report.add("exists", False, f"file not found: {path}")
        return report
    report.add("exists", True, f"found {path.name}")

    size = path.stat().st_size
    report.file_size_bytes = size
    if size < MIN_PLAUSIBLE_BYTES:
        report.add("plausible_size", False,
                   f"file is only {size} bytes - likely an error page or stub")
        return report
    report.add("plausible_size", True, f"{size:,} bytes")

    detected = sniff_format(path)
    report.detected_format = detected
    if detected == "html-error-page":
        report.add("signature", False,
                   "content looks like an HTML page, not the expected data format")
        return report
    report.add("signature", True, f"detected {detected}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        _validate_csv(path, report, spec, max_rows=max_rows)
    elif suffix == ".json":
        _validate_json(path, report, spec)
    elif suffix == ".jsonl":
        _validate_jsonl(path, report, spec)
    else:
        report.add("schema", True, f"no schema check defined for {detected}")
        report.warn(f"No column-level validation implemented for '{detected}' files yet")

    report.ok = not report.errors
    return report


def validate_dataset(name: str, *, root: Path | None = None,
                     max_rows: int = 200_000) -> ValidationReport:
    """Validate every file present for a dataset.

    A dataset with several CSVs (e.g. CIC-IDS2017 per-day files) reports on
    the aggregate; individual per-file results stay in ``checks``.
    """
    from .registry import get_dataset

    spec = get_dataset(name)
    if root is None:
        from ..core.config import get_settings
        root = get_settings().paths.raw_dir / spec.name

    files = find_dataset_files(Path(root))
    if not files:
        report = ValidationReport(name=spec.name, file_path=str(root))
        hint = (
            f"Automated download: `datasets download --name {spec.name}`."
            if spec.access != "manual"
            else "This dataset needs manual acquisition - see "
                 f"`datasets list --name {spec.name}`."
        )
        report.add("files_present", False, f"no data files under {root}. {hint}")
        return report

    combined = ValidationReport(name=spec.name, file_path=str(root))
    combined.add("files_present", True, f"{len(files)} file(s) under {root}")
    for file_path in files:
        report = validate_file(file_path, spec, max_rows=max_rows)
        combined.row_count += report.row_count
        combined.columns = report.columns or combined.columns
        combined.warnings.extend(f"{file_path.name}: {w}" for w in report.warnings)
        combined.errors.extend(f"{file_path.name}: {e}" for e in report.errors)
        combined.checks.append({"check": f"file:{file_path.name}", **report.as_dict()})
        if report.label_distribution:
            combined.label_distribution = report.label_distribution
    combined.file_size_bytes = sum(f.stat().st_size for f in files)
    combined.ok = not combined.errors
    return combined


__all__ = [
    "ValidationReport", "validate_dataset", "validate_file",
    "sniff_format", "find_dataset_files",
]

