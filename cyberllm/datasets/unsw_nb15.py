"""Acquisition of the UNSW-NB15 dataset through the publisher's official share.

Provenance and terms
--------------------
The dataset page at :data:`SOURCE_PAGE` (UNSW Canberra at ADFA) is the
authority. It states verbatim:

    "Free use of the UNSW-NB15 dataset for academic research purposes is hereby
     granted in perpetuity. Use for commercial purposes should be agreed by
     the authors. ... To whom intend the use of the UNSW-NB15 dataset have to
     cite the above five papers."

Attribution is therefore a *condition* of use, not a courtesy: every artefact
CyberLLM writes from this data carries the citation block in :data:`CITATION`.

How the download works
----------------------
The page's only download link ("HERE") points at a UNSW OneDrive/SharePoint
folder published as *anyone with the link*. Consuming that link issues an
anonymous ``FedAuth`` cookie; the same cookie then authorises SharePoint's own
file endpoint, which is exactly the request a browser makes when a researcher
clicks "Download". :class:`UnswShare` does precisely that and nothing else:

* it sends **no credentials** and stores **no** credential;
* it never submits an agreement on the operator's behalf - a human accepting
  the terms is a human decision, and :func:`manual_steps` documents it;
* it refuses to guess file names. Files are discovered from the folder
  listing the publisher exposes and requested by server-relative URL;
* if the share ever stops being public, the typed
  :class:`~cyberllm.core.errors.DatasetDownloadError` raised here names the
  exact manual procedure rather than silently degrading.

Licence compatibility: academic research use is permitted, which is how this
project uses it. Commercial deployment is out of scope and is refused in the
model cards.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from ..core.errors import DatasetDownloadError
from ..core.logging_config import get_logger
from .acquire import DownloadResult, download_file, sha256_file

logger = get_logger("datasets.unsw_nb15")

#: The publisher's authoritative dataset page. Cite this, not a mirror.
SOURCE_PAGE = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"

#: The "HERE" link on that page: a public OneDrive/SharePoint folder share.
SHARE_URL = (
    "https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/"
    "EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS"
)

#: SharePoint site root the share resolves into.
WEB_ROOT = "https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au"

#: Server-relative folder holding the tabular release.
CSV_FOLDER = "/personal/z5025758_ad_unsw_edu_au/Documents/UNSW-NB15 dataset/CSV Files"

#: Server-relative folder holding the publisher's own train/test partition.
SPLIT_FOLDER = CSV_FOLDER + "/Training and Testing Sets"

#: Files needed to train and benchmark, with the role each one plays.
#:
#: ``features``  the feature dictionary - the schema authority for the columns.
#: ``train``     the publisher's own 175,341-row training partition.
#: ``test``      the publisher's own 82,332-row test partition.
#:
#: The publisher misspells "UNSW" as "NUSW" in two of these filenames; the
#: names below are the ones actually served, so they must not be "corrected".
CORE_FILES: dict[str, str] = {
    "features": "NUSW-NB15_features.csv",
    "train": "UNSW_NB15_training-set.csv",
    "test": "UNSW_NB15_testing-set.csv",
}

#: The four ~2M-row source files, offered as an opt-in bulk group.
FULL_FILES: tuple[str, ...] = (
    "UNSW-NB15_1.csv", "UNSW-NB15_2.csv", "UNSW-NB15_3.csv", "UNSW-NB15_4.csv",
)

USER_AGENT = (
    "CyberLLM/0.1 (defensive security research; academic use of UNSW-NB15)"
)

#: Citation block required by the publisher's terms. Embedded in every model
#: card and report derived from this dataset.
CITATION: tuple[str, ...] = (
    "Moustafa, N. and Slay, J. (2015) 'UNSW-NB15: a comprehensive data set for "
    "network intrusion detection systems (UNSW-NB15 network data set)', "
    "Military Communications and Information Systems Conference (MilCIS), 2015.",
    "Moustafa, N. and Slay, J. (2016) 'The evaluation of Network Anomaly "
    "Detection Systems: Statistical analysis of the UNSW-NB15 dataset and the "
    "comparison with the KDD99 dataset', Information Security Journal, 1-14.",
    "Moustafa, N. et al. (2017) 'Novel geometric area analysis technique for "
    "anomaly detection using trapezoidal area estimation on large-scale "
    "networks', IEEE Transactions on Big Data.",
    "Moustafa, N. et al. (2017) 'Big data analytics for intrusion detection "
    "system: statistical decision-making using finite dirichlet mixture "
    "models', Data Analytics and Decision Support for Cybersecurity, Springer, "
    "pp. 127-156.",
    "Sarhan, M., Layeghy, S., Moustafa, N. and Portmann, M. (2020) 'NetFlow "
    "Datasets for Machine Learning-Based Network Intrusion Detection Systems', "
    "Big Data Technologies and Applications, 10th EAI International Conference "
    "(BDTA 2020), p. 117.",
)

#: Honest description of what the data actually is, for model-card caveats.
DATASET_CAVEAT = (
    "UNSW-NB15 is a hybrid corpus: the authors combined real modern normal "
    "traffic with synthetic contemporary attack behaviours generated by the "
    "IXIA PerfectStorm tool in a cyber range. It is real published research "
    "data, not production captures, so accuracy here is an upper bound rather "
    "than a field estimate."
)


def manual_steps() -> list[str]:
    """The human procedure, used when automation is not possible."""
    return [
        f"1. Open {SOURCE_PAGE} and read the usage terms (free for academic "
        "research; the five papers must be cited).",
        "2. Click the 'HERE' link to open the shared OneDrive folder.",
        f"   Direct share URL: {SHARE_URL}",
        "3. If OneDrive asks you to sign in, stop - the share is no longer "
        "public. Contact the dataset author (nour.moustafa@unsw.edu.au) rather "
        "than working around the gate.",
        "4. Download 'CSV Files/NUSW-NB15_features.csv' (the feature "
        "dictionary), 'CSV Files/UNSW_NB15_training-set.csv' and "
        "'CSV Files/UNSW_NB15_testing-set.csv'.",
        "5. Place them in  data/raw/unsw_nb15/  keeping these exact filenames.",
        "6. Validate and benchmark:",
        "     python -m cyberllm.cli datasets validate --name unsw_nb15",
        "     python -m cyberllm.cli models benchmark --dataset unsw_nb15",
    ]


@dataclass
class ShareEntry:
    """One item in a SharePoint folder listing."""

    name: str
    server_relative_url: str
    length: int = 0
    modified: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "server_relative_url": self.server_relative_url,
            "length": self.length,
            "modified": self.modified,
        }


@dataclass
class UnswShare:
    """A session bound to the publisher's public UNSW-NB15 folder share.

    Construction is cheap and performs no I/O; call :meth:`open` first. The
    session keeps the anonymous ``FedAuth`` cookie the share link hands out,
    which is what authorises the subsequent file reads.
    """

    share_url: str = SHARE_URL
    web_root: str = WEB_ROOT
    timeout: int = 120
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            # SharePoint answers with Atom XML unless JSON is requested
            # explicitly, and the XML form is far noisier to parse.
            "Accept": "application/json;odata=nometadata",
        })

    # ------------------------------------------------------------------
    def open(self) -> "UnswShare":
        """Consume the share link and confirm anonymous read access.

        Raises a typed error naming the manual procedure if the share is
        closed, requires sign-in, or the host is unreachable - never a guess.
        """
        try:
            response = self.session.get(self.share_url, timeout=self.timeout)
        except requests.RequestException as exc:
            raise DatasetDownloadError(
                f"Could not reach the UNSW-NB15 share: {type(exc).__name__}: {exc}",
                details={"share_url": self.share_url,
                         "setup_hint": " ".join(manual_steps())},
            ) from exc

        if response.status_code in (401, 403):
            raise DatasetDownloadError(
                "The UNSW-NB15 share returned HTTP "
                f"{response.status_code}: it is no longer publicly readable.",
                details={"http_status": response.status_code,
                         "setup_hint": " ".join(manual_steps())},
            )
        if response.status_code >= 400:
            raise DatasetDownloadError(
                f"The UNSW-NB15 share returned HTTP {response.status_code}.",
                details={"http_status": response.status_code,
                         "setup_hint": " ".join(manual_steps())},
            )
        if not self.session.cookies.get("FedAuth"):
            raise DatasetDownloadError(
                "The UNSW-NB15 share issued no anonymous session cookie, so the "
                "folder cannot be read without credentials. CyberLLM does not "
                "sign in on an operator's behalf.",
                details={"share_url": self.share_url,
                         "setup_hint": " ".join(manual_steps())},
            )
        logger.info("anonymous access to the UNSW-NB15 share confirmed")
        return self

    # ------------------------------------------------------------------
    def _api(self, suffix: str) -> dict[str, Any]:
        url = f"{self.web_root}/_api/{suffix}"
        try:
            response = self.session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            raise DatasetDownloadError(
                f"UNSW-NB15 folder listing failed: {type(exc).__name__}: {exc}",
                details={"url": url, "setup_hint": " ".join(manual_steps())},
            ) from exc
        if response.status_code >= 400:
            raise DatasetDownloadError(
                f"UNSW-NB15 folder listing returned HTTP {response.status_code}.",
                details={"http_status": response.status_code, "url": url,
                         "setup_hint": " ".join(manual_steps())},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise DatasetDownloadError(
                "UNSW-NB15 folder listing did not return JSON; the share may "
                "have changed or now requires sign-in.",
                details={"url": url, "setup_hint": " ".join(manual_steps())},
            ) from exc
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _entries(payload: dict[str, Any], *, with_length: bool) -> list[ShareEntry]:
        """Convert a SharePoint listing payload into typed entries."""
        entries: list[ShareEntry] = []
        for item in payload.get("value", []):
            if not isinstance(item, dict):
                continue
            raw_length = str(item.get("Length") or 0)
            entries.append(ShareEntry(
                name=str(item.get("Name", "")),
                server_relative_url=str(item.get("ServerRelativeUrl", "")),
                length=int(raw_length) if with_length and raw_length.isdigit() else 0,
                modified=str(item.get("TimeLastModified", "")),
            ))
        return entries

    def list_folder(self, server_relative_url: str) -> list[ShareEntry]:
        """List the files in a folder of the shared dataset."""
        payload = self._api(
            f"web/GetFolderByServerRelativeUrl("
            f"'{urllib.parse.quote(server_relative_url)}')/Files"
        )
        return self._entries(payload, with_length=True)

    def list_subfolders(self, server_relative_url: str) -> list[ShareEntry]:
        """List the sub-folders of a folder in the shared dataset."""
        payload = self._api(
            f"web/GetFolderByServerRelativeUrl("
            f"'{urllib.parse.quote(server_relative_url)}')/Folders"
        )
        return self._entries(payload, with_length=False)

    def file_url(self, server_relative_url: str) -> str:
        """The public read URL for one file in the shared dataset.

        ``GetFileByServerRelativeUrl(...)/$value`` is the endpoint SharePoint's
        own client uses; on this tenant the bare server-relative path returns
        404, so the API form is the correct one and is not a workaround.
        """
        quoted = urllib.parse.quote(server_relative_url, safe="/:@'")
        return f"{self.web_root}/_api/web/GetFileByServerRelativeUrl('{quoted}')/$value"

    # ------------------------------------------------------------------
    def download(self, entry: ShareEntry, dest: Path, *,
                 timeout: int | None = None,
                 retries: int = 3) -> DownloadResult:
        """Stream one shared file to ``dest`` using the shared primitives."""
        return download_file(
            self.file_url(entry.server_relative_url), dest,
            timeout=timeout or self.timeout, retries=retries, session=self.session,
            # The session asks for JSON for listings; file content must be
            # asked for as an octet stream or SharePoint answers with metadata.
            headers={"Accept": "application/octet-stream"},
        )


def provenance(files: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Machine-readable provenance block stored beside every artefact."""
    record: dict[str, Any] = {
        "dataset": "UNSW-NB15",
        "publisher": "UNSW Canberra at ADFA (Cyber Range Lab)",
        "source_page": SOURCE_PAGE,
        "share_url": SHARE_URL,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "license": (
            "Free use for academic research granted in perpetuity by the "
            "authors; commercial use requires the authors' agreement."
        ),
        "attribution_required": True,
        "citation": list(CITATION),
        "caveat": DATASET_CAVEAT,
        "user_agent": USER_AGENT,
    }
    if files:
        record["files"] = files
    return record


def core_files_present(directory: Path) -> dict[str, Path]:
    """Map role -> path for the core files that exist under ``directory``."""
    directory = Path(directory)
    found: dict[str, Path] = {}
    for role, filename in CORE_FILES.items():
        candidate = directory / filename
        if candidate.exists():
            found[role] = candidate
    return found


def resolve_core_entries(share: UnswShare) -> list[ShareEntry]:
    """Locate the core files across the publisher's folder layout.

    The files are searched in both the CSV folder and the train/test folder
    because the publisher's layout has changed between releases; resolving
    them from a listing is what keeps this working without inventing URLs.
    """
    available: dict[str, ShareEntry] = {}
    for folder in (CSV_FOLDER, SPLIT_FOLDER):
        try:
            for entry in share.list_folder(folder):
                available.setdefault(entry.name, entry)
        except DatasetDownloadError as exc:
            logger.warning("could not list %s: %s", folder, exc.message)

    resolved: list[ShareEntry] = []
    missing: list[str] = []
    for role, filename in CORE_FILES.items():
        entry = available.get(filename)
        if entry is None or not entry.server_relative_url:
            missing.append(f"{role} -> {filename}")
            continue
        resolved.append(entry)
    if missing:
        raise DatasetDownloadError(
            "The UNSW-NB15 share is reachable but does not contain the expected "
            f"files ({'; '.join(missing)}). The publisher may have reorganised "
            "the folder.",
            details={"available": sorted(available),
                     "setup_hint": " ".join(manual_steps())},
        )
    logger.info("resolved %d core UNSW-NB15 files", len(resolved))
    return resolved


def download_unsw_nb15(
    target_dir: Path,
    *,
    group: str = "core",
    force: bool = False,
    timeout: int = 120,
) -> list[DownloadResult]:
    """Fetch UNSW-NB15 from the publisher's public share into ``target_dir``.

    ``group="core"`` retrieves the feature dictionary plus the publisher's
    train/test partition (about 48 MB). ``group="full"`` additionally
    retrieves the four ~2M-row source files (about 586 MB) needed for a
    corpus-wide duplicate and leakage audit.

    Returns one :class:`DownloadResult` per file; files already on disk are
    reported as present with their checksum rather than refetched, unless
    ``force`` is set.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    share = UnswShare(timeout=timeout).open()
    entries = resolve_core_entries(share)
    if group == "full":
        known = {entry.name for entry in entries}
        entries.extend(e for e in share.list_folder(CSV_FOLDER)
                       if e.name in FULL_FILES and e.name not in known)
    elif group != "core":
        raise DatasetDownloadError(
            f"Unknown UNSW-NB15 download group '{group}'.",
            details={"available": ["core", "full"]},
        )

    results: list[DownloadResult] = []
    for entry in entries:
        dest = target_dir / entry.name
        if dest.exists() and not force:
            size = dest.stat().st_size
            logger.info("%s already present (%.1f MB)", entry.name, size / 1_048_576)
            result = DownloadResult(
                url=share.file_url(entry.server_relative_url), path=dest, ok=True,
                status="present", bytes_downloaded=size, sha256=sha256_file(dest),
                headers={"content-length": str(size)},
            )
        else:
            result = share.download(entry, dest)
            if not result.ok:
                logger.error("failed to download %s: %s", entry.name, result.error)
        results.append(result)
    return results


__all__ = [
    "UnswShare", "ShareEntry", "download_unsw_nb15", "resolve_core_entries",
    "core_files_present", "provenance", "manual_steps", "CITATION",
    "DATASET_CAVEAT", "SOURCE_PAGE", "SHARE_URL", "CORE_FILES", "FULL_FILES",
    "CSV_FOLDER", "SPLIT_FOLDER", "WEB_ROOT", "USER_AGENT",
]
