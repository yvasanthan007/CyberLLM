"""Dataset registry: the single source of truth for what CyberLLM uses.

Every dataset is declared once in :data:`DATASETS` with its provenance,
licence, expected format and acquisition route. The CLI, the API health
endpoint and the Streamlit inventory all read from this registry, so a
dataset's status is never duplicated or hand-edited in several places.

Status vocabulary (``DatasetStatus``)
-------------------------------------
``registered``   Declared, nothing fetched yet.
``downloadable`` An automated download path exists.
``manual``       Requires registration/acceptance - exact steps are recorded.
``downloaded``   File present on disk.
``validated``    File passed schema/signature checks.
``preprocessed`` Preprocessing artefacts exist.
``unavailable``  Source unreachable; the failure is recorded, not hidden.
``synthetic``    Locally generated fixture, NOT real-world data.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

from ..core.errors import DatasetNotFoundError


class DatasetStatus(str, Enum):
    """Lifecycle state of a dataset in the registry."""

    REGISTERED = "registered"
    DOWNLOADABLE = "downloadable"
    MANUAL = "manual"
    DOWNLOADED = "downloaded"
    VALIDATED = "validated"
    PREPROCESSED = "preprocessed"
    UNAVAILABLE = "unavailable"
    SYNTHETIC = "synthetic"

    @property
    def is_acquired(self) -> bool:
        return self in {
            DatasetStatus.DOWNLOADED, DatasetStatus.VALIDATED, DatasetStatus.PREPROCESSED
        }


class DatasetKind(str, Enum):
    """What the dataset is used for."""

    THREAT_INTEL = "threat_intel"      # CVE / CISA KEV / MITRE
    NETWORK_IDS = "network_ids"        # flow-based intrusion detection
    MALWARE = "malware"                # malware memory / static features
    PHISHING = "phishing"              # URL datasets
    LLM_TRAINING = "llm_training"      # instruction / QA corpora


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable declaration of one dataset.

    Attributes
    ----------
    name:
        Stable identifier used by every CLI/API call.
    kind:
        Purpose within the platform.
    description:
        One-line human summary shown in the UI.
    source_url:
        Official documentation page (not a mirror, not a direct file link).
    license:
        Licence string as stated by the publisher. ``"see-source"`` when the
        publisher does not publish an explicit licence.
    download_url:
        Direct download endpoint when one exists and is permitted.
    access:
        ``public`` (no gate), ``api-key`` (NVD), ``manual`` (registration form).
    expected_format:
        Primary file extension(s) expected after download.
    expected_columns:
        Columns a schema check will look for (empty = no column check).
    approx_size_mb:
        Published or measured size, used for disk-space planning.
    notes:
        Attribution / citation text reproduced in the UI and docs.
    manual_steps:
        Exact remaining steps when ``access == "manual"``.
    priority:
        1 = essential, 2 = useful, 3 = optional. Drives download ordering.
    """

    name: str
    kind: DatasetKind
    description: str
    source_url: str
    license: str
    access: str = "public"
    download_url: str | None = None
    expected_format: tuple[str, ...] = (".json",)
    expected_columns: tuple[str, ...] = ()
    approx_size_mb: float = 0.0
    notes: str = ""
    manual_steps: tuple[str, ...] = ()
    priority: int = 2
    year: str = ""
    version: str = "unknown"
    synthetic: bool = False
    tags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#: Every dataset CyberLLM knows about, ordered by priority.
DATASETS: tuple[DatasetSpec, ...] = (
    # ---------------- Threat intelligence (RAG corpus) -------------------
    DatasetSpec(
        name="cisa_kev",
        kind=DatasetKind.THREAT_INTEL,
        description="CISA Known Exploited Vulnerabilities catalog (actively exploited flaws).",
        source_url="https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
        license="U.S. Government work - public domain (CISA terms of use apply)",
        download_url=(
            "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
        ),
        expected_format=(".json",),
        expected_columns=("cveID", "vendorProject", "product", "dateAdded"),
        approx_size_mb=0.2,
        notes=(
            "Catalog of CVEs with evidence of active exploitation, plus mandated "
            "federal remediation deadlines. Highly authoritative, small and "
            "frequently updated - ideal RAG seed content."
        ),
        priority=1,
        version="live-feed",
        tags=("cve", "exploitation", "authoritative"),
    ),
    DatasetSpec(
        name="mitre_attack",
        kind=DatasetKind.THREAT_INTEL,
        description="MITRE ATT&CK Enterprise matrix (STIX 2.1 bundle).",
        source_url="https://attack.mitre.org/",
        license="Apache-2.0 / MITRE ATT&CK terms of use (attribution required)",
        download_url=(
            "https://raw.githubusercontent.com/mitre/cti/master/"
            "enterprise-attack/enterprise-attack.json"
        ),
        expected_format=(".json",),
        expected_columns=("type", "id", "name"),
        approx_size_mb=45.0,
        notes=(
            "Adversary tactics, techniques and procedures, distributed as a STIX "
            "2.1 bundle from the official mitre/cti repository. Attribution to "
            "MITRE ATT&CK is required in any derivative work."
        ),
        priority=1,
        version="stix2.1",
        tags=("threat-intel", "ttp", "stix"),
    ),
    DatasetSpec(
        name="nvd_cve",
        kind=DatasetKind.THREAT_INTEL,
        description="NVD CVE API 2.0 - structured vulnerability records.",
        source_url="https://nvd.nist.gov/vuln/data-feeds",
        license="U.S. Government work - NVD terms permit free use with attribution",
        download_url="https://services.nvd.nist.gov/rest/json/cves/2.0",
        access="api-key",
        expected_format=(".json",),
        expected_columns=("id", "descriptions", "metrics"),
        approx_size_mb=0.0,
        notes=(
            "Official NVD REST API. A free API key raises the rate limit from "
            "~5 requests/30s to 50/30s. Without a key the API still works, just "
            "slowly - set CYBERLLM_NVD_MAX_RECORDS to a small number when keyless."
        ),
        manual_steps=(
            "1. Register at https://nvd.nist.gov/developers/request-an-api-key",
            "2. Confirm the emailed key.",
            "3. Export it: set NVD_API_KEY=<your-key>  (PowerShell: $env:NVD_API_KEY='<key>')",
            "4. Run:  python -m cyberllm.cli datasets download --name nvd_cve",
        ),
        priority=2,
        version="api-2.0",
        tags=("cve", "vulnerability", "nvd"),
    ),


    # ---------------- Network intrusion detection -------------------------
    DatasetSpec(
        name="unsw_nb15",
        kind=DatasetKind.NETWORK_IDS,
        description="UNSW-NB15 network flow dataset (real + synthetic traffic).",
        source_url="https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        license=(
            "UNSW dataset terms - free for ACADEMIC RESEARCH in perpetuity; "
            "commercial use must be agreed with the authors; the five "
            "published papers MUST be cited"
        ),
        download_url=(
            "https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/"
            "EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS"
        ),
        access="public",
        expected_format=(".csv",),
        expected_columns=("proto", "service", "state", "dur", "sbytes", "label",
                          "attack_cat"),
        approx_size_mb=48.0,
        notes=(
            "Ten classes (Normal + 9 attack families) over 40 Argus/Bro-IDS "
            "flow statistics. The publisher's folder share is public (anyone "
            "with the link) so no registration is needed, but citing the five "
            "papers is a condition of the licence. Hybrid corpus - real normal "
            "traffic plus tool-generated attacks - so accuracy is an upper "
            "bound, not a field estimate. See cyberllm/datasets/unsw_nb15.py."
        ),
        manual_steps=(
            "1. Open the UNSW-NB15 dataset page and read the usage terms.",
            "2. Click 'HERE' to open the public OneDrive folder, then download",
            "   CSV Files/NUSW-NB15_features.csv,",
            "   CSV Files/UNSW_NB15_training-set.csv and",
            "   CSV Files/UNSW_NB15_testing-set.csv.",
            "3. Place them in  data/raw/unsw_nb15/  keeping these exact names.",
            "4. python -m cyberllm.cli datasets validate --name unsw_nb15",
            "5. python -m cyberllm.cli models benchmark --dataset unsw_nb15",
            "If the share ever requires sign-in, contact the dataset author "
            "(nour.moustafa@unsw.edu.au) - do not work around the gate.",
        ),
        priority=1,
        version="1.0",
        year="2015",
        tags=("ids", "network-flows", "acquired"),
    ),
    DatasetSpec(
        name="cic_ids2017",
        kind=DatasetKind.NETWORK_IDS,
        description="CIC-IDS2017 intrusion detection dataset (8 attack classes).",
        source_url="https://www.unb.ca/cic/datasets/ids-2017.html",
        license="CIC datasets - free for research/educational use with citation",
        access="manual",
        expected_format=(".csv",),
        expected_columns=("Destination Port", "Flow Duration"),
        approx_size_mb=900.0,
        notes=(
            "Labelled CSVs generated by running attacks against a victim network. "
            "Very large; consider preprocessing a single day to bound disk and "
            "training time."
        ),
        manual_steps=(
            "1. Open the CIC-IDS2017 page and accept the terms.",
            "2. Download the CSV archive (MachineLearningCVE.zip or per-day CSVs).",
            "3. Extract into  data/raw/cic_ids2017/  preserving the CSV filenames.",
            "4. Run:  python -m cyberllm.cli datasets validate --name cic_ids2017",
            "5. Run:  python -m cyberllm.cli datasets preprocess --name cic_ids2017",
        ),
        priority=3,
        version="1.0",
        year="2017",
        tags=("ids", "network-flows", "large"),
    ),
    DatasetSpec(
        name="cse_cic_ids2018",
        kind=DatasetKind.NETWORK_IDS,
        description="CSE-CIC-IDS2018 IDS dataset (Bots, DoS, infiltration, web attacks).",
        source_url="https://www.unb.ca/cic/datasets/ids-2018.html",
        license="CIC datasets - free for research/educational use with citation",
        access="manual",
        expected_format=(".csv",),
        expected_columns=("Start Time", "Destination IP"),
        approx_size_mb=2000.0,
        notes=(
            "Larger than CIC-IDS2017 and includes infiltration attacks. "
            "Multi-gigabyte download."
        ),
        manual_steps=(
            "1. Open the CSE-CIC-IDS2018 page and accept the terms.",
            "2. Download the CSV archive.",
            "3. Extract into  data/raw/cse_cic_ids2018/  keeping the nested CSVs.",
            "4. Run:  python -m cyberllm.cli datasets validate --name cse_cic_ids2018",
        ),
        priority=3,
        version="1.0",
        year="2018",
        tags=("ids", "network-flows", "large"),
    ),


    # ---------------- Malware ---------------------------------------------
    DatasetSpec(
        name="cic_malmem_2022",
        kind=DatasetKind.MALWARE,
        description="CIC-MalMem-2022 memory-based malware analysis dataset.",
        source_url="https://www.unb.ca/cic/datasets/malmem2022.html",
        license="CIC datasets - free for research/educational use with citation",
        access="manual",
        expected_format=(".csv",),
        expected_columns=("Class", "PID"),
        approx_size_mb=500.0,
        notes=(
            "Volatile-memory (RAM) artefacts from 12 malware families plus benign "
            "families, including obfuscation variants. Malware samples are NEVER "
            "executed by CyberLLM; only pre-extracted feature tables are used."
        ),
        manual_steps=(
            "1. Open the CIC-MalMem-2022 page and accept the terms.",
            "2. Download the memory dump image(s) and the accompanying "
            "   MemoryDataset_Analysis / feature CSV files.",
            "3. Extract ONLY the feature CSVs into  data/raw/cic_malmem_2022/  "
            "   (do not run or mount the memory images).",
            "4. Run:  python -m cyberllm.cli datasets validate --name cic_malmem_2022",
            "5. Run:  python -m cyberllm.cli datasets preprocess --name cic_malmem_2022",
        ),
        priority=3,
        version="1.0",
        year="2022",
        tags=("malware", "memory", "features-only"),
    ),
    # ---------------- Phishing --------------------------------------------
    DatasetSpec(
        name="phishing_urls",
        kind=DatasetKind.PHISHING,
        description="Labelled phishing URL corpus (legitimate research release).",
        source_url="https://phishtank.org/",
        license="PhishTank developer terms - attribution required, see source",
        access="manual",
        expected_format=(".csv",),
        expected_columns=("url", "label"),
        approx_size_mb=5.0,
        notes=(
            "URLs are analysed entirely offline. CyberLLM never fetches a submitted "
            "URL and never contacts PhishTank unless the operator supplies their "
            "own registered API key."
        ),
        manual_steps=(
            "Option A (recommended for reproducibility) - published academic CSV:",
            "  1. Obtain a licensed phishing/benign URL CSV (e.g. the Mendeley or",
            "     IEEE DataPort releases of the 'Phishing URL Datasets').",
            "  2. Place it at  data/raw/phishing_urls/urls.csv  with columns url,label",
            "     where label is 1 for phishing and 0 for benign.",
            "Option B - PhishTank online-URL feed (needs a free app key):",
            "  1. Register at https://phishtank.com/api_info.php and get an app key.",
            "  2. Download the online-URL CSV feed.",
            "  3. Place it at  data/raw/phishing_urls/urls.csv (url,label).",
            "  4. Run:  python -m cyberllm.cli datasets validate --name phishing_urls",
        ),
        priority=2,
        version="research-mix",
        tags=("url", "phishing"),
    ),
    # ---------------- LLM training corpora --------------------------------
    DatasetSpec(
        name="cyberllm_sft",
        kind=DatasetKind.LLM_TRAINING,
        description=(
            "CyberLLM supervised fine-tuning corpus built from MITRE ATT&CK, "
            "CISA KEV and NVD records (in-repo, generated by scripts/)."
        ),
        source_url="https://github.com/yvasanthan007/CyberLLM",
        license="MIT (code); underlying records retain their original licences",
        download_url=None,
        expected_format=(".jsonl",),
        expected_columns=("instruction", "response"),
        approx_size_mb=0.0,
        notes=(
            "Present in the repository at dataset/final/cyberllm_train.jsonl. "
            "Instruction/response pairs only - network-flow tables are never mixed "
            "into language-model training data."
        ),
        priority=1,
        version="v1",
        tags=("sft", "instruction", "rag-corpus"),
    ),
)

DATASETS_BY_NAME: dict[str, DatasetSpec] = {d.name: d for d in DATASETS}


def get_dataset(name: str) -> DatasetSpec:
    """Look up a dataset declaration, raising a typed error when unknown."""
    key = name.strip().lower()
    if key not in DATASETS_BY_NAME:
        raise DatasetNotFoundError(
            f"Unknown dataset '{name}'.",
            details={"available": sorted(DATASETS_BY_NAME)},
        )
    return DATASETS_BY_NAME[key]


def list_datasets(*, kind: DatasetKind | None = None,
                  access: str | None = None) -> list[DatasetSpec]:
    """Filter the registry. Results are ordered by priority then name."""
    items = list(DATASETS)
    if kind is not None:
        items = [d for d in items if d.kind == kind]
    if access is not None:
        items = [d for d in items if d.access == access]
    return sorted(items, key=lambda d: (d.priority, d.name))


__all__ = [
    "DatasetSpec", "DatasetStatus", "DatasetKind", "DATASETS", "DATASETS_BY_NAME",
    "get_dataset", "list_datasets",
]

