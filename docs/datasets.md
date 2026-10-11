# Datasets

Every dataset is declared once in `cyberllm/datasets/registry.py` with its
provenance, licence and acquisition route. `data/dataset_manifest.csv` records
the *actual* state of each one on disk.

**Nothing here is downloaded by default.** Only sources that permit automated
access are fetched; the rest record the exact manual steps and their licences.

## Acquisition status

| Dataset | Kind | Access | Size | Status |
|---|---|---|---|---|
| `cisa_kev` | threat intel | public | 1.7 MB | **acquired + validated** (1,730 records) |
| `mitre_attack` | threat intel | public | 46 MB | **acquired + validated** (26,085 STIX objects) |
| `nvd_cve` | threat intel | API key | varies | endpoint verified; needs a key or a small pull |
| `unsw_nb15` | network flows | **public share** | 46 MB | **acquired + validated** (257,673 flows) |
| `cic_ids2017` | network flows | manual | 900 MB | **manual** - terms acceptance required |
| `cse_cic_ids2018` | network flows | manual | 2 GB | **manual** - terms acceptance required |
| `cic_malmem_2022` | malware | manual | 500 MB | **manual** - terms acceptance required |
| `phishing_urls` | phishing | manual | 5 MB | **manual** - licensed research release |
| `cyberllm_sft` | LLM training | in-repo | small | present at `dataset/final/cyberllm_train.jsonl` |

Total if everything were acquired: **~3.7 GB**. Prioritise by need; the
knowledge base works with CISA KEV + MITRE ATT&CK alone.

## Automated datasets

```bash
python -m cyberllm.cli datasets download --all
python -m cyberllm.cli datasets validate
python -m cyberllm.cli datasets preprocess   # indexes into the vector store
```

### CISA KEV (acquired)
- **Source**: <https://www.cisa.gov/known-exploited-vulnerabilities-catalog>
- **Licence**: US Government work; CISA terms of use apply
- **Why it matters**: the highest-signal source in the project - CVEs with
  *evidence of active exploitation*, plus mandated federal remediation dates.
- Attribution: CISA. NVD links are generated for each record.

### MITRE ATT&CK (acquired)
- **Source**: <https://attack.mitre.org/>, STIX 2.1 from `mitre/cti`
- **Licence**: MITRE ATT&CK terms of use; **attribution required**
- **Note**: revoked and deprecated objects are skipped at ingestion, so stale
  entries are never retrieved as if current.

### UNSW-NB15 (acquired)

This one is worth reading in full, because "automated" has limits.

- **Source page**: <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
  (UNSW Canberra at ADFA, Cyber Range Lab).
- **Download route**: the page's only link points at a OneDrive/SharePoint
  folder published as *anyone with the link*. Consuming it issues an anonymous
  `FedAuth` cookie which then authorises SharePoint's own file endpoint - the
  same request a browser makes when a researcher clicks Download. No account,
  no registration, no credentials.
- **Licence**: *"Free use of the UNSW-NB15 dataset for academic research
  purposes is hereby granted in perpetuity. Use for commercial purposes should
  be agreed by the authors."* **Citing the five published papers is a
  condition of the licence**, not a courtesy; the citation block is embedded in
  `cyberllm/datasets/unsw_nb15.py:CITATION` and copied into every model card,
  benchmark report and provenance record CyberLLM writes.
- **What CyberLLM will not do**: sign in with an operator's credentials,
  accept terms on their behalf, or work around a gate. If the share ever
  requires sign-in, `UnswShare.open()` raises a typed error containing the
  author's contact address and the manual steps, and acquisition stops there.

```bash
python -m cyberllm.cli datasets download --name unsw_nb15   # ~46 MB
python -m cyberllm.cli datasets validate --name unsw_nb15
python -m cyberllm.cli models benchmark --dataset unsw_nb15 # ~4 min, CPU only
```

Files land in `data/raw/unsw_nb15/` with their publisher filenames intact,
including the `NUSW-NB15_features.csv` typo - "correcting" it would make the
download fail.

Implementation: `cyberllm/datasets/unsw_nb15.py`. Results:
[docs/benchmarks.md](benchmarks.md).

### NVD (needs a key for anything large)
- **Source**: <https://nvd.nist.gov/vuln/data-feeds>
- **Rate limit**: ~5 requests/30 s keyless, ~50 requests/30 s with a free key.
- Get a key: <https://nvd.nist.gov/developers/request-an-api-key>, then
  `set NVD_API_KEY=<key>` and
  `python -m cyberllm.cli datasets download --name nvd_cve`.
- Keyless runs are capped by `CYBERLLM_NVD_MAX_RECORDS` (default 400).

## Manual datasets

These require accepting a publisher's terms. CyberLLM will not bypass
authentication, paywalls or access controls, and does not fabricate download
URLs. Follow the exact steps below, then run `datasets validate` and
`datasets preprocess`.

### UNSW-NB15 (manual fallback)

The automated route above normally works. If the share stops being public,
these are the steps - the CLI prints them verbatim on failure.

1. Open <https://research.unsw.edu.au/projects/unsw-nb15-dataset> and read the
   usage terms.
2. Click "HERE" to open the shared OneDrive folder.
3. If OneDrive asks you to sign in, **stop**. Do not work around the gate -
   contact the dataset author (nour.moustafa@unsw.edu.au).
4. Download `CSV Files/NUSW-NB15_features.csv`,
   `CSV Files/UNSW_NB15_training-set.csv` and
   `CSV Files/UNSW_NB15_testing-set.csv`.
5. Place them in `data/raw/unsw_nb15/` keeping those exact filenames.
6. `python -m cyberllm.cli datasets validate --name unsw_nb15`
7. `python -m cyberllm.cli models benchmark --dataset unsw_nb15`

*Licence*: UNSW dataset terms - academic research use; commercial use requires
the authors' agreement; the five papers must be cited.
*Caveat*: hybrid corpus (real normal traffic + tool-generated attacks) and no
IP/port columns in the released partitions - see
[docs/benchmarks.md](benchmarks.md) for what that does to the numbers.

### CIC-IDS2017
1. Open <https://www.unb.ca/cic/datasets/ids-2017.html> and accept the terms.
2. Download the CSV archive.
3. Extract into `data/raw/cic_ids2017/` preserving filenames.
4. Validate, then train with `--data <one-csv>`.
5. Prefer a single day (e.g. Tuesday) to bound disk and training time.

### CSE-CIC-IDS2018
As above, from <https://www.unb.ca/cic/datasets/ids-2018.html>; extract into
`data/raw/cse_cic_ids2018/` keeping the nested 4-level CSVs. ~2 GB.

### CIC-MalMem-2022
1. Open <https://www.unb.ca/cic/datasets/malmem2022.html> and accept terms.
2. Download the memory images **and** the feature CSVs.
3. Extract **only the feature CSVs** into `data/raw/cic_malmem_2022/`.
   Do not run or mount the memory images - CyberLLM never executes samples.
4. `python -m cyberllm.cli models train --task malware --data <features.csv>`

### Phishing URLs
Option A (recommended, reproducible): obtain a licensed academic
phishing/benign URL release and place it at
`data/raw/phishing_urls/urls.csv` with columns `url,label` (1 = phishing,
0 = benign).

Option B: a PhishTank app key from <https://phishtank.com/api_info.php>, then
the online-URL feed, same layout.

CyberLLM never contacts PhishTank on its own and never fetches a submitted
URL.

## Synthetic fixtures

`cyberllm.detection.fixtures` generates a labelled flow table when no real
dataset is present, so the pipeline is testable in CI. It is:

- written to `data/raw/synthetic/` (git-ignored);
- flagged `is_synthetic=True` in every model card and API response;
- **never** ranked above a real-data model in a comparison;
- excluded from any accuracy claim.

## Adding a dataset

1. Add a `DatasetSpec` to `cyberllm/datasets/registry.py` with the source URL,
   licence, expected format, expected columns and size.
2. If it needs a loader, add one in `cyberllm/rag/loaders.py` and register it
   in `LOADERS`.
3. Run `python -m cyberllm.cli datasets list --name <id>` to confirm it
   appears, then `download`, `validate` and `preprocess`.
4. `data/dataset_manifest.csv` updates automatically; commit it - it is the
   shared record of what each developer actually has.
