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
| `unsw_nb15` | network flows | manual | 250 MB | **manual** - terms acceptance required |
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

### UNSW-NB15
1. Open <https://research.unsw.edu.au/projects/unsw-nb15-dataset> and accept
   the agreement.
2. Download the CSV feature sets.
3. Place them in `data/raw/unsw_nb15/` (`UNSW_NB15_features.csv`,
   `UNSW_NB15_list_of_labelled_flows.csv`, `NUSW-NB15_features.csv`).
4. `python -m cyberllm.cli datasets validate --name unsw_nb15`
5. `python -m cyberllm.cli models train --task nids --data data/raw/unsw_nb15/UNSW_NB15_features.csv`

*Licence*: UNSW dataset terms - academic use, citation required.
*Caveat*: contains both real and tool-generated synthetic flows; report that
when quoting accuracy.

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
