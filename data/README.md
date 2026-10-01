# Data directory

Datasets are **not** in Git. This directory holds only the manifest and the
documentation needed to reproduce an acquisition.

## Layout

```
data/
  README.md              this file
  dataset_manifest.csv   per-dataset acquisition state (committed)
  raw/<dataset_id>/      downloaded source files      (git-ignored)
  processed/<dataset_id>/ derived artefacts            (git-ignored)
  synthetic/             locally generated fixtures    (git-ignored)
```

`data/raw/` and `data/processed/` are git-ignored by design: they are large,
separately licensed, and reproducible from the registry.

## `dataset_manifest.csv`

One row per dataset, always committed, so both developers can see what the
other actually has.

| Column | Meaning |
|---|---|
| `name`, `kind` | Registry identity |
| `status` | `registered` / `downloadable` / `manual` / `downloaded` / `validated` / `preprocessed` / `unavailable` / `synthetic` |
| `access` | `public` / `api-key` / `manual` |
| `license`, `version`, `year` | Provenance |
| `source_url`, `download_url` | Where it came from |
| `retrieved_at`, `file_size_bytes`, `sha256` | What was fetched, and its digest |
| `row_count`, `column_count` | Shape of the validated file |
| `preprocess_status`, `preprocess_path`, `preprocessed_at` | Derived-artefact state |
| `error` | Last failure, if any - recorded, not hidden |

**Status is reconciled against the filesystem, not trusted blindly.** If a file
is deleted, `datasets list` reports `registered` again rather than continuing
to claim `downloaded`.

## Regenerating

```bash
python -m cyberllm.cli datasets list
python -m cyberllm.cli datasets validate
```

Both rewrite the manifest from real state. The file is written to a temporary
path and swapped, so an interrupted write cannot corrupt it.

## Acquiring data

Automated sources:

```bash
python -m cyberllm.cli datasets download --all
```

Manual sources (terms must be accepted by a human) have exact steps recorded
in the registry and surfaced by `datasets list --name <id>`; see
[docs/datasets.md](../docs/datasets.md).

CyberLLM does not bypass authentication, paywalls or access controls, and does
not invent download URLs.
