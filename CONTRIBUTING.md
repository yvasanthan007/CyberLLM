# Contributing

CyberLLM is developed by **two people**. This document defines who owns what,
how branches are used, and the rules that keep each developer unblocked.

## Ground rules

1. **Never commit directly to `main` or `develop`.** Work on a feature branch.
2. **Never commit datasets, model weights or artefacts.** They are git-ignored.
   Commit `data/dataset_manifest.csv` instead - it records the same information
   without the bulk.
3. **Never commit a secret.** No API keys in code, config, logs or notebooks.
   The log filter redacts credential-shaped strings, but that is a backstop,
   not permission.
4. **Green before merge.** `python -m pytest tests -q` and
   `python scripts/static_check.py` must both be clean.
5. **Update `PROJECT_STATUS.md`** in the same PR that changes what is verified.

## Branching

```
main        stable, releasable
  └── develop        integration branch
        ├── feature/<scope>-<short-description>
```

Workflow per unit of work:

```bash
git checkout develop
git pull
git checkout -b feature/detection-unsw-nb15
# ... work ...
python -m pytest tests -q
python scripts/static_check.py
git add -A && git commit
git push -u origin feature/detection-unsw-nb15
# open a PR into develop
```

After review and CI: squash-merge into `develop`. `develop` moves to `main` on
release. **Do not merge, push, delete branches or rewrite history without
explicit agreement from the other developer.**

## Module ownership

Each area has one primary owner who reviews changes there. Review is
required on the public contracts, not on every line.

| Area | Owner (A) | Owner (B) |
|---|---|---|
| `cyberllm/rag/*`, `cyberllm/datasets/*` | ✅ | consult |
| `cyberllm/detection/*`, `cyberllm/phishing/*`, `cyberllm/malware/*` | | ✅ |
| `cyberllm/api/*`, `cyberllm/app/*` | ✅ | consult |
| `cyberllm/llm/*`, `cyberllm/service/*` | | ✅ |
| `tests/*` | ✅ (author of the change) | ✅ (author of the change) |
| `scripts/*`, CI | | ✅ |

**Shared files - coordinate before editing.** These change rarely but affect
everyone; announce an intent to change one in the PR description or in chat
first, and avoid having both people edit them in the same session:

- `cyberllm/core/config.py` - adding a config field touches every module
- `cyberllm/core/errors.py` - the error contract
- `cyberllm/cli.py` - command surface
- `requirements.txt`, `.github/workflows/*` - dependency and CI changes
- `.env.example`, `PROJECT_STATUS.md`

## Conflict resolution

- Rebase on `develop` before requesting review; do not merge `develop` into a
  feature branch.
- Resolve conflicts **in the feature branch**, never by editing `develop`.
- If a conflict touches a shared file listed above, the branch owner resolves
  it and the other developer confirms.
- Never "resolve" by deleting the other person's tests or checks to make CI
  pass. If a test is genuinely wrong, change it in a separate, explained
  commit.

## Code standards

See [docs/coding_standards.md](docs/coding_standards.md). The short version:

- type hints on public functions; `from __future__ import annotations`
- docstrings that explain *why*, not what
- no bare `except Exception: pass` - catch specifically or log
- no backslash line continuations - use parentheses
- every new module must have a real purpose; no filler modules
- tests cover the success path, invalid input and the unconfigured path

## Pull request checklist

- [ ] Branch is not `main`/`develop`
- [ ] `python -m pytest tests -q` passes
- [ ] `python scripts/static_check.py` reports 0 issues
- [ ] No dataset, artefact, weight or secret in the diff
- [ ] New unconfigured behaviour returns a typed error with a `setup_hint`
- [ ] `PROJECT_STATUS.md` updated with results, not intentions
- [ ] Synthetic results are labelled as such and not presented as benchmarks
