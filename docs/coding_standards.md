# Coding standards

Shared conventions for both developers. These exist because this project has
already been bitten by the failure modes they prevent.

## Formatting

- 4-space indent, ~88 column soft limit.
- `from __future__ import annotations` at the top of every module.
- Type hints on public functions. Internal helpers may be looser.
- `snake_case` for functions and variables, `PascalCase` for classes,
  `UPPER_SNAKE` for module constants.

## No backslash line continuations

```python
# Bad - and it hides broken conditionals
value = f"a {x}" \
        f"b {y}"
if cond \
        if other:

# Good
value = (f"a {x}"
         f"b {y}")
```

`scripts/static_check.py` fails the build on these. They are legal Python,
which is exactly why they are dangerous: an interrupted edit can leave a
dangling `\` that still parses.

## Docstrings explain why

```python
# Bad - restates the signature
def search(self, query):
    """Search the store."""

# Good - explains the decision
def search(self, query, top_k=None):
    """Search by exact identifier first, then by vector similarity.

    Identifiers are keys, not fuzzy terms: at a few hundred dimensions
    hash collisions rank unrelated documents above the one containing the
    CVE id the user asked about.
    """
```

## Error handling

- Raise a typed error from `core/errors.py` with a `setup_hint`, so an
  unconfigured capability teaches instead of failing.
- Catch specific exception types.
- Never `except Exception: pass` - the static check fails on it.
- Never log from inside a log filter, or from an exception handler on the
  logging path; that recurses.

## Untrusted input

All external strings pass through `security/guards.py`. No ad-hoc validation
in handlers. Bounds are declared in the Pydantic schema, not enforced ad hoc.

## Honesty requirements

These are project invariants, not style preferences:

- A capability that is not configured returns a typed error with setup
  instructions - never a placeholder result.
- Demo mode is labelled at every surface: response field, CLI banner, UI
  banner, model card.
- Synthetic training data is flagged in the model card, the API response and
  the comparison, and is never ranked above a real-data model.
- Model cards record seed, split strategy, library versions, data hash and
  stated limitations.

## Tests

Each new behaviour needs:

1. the success path,
2. invalid input,
3. the unconfigured path,
4. a regression test for any bug fixed (name it after the failure mode).

Use synthetic fixtures in tests. Real-data benchmarks live outside the suite.

## Artefacts

- Never commit data, weights or artefacts; they are git-ignored.
- Commit `data/dataset_manifest.csv` - it is the shared record of what each
  developer has acquired.
- Give every artefact a provenance header: source, licence, version, hash.

