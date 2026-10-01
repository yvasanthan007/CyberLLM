# Security, privacy and limitations

CyberLLM is a **defensive** security tool. This document states what it
protects against, what it deliberately does not do, and where its output can be
wrong.

## Absolute constraints

These are enforced in code, not just documented:

| Constraint | Where enforced |
|---|---|
| Malware is never executed, loaded, mounted or emulated | `cyberllm/malware/` accepts numeric features only; no loader, sandbox or emulator exists |
| Submitted URLs are never fetched | `phishing/features.py` imports nothing that opens a socket; no DNS/HTTP/WHOIS call exists |
| The LLM cannot execute anything | No tool-calling, no shell, no subprocess in the generation path |
| Retrieved text is data, never instructions | `scan_prompt_injection` + `wrap_untrusted` before any prompt assembly |
| No secret is logged | `SecretRedactingFilter` on every handler, plus no secret is ever passed to a logger |
| Oversized inputs are rejected before parsing | `Content-Length` check in API middleware; size checks before every read |

## Threat model

**Assets**: threat-intel corpus integrity, host credentials, model artefacts,
analyst-supplied URLs and flow records.

| Threat | Control |
|---|---|
| Prompt injection via a poisoned document | Injection scan, delimiter wrapping, trust notice outside content tags, system preamble stating the boundary |
| Path traversal via a dataset id | `safe_local_path` resolves and checks containment before any file access |
| SSRF / credential harvesting through submitted URLs | Phishing analysis is purely lexical; the URL is never resolved |
| Oversized request / memory exhaustion | Body-size middleware, per-field Pydantic bounds, batch caps, row caps |
| Secret leakage in logs | Redaction filter; malformed records are rewritten rather than dumped by `Handler.handleError` |
| Model artefact tampering | Artefacts live outside the web root; the API only reads them |
| Unauthorised network access | Service refuses a non-localhost bind unless `CYBERLLM_API_KEY` is set |

## Network exposure

The service ships **without TLS and with authentication opt-in**. That is a
deliberate, documented posture, not an oversight:

- default bind is `127.0.0.1` only;
- `cyberllm.cli app --host 0.0.0.0` is **refused** unless
  `CYBERLLM_API_KEY` is set;
- even with a key, put the service behind a TLS-terminating reverse proxy and
  a real identity provider before exposing it to a network.

The optional `X-API-Key` check uses `hmac.compare_digest` so the key cannot be
recovered from response timing.

## Privacy

- Chat and analysis history is held in the browser session only. The console
  states this and provides a clear control; nothing is persisted server-side by
  the UI.
- A submitted URL never leaves the process.
- A submitted flow record is scored in-process and not stored.
- Dataset downloads contact only the publishers listed in the registry.

## Known limitations

**Classifiers are probabilistic.** A network-intrusion or phishing model
estimates a probability. False positives and false negatives are expected on
real traffic. A high score is a reason to investigate, not proof of an attack.

**Lab data is not production traffic.** CIC-IDS2017, CSE-CIC-IDS2018 and
UNSW-NB15 are generated in a lab, with known tooling and known attacks.
Expect degradation under concept drift, novel attacks and encrypted traffic.
Some published releases are anonymised or stripped, which removes signal the
model would otherwise use.

**Synthetic results are not benchmarks.** Any model trained on
`data/raw/synthetic/` is marked `is_synthetic` in its model card, is never
ranked above a real-data model, and its metrics must not be quoted as
field accuracy.

**Retrieval is lexical by default.** The hashing embedder is a deterministic
bag-of-words projection, not a semantic model. It is fast and dependency-free
but recall is keyword-driven. Identifiers are handled by an exact index, which
is why CVE lookups are reliable; free-text semantic recall is weaker until
`sentence-transformers` is enabled.

**Knowledge is a snapshot.** CISA KEV and NVD change daily. Re-run
`datasets download` and `rag ingest` to refresh, and check the document's
`retrieved_at` before relying on it for an incident.

**The LLM can still be wrong.** It is instructed not to invent CVE ids or CVSS
scores, but instruction is not enforcement. Always confirm against the cited
source, which is why every answer carries numbered citations.

**Weak evidence is flagged, not hidden.** When retrieval scores are low the
response sets `weak_evidence=true` and says the answer is not sourced.

## Responsible use

CyberLLM supports authorised defensive work: detection engineering, triage,
threat research, incident response. It declines offensive tooling, credential
theft and evasion techniques, and its system preamble states these limits.
Operators remain responsible for how they act on its output.
