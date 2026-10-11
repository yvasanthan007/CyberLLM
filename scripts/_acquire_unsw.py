"""One-off driver: run the UNSW-NB15 acquisition and report what landed."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cyberllm.datasets.unsw_nb15 import download_unsw_nb15, provenance  # noqa: E402

target = Path(sys.argv[1] if len(sys.argv) > 1 else r"d:\CyberLLM\data\raw\unsw_nb15")
group = sys.argv[2] if len(sys.argv) > 2 else "core"

started = time.monotonic()
results = download_unsw_nb15(target, group=group, timeout=300)
elapsed = time.monotonic() - started

summary = []
for result in results:
    summary.append({
        "file": Path(result.path).name if result.path else None,
        "ok": result.ok,
        "status": result.status,
        "bytes": result.bytes_downloaded,
        "sha256": result.sha256,
        "error": result.error,
    })
    print(f"{'OK ' if result.ok else 'FAIL'} {result.status:<10} "
          f"{result.bytes_downloaded:>14,}  "
          f"{Path(result.path).name if result.path else '?'}  {result.error or ''}")

record = provenance({
    Path(r.path).name: {"bytes": r.bytes_downloaded, "sha256": r.sha256}
    for r in results if r.path and r.ok
})
out = target / "PROVENANCE.json"
out.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(f"\nelapsed {elapsed:.1f}s -> wrote {out}")
