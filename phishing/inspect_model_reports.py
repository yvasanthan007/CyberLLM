import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "artifacts" / "reports" / "phishing"

REPORTS = [
    "baseline_url_evaluation.json",
    "candidate_evaluation.json",
    "hostname_candidate_evaluation.json",
    "targeted_evaluation.json",
    "model_diagnostic_comparison.json",
]

for filename in REPORTS:
    path = REPORT_DIR / filename

    print(f"\n{'=' * 70}")
    print(f"REPORT: {filename}")
    print("=" * 70)

    if not path.exists():
        print("Report not found.")
        continue

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(data, indent=2)[:12000])
    except Exception as exc:
        print(f"Could not read report: {type(exc).__name__}: {exc}")
