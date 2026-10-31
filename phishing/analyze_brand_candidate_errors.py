import json
from pathlib import Path

report_path = Path(
    "artifacts/reports/phishing/brand_candidate_evaluation.json"
)

report = json.loads(
    report_path.read_text(encoding="utf-8")
)

results = report["targeted_diagnostic"]["results"]

print("=" * 90)
print("CYBERLLM TARGETED ERROR ANALYSIS")
print("=" * 90)

for item in results:
    expected = item["expected_label"]
    candidate = item["candidate_label"]
    production = item["production_label"]

    expected_name = "PHISHING" if expected == 0 else "LEGITIMATE"
    candidate_name = "PHISHING" if candidate == 0 else "LEGITIMATE"
    production_name = "PHISHING" if production == 0 else "LEGITIMATE"

    candidate_status = "PASS" if item["candidate_correct"] else "FAIL"
    production_status = "PASS" if item["production_correct"] else "FAIL"

    print(f"\n{item['name']}")
    print(f"  URL: {item['url']}")
    print(f"  Expected: {expected_name}")
    print(
        f"  Candidate: {candidate_name} "
        f"(score={item['candidate_phishing_score']:.6f}, "
        f"{candidate_status})"
    )
    print(
        f"  Production: {production_name} "
        f"(score={item['production_phishing_score']:.6f}, "
        f"{production_status})"
    )

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

candidate_failures = [
    item for item in results
    if not item["candidate_correct"]
]

production_failures = [
    item for item in results
    if not item["production_correct"]
]

print(f"Candidate failures: {len(candidate_failures)}")
print(f"Production failures: {len(production_failures)}")

print("\nCandidate failures:")
for item in candidate_failures:
    print(f"- {item['name']}: {item['url']}")

print("\nProduction failures:")
for item in production_failures:
    print(f"- {item['name']}: {item['url']}")

print("\nProduction model unchanged:",
      report["production_model"]["unchanged"])
