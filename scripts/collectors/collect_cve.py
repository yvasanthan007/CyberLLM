"""
CyberLLM
CVE Collector
"""

import json
import time
import traceback
import requests

from configs.config import NVD_API_KEY, CVE_DATASET

BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

RESULTS_PER_PAGE = 2000


def download_batch(start_index):

    headers = {}

    if NVD_API_KEY:
        headers["apiKey"] = NVD_API_KEY

    params = {
        "startIndex": start_index,
        "resultsPerPage": RESULTS_PER_PAGE
    }

    print(f"Downloading from index {start_index}")

    response = requests.get(
        BASE_URL,
        headers=headers,
        params=params,
        timeout=60
    )

    print("Status Code:", response.status_code)

    response.raise_for_status()

    return response.json()


def save_batch(data, batch):

    filename = CVE_DATASET / f"batch_{batch:05d}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Saved", filename)


def main():

    print("=" * 60)
    print("CyberLLM CVE Collector")
    print("=" * 60)

    print("API Key Loaded:", bool(NVD_API_KEY))

    start = 0
    batch = 1

    while True:

        try:

            data = download_batch(start)

            vulns = data.get("vulnerabilities", [])

            if len(vulns) == 0:
                print("Finished")
                break

            save_batch(data, batch)

            start += RESULTS_PER_PAGE
            batch += 1

            time.sleep(2)

        except Exception:

            traceback.print_exc()

            break


if __name__ == "__main__":
    main()