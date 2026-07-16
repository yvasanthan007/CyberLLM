"""
CyberLLM
MITRE ATT&CK Collector

Downloads the MITRE ATT&CK Enterprise dataset.
"""

import requests

from configs.config import MITRE_DATASET

URL = (
    "https://raw.githubusercontent.com/"
    "mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)

OUTPUT_FILE = MITRE_DATASET / "enterprise-attack.json"


def download_dataset():
    print("=" * 60)
    print("Downloading MITRE ATT&CK Enterprise Dataset")
    print("=" * 60)

    response = requests.get(URL, timeout=60)
    response.raise_for_status()

    with open(OUTPUT_FILE, "wb") as f:
        f.write(response.content)

    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    download_dataset()