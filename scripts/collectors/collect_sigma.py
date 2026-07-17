"""
CyberLLM
Sigma Collector

Downloads Sigma Rules from GitHub.
"""

from pathlib import Path
from git import Repo

from configs.config import SIGMA_DATASET

SIGMA_REPO = "https://github.com/SigmaHQ/sigma.git"


def clone_sigma():

    if any(SIGMA_DATASET.iterdir()):
        print("=" * 60)
        print("Sigma dataset already exists.")
        print(SIGMA_DATASET)
        print("=" * 60)
        return

    print("=" * 60)
    print("Downloading Sigma Rules")
    print("=" * 60)

    Repo.clone_from(
        SIGMA_REPO,
        SIGMA_DATASET
    )

    print("=" * 60)
    print("Download Complete")
    print(SIGMA_DATASET)
    print("=" * 60)


if __name__ == "__main__":
    clone_sigma()