"""One-off patch: update the unsw_nb15 registry entry.

Split out as a script because the file's CRLF line endings defeat exact-text
matching, and this block is being rewritten wholesale rather than tweaked.
"""
from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "cyberllm" / "datasets" / "registry.py"

NEW_BLOCK = '''    DatasetSpec(
        name="unsw_nb15",
        kind=DatasetKind.NETWORK_IDS,
        description="UNSW-NB15 network flow dataset (real + synthetic traffic).",
        source_url="https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        license=(
            "UNSW dataset terms - free for ACADEMIC RESEARCH in perpetuity; "
            "commercial use must be agreed with the authors; the five "
            "published papers MUST be cited"
        ),
        download_url=(
            "https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/"
            "EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS"
        ),
        access="public",
        expected_format=(".csv",),
        expected_columns=("proto", "service", "state", "dur", "sbytes", "label",
                          "attack_cat"),
        approx_size_mb=48.0,
        notes=(
            "Ten classes (Normal + 9 attack families) over 40 Argus/Bro-IDS "
            "flow statistics. The publisher's folder share is public (anyone "
            "with the link) so no registration is needed, but citing the five "
            "papers is a condition of the licence. Hybrid corpus - real normal "
            "traffic plus tool-generated attacks - so accuracy is an upper "
            "bound, not a field estimate. See cyberllm/datasets/unsw_nb15.py."
        ),
        manual_steps=(
            "1. Open the UNSW-NB15 dataset page and read the usage terms.",
            "2. Click 'HERE' to open the public OneDrive folder, then download",
            "   CSV Files/NUSW-NB15_features.csv,",
            "   CSV Files/UNSW_NB15_training-set.csv and",
            "   CSV Files/UNSW_NB15_testing-set.csv.",
            "3. Place them in  data/raw/unsw_nb15/  keeping these exact names.",
            "4. python -m cyberllm.cli datasets validate --name unsw_nb15",
            "5. python -m cyberllm.cli models benchmark --dataset unsw_nb15",
            "If the share ever requires sign-in, contact the dataset author "
            "(nour.moustafa@unsw.edu.au) - do not work around the gate.",
        ),
        priority=1,
        version="1.0",
        year="2015",
        tags=("ids", "network-flows", "acquired"),
    ),'''

text = TARGET.read_text(encoding="utf-8")
lines = text.splitlines(keepends=True)

start = next(i for i, line in enumerate(lines)
             if line.strip() == 'name="unsw_nb15",') - 1
end = next(i for i in range(start, len(lines)) if lines[i].rstrip() == "    ),")
print("replacing lines", start + 1, "to", end + 1)
print("first:", lines[start].strip())
print("last :", lines[end].strip())

newline = "\r\n" if lines[start].endswith("\r\n") else "\n"
block = NEW_BLOCK.replace("\n", newline) + newline
TARGET.write_text("".join(lines[:start]) + block + "".join(lines[end + 1:]),
                  encoding="utf-8")
print("written", TARGET)
