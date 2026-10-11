"""Determine whether the publisher's UNSW-NB15 share grants anonymous read.

Follows the publisher's own link, then asks the *same* OneDrive/SharePoint
session to list the shared folder - the API a browser uses when a human clicks
"Download". No credentials are supplied and nothing is bypassed: if the share
is not public, this reports a gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

SOURCE = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
LINK = ("https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/"
        "EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS")
WEB = "https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au"
FOLDER = "/personal/z5025758_ad_unsw_edu_au/Documents/UNSW-NB15 dataset"

OUT = Path(__file__).with_name("_unsw_probe_result.json")
report: dict[str, object] = {"source_page": SOURCE, "download_link": LINK}

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) cyberllm-dataset-audit/1.0",
    "Accept": "application/json;odata=nometadata, text/html, */*",
})

# 1. Consume the share link to obtain whatever anonymous grant exists.
landing = session.get(LINK, timeout=90)
report["landing_status"] = landing.status_code
report["landing_url"] = landing.url
report["cookies"] = sorted(session.cookies.get_dict().keys())

# 2. Ask that same session to enumerate the shared folder.
attempts = {
    "rest_files": f"{WEB}/_api/web/GetFolderByServerRelativeUrl('{FOLDER}')/Files",
    "rest_folders": f"{WEB}/_api/web/GetFolderByServerRelativeUrl('{FOLDER}')/Folders",
    "onedrive_root": f"{WEB}/_api/v2.0/drives/root/children",
}
results: dict[str, object] = {}
for key, url in attempts.items():
    try:
        resp = session.get(url, timeout=60)
        entry: dict[str, object] = {
            "status": resp.status_code,
            "content_type": resp.headers.get("Content-Type", "")[:60],
            "bytes": len(resp.content),
        }
        if "json" in resp.headers.get("Content-Type", ""):
            try:
                entry["json"] = resp.json()
            except ValueError:
                entry["body_head"] = resp.text[:400]
        else:
            entry["body_head"] = resp.text[:400]
        results[key] = entry
    except requests.RequestException as exc:
        results[key] = {"error": f"{type(exc).__name__}: {exc}"}
report["api_attempts"] = results

OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
print(json.dumps(report, indent=2, default=str)[:4000])
print(f"\nwritten -> {OUT}")
