"""Enumerate the UNSW-NB15 CSV folder through the publisher's public share."""
from __future__ import annotations

import json
from pathlib import Path

import requests

LINK = ("https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/"
        "EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS")
WEB = "https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au"
ROOT = "/personal/z5025758_ad_unsw_edu_au/Documents/UNSW-NB15 dataset"

session = requests.Session()
session.headers.update({"User-Agent": "cyberllm-dataset-audit/1.0",
                        "Accept": "application/json;odata=nometadata"})
session.get(LINK, timeout=90)


def list_folder(path: str, depth: int = 0) -> list[dict]:
    rows: list[dict] = []
    files = session.get(
        f"{WEB}/_api/web/GetFolderByServerRelativeUrl('{path}')/Files",
        timeout=60).json().get("value", [])
    folders = session.get(
        f"{WEB}/_api/web/GetFolderByServerRelativeUrl('{path}')/Folders",
        timeout=60).json().get("value", [])
    for item in files:
        rows.append({"kind": "file", "name": item["Name"],
                     "bytes": int(item.get("Length") or 0),
                     "modified": item.get("TimeLastModified"),
                     "url": WEB + item["ServerRelativeUrl"] + "?download=1"})
    for item in folders:
        rows.append({"kind": "folder", "name": item["Name"],
                     "item_count": item.get("ItemCount")})
        if depth < 2:
            rows.extend(list_folder(item["ServerRelativeUrl"], depth + 1))
    return rows


entries = list_folder(ROOT)
out = Path(__file__).with_name("_unsw_file_listing.json")
out.write_text(json.dumps(entries, indent=2), encoding="utf-8")
total = 0
for row in entries:
    if row["kind"] == "file":
        total += row["bytes"]
        print(f"FILE   {row['bytes']:>14,}  {row['name']}")
    else:
        print(f"DIR    {row.get('item_count', '?'):>14}  {row['name']}/")
print(f"\ntotal bytes: {total:,} ({total / 1048576:.1f} MB)")
print("written ->", out)
