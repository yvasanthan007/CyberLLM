"""Find the working anonymous file-content endpoint on the public share."""
from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

import requests

LISTING = Path(__file__).with_name("_unsw_file_listing.json")
LINK = ("https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/"
        "EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS")
WEB = "https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au"

entries = json.loads(LISTING.read_text(encoding="utf-8"))
s = requests.Session()
s.headers.update({"User-Agent": "cyberllm-dataset-audit/1.0"})
s.get(LINK, timeout=90)

entry = next(e for e in entries if e["kind"] == "file" and "features" in e["name"])
raw = entry["url"].split(WEB, 1)[1].split("?download=1", 1)[0]
quoted = urllib.parse.quote(raw, safe="/:@")
api_quoted = raw.replace("'", "''")

candidates = [
    ("file_value", f"{WEB}/_api/web/GetFileByServerRelativeUrl('{api_quoted}')/$value"),
    ("file_downloadurl",
     f"{WEB}/_api/web/GetFileByServerRelativeUrl('{api_quoted}')?$select=ServerRelativeUrl,Length"),
    ("layout_download",
     "https://unsw-my.sharepoint.com/_layouts/15/download.aspx?SourceUrl="
     + quoted),
    ("raw_path", WEB + quoted),
]

for name, url in candidates:
    try:
        r = s.get(url, timeout=90, stream=True, allow_redirects=True)
        ctype = r.headers.get("Content-Type", "")
        body = b""
        if r.status_code == 200 and "json" not in ctype:
            body = r.raw.read(300, decode_content=True)
        print(f"\n[{name}] status={r.status_code} type={ctype} "
              f"len={r.headers.get('Content-Length')}")
        print("   disp:", r.headers.get("Content-Disposition"))
        print("   head:", (body or r.content[:300])[:250])
    except requests.RequestException as exc:
        print(f"\n[{name}] ERROR {type(exc).__name__}: {exc}")
