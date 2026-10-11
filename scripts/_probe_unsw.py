"""Check whether the publisher's official UNSW-NB15 download is openly reachable.

This performs a plain, unauthenticated GET against the exact link published on
https://research.unsw.edu.au/projects/unsw-nb15-dataset. It supplies no
credentials, submits no agreement and circumvents no access control. The only
question it answers is: "does an anonymous operator get the data, or a gate?"
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests

SOURCE = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
LINK = ("https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/"
        "EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS")

OUT = Path(__file__).with_name("_unsw_probe_result.json")
report: dict[str, object] = {"source_page": SOURCE, "download_link": LINK}

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) cyberllm-dataset-audit/1.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})

try:
    resp = session.get(LINK, timeout=90, allow_redirects=True)
    report.update({
        "http_status": resp.status_code,
        "final_url": resp.url,
        "content_type": resp.headers.get("Content-Type", ""),
        "content_length": resp.headers.get("Content-Length", ""),
        "history": [f"{h.status_code} -> {h.headers.get('Location', '')}"
                    for h in resp.history],
        "body_bytes": len(resp.content),
    })
    ctype = resp.headers.get("Content-Type", "")
    if "text/html" in ctype or "html" in ctype.lower():
        text = resp.text
        report["signals"] = {
            token: bool(re.search(token, text, flags=re.I))
            for token in (
                "sign in", "signin", "login", "authenticate", "request access",
                "access denied", "you don.t have permission", "session expired",
                "this site is being shared", "download", "folders",
            )
        }
        titles = re.findall(r"<title>(.*?)</title>", text, flags=re.I | re.S)
        report["title"] = titles[0].strip() if titles else ""
        report["html_head"] = re.sub(r"\s+", " ", text[:600])
except requests.RequestException as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"

OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
print(json.dumps(report, indent=2, default=str)[:3000])
print(f"\nwritten -> {OUT}")
