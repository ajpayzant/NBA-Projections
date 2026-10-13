"""Where manual lineup overrides (who plays, who starts, minutes) live.

On Streamlit Cloud the container disk is wiped on every redeploy, so overrides go to a secret
GitHub gist (one shared set for every visitor), configured in Streamlit secrets:

    gist_id      = "<id of a secret gist (it may be the NHL tool's: this tool uses its own file, nba_lineups.json)>"
    github_token = "<fine-grained token with Gists: read and write>"

Without those secrets (e.g. running locally) overrides are the file data/overrides/lineups.json.
"""
from __future__ import annotations

import json
import os
import time

import requests

import config as C

FILE = C.OVERRIDES / "lineups.json"
GIST_FILE = "nba_lineups.json"
KEEP_DAYS = 7          # overrides older than this are dropped when saving
_cache: dict = {"at": 0.0, "data": None}
CACHE_SECS = 10        # a page renders several times per click; don't hit the API each time


def _secret(name: str) -> str | None:
    v = os.environ.get(name.upper())
    if v:
        return v
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def backend() -> str:
    return "gist" if _secret("gist_id") and _secret("github_token") else "file"


def _headers() -> dict:
    return {"Authorization": f"Bearer {_secret('github_token')}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def load() -> dict:
    if backend() == "file":
        return json.loads(FILE.read_text()) if FILE.exists() else {}
    if _cache["data"] is not None and time.time() - _cache["at"] < CACHE_SECS:
        return dict(_cache["data"])
    r = requests.get(f"https://api.github.com/gists/{_secret('gist_id')}", headers=_headers(), timeout=20)
    r.raise_for_status()
    f = r.json().get("files", {}).get(GIST_FILE) or {}
    content = f.get("content") or "{}"
    if f.get("truncated") and f.get("raw_url"):
        content = requests.get(f["raw_url"], headers=_headers(), timeout=20).text
    data = json.loads(content or "{}")
    _cache.update(at=time.time(), data=data)
    return dict(data)


def save(data: dict) -> None:
    cutoff = time.strftime("%Y-%m-%d", time.localtime(time.time() - KEEP_DAYS * 86400))
    data = {k: v for k, v in data.items() if k.split(":")[0] >= cutoff}
    if backend() == "file":
        C.OVERRIDES.mkdir(parents=True, exist_ok=True)
        FILE.write_text(json.dumps(data, indent=1))
        return
    body = {"files": {GIST_FILE: {"content": json.dumps(data, indent=1)}}}
    r = requests.patch(f"https://api.github.com/gists/{_secret('gist_id')}", headers=_headers(), json=body, timeout=20)
    r.raise_for_status()
    _cache.update(at=time.time(), data=data)
