"""Download NBA games from ESPN's public API.

    python fetch.py                    # every history season + the current one
    python fetch.py --season 2025      # one season (2025 = 2025-26)

One scoreboard call per date finds the games; each finished regular-season, play-in or
playoff game's full summary (box score, play-by-play, injuries, game info) is saved gzipped
to data/raw/<season>/<game id>.json.gz. Finished games never change, so a file that exists is
never fetched again: re-runs only pick up what is new, and an interrupted run loses nothing.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import requests

import config as C

_session = requests.Session()
_session.headers["User-Agent"] = C.USER_AGENT


def get_json(url: str, params: dict | None = None, tries: int = 4) -> dict:
    for attempt in range(tries):
        try:
            r = _session.get(url, params=params, timeout=C.REQUEST_TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(str(r.status_code), response=r)
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def raw_path(season: int, gid: str) -> Path:
    return C.RAW / str(season) / f"{gid}.json.gz"


def scoreboard(day: date) -> list[dict]:
    """The day's games: id, status, season type."""
    d = get_json(f"{C.ESPN}/scoreboard", {"dates": day.strftime("%Y%m%d"), "limit": 50})
    return [{"gid": e["id"], "date": day.isoformat(), "type": e["season"]["type"],
             "final": e["status"]["type"].get("completed", False)} for e in d.get("events", [])]


def season_dates(season: int) -> list[date]:
    start, end = date(season, 9, 25), min(date(season + 1, 6, 30), date.today())
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _save(season: int, gid: str) -> str:
    p = raw_path(season, gid)
    if p.exists():
        return "cached"
    d = get_json(f"{C.ESPN}/summary", {"event": gid})
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(d, fh)
    tmp.replace(p)
    return "fetched"


def fetch_season(season: int, verbose: bool = True) -> dict:
    with ThreadPoolExecutor(C.FETCH_WORKERS) as ex:
        days = list(ex.map(scoreboard, season_dates(season)))
    games = [g for day in days for g in day if g["final"] and g["type"] in C.SEASON_TYPES]
    counts = {"fetched": 0, "cached": 0, "failed": 0}

    def one(g):
        try:
            return _save(season, g["gid"])
        except Exception:
            return "failed"
    with ThreadPoolExecutor(C.FETCH_WORKERS) as ex:
        for r in ex.map(one, games):
            counts[r] += 1
    # the schedule itself, for parse.py (date, season type of every game)
    (C.RAW / str(season)).mkdir(parents=True, exist_ok=True)
    (C.RAW / str(season) / "schedule.json").write_text(json.dumps(games))
    if verbose:
        print(f"{season}-{str(season + 1)[2:]}: {len(games)} games, {counts}", flush=True)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, action="append")
    args = ap.parse_args()
    for yr in args.season or [*C.HISTORY_SEASONS, C.CURRENT_SEASON]:
        fetch_season(yr)


if __name__ == "__main__":
    main()
