"""Sportsbook odds from OpticOdds (api.opticodds.com, v3), for the Edges tab.

    python odds.py            # fetch game lines + player props -> data/odds/odds.parquet

Odds are fetched only on demand: the Edges tab's "Refresh odds" button (or this script), never
by the scheduled refresh. Needs an API key: opticodds_api_key in the Streamlit app's secrets or
in .streamlit/secrets.toml locally (or OPTICODDS_API_KEY in the environment).

OpticOdds has no per-call credits, only a rate limit (the key is the company's, shared). A fetch
is one call for the slate plus one per 5 games (the most fixtures a call takes; 5 books is also
the most per call), for the games starting in the next HOURS hours. Only each book's main line
is kept (OpticOdds also sends every alternate line), as The Odds API used to give us.

Rows (one per book, market and outcome), with The Odds API's market keys so edges.py is unchanged
(double-double / triple-double come as Over 0.5 and are written as name "Yes", no point):
    event_id, commence_time, home, away, book, market, name, player, point, price, last_update, fetched_at
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

import config as C

API = "https://api.opticodds.com/api/v3"
ODDS_DIR = C.DATA / "odds"
ODDS_FILE = ODDS_DIR / "odds.parquet"
# The books the Edges tab compares (OpticOdds ids -> the names its odds come back with)
BOOKS = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "caesars": "Caesars",
         "fanatics": "Fanatics"}
# OpticOdds market ids -> the market keys edges.py uses
MARKETS = {"moneyline": "h2h", "point_spread": "spreads", "total_points": "totals",
           "player_points": "player_points", "player_rebounds": "player_rebounds", "player_assists": "player_assists",
           "player_made_threes": "player_threes", "player_points_+_rebounds_+_assists": "player_points_rebounds_assists",
           "player_points_+_rebounds": "player_points_rebounds", "player_points_+_assists": "player_points_assists",
           "player_rebounds_+_assists": "player_rebounds_assists", "player_double_double": "player_double_double",
           "player_triple_double": "player_triple_double"}
GAME_MARKETS = {"h2h", "spreads", "totals"}
YES_NO = {"player_double_double", "player_triple_double"}
HOURS = 48
PER_CALL = 5   # fixtures per /fixtures/odds call


def api_key() -> str | None:
    k = os.environ.get("OPTICODDS_API_KEY")
    if k:
        return k
    try:
        import streamlit as st
        return st.secrets.get("opticodds_api_key")
    except Exception:
        pass
    p = C.ROOT / ".streamlit" / "secrets.toml"   # local runs outside streamlit
    if p.exists():
        import tomllib
        return tomllib.loads(p.read_text()).get("opticodds_api_key")
    return None


def _get(path: str, key: str, **params) -> tuple[dict, str | None]:
    r = requests.get(f"{API}{path}", params=params, headers={"X-Api-Key": key}, timeout=C.REQUEST_TIMEOUT)
    if not r.ok:   # OpticOdds explains a bad request in {"error": ...}
        try:
            msg = r.json().get("error")
        except Exception:
            msg = r.text[:200]
        raise requests.HTTPError(f"{r.status_code} from OpticOdds {path}: {msg}", response=r)
    return r.json(), r.headers.get("x-ratelimit-remaining")


def _rows(fx: dict, fetched_at: str) -> list[dict]:
    home, away = fx["home_competitors"][0]["name"], fx["away_competitors"][0]["name"]
    out = []
    for o in fx.get("odds", []):
        market = MARKETS.get(o.get("market_id"))
        if market is None or not o.get("is_main"):
            continue
        prop = market not in ("h2h", "spreads")
        side = o.get("selection_line")
        if prop and side not in ("over", "under"):
            continue
        if market in YES_NO:
            name, point = ("Yes" if side == "over" else "No"), None
        else:
            name, point = (side.title() if prop else o.get("selection")), o.get("points")
        ts = o.get("timestamp")
        out.append({"event_id": fx["id"], "commence_time": fx["start_date"], "home": home, "away": away,
                    "book": o["sportsbook"], "market": market, "name": name,
                    "player": o.get("selection") if market not in GAME_MARKETS else None,
                    "point": point, "price": o.get("price"),
                    "last_update": datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds") if ts else None,
                    "fetched_at": fetched_at})
    return out


def fetch(key: str | None = None, verbose: bool = True) -> pd.DataFrame | None:
    """Main game lines and props for the games starting in the next HOURS hours. Writes ODDS_FILE."""
    key = key or api_key()
    if not key:
        if verbose:
            print("odds: no OPTICODDS_API_KEY, skipped")
        return None
    now = datetime.now(timezone.utc)
    fetched_at = now.isoformat(timespec="seconds")
    before = (now + timedelta(hours=HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    games, page = [], 1
    while True:
        js, left = _get("/fixtures/active", key, league="nba", start_date_before=before, page=page)
        games += [g for g in js.get("data", []) if g.get("status") == "unplayed"]
        if not js.get("has_more"):
            break
        page += 1
    rows = []
    for i in range(0, len(games), PER_CALL):
        ids = [g["id"] for g in games[i:i + PER_CALL]]
        js, left = _get("/fixtures/odds", key, fixture_id=ids, sportsbook=list(BOOKS), market=list(MARKETS),
                        odds_format="AMERICAN")
        for fx in js.get("data", []):
            rows += _rows(fx, fetched_at)
    df = pd.DataFrame(rows, columns=["event_id", "commence_time", "home", "away", "book", "market", "name",
                                     "player", "point", "price", "last_update", "fetched_at"])
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ODDS_FILE, index=False)
    if verbose:
        print(f"odds: {len(games)} games, {len(df):,} prices from {df.book.nunique()} books; "
              f"rate limit left {left}")
    return df


def load() -> pd.DataFrame | None:
    """The saved odds, kept to BOOKS."""
    if not ODDS_FILE.exists():
        return None
    o = pd.read_parquet(ODDS_FILE)
    return o[o.book.isin(BOOKS.values())].reset_index(drop=True)


if __name__ == "__main__":
    try:
        fetch()
    except Exception as e:  # never fail the refresh over odds
        print(f"odds: fetch failed: {e!r}", file=sys.stderr)
