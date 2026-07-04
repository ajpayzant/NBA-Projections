"""
NBA Current Roster Cache
========================
Fetches and caches the current-season roster for all 30 NBA teams
using the commonteamroster endpoint.

Used by the projection engine to filter historical player data
down to only players currently on each team.

Usage:
    python nba_roster_cache.py          # update cache
    from nba_roster_cache import load_current_rosters
"""
from __future__ import annotations

import json
import logging
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("nba.rosters")

_ROOT = Path(__file__).resolve().parent
_CACHE_PATH = _ROOT / "data" / "reference_tables" / "current_rosters.json"

# All 30 NBA team IDs
NBA_TEAMS = {
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751,
    "CHA": 1610612766, "CHI": 1610612741, "CLE": 1610612739,
    "DAL": 1610612742, "DEN": 1610612743, "DET": 1610612765,
    "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763,
    "MIA": 1610612748, "MIL": 1610612749, "MIN": 1610612750,
    "NOP": 1610612740, "NYK": 1610612752, "OKC": 1610612760,
    "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759,
    "TOR": 1610612761, "UTA": 1610612762, "WAS": 1610612764,
}


def _fetch_team_roster(team_id: int, season: str) -> List[Dict]:
    """Fetch current roster for one team via commonteamroster."""
    try:
        sys.path.insert(0, str(_ROOT))
        from scripts.build_warehouse import _fetch_stats_json
        data = _fetch_stats_json("commonteamroster", {
            "TeamID":   str(team_id),
            "Season":   season,
            "LeagueID": "00",
        }, max_retries=3, base_sleep=2.0)
        if data is None:
            return []
        rs = data["resultSets"][0]
        headers = rs["headers"]
        rows = []
        for row_data in rs["rowSet"]:
            row = dict(zip(headers, row_data))
            rows.append({
                "player_id":   int(row.get("PLAYER_ID", 0) or 0),
                "player_name": str(row.get("PLAYER", "")),
                "number":      str(row.get("NUM", "")),
                "position":    str(row.get("POSITION", "UNK")),
                "height":      str(row.get("HEIGHT", "")),
                "weight":      str(row.get("WEIGHT", "")),
            })
        return rows
    except Exception as e:
        logger.warning("Roster fetch failed for team_id=%d: %s", team_id, e)
        return []


def update_roster_cache(season: str = "2025-26") -> Dict[str, List[Dict]]:
    """
    Fetch current rosters for all 30 teams and save to cache.
    Returns {team_abbr: [player_dict, ...]}
    """
    rosters: Dict[str, List[Dict]] = {}
    logger.info("Fetching current rosters for all 30 teams (season %s)...", season)

    for abbr, team_id in NBA_TEAMS.items():
        time.sleep(random.uniform(0.5, 1.2))
        players = _fetch_team_roster(team_id, season)
        if players:
            rosters[abbr] = players
            logger.info("  %s: %d players", abbr, len(players))
        else:
            logger.warning("  %s: no data", abbr)

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"season": season, "rosters": rosters}
    _CACHE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved roster cache: %d teams", len(rosters))
    return rosters


def load_current_rosters() -> Dict[str, List[Dict]]:
    """
    Load roster cache from disk. Returns {team_abbr: [player_dict, ...]}.
    Returns empty dict if cache doesn't exist.
    """
    if not _CACHE_PATH.exists():
        return {}
    try:
        payload = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        return payload.get("rosters", {})
    except Exception as e:
        logger.warning("Could not load roster cache: %s", e)
        return {}


def get_active_player_ids(team_abbr: str) -> List[int]:
    """Return list of player_ids currently on a team's roster."""
    rosters = load_current_rosters()
    team = rosters.get(team_abbr.upper(), [])
    return [p["player_id"] for p in team if p.get("player_id")]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    import datetime as dt
    _today = dt.date.today()
    # Determine current season
    if _today.month >= 10:
        season = f"{_today.year}-{str(_today.year + 1)[-2:]}"
    else:
        season = f"{_today.year - 1}-{str(_today.year)[-2:]}"
    update_roster_cache(season)
    print(f"Roster cache updated for {season}")
