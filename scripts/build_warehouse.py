"""
NBA Warehouse Builder
=====================
Pulls data from nba_api and loads into DuckDB.

Usage:
    python scripts/build_warehouse.py            # incremental update
    python scripts/build_warehouse.py --full     # full rebuild from scratch

Sections:
    1. Player game logs (LeagueGameLog)
    2. Team game logs   (LeagueGameLog — team side)
    3. Player info / positions (CommonPlayerInfo, cached)
    4. Schedule / upcoming games (ScoreboardV2 + LeagueSchedule)
    5. Feature engineering (rolling EWM, pace, USG%, opp context)
    6. Load into DuckDB
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Path bootstrap ────────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nba.warehouse")

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = _ROOT / "data" / "analytics_database" / "nba_warehouse.duckdb"
RAW_DIR = _ROOT / "data" / "raw_cache"
REF_DIR = _ROOT / "data" / "reference_tables"

def _current_nba_season() -> str:
    """
    Return the current NBA season string (e.g. '2025-26').
    NBA seasons start in October — if today is Oct or later, we're in
    the season that started this year; otherwise we're still in last year's.
    """
    today = dt.date.today()
    if today.month >= 10:
        start_year = today.year
    else:
        start_year = today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _build_season_list(n_back: int = 5) -> List[str]:
    """Return the last n_back seasons up to and including the current one."""
    current = _current_nba_season()
    start_year = int(current.split("-")[0])
    seasons = []
    for i in range(n_back - 1, -1, -1):
        y = start_year - i
        seasons.append(f"{y}-{str(y + 1)[-2:]}")
    return seasons


CURRENT_SEASON = _current_nba_season()
SEASONS = _build_season_list(n_back=5)

logger.info("Current NBA season: %s | Seasons to load: %s", CURRENT_SEASON, SEASONS)
SEASON_TYPE = "Regular Season"

TIMEOUT_KEYWORDS = ("Read timed out", "ReadTimeout", "Max retries", "HTTPSConnectionPool")

# ── Direct HTTP client using curl_cffi (real browser TLS fingerprint) ────────
# stats.nba.com uses TLS fingerprint detection — it blocks Python's urllib3
# regardless of headers because the TLS handshake looks like a bot.
# curl_cffi uses libcurl with Chrome's actual TLS fingerprint, bypassing this.

try:
    from curl_cffi import requests as _cffi_requests
    _USE_CFFI = True
    logger.info("curl_cffi available — using Chrome TLS fingerprint for NBA API")
except ImportError:
    import requests as _cffi_requests  # fallback, may fail in CI
    _USE_CFFI = False
    logger.warning("curl_cffi not available — falling back to requests (may be blocked by stats.nba.com)")

_NBA_HEADERS = {
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token":  "true",
    "Origin":             "https://www.nba.com",
    "Referer":            "https://www.nba.com/",
    "Accept":             "application/json, text/plain, */*",
    "Accept-Language":    "en-US,en;q=0.9",
}

_NBA_BASE = "https://stats.nba.com/stats"


def _fetch_stats_json(endpoint: str, params: Dict,
                      max_retries: int = 8,
                      base_sleep: float = 5.0) -> Optional[Dict]:
    """
    Fetch a stats.nba.com endpoint using curl_cffi with Chrome TLS fingerprint.
    Falls back to regular requests if curl_cffi is unavailable.
    """
    url = f"{_NBA_BASE}/{endpoint}"
    for attempt in range(1, max_retries + 1):
        try:
            if _USE_CFFI:
                resp = _cffi_requests.get(url, headers=_NBA_HEADERS, params=params,
                                          impersonate="chrome136", timeout=60)
            else:
                resp = _cffi_requests.get(url, headers=_NBA_HEADERS, params=params,
                                          timeout=60)
            resp.raise_for_status()
            time.sleep(random.uniform(1.0, 2.5))
            return resp.json()
        except Exception as e:
            msg = str(e)
            sleep_s = base_sleep * (2 ** min(attempt - 1, 4)) + random.uniform(0, 3.0)
            if any(k in msg for k in TIMEOUT_KEYWORDS):
                sleep_s = max(sleep_s, 20.0)
            logger.warning("HTTP attempt %d/%d for %s: %s — sleeping %.1fs",
                           attempt, max_retries, endpoint, msg[:80], sleep_s)
            time.sleep(sleep_s)
    logger.error("Max retries reached for endpoint: %s", endpoint)
    return None


def _json_to_df(data: Dict, result_set_idx: int = 0) -> pd.DataFrame:
    """Convert stats.nba.com JSON response to a DataFrame."""
    try:
        rs = data["resultSets"][result_set_idx]
        return pd.DataFrame(rs["rowSet"], columns=rs["headers"])
    except Exception as e:
        logger.warning("Failed to parse JSON response: %s", e)
        return pd.DataFrame()


def safe_api_call(api_cls, max_retries=8, base_sleep=5.0, **kwargs):
    """
    nba_api wrapper — kept for CommonPlayerInfo and ScoreboardV2
    which we still call via nba_api (lower volume, less affected by blocking).
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = api_cls(**kwargs)
            time.sleep(random.uniform(1.0, 2.0))
            return resp
        except Exception as e:
            msg = str(e)
            sleep_s = base_sleep * (2 ** min(attempt - 1, 4)) + random.uniform(0, 3.0)
            if any(k in msg for k in TIMEOUT_KEYWORDS):
                sleep_s = max(sleep_s, 20.0)
            logger.warning("API attempt %d/%d failed (%s). Sleeping %.1fs",
                           attempt, max_retries, msg[:80], sleep_s)
            time.sleep(sleep_s)
    logger.error("Max retries reached for %s", api_cls.__name__)
    return None


# ── Section 1 & 2: Game logs (direct HTTP — bypasses nba_api session) ────────

def fetch_player_logs(seasons: List[str]) -> pd.DataFrame:
    """Fetch player game logs directly from stats.nba.com."""
    frames = []
    for season in seasons:
        logger.info("Fetching player logs: %s", season)
        time.sleep(random.uniform(2.0, 4.0))
        data = _fetch_stats_json("leaguegamelog", {
            "Season":            season,
            "SeasonType":        SEASON_TYPE,
            "PlayerOrTeam":      "P",
            "Direction":         "ASC",
            "Sorter":            "DATE",
            "LeagueID":          "00",
            "Counter":           "0",
            "DateFrom":          "",
            "DateTo":            "",
        })
        if data is None:
            logger.warning("Skipping player logs for %s", season)
            continue
        df = _json_to_df(data)
        if df.empty:
            logger.warning("Empty player logs for %s", season)
            continue
        df["SEASON"] = season
        frames.append(df)
        logger.info("  Got %d player-game rows for %s", len(df), season)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_team_logs(seasons: List[str]) -> pd.DataFrame:
    """Fetch team game logs directly from stats.nba.com."""
    frames = []
    for season in seasons:
        logger.info("Fetching team logs: %s", season)
        time.sleep(random.uniform(2.0, 4.0))
        data = _fetch_stats_json("leaguegamelog", {
            "Season":            season,
            "SeasonType":        SEASON_TYPE,
            "PlayerOrTeam":      "T",
            "Direction":         "ASC",
            "Sorter":            "DATE",
            "LeagueID":          "00",
            "Counter":           "0",
            "DateFrom":          "",
            "DateTo":            "",
        })
        if data is None:
            logger.warning("Skipping team logs for %s", season)
            continue
        df = _json_to_df(data)
        if df.empty:
            logger.warning("Empty team logs for %s", season)
            continue
        df["SEASON"] = season
        frames.append(df)
        logger.info("  Got %d team-game rows for %s", len(df), season)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Section 3: Player info / positions (curl_cffi) ───────────────────────────

def fetch_player_info(player_ids: List[int], cache_path: Path) -> pd.DataFrame:
    """
    Fetch and cache player positions via commonplayerinfo endpoint.
    Uses curl_cffi directly to avoid TLS blocking.
    """
    cache: Dict[int, dict] = {}
    if cache_path.exists():
        try:
            cache = {int(k): v for k, v in json.loads(cache_path.read_text()).items()}
        except Exception:
            cache = {}

    missing = [pid for pid in player_ids if pid not in cache]
    logger.info("Player info cache: %d known, %d to fetch", len(cache), len(missing))

    for i, pid in enumerate(missing):
        if i > 0 and i % 50 == 0:
            logger.info("  Player info progress: %d/%d", i, len(missing))
            cache_path.write_text(json.dumps(cache))

        data = _fetch_stats_json("commonplayerinfo", {
            "PlayerID":  str(pid),
            "LeagueID":  "00",
        })
        if data is None:
            continue
        try:
            rs = data["resultSets"][0]
            headers = rs["headers"]
            row_data = rs["rowSet"]
            if not row_data:
                continue
            row = dict(zip(headers, row_data[0]))
            cache[pid] = {
                "PLAYER_NAME":       str(row.get("DISPLAY_FIRST_LAST", "")),
                "POSITION":          str(row.get("POSITION", "UNK")),
                "HEIGHT":            str(row.get("HEIGHT", "")),
                "WEIGHT":            str(row.get("WEIGHT", "")),
                "TEAM_ID":           int(row.get("TEAM_ID", 0) or 0),
                "TEAM_ABBREVIATION": str(row.get("TEAM_ABBREVIATION", "")),
            }
        except Exception as e:
            logger.debug("Parse error for player %d: %s", pid, e)

    cache_path.write_text(json.dumps(cache))
    records = [{"PLAYER_ID": pid, **info} for pid, info in cache.items()]
    return pd.DataFrame(records)


# ── Section 4: Schedule (curl_cffi) ──────────────────────────────────────────

def fetch_upcoming_games(days_ahead: int = 14) -> pd.DataFrame:
    """Return upcoming games within the next N days via scoreboard endpoint."""
    rows = []
    today = dt.date.today()
    for delta in range(days_ahead):
        d = today + dt.timedelta(days=delta)
        # Only 2 retries for schedule — 500 means server down, no point hammering
        data = _fetch_stats_json("scoreboardv2", {
            "GameDate":  d.strftime("%Y-%m-%d"),
            "DayOffset": "0",
            "LeagueID":  "00",
        }, max_retries=2, base_sleep=3.0)
        if data is None:
            continue
        try:
            # GameHeader result set contains game info
            for rs in data.get("resultSets", []):
                if rs["name"] != "GameHeader":
                    continue
                headers = rs["headers"]
                for row_data in rs["rowSet"]:
                    row = dict(zip(headers, row_data))
                    game_id    = str(row.get("GAME_ID", ""))
                    home_id    = int(row.get("HOME_TEAM_ID", 0) or 0)
                    away_id    = int(row.get("VISITOR_TEAM_ID", 0) or 0)
                    if not game_id or not home_id or not away_id:
                        continue
                    # Look up abbreviations from static data
                    from nba_api.stats.static import teams as _nba_teams
                    _id_map = {t["id"]: t for t in _nba_teams.get_teams()}
                    home_info = _id_map.get(home_id, {})
                    away_info = _id_map.get(away_id, {})
                    rows.append({
                        "game_id":        game_id,
                        "game_date":      d.isoformat(),
                        "season":         CURRENT_SEASON,
                        "home_team_id":   home_id,
                        "away_team_id":   away_id,
                        "home_team_abbr": home_info.get("abbreviation", ""),
                        "away_team_abbr": away_info.get("abbreviation", ""),
                        "home_team_name": home_info.get("full_name", ""),
                        "away_team_name": away_info.get("full_name", ""),
                        "status":         str(row.get("GAME_STATUS_TEXT", "Scheduled")),
                    })
        except Exception as e:
            logger.debug("Scoreboard parse error %s: %s", d, e)
        time.sleep(random.uniform(0.5, 1.0))

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("game_id")
    return df[df["home_team_id"] > 0].sort_values("game_date").reset_index(drop=True)


def _fetch_upcoming_games_fallback(days_ahead: int = 14) -> pd.DataFrame:
    """Fallback using nba_api ScoreboardV2 if direct fetch fails."""
    from nba_api.stats.endpoints import scoreboardv2
    rows = []
    today = dt.date.today()
    for delta in range(days_ahead):
        d = today + dt.timedelta(days=delta)
        resp = safe_api_call(scoreboardv2.ScoreboardV2,
                             game_date=d.strftime("%Y-%m-%d"),
                             day_offset=0,
                             league_id="00",
                             timeout=30)
        if resp is None:
            continue
        try:
            games_df = resp.get_data_frames()[0]
            if games_df.empty:
                continue
            for _, g in games_df.iterrows():
                rows.append({
                    "game_id":        str(g.get("GAME_ID", "")),
                    "game_date":      d.isoformat(),
                    "season":         CURRENT_SEASON,
                    "home_team_id":   int(g.get("HOME_TEAM_ID", 0)),
                    "away_team_id":   int(g.get("VISITOR_TEAM_ID", 0)),
                    "home_team_abbr": str(g.get("HOME_TEAM_ABBREVIATION", g.get("HOME_TEAM_CITY", ""))),
                    "away_team_abbr": str(g.get("VISITOR_TEAM_ABBREVIATION", g.get("VISITOR_TEAM_CITY", ""))),
                    "home_team_name": str(g.get("HOME_TEAM_CITY", "") + " " + g.get("HOME_TEAM_NICKNAME", "")),
                    "away_team_name": str(g.get("VISITOR_TEAM_CITY", "") + " " + g.get("VISITOR_TEAM_NICKNAME", "")),
                    "status":         str(g.get("GAME_STATUS_TEXT", "Scheduled")),
                })
        except Exception as e:
            logger.debug("Scoreboard parse error %s: %s", d, e)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("game_id")
    df = df[df["home_team_id"] > 0].copy()
    return df.sort_values("game_date").reset_index(drop=True)


# ── Section 5: Feature engineering ───────────────────────────────────────────

def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return np.where(den > 0, num / den, np.nan)


def build_team_features(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-game advanced metrics and leakage-safe rolling features.
    All rolling values use .shift(1) so game N only sees games 0..N-1.
    """
    df = team_df.copy()
    df.columns = [c.strip() for c in df.columns]
    df["GAME_DATE"] = pd.to_datetime(df.get("GAME_DATE", pd.Series(dtype="object")),
                                     errors="coerce", format="mixed")

    # Normalize column names from LeagueGameLog team side
    for old, new in [("PTS", "TEAM_PTS"), ("REB", "TEAM_REB"), ("AST", "TEAM_AST"),
                     ("TOV", "TEAM_TOV"), ("STL", "TEAM_STL"), ("BLK", "TEAM_BLK"),
                     ("FGA", "TEAM_FGA"), ("FGM", "TEAM_FGM"),
                     ("FG3A", "TEAM_FG3A"), ("FG3M", "TEAM_FG3M"),
                     ("FTA", "TEAM_FTA"), ("FTM", "TEAM_FTM"),
                     ("OREB", "TEAM_OREB"), ("DREB", "TEAM_DREB")]:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # Home/away from MATCHUP
    if "MATCHUP" in df.columns:
        df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", regex=False).astype(int)
    else:
        df["IS_HOME"] = 0

    # Rest days / back-to-back
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    df["PREV_GAME_DATE"] = df.groupby("TEAM_ABBREVIATION")["GAME_DATE"].shift(1)
    df["REST_DAYS"] = (df["GAME_DATE"] - df["PREV_GAME_DATE"]).dt.days.clip(upper=7)
    df["IS_B2B"] = (df["REST_DAYS"] == 1).astype(int)

    # Possessions and pace
    if all(c in df.columns for c in ["TEAM_FGA", "TEAM_FTA", "TEAM_OREB", "TEAM_TOV"]):
        df["POSS"] = df["TEAM_FGA"] + 0.44 * df["TEAM_FTA"] - df["TEAM_OREB"] + df["TEAM_TOV"]
    else:
        df["POSS"] = np.nan

    if "POSS" in df.columns and "MIN" in df.columns:
        df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
        df["PACE"] = np.where(df["MIN"] > 0, 48.0 * df["POSS"] / (df["MIN"] / 5.0), np.nan)
    else:
        df["PACE"] = np.nan

    # Shooting efficiencies
    if "TEAM_FGA" in df.columns:
        df["TEAM_FG_PCT"] = _safe_div(df.get("TEAM_FGM", 0), df["TEAM_FGA"])
        df["TEAM_FG3_PCT"] = _safe_div(df.get("TEAM_FG3M", 0), df.get("TEAM_FG3A", 1))
        df["TEAM_FT_PCT"] = _safe_div(df.get("TEAM_FTM", 0), df.get("TEAM_FTA", 1))
        df["TEAM_EFG"] = _safe_div(
            df.get("TEAM_FGM", 0) + 0.5 * df.get("TEAM_FG3M", 0), df["TEAM_FGA"]
        )
        df["TEAM_TOV_RATE"] = _safe_div(
            df.get("TEAM_TOV", 0),
            df["TEAM_FGA"] + 0.44 * df.get("TEAM_FTA", 0) + df.get("TEAM_TOV", 0)
        )

    # Opponent GAME_ID mapping
    opp = df[["GAME_ID", "TEAM_ABBREVIATION"]].rename(
        columns={"TEAM_ABBREVIATION": "OPP_TEAM_ABBREVIATION"}
    )
    df = df.merge(
        df[["GAME_ID", "TEAM_ABBREVIATION"]].merge(opp, on="GAME_ID")
        .query("TEAM_ABBREVIATION != OPP_TEAM_ABBREVIATION")
        .drop_duplicates(["GAME_ID", "TEAM_ABBREVIATION"]),
        on=["GAME_ID", "TEAM_ABBREVIATION"], how="left"
    )

    # Offensive and Defensive Ratings (pts per 100 possessions)
    # These are the gold standard for team quality measurement in NBA analytics
    if "POSS" in df.columns:
        df["OFF_RTG"] = np.where(df["POSS"] > 0, (df["TEAM_PTS"] / df["POSS"]) * 100.0, np.nan)
        # DEF_RTG requires opponent pts — will be filled after opponent merge below
        # For now compute from opponent in the merge
    else:
        df["OFF_RTG"] = np.nan

    # Opponent game stats for DEF_RTG — merge opponent scoring
    if "OPP_TEAM_ABBREVIATION" in df.columns:
        opp_pts = df[["GAME_ID","TEAM_ABBREVIATION","TEAM_PTS","POSS"]].rename(
            columns={"TEAM_ABBREVIATION":"OPP_TEAM_ABBREVIATION",
                     "TEAM_PTS":"OPP_PTS_SCORED","POSS":"OPP_POSS"}
        )
        df = df.merge(opp_pts, on=["GAME_ID","OPP_TEAM_ABBREVIATION"], how="left")
        df["DEF_RTG"] = np.where(
            df["POSS"].fillna(0) > 0,
            (df["OPP_PTS_SCORED"].fillna(df["TEAM_PTS"]) / df["POSS"]) * 100.0,
            np.nan
        )
        df["NET_RTG"] = df["OFF_RTG"] - df["DEF_RTG"]
    else:
        df["DEF_RTG"] = np.nan
        df["NET_RTG"] = np.nan

    # Rolling EWM and window features (leakage-safe)
    HL = 8   # EWM half-life in games
    roll_stats = ["TEAM_PTS", "TEAM_REB", "TEAM_AST", "TEAM_TOV",
                  "TEAM_FG_PCT", "TEAM_FG3_PCT", "TEAM_EFG",
                  "TEAM_TOV_RATE", "PACE", "POSS",
                  "OFF_RTG", "DEF_RTG", "NET_RTG"]   # ← Added
    roll_stats = [c for c in roll_stats if c in df.columns]

    df = df.sort_values(["TEAM_ABBREVIATION", "SEASON", "GAME_DATE"]).reset_index(drop=True)
    for col in roll_stats:
        grp = df.groupby(["TEAM_ABBREVIATION", "SEASON"])[col]
        df[f"{col}_EWM"] = grp.transform(
            lambda x: x.shift(1).ewm(halflife=HL, min_periods=1).mean()
        )
        df[f"{col}_AVG5"] = grp.transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        df[f"{col}_SEASON_AVG"] = grp.transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

    return df


def build_player_features(player_df: pd.DataFrame,
                           team_df: pd.DataFrame,
                           player_info: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-game player features with leakage-safe rolling.
    Merges team context and opponent-allowed context.
    """
    df = player_df.copy()
    df.columns = [c.strip() for c in df.columns]
    df["GAME_DATE"] = pd.to_datetime(df.get("GAME_DATE", pd.Series(dtype="object")),
                                     errors="coerce", format="mixed")

    # Merge position info
    if not player_info.empty and "PLAYER_ID" in player_info.columns:
        df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce")
        player_info["PLAYER_ID"] = pd.to_numeric(player_info["PLAYER_ID"], errors="coerce")
        df = df.merge(
            player_info[["PLAYER_ID", "POSITION"]].drop_duplicates("PLAYER_ID"),
            on="PLAYER_ID", how="left"
        )
    if "POSITION" not in df.columns:
        df["POSITION"] = "UNK"
    df["POSITION"] = df["POSITION"].fillna("UNK")

    # Position group
    def pos_group(p):
        s = str(p).lower().strip()
        # Exact single-position codes first
        if s in ("pg", "sg", "g"):
            return "G"
        if s in ("sf", "pf", "f"):
            return "F"
        if s in ("c",):
            return "C"
        # Hyphenated compound: first word is primary role
        # NBA API format: "Guard-Forward" = guard primary, "Forward-Guard" = forward primary
        if "-" in s:
            primary = s.split("-")[0].strip()
            if "guard" in primary:
                return "G"
            if "forward" in primary:
                return "F"
            if "center" in primary:
                return "C"
        # Simple text match fallback
        if s == "guard":
            return "G"
        if s == "forward":
            return "F"
        if s == "center":
            return "C"
        return "UNK"

    df["POS_GROUP"] = df["POSITION"].apply(pos_group)

    # Numeric coercion
    stat_cols = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
                 "FGA", "FGM", "FG3A", "FG3M", "FTA", "FTM", "OREB", "DREB",
                 "PLUS_MINUS"]
    for c in stat_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Per-minute rates
    for stat in ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FGA"]:
        if stat in df.columns:
            df[f"{stat}_PM"] = np.where(df["MIN"] > 0, df[stat] / df["MIN"], np.nan)

    # Shooting splits
    df["FG_PCT"]  = _safe_div(df.get("FGM", 0), df.get("FGA", 1))
    df["FG3_PCT"] = _safe_div(df.get("FG3M", 0), df.get("FG3A", 1))
    df["FT_PCT"]  = _safe_div(df.get("FTM", 0), df.get("FTA", 1))
    df["EFG_PCT"] = _safe_div(
        df.get("FGM", 0) + 0.5 * df.get("FG3M", 0), df.get("FGA", 1)
    )

    # Usage rate
    team_usage = team_df[["GAME_ID", "TEAM_ABBREVIATION",
                           "FGA", "FTA", "TOV", "MIN"]].rename(
        columns={"FGA": "T_FGA", "FTA": "T_FTA", "TOV": "T_TOV", "MIN": "T_MIN"}
    ).drop_duplicates(["GAME_ID", "TEAM_ABBREVIATION"])
    df = df.merge(team_usage, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")

    p_num = df.get("FGA", 0).fillna(0) + 0.44 * df.get("FTA", 0).fillna(0) + df.get("TOV", 0).fillna(0)
    t_den = df.get("T_FGA", 0).fillna(0) + 0.44 * df.get("T_FTA", 0).fillna(0) + df.get("T_TOV", 0).fillna(0)
    df["USG_PCT"] = np.where(
        (df["MIN"].fillna(0) > 0) & (t_den > 0),
        100.0 * p_num * (df.get("T_MIN", 240).fillna(240) / 5.0) / (df["MIN"] * t_den),
        np.nan
    )

    # Merge team context (EWM ratings, rest, pace)
    team_ctx_cols = ["GAME_ID", "TEAM_ABBREVIATION", "IS_HOME", "REST_DAYS", "IS_B2B",
                     "OPP_TEAM_ABBREVIATION",
                     "PACE_EWM", "TEAM_PTS_EWM", "TEAM_AST_EWM", "TEAM_TOV_EWM"]
    team_ctx_cols = [c for c in team_ctx_cols if c in team_df.columns or c in ["GAME_ID", "TEAM_ABBREVIATION"]]
    avail = [c for c in team_ctx_cols if c in team_df.columns]
    df = df.merge(
        team_df[avail].drop_duplicates(["GAME_ID", "TEAM_ABBREVIATION"]),
        on=["GAME_ID", "TEAM_ABBREVIATION"], how="left"
    )

    # Opponent rest
    opp_rest = team_df[["GAME_ID", "TEAM_ABBREVIATION", "REST_DAYS", "IS_B2B"]].rename(
        columns={"TEAM_ABBREVIATION": "OPP_TEAM_ABBREVIATION",
                 "REST_DAYS": "OPP_REST_DAYS", "IS_B2B": "OPP_IS_B2B"}
    ).drop_duplicates(["GAME_ID", "OPP_TEAM_ABBREVIATION"])
    if "OPP_TEAM_ABBREVIATION" in df.columns:
        df = df.merge(opp_rest, on=["GAME_ID", "OPP_TEAM_ABBREVIATION"], how="left")

    # Opponent allowed by position (avg pts/reb/ast allowed to G/F/C in last 10 games)
    if "OPP_TEAM_ABBREVIATION" in df.columns and "POS_GROUP" in df.columns:
        pos_game = df[["GAME_ID", "GAME_DATE", "OPP_TEAM_ABBREVIATION", "POS_GROUP",
                        "PTS", "REB", "AST", "FG3M"]].copy()
        pos_game = pos_game.dropna(subset=["OPP_TEAM_ABBREVIATION", "POS_GROUP"])
        pos_game["GAME_DATE"] = pd.to_datetime(pos_game["GAME_DATE"], errors="coerce")
        pos_agg = pos_game.groupby(["GAME_ID", "GAME_DATE", "OPP_TEAM_ABBREVIATION", "POS_GROUP"],
                                    as_index=False).agg(
            POS_PTS=("PTS", "sum"), POS_REB=("REB", "sum"),
            POS_AST=("AST", "sum"), POS_3PM=("FG3M", "sum"),
        )
        pos_agg = pos_agg.sort_values(["OPP_TEAM_ABBREVIATION", "POS_GROUP", "GAME_DATE"])
        for stat, col in [("PTS", "POS_PTS"), ("REB", "POS_REB"),
                           ("AST", "POS_AST"), ("3PM", "POS_3PM")]:
            pos_agg[f"OPP_ALLOWED_{stat}_POS_AVG10"] = (
                pos_agg.groupby(["OPP_TEAM_ABBREVIATION", "POS_GROUP"])[col]
                .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
            )
        merge_cols = ["GAME_ID", "OPP_TEAM_ABBREVIATION", "POS_GROUP",
                      "OPP_ALLOWED_PTS_POS_AVG10", "OPP_ALLOWED_REB_POS_AVG10",
                      "OPP_ALLOWED_AST_POS_AVG10", "OPP_ALLOWED_3PM_POS_AVG10"]
        merge_cols = [c for c in merge_cols if c in pos_agg.columns]
        df = df.merge(
            pos_agg[merge_cols].drop_duplicates(["GAME_ID", "OPP_TEAM_ABBREVIATION", "POS_GROUP"]),
            on=["GAME_ID", "OPP_TEAM_ABBREVIATION", "POS_GROUP"], how="left"
        )

    # Player rolling EWM (leakage-safe)
    HL = 8
    roll_player = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M",
                   "PTS_PM", "REB_PM", "AST_PM", "STL_PM", "BLK_PM", "TOV_PM",
                   "FG3M_PM", "FG_PCT", "FG3_PCT", "FT_PCT", "EFG_PCT", "USG_PCT"]
    roll_player = [c for c in roll_player if c in df.columns]

    df = df.sort_values(["PLAYER_ID", "SEASON", "GAME_DATE"]).reset_index(drop=True)
    for col in roll_player:
        grp = df.groupby(["PLAYER_ID", "SEASON"])[col]
        df[f"{col}_EWM"] = grp.transform(
            lambda x: x.shift(1).ewm(halflife=HL, min_periods=1).mean()
        )
        df[f"{col}_AVG5"] = grp.transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        df[f"{col}_SEASON_AVG"] = grp.transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
        df[f"{col}_CAREER_AVG"] = grp.transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )
        df[f"GAMES_PLAYED"] = grp.transform(lambda x: x.shift(1).expanding().count())

    return df


# ── Section 6: DuckDB loader ─────────────────────────────────────────────────

def load_to_duckdb(player_features: pd.DataFrame,
                   team_features: pd.DataFrame,
                   schedule: pd.DataFrame,
                   player_info: pd.DataFrame) -> None:
    import duckdb

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS clean")
        con.execute("CREATE SCHEMA IF NOT EXISTS marts")

        # player_game_stats
        con.execute("DROP TABLE IF EXISTS clean.player_game_stats")
        con.execute("CREATE TABLE clean.player_game_stats AS SELECT * FROM player_features")
        logger.info("Loaded player_game_stats: %d rows", len(player_features))

        # team_game_stats
        con.execute("DROP TABLE IF EXISTS clean.team_game_stats")
        con.execute("CREATE TABLE clean.team_game_stats AS SELECT * FROM team_features")
        logger.info("Loaded team_game_stats: %d rows", len(team_features))

        # schedule
        if not schedule.empty:
            con.execute("DROP TABLE IF EXISTS clean.game_schedule")
            con.execute("CREATE TABLE clean.game_schedule AS SELECT * FROM schedule")
            logger.info("Loaded game_schedule: %d rows", len(schedule))

        # player_info
        if not player_info.empty:
            con.execute("DROP TABLE IF EXISTS clean.player_info")
            con.execute("CREATE TABLE clean.player_info AS SELECT * FROM player_info")
            logger.info("Loaded player_info: %d rows", len(player_info))

        # marts: current season aggregates
        con.execute("DROP TABLE IF EXISTS marts.player_season_stats")
        con.execute("""
            CREATE TABLE marts.player_season_stats AS
            SELECT
                PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION, SEASON,
                COUNT(*) AS games,
                AVG(MIN) AS min_pg, AVG(PTS) AS pts_pg, AVG(REB) AS reb_pg,
                AVG(AST) AS ast_pg, AVG(STL) AS stl_pg, AVG(BLK) AS blk_pg,
                AVG(TOV) AS tov_pg, AVG(FG3M) AS fg3m_pg,
                AVG(FG_PCT) AS fg_pct, AVG(FG3_PCT) AS fg3_pct,
                AVG(FT_PCT) AS ft_pct, AVG(USG_PCT) AS usg_pct
            FROM clean.player_game_stats
            WHERE MIN IS NOT NULL AND MIN > 0
            GROUP BY PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION, SEASON
        """)
        logger.info("Built player_season_stats mart")

    finally:
        con.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def _fetch_recent_games(season: str, days_back: int = 30) -> tuple:
    """
    Fetch only the most recent games for a season (last N days).
    Uses DateFrom/DateTo to minimize API calls and reduce blocking risk.
    Returns (player_df, team_df).
    """
    date_from = (dt.date.today() - dt.timedelta(days=days_back)).strftime("%m/%d/%Y")
    date_to   = dt.date.today().strftime("%m/%d/%Y")
    logger.info("Fetching recent games %s → %s for %s", date_from, date_to, season)

    p_data = _fetch_stats_json("leaguegamelog", {
        "Season":       season,
        "SeasonType":   SEASON_TYPE,
        "PlayerOrTeam": "P",
        "Direction":    "DESC",
        "Sorter":       "DATE",
        "LeagueID":     "00",
        "Counter":      "0",
        "DateFrom":     date_from,
        "DateTo":       date_to,
    })
    t_data = _fetch_stats_json("leaguegamelog", {
        "Season":       season,
        "SeasonType":   SEASON_TYPE,
        "PlayerOrTeam": "T",
        "Direction":    "DESC",
        "Sorter":       "DATE",
        "LeagueID":     "00",
        "Counter":      "0",
        "DateFrom":     date_from,
        "DateTo":       date_to,
    })

    p_df = _json_to_df(p_data) if p_data else pd.DataFrame()
    t_df = _json_to_df(t_data) if t_data else pd.DataFrame()
    if not p_df.empty:
        p_df["SEASON"] = season
    if not t_df.empty:
        t_df["SEASON"] = season
    return p_df, t_df


def main(full_rebuild: bool = False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)

    player_raw_path = RAW_DIR / "player_game_logs.parquet"
    team_raw_path   = RAW_DIR / "team_game_logs.parquet"
    pos_cache_path  = REF_DIR / "player_positions.json"

    # Off-season detection — defined here so all code paths can use it
    _today = dt.date.today()
    is_offseason = (
        (_today.month >= 7 and _today.month <= 9) or
        (_today.month == 6 and _today.day > 20) or
        (_today.month == 10 and _today.day < 5)
    )
    if is_offseason:
        logger.info("NBA off-season (%s) — live API calls will be skipped.", _today)

    # ── Load or scrape raw logs ───────────────────────────────────────────────
    if not full_rebuild and player_raw_path.exists() and team_raw_path.exists():
        logger.info("Loading cached raw logs...")
        player_raw = pd.read_parquet(player_raw_path)
        team_raw   = pd.read_parquet(team_raw_path)

        existing_seasons = set(player_raw["SEASON"].unique()) if "SEASON" in player_raw.columns else set()

        # Fetch any completely missing seasons (new season start)
        missing = [s for s in SEASONS if s not in existing_seasons]
        if missing:
            logger.info("Fetching new seasons from scratch: %s", missing)
            new_p = fetch_player_logs(missing)
            new_t = fetch_team_logs(missing)
            if not new_p.empty:
                player_raw = pd.concat([player_raw, new_p], ignore_index=True)
            if not new_t.empty:
                team_raw = pd.concat([team_raw, new_t], ignore_index=True)

        # Incremental update: fetch only recent games for the current season.
        # Skipped during off-season (is_offseason defined at top of main()).
        if is_offseason:
            logger.info("Off-season detected (%s) — skipping incremental game fetch. "
                        "Cache is up to date from end of last season.", _today)
        else:
            logger.info("Incremental update: fetching recent games for %s", CURRENT_SEASON)
            try:
                new_p, new_t = _fetch_recent_games(CURRENT_SEASON, days_back=7)
                if not new_p.empty:
                    logger.info("  Got %d new player rows", len(new_p))
                    if "GAME_ID" in player_raw.columns and "GAME_ID" in new_p.columns:
                        existing_ids = set(player_raw["GAME_ID"].astype(str))
                        new_p = new_p[~new_p["GAME_ID"].astype(str).isin(existing_ids)]
                    if not new_p.empty:
                        player_raw = pd.concat([player_raw, new_p], ignore_index=True)
                if not new_t.empty:
                    logger.info("  Got %d new team rows", len(new_t))
                    if "GAME_ID" in team_raw.columns and "GAME_ID" in new_t.columns:
                        existing_ids = set(team_raw["GAME_ID"].astype(str))
                        new_t = new_t[~new_t["GAME_ID"].astype(str).isin(existing_ids)]
                    if not new_t.empty:
                        team_raw = pd.concat([team_raw, new_t], ignore_index=True)
            except Exception as e:
                logger.warning("Incremental fetch failed (%s) — using existing cached data.", e)

    else:
        # No cache — full scrape. Fetch all seasons one at a time with delay.
        logger.info("No cache found. Full scrape of all seasons (this takes ~10 minutes)...")
        player_frames, team_frames = [], []
        for season in SEASONS:
            time.sleep(random.uniform(3.0, 6.0))  # polite delay between seasons
            p = fetch_player_logs([season])
            t = fetch_team_logs([season])
            if not p.empty:
                player_frames.append(p)
            if not t.empty:
                team_frames.append(t)
        player_raw = pd.concat(player_frames, ignore_index=True) if player_frames else pd.DataFrame()
        team_raw   = pd.concat(team_frames,   ignore_index=True) if team_frames   else pd.DataFrame()

    # Standardize GAME_ID
    for df in [player_raw, team_raw]:
        if "Game_ID" in df.columns:
            df.rename(columns={"Game_ID": "GAME_ID"}, inplace=True)
        df.columns = [c.strip() for c in df.columns]
        # Deduplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

    player_raw.to_parquet(player_raw_path, index=False)
    team_raw.to_parquet(team_raw_path, index=False)
    logger.info("Raw logs: %d player rows, %d team rows", len(player_raw), len(team_raw))

    # ── Player info / positions ───────────────────────────────────────────────
    player_ids = player_raw["PLAYER_ID"].dropna().astype(int).unique().tolist()
    player_info_df = fetch_player_info(player_ids, pos_cache_path)

    # ── Schedule ──────────────────────────────────────────────────────────────
    # Load from cache if it exists; refresh from API but skip gracefully if down
    schedule_cache = RAW_DIR / "schedule.parquet"
    schedule_df = pd.DataFrame()
    if schedule_cache.exists():
        try:
            schedule_df = pd.read_parquet(schedule_cache)
            logger.info("Loaded cached schedule: %d games", len(schedule_df))
        except Exception:
            pass

    if is_offseason:
        logger.info("Off-season — skipping schedule fetch (no games scheduled).")
    else:
        logger.info("Fetching upcoming schedule (max 3 retries, skip if unavailable)...")
        try:
            fresh = fetch_upcoming_games(days_ahead=14)
            if not fresh.empty:
                schedule_df = fresh
                schedule_df.to_parquet(schedule_cache, index=False)
                logger.info("Schedule updated: %d upcoming games", len(schedule_df))
        except Exception as e:
            logger.warning("Schedule fetch failed (%s) — using cached or empty schedule", e)

    logger.info("Schedule: %d games available", len(schedule_df))

    # ── Feature engineering ───────────────────────────────────────────────────
    logger.info("Building team features...")
    team_features = build_team_features(team_raw)

    logger.info("Building player features...")
    player_features = build_player_features(player_raw, team_features, player_info_df)

    # ── Load to DuckDB ────────────────────────────────────────────────────────
    logger.info("Loading into DuckDB: %s", DB_PATH)
    load_to_duckdb(player_features, team_features, schedule_df, player_info_df)
    logger.info("Warehouse build complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build NBA data warehouse")
    parser.add_argument("--full", action="store_true", help="Full rebuild from scratch")
    args = parser.parse_args()
    main(full_rebuild=args.full)
