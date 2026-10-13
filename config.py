"""Paths, seasons and source URLs for the NBA player game projections."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"            # gzipped ESPN game summaries, one file per game (gitignored)
LAKE = DATA / "lake"          # parsed parquet tables built from RAW
MODELS = DATA / "models"
OVERRIDES = DATA / "overrides"
LINEUP_DIR = DATA / "lineups"  # Rotowire lineup and ESPN injury snapshots

# Season = the year it starts: 2025 = 2025-26. History from 2018-19.
HISTORY_SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
CURRENT_SEASON = 2026
SEASON_TYPES = (2, 3, 5)      # ESPN: 2 regular season, 3 playoffs, 5 play-in

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"
REQUEST_TIMEOUT = 30
FETCH_WORKERS = 6

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ROTOWIRE_LINEUPS = "https://www.rotowire.com/basketball/nba-lineups.php"


def season_of(date: str) -> int:
    """'2025-03-15' -> 2024 (the season that started in autumn 2024)."""
    y, m = int(date[:4]), int(date[5:7])
    return y if m >= 8 else y - 1
