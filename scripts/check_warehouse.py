"""
NBA Warehouse Diagnostic
========================
Run this to verify all data is present and correct before using the app.

Usage:
    python scripts/check_warehouse.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import datetime as dt
import numpy as np
import pandas as pd

DB_PATH  = _ROOT / "data" / "analytics_database" / "nba_warehouse.duckdb"
RAW_DIR  = _ROOT / "data" / "raw_cache"
REF_DIR  = _ROOT / "data" / "reference_tables"

PASS = "✅"
WARN = "⚠️ "
FAIL = "❌"

issues = []

def check(label, condition, detail="", warn_only=False):
    if condition:
        print(f"  {PASS}  {label}")
        if detail:
            print(f"         {detail}")
    else:
        sym = WARN if warn_only else FAIL
        print(f"  {sym}  {label}")
        if detail:
            print(f"         {detail}")
        if not warn_only:
            issues.append(label)

print()
print("=" * 60)
print("NBA WAREHOUSE DIAGNOSTIC")
print(f"Run at: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)

# ── 1. Raw cache files ────────────────────────────────────────────────────────
print("\n[1] Raw cache files")

player_path = RAW_DIR / "player_game_logs.parquet"
team_path   = RAW_DIR / "team_game_logs.parquet"
pos_path    = REF_DIR / "player_positions.json"

check("player_game_logs.parquet exists", player_path.exists(),
      f"Path: {player_path}")
check("team_game_logs.parquet exists", team_path.exists(),
      f"Path: {team_path}")
check("player_positions.json exists", pos_path.exists(),
      f"Path: {pos_path}")

if player_path.exists() and team_path.exists():
    p_raw = pd.read_parquet(player_path)
    t_raw = pd.read_parquet(team_path)
    import json
    pos_data = json.loads(pos_path.read_text()) if pos_path.exists() else {}

    check("Player rows > 100,000", len(p_raw) > 100_000,
          f"Got {len(p_raw):,} rows")
    check("Team rows > 10,000", len(t_raw) > 10_000,
          f"Got {len(t_raw):,} rows")
    check("Player positions > 500", len(pos_data) > 500,
          f"Got {len(pos_data):,} players")

    if "SEASON" in p_raw.columns:
        seasons = sorted(p_raw["SEASON"].unique())
        current = "2025-26"
        check(f"Current season {current} present", current in seasons,
              f"Seasons available: {seasons}")
        check("At least 4 seasons of data", len(seasons) >= 4,
              f"Seasons: {seasons}")

    if "GAME_DATE" in p_raw.columns:
        p_raw["GAME_DATE"] = pd.to_datetime(p_raw["GAME_DATE"], errors="coerce")
        latest = p_raw["GAME_DATE"].max()
        # During off-season, latest game should be from June at the latest
        today = dt.date.today()
        is_offseason = today.month in range(7, 10) or \
                       (today.month == 6 and today.day > 20)
        if is_offseason:
            check("Latest game date is from this season",
                  latest.year >= 2025,
                  f"Latest game: {latest.date()}")
        else:
            days_old = (dt.datetime.now() - latest).days
            check("Data is fresh (< 3 days old)", days_old < 3,
                  f"Latest game: {latest.date()} ({days_old} days ago)",
                  warn_only=days_old >= 3)

# ── 2. DuckDB warehouse ───────────────────────────────────────────────────────
print("\n[2] DuckDB warehouse")

check("nba_warehouse.duckdb exists", DB_PATH.exists(),
      f"Path: {DB_PATH}")

if DB_PATH.exists():
    try:
        import duckdb
        con = duckdb.connect(str(DB_PATH), read_only=True)

        # Tables
        tables = con.execute("""
            SELECT table_schema || '.' || table_name AS tbl
            FROM information_schema.tables
            ORDER BY 1
        """).df()["tbl"].tolist()

        required = [
            "clean.player_game_stats",
            "clean.team_game_stats",
            "clean.player_info",
            "marts.player_season_stats",
        ]
        for tbl in required:
            exists = tbl in tables
            check(f"Table {tbl} exists", exists)
            if exists:
                n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                check(f"  {tbl} has rows", n > 0, f"{n:,} rows")

        # Key columns in player_game_stats
        print("\n[3] Key columns and data quality")
        if "clean.player_game_stats" in tables:
            cols = con.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema='clean' AND table_name='player_game_stats'
                ORDER BY column_name
            """).df()["column_name"].tolist()

            required_cols = [
                "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_DATE",
                "SEASON", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
                "FG3M", "FGA", "FGM", "USG_PCT",
                "PTS_PM_EWM", "REB_PM_EWM", "AST_PM_EWM", "MIN_EWM",
                "FG3M_PM_EWM", "FG_PCT_EWM", "FG3_PCT_EWM", "FT_PCT_EWM",
                "PACE_EWM", "POS_GROUP",
            ]
            missing_cols = [c for c in required_cols if c not in cols]
            check("All required player columns present",
                  len(missing_cols) == 0,
                  f"Missing: {missing_cols}" if missing_cols else f"{len(cols)} columns total")

            # Data quality
            sample = con.execute("""
                SELECT
                    COUNT(*) as n,
                    COUNT(DISTINCT PLAYER_ID) as players,
                    COUNT(DISTINCT TEAM_ABBREVIATION) as teams,
                    COUNT(DISTINCT SEASON) as seasons,
                    AVG(PTS) as avg_pts,
                    AVG(MIN) as avg_min,
                    AVG(USG_PCT) as avg_usg,
                    SUM(CASE WHEN MIN_EWM IS NULL THEN 1 ELSE 0 END) as null_min_ewm,
                    SUM(CASE WHEN PTS_PM_EWM IS NULL THEN 1 ELSE 0 END) as null_pts_ewm
                FROM clean.player_game_stats
                WHERE MIN > 0
            """).fetchone()

            check("Reasonable avg PTS (8-20 expected)", 8 < sample[4] < 25,
                  f"avg_pts={sample[4]:.1f}")
            check("Reasonable avg MIN (20-30 expected)", 15 < sample[5] < 40,
                  f"avg_min={sample[5]:.1f}")
            check("USG% populated", sample[6] is not None and sample[6] > 0,
                  f"avg_usg={sample[6]:.1f}%" if sample[6] else "NULL")
            check("MIN_EWM has no nulls in active players",
                  sample[7] == 0, f"{sample[7]} null rows")
            check("PTS_PM_EWM has no nulls in active players",
                  sample[8] == 0, f"{sample[8]} null rows")

            print(f"\n         Summary: {sample[0]:,} game rows · "
                  f"{sample[1]:,} players · {sample[2]} teams · {sample[3]} seasons")

        # Team stats
        if "clean.team_game_stats" in tables:
            t_sample = con.execute("""
                SELECT AVG(TEAM_PTS) as avg_pts, AVG(PACE) as avg_pace,
                       COUNT(DISTINCT TEAM_ABBREVIATION) as teams
                FROM clean.team_game_stats
                WHERE TEAM_PTS IS NOT NULL
            """).fetchone()
            check("Team avg PTS reasonable (100-125)", 95 < t_sample[0] < 130,
                  f"avg_team_pts={t_sample[0]:.1f}")
            check("Pace populated", t_sample[1] is not None and t_sample[1] > 80,
                  f"avg_pace={t_sample[1]:.1f}")
            check("All 30 teams present", t_sample[2] == 30,
                  f"Got {t_sample[2]} teams")

        # Per-team sample for current season
        print("\n[4] Current season team coverage (2025-26)")
        team_coverage = con.execute("""
            SELECT TEAM_ABBREVIATION, COUNT(*) as games,
                   AVG(PTS) as avg_pts, MAX(GAME_DATE) as last_game
            FROM clean.player_game_stats
            WHERE SEASON = '2025-26' AND MIN > 10
            GROUP BY TEAM_ABBREVIATION
            ORDER BY TEAM_ABBREVIATION
        """).df()

        check("All 30 teams have 2025-26 data", len(team_coverage) == 30,
              f"Got {len(team_coverage)} teams")
        if not team_coverage.empty:
            min_games = team_coverage["games"].min()
            max_games = team_coverage["games"].max()
            check("Each team has > 50 player-game rows", min_games > 50,
                  f"Min: {min_games}, Max: {max_games} player-game rows per team")
            print(f"\n         Sample teams (2025-26):")
            for _, row in team_coverage.head(5).iterrows():
                print(f"           {row['TEAM_ABBREVIATION']}: "
                      f"{row['games']} rows, avg {row['avg_pts']:.1f} pts")

        # EWM sanity check — pick a known star player
        print("\n[5] EWM rating sanity check")
        known_players = ["LeBron James", "Stephen Curry", "Kevin Durant",
                         "Nikola Jokic", "Giannis Antetokounmpo"]
        for name in known_players:
            row = con.execute("""
                SELECT PLAYER_NAME, TEAM_ABBREVIATION, MIN_EWM, PTS_PM_EWM,
                       USG_PCT, GAMES_PLAYED
                FROM clean.player_game_stats
                WHERE PLAYER_NAME LIKE ?
                ORDER BY GAME_DATE DESC LIMIT 1
            """, [f"%{name.split()[0]}%"]).fetchone()
            if row:
                proj_pts = row[2] * row[3] if row[2] and row[3] else 0
                check(f"{row[0]} ({row[1]}) looks reasonable",
                      10 < proj_pts < 50 and row[5] > 50,
                      f"MIN_EWM={row[2]:.1f}, PTS_PM={row[3]:.3f}, "
                      f"proj_pts≈{proj_pts:.1f}, games={row[5]}")
                break

        con.close()

    except Exception as e:
        check("DuckDB readable", False, str(e))

# ── 5. Engine smoke test ──────────────────────────────────────────────────────
print("\n[6] Engine smoke test")
try:
    from nba_engine import NBAProjectionEngine
    eng = NBAProjectionEngine(db_path=str(DB_PATH))
    eng.load()
    check("Engine loads", True, f"Player rows: {len(eng.player_games):,}")

    eng.fit()
    check("Engine fits", True)

    # Try projecting a real upcoming matchup or a known rivalry
    test_matchup = ("BOS", "LAL")
    r = eng.project(test_matchup[0], test_matchup[1])
    check(f"Projection {test_matchup[0]} vs {test_matchup[1]} runs",
          r.home_proj.proj_pts > 0,
          f"{test_matchup[0]}: {r.home_proj.proj_pts:.1f} pts  "
          f"{test_matchup[1]}: {r.away_proj.proj_pts:.1f} pts  "
          f"win prob: {r.game_sim.home_win_prob:.1%}")

    active_home = [p for p in r.home_players if p.active]
    check(f"{test_matchup[0]} has active players", len(active_home) >= 8,
          f"{len(active_home)} active players, top scorer: "
          f"{max(active_home, key=lambda x: x.proj_pts).player_name} "
          f"({max(active_home, key=lambda x: x.proj_pts).proj_pts:.1f} pts)")

    check("Player simulations generated",
          len(r.home_player_sims) > 0,
          f"{len(r.home_player_sims) + len(r.away_player_sims)} total player sims")

    check("Prop markets generated",
          len(r.player_markets) > 0,
          f"{len(r.player_markets)} players priced")

except Exception as e:
    check("Engine smoke test", False, str(e))

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
if not issues:
    print(f"{PASS}  ALL CHECKS PASSED — data is ready to use")
else:
    print(f"{FAIL}  {len(issues)} ISSUE(S) FOUND:")
    for issue in issues:
        print(f"     • {issue}")
    print()
    print("Run: python scripts/build_warehouse.py --full")
    print("to rebuild from scratch.")
print("=" * 60)
print()
