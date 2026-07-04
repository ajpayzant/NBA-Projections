"""
Google Sheets writer for NBA projection snapshots.
Adapted from PLL gsheets_writer.py for NBA stats.

One master spreadsheet, each game gets its own tab.
Tab name format: Away@Home_YYYY-MM-DD
"""
from __future__ import annotations

import datetime as _dt
import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("nba.sheets")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

STAT_LABELS = {
    "PTS": "Points", "REB": "Rebounds", "AST": "Assists",
    "FG3M": "3-Pt Made", "STL": "Steals", "BLK": "Blocks",
    "TOV": "Turnovers", "PRA": "Pts+Reb+Ast", "PR": "Pts+Reb",
    "PA": "Pts+Ast", "RA": "Reb+Ast", "MIN": "Minutes",
}


def _get_credentials():
    import streamlit as st
    from google.oauth2.service_account import Credentials
    return Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=SCOPES
    )


def _get_client():
    import gspread
    return gspread.authorize(_get_credentials())


def _get_sheet_id() -> str:
    import streamlit as st
    return str(st.secrets["google_drive"]["nba_sheet_id"])


def _ensure_access(sheet_id: str) -> None:
    """
    Verify access before every write. If a 403 is detected, try the
    Apps Script reshare trigger (if configured), then raise a clear
    error with instructions if access still fails.
    """
    gc = _get_client()
    try:
        gc.open_by_key(sheet_id)
        return  # access OK
    except Exception as e:
        if "403" not in str(e) and "permission" not in str(e).lower():
            raise  # not a permission error — re-raise

    logger.warning("Sheet access lost (403) — trying Apps Script reshare...")
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from scripts.reshare_sheets import trigger_apps_script_reshare
        triggered = trigger_apps_script_reshare()
        if triggered:
            logger.info("Apps Script reshare triggered — retrying access...")
            try:
                gc2 = _get_client()
                gc2.open_by_key(sheet_id)
                return  # access restored
            except Exception:
                pass
    except Exception as re2:
        logger.debug("Apps Script reshare attempt failed: %s", re2)

    raise PermissionError(
        "Google Sheets access lost.\n\n"
        "Fix: Open the Season Index sheet → click NBA Tools → Reshare with Service Account\n"
        "Or run: python scripts/reshare_sheets.py"
    )


def _tab_name(game: Dict) -> str:
    away = str(game.get("away_team_abbr", game.get("away_team", "Away")))
    home = str(game.get("home_team_abbr", game.get("home_team", "Home")))
    date = str(game.get("game_date", ""))[:10]
    return f"{away}@{home}_{date}"


def _rgb(r: int, g: int, b: int) -> Dict:
    return {"red": r / 255, "green": g / 255, "blue": b / 255}


def _range(sid: int, r0: int, r1: int, c0: int, c1: int) -> Dict:
    return {"sheetId": sid, "startRowIndex": r0, "endRowIndex": r1,
            "startColumnIndex": c0, "endColumnIndex": c1}


def _repeat(sid: int, r0: int, r1: int, c0: int, c1: int,
            fmt: Dict, fields: str) -> Dict:
    return {"repeatCell": {
        "range": _range(sid, r0, r1, c0, c1),
        "cell": {"userEnteredFormat": fmt},
        "fields": fields,
    }}


def _build_sections(result, game: Dict, engine) -> List[List[Any]]:
    """Build all worksheet rows as a flat list-of-lists."""
    from pages._engine_state import team_name

    rows: List[List[Any]] = []

    def _header(text):
        rows.append([text])

    def _blank():
        rows.append([])

    # ── METADATA ──────────────────────────────────────────────────────────────
    _header("METADATA")
    for field, val in [
        ("Game",         f"{team_name(result.away_team)} @ {team_name(result.home_team)}"),
        ("Game Date",    str(game.get("game_date", ""))[:10]),
        ("Away Team",    team_name(result.away_team)),
        ("Home Team",    team_name(result.home_team)),
        ("Saved At",     _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")),
        ("Model",        "NBA EWM Rate Model v1"),
    ]:
        rows.append([field, str(val)])

    _blank(); _blank()

    # ── TEAM PROJECTIONS ──────────────────────────────────────────────────────
    _header("TEAM PROJECTIONS")
    rows.append(["Team", "Proj PTS", "Proj REB", "Proj AST", "Proj 3PM",
                 "Win Prob", "Spread", "Total Line",
                 "Actual PTS", "Actual Score"])
    h, a = result.home_proj, result.away_proj
    gm = result.game_market
    for proj, wp in [(a, a.proj_win_prob), (h, h.proj_win_prob)]:
        rows.append([
            proj.team_abbr,
            round(proj.proj_pts, 1), round(proj.proj_reb, 1),
            round(proj.proj_ast, 1), round(proj.proj_fg3m, 1),
            round(wp, 3),
            f"{gm.spread_home:+.1f}" if proj.team_abbr == h.team_abbr else f"{-gm.spread_home:+.1f}",
            gm.total_line,
            "", "",  # Actual PTS, Actual Score — filled by sync
        ])

    _blank(); _blank()

    # ── GAME LINES ────────────────────────────────────────────────────────────
    _header("GAME LINES")
    rows.append(["Market", "Line", "Odds", "Fair Prob"])
    gs = result.game_sim
    fair_over = float(np.mean(gs.total_distribution > gm.total_line))
    for mkt, line, odds, fair in [
        (f"{result.away_team} ML",    "--", gm.away_ml,          f"{gm.away_win_prob*100:.1f}%"),
        (f"{result.home_team} ML",    "--", gm.home_ml,          f"{gm.home_win_prob*100:.1f}%"),
        (f"{result.away_team} Spread", f"{-gm.spread_home:+.1f}", gm.spread_away_odds, "--"),
        (f"{result.home_team} Spread", f"{gm.spread_home:+.1f}",  gm.spread_home_odds, "--"),
        ("Total Over",  f"{gm.total_line:.1f}", gm.over_odds,  f"{fair_over*100:.1f}%"),
        ("Total Under", f"{gm.total_line:.1f}", gm.under_odds, f"{(1-fair_over)*100:.1f}%"),
    ]:
        rows.append([mkt, line, str(odds), fair])

    _blank(); _blank()

    # ── PLAYER PROPS ──────────────────────────────────────────────────────────
    _header("PLAYER PROPS")
    rows.append([
        "Player", "Team", "Pos", "Stat", "Projection",
        "Main Line", "Over Odds", "Under Odds", "Fair P(Over)",
        "P10", "P25", "P50", "P75", "P90",
        "Actual Result", "Hit/Miss",
    ])

    all_players = {p.player_id: p for p in result.home_players + result.away_players}
    sims_all    = result.home_player_sims + result.away_player_sims
    prop_rows   = []

    PROJ_STATS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "PRA", "PR", "PA", "RA"]

    for sim in sims_all:
        proj = all_players.get(sim.player_id)
        if proj is None or not proj.active:
            continue
        mkt = result.player_markets.get(proj.player_id, {})
        pv  = sim.proj_values

        for stat in PROJ_STATS:
            if stat not in pv:
                continue
            proj_val = round(float(pv[stat]), 2)
            if proj_val < 0.5 and stat not in ("BLK", "STL"):
                continue
            m    = mkt.get(stat, {})
            dist = sim.stat_distributions.get(stat, np.array([]))
            prop_rows.append([
                proj.player_name, proj.team_abbr, proj.pos_group,
                STAT_LABELS.get(stat, stat),
                proj_val,
                m.get("line", ""),
                str(m.get("over_odds", "")),
                str(m.get("under_odds", "")),
                round(float(m.get("fair_over", 0)), 3),
                round(float(m.get("p10", 0)), 1) if m else "",
                round(float(m.get("p25", 0)), 1) if m else "",
                round(float(m.get("p50", 0)), 1) if m else "",
                round(float(m.get("p75", 0)), 1) if m else "",
                round(float(m.get("p90", 0)), 1) if m else "",
                "",  # Actual Result
                "",  # Hit/Miss
            ])

    prop_rows.sort(key=lambda r: (r[1], r[2], r[0], r[3]))
    rows.extend(prop_rows)
    return rows


def save_snapshot(result, game: Dict, engine) -> str:
    """Write projection snapshot to Google Sheets. Returns tab name."""
    sheet_id = _get_sheet_id()
    _ensure_access(sheet_id)   # auto-reshare if 403
    gc  = _get_client()
    sh  = gc.open_by_key(sheet_id)
    tab = _tab_name(game)

    existing = next((ws for ws in sh.worksheets() if ws.title == tab), None)
    if existing:
        sh.del_worksheet(existing)

    rows   = _build_sections(result, game, engine)
    n_rows = max(len(rows) + 5, 50)
    n_cols = 16

    ws  = sh.add_worksheet(title=tab, rows=n_rows, cols=n_cols)
    sid = ws.id
    ws.update(values=rows, range_name="A1")

    # ── Formatting ────────────────────────────────────────────────────────────
    NAVY     = _rgb(23, 37, 63)
    MED_BLUE = _rgb(37, 77, 130)
    WHITE    = _rgb(255, 255, 255)
    BLACK    = _rgb(0, 0, 0)
    LIGHT    = _rgb(235, 242, 252)
    GREEN_BG = _rgb(198, 239, 206); GREEN_FG = _rgb(0, 97, 0)
    RED_BG   = _rgb(255, 199, 206); RED_FG   = _rgb(156, 0, 6)
    AMBER_BG = _rgb(255, 243, 205); AMBER_FG = _rgb(133, 77, 14)

    SECTIONS = {"METADATA", "TEAM PROJECTIONS", "GAME LINES", "PLAYER PROPS"}
    section_rows: Dict[str, int] = {}
    col_header_rows: List[int] = []
    data_bands: List[tuple] = []

    i = 0
    while i < len(rows):
        cell0 = rows[i][0].strip().upper() if rows[i] else ""
        if cell0 in SECTIONS and len(rows[i]) == 1:
            section_rows[cell0] = i
            j = i + 1
            while j < len(rows) and not any(rows[j]):
                j += 1
            if j < len(rows) and any(rows[j]):
                col_header_rows.append(j)
                k = j + 1
                band_start = k
                while k < len(rows):
                    if not any(rows[k]):
                        break
                    k += 1
                if k > band_start:
                    data_bands.append((band_start, k))
        i += 1

    reqs = []
    # Base font
    reqs.append(_repeat(sid, 0, n_rows, 0, n_cols,
        {"textFormat": {"fontFamily": "Arial", "fontSize": 10},
         "verticalAlignment": "MIDDLE"},
        "userEnteredFormat(textFormat,verticalAlignment)"))
    # Explicit black on data rows
    reqs.append(_repeat(sid, 0, n_rows, 0, n_cols,
        {"textFormat": {"foregroundColor": BLACK}},
        "userEnteredFormat.textFormat.foregroundColor"))
    # Section headers
    for ri in section_rows.values():
        reqs.append(_repeat(sid, ri, ri+1, 0, n_cols,
            {"backgroundColor": NAVY,
             "textFormat": {"bold": True, "fontSize": 11,
                            "foregroundColor": WHITE, "fontFamily": "Arial"}},
            "userEnteredFormat(backgroundColor,textFormat)"))
    # Column headers
    for ri in col_header_rows:
        reqs.append(_repeat(sid, ri, ri+1, 0, n_cols,
            {"backgroundColor": MED_BLUE,
             "textFormat": {"bold": True, "foregroundColor": WHITE,
                            "fontSize": 10, "fontFamily": "Arial"},
             "horizontalAlignment": "CENTER"},
            "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)"))
    # Zebra rows
    for bs, be in data_bands:
        for ri in range(bs, be):
            bg = LIGHT if (ri - bs) % 2 == 1 else WHITE
            reqs.append(_repeat(sid, ri, ri+1, 0, n_cols,
                {"backgroundColor": bg}, "userEnteredFormat(backgroundColor)"))
    # Center data cols C+
    for bs, be in data_bands:
        reqs.append(_repeat(sid, bs, be, 2, n_cols,
            {"horizontalAlignment": "CENTER"},
            "userEnteredFormat(horizontalAlignment)"))
    # Metadata col B left-aligned
    if data_bands:
        meta_start, meta_end = min(data_bands, key=lambda b: b[0])
        reqs.append(_repeat(sid, meta_start, meta_end, 1, 2,
            {"horizontalAlignment": "LEFT"},
            "userEnteredFormat(horizontalAlignment)"))
    # Player name col A left-aligned in player props band
    if data_bands:
        pp_start, pp_end = max(data_bands, key=lambda b: b[1] - b[0])
        reqs.append(_repeat(sid, pp_start, pp_end, 0, 1,
            {"horizontalAlignment": "LEFT"},
            "userEnteredFormat(horizontalAlignment)"))
    # Merge B2:D2 for matchup title
    reqs.append({"mergeCells": {"range": _range(sid, 1, 2, 1, 4), "mergeType": "MERGE_ALL"}})
    # Col widths
    reqs.append({"updateDimensionProperties": {
        "range": {"sheetId": sid, "dimension": "COLUMNS", "startIndex": 0, "endIndex": 1},
        "properties": {"pixelSize": 180}, "fields": "pixelSize"}})
    reqs.append({"updateDimensionProperties": {
        "range": {"sheetId": sid, "dimension": "COLUMNS", "startIndex": 1, "endIndex": 2},
        "properties": {"pixelSize": 100}, "fields": "pixelSize"}})
    # Hit/Miss conditional formatting
    for bs, be in data_bands:
        reqs.append({"addConditionalFormatRule": {"rule": {
            "ranges": [_range(sid, bs, be, 15, 16)],
            "booleanRule": {
                "condition": {"type": "TEXT_EQ", "values": [{"userEnteredValue": "Hit"}]},
                "format": {"backgroundColor": GREEN_BG,
                           "textFormat": {"foregroundColor": GREEN_FG, "bold": True}},
            }}, "index": 0}})
        reqs.append({"addConditionalFormatRule": {"rule": {
            "ranges": [_range(sid, bs, be, 15, 16)],
            "booleanRule": {
                "condition": {"type": "TEXT_EQ", "values": [{"userEnteredValue": "Miss"}]},
                "format": {"backgroundColor": RED_BG,
                           "textFormat": {"foregroundColor": RED_FG, "bold": True}},
            }}, "index": 1}})
        # Fair P(Over) > 55% green, < 45% red (col 8)
        reqs.append({"addConditionalFormatRule": {"rule": {
            "ranges": [_range(sid, bs, be, 8, 9)],
            "booleanRule": {
                "condition": {"type": "NUMBER_GREATER", "values": [{"userEnteredValue": "0.55"}]},
                "format": {"backgroundColor": GREEN_BG},
            }}, "index": 2}})
        reqs.append({"addConditionalFormatRule": {"rule": {
            "ranges": [_range(sid, bs, be, 8, 9)],
            "booleanRule": {
                "condition": {"type": "NUMBER_LESS", "values": [{"userEnteredValue": "0.45"}]},
                "format": {"backgroundColor": RED_BG},
            }}, "index": 3}})
    # Tab colour
    reqs.append({"updateSheetProperties": {
        "properties": {"sheetId": sid, "tabColorStyle": {"rgbColor": _rgb(23, 37, 63)}},
        "fields": "tabColorStyle"}})

    sh.batch_update({"requests": reqs})

    # Autofit after filter
    if "PLAYER PROPS" in section_rows:
        pp_hdr = next((r for r in col_header_rows if r > section_rows["PLAYER PROPS"]), None)
        if pp_hdr and data_bands:
            pp_band = next(((s, e) for s, e in data_bands if s > section_rows["PLAYER PROPS"]), None)
            if pp_band:
                sh.batch_update({"requests": [{"setBasicFilter": {"filter": {
                    "range": _range(sid, pp_hdr, pp_band[1], 0, n_cols)}}}]})

    sh.batch_update({"requests": [{"autoResizeDimensions": {
        "dimensions": {"sheetId": sid, "dimension": "COLUMNS",
                       "startIndex": 2, "endIndex": n_cols}}}]})

    # Update Overview dashboard
    _update_dashboard(sh, tab)

    logger.info("NBA snapshot saved: %s", tab)
    return tab


def _update_dashboard(sh, latest_tab: str) -> None:
    """Rewrite the Overview tab with game index and instructions."""
    try:
        ws = next((w for w in sh.worksheets() if w.title in ("Sheet1", "Overview")), None)
        if ws is None:
            ws = sh.add_worksheet(title="Overview", rows=100, cols=8)
        elif ws.title == "Sheet1":
            ws.update_title("Overview")
        sid = ws.id

        games = []
        for w in sh.worksheets():
            t = w.title
            if "@" not in t or "_" not in t:
                continue
            try:
                matchup, date = t.rsplit("_", 1)
                away, home = matchup.split("@", 1)
                # Check actuals
                vals = w.col_values(15)
                has_actuals = any(
                    v.strip() not in ("", "Actual Result")
                    and v.strip().lstrip("-").replace(".", "", 1).isdigit()
                    for v in vals
                )
                games.append({"tab": t, "away": away, "home": home,
                              "date": date, "actuals": has_actuals})
            except Exception:
                continue
        games.sort(key=lambda g: g["date"], reverse=True)

        rows: List[List[Any]] = []
        rows.append(["NBA PROJECTION SNAPSHOTS"])
        rows.append(["Master tracking sheet for all NBA game projections and actuals"])
        rows.append([f"Last updated: {_dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"])
        rows.append([])
        rows.append(["SAVED GAMES"])
        rows.append(["Game Date", "Matchup", "Tab Name", "Actuals Synced"])
        for g in games:
            rows.append([g["date"], f"{g['away']} @ {g['home']}", g["tab"],
                         "Yes" if g["actuals"] else "Pending"])
        if not games:
            rows.append(["No games saved yet.", "", "", ""])
        rows.append([]); rows.append([])
        rows.append(["HOW TO USE"])
        rows.append(["Step", "Action", "Details"])
        rows.append(["1", "Run projection", "NBA app: select matchup, run projection"])
        rows.append(["2", "Save snapshot", "Click Save to Google Sheets button"])
        rows.append(["3", "Sync actuals", "After game ends, click Sync Actuals"])
        rows.append(["4", "Review history", "Page 5 in the NBA app"])
        rows.append([]); rows.append([])
        rows.append(["PLAYER PROPS COLUMN GUIDE"])
        rows.append(["Column", "Description"])
        rows.append(["Projection", "Model expected value"])
        rows.append(["Main Line", "Balanced prop line"])
        rows.append(["Fair P(Over)", "Model true probability — green >55%, red <45%"])
        rows.append(["P10-P90", "Simulation percentiles from 20,000 draws"])
        rows.append(["Actual Result", "Auto-filled by Sync Actuals"])
        rows.append(["Hit/Miss", "Green=Hit, Red=Miss"])

        ws.clear()
        sh.batch_update({"requests": [{"updateCells": {
            "range": {"sheetId": sid, "startRowIndex": 0, "endRowIndex": 100,
                      "startColumnIndex": 0, "endColumnIndex": 8},
            "fields": "userEnteredFormat"}}]})
        ws.update(values=rows, range_name="A1")

        NAVY = _rgb(23, 37, 63); MED_BLUE = _rgb(37, 77, 130)
        WHITE = _rgb(255, 255, 255); BLACK = _rgb(0, 0, 0)
        LIGHT = _rgb(235, 242, 252); WHITE_BG = _rgb(255, 255, 255)
        GREEN_BG = _rgb(198, 239, 206); GREEN_FG = _rgb(0, 97, 0)
        AMBER_BG = _rgb(255, 243, 205); AMBER_FG = _rgb(133, 77, 14)

        games_hdr = next((i for i, r in enumerate(rows) if r and r[0] == "SAVED GAMES"), None)
        games_col_hdr = games_hdr + 1 if games_hdr is not None else None
        games_data_start = games_col_hdr + 1 if games_col_hdr is not None else None
        games_data_end   = games_data_start + len(games) if games_data_start is not None else None
        how_hdr   = next((i for i, r in enumerate(rows) if r and r[0] == "HOW TO USE"), None)
        guide_hdr = next((i for i, r in enumerate(rows) if r and r[0] == "PLAYER PROPS COLUMN GUIDE"), None)

        reqs = []
        reqs.append(_repeat(sid, 0, len(rows)+5, 0, 8,
            {"textFormat": {"fontFamily": "Arial", "fontSize": 10},
             "verticalAlignment": "MIDDLE"},
            "userEnteredFormat(textFormat,verticalAlignment)"))
        reqs.append(_repeat(sid, 0, len(rows)+5, 0, 8,
            {"textFormat": {"foregroundColor": BLACK}},
            "userEnteredFormat.textFormat.foregroundColor"))
        reqs.append(_repeat(sid, 0, 1, 0, 8,
            {"backgroundColor": NAVY,
             "textFormat": {"bold": True, "fontSize": 14,
                            "foregroundColor": WHITE, "fontFamily": "Arial"}},
            "userEnteredFormat(backgroundColor,textFormat)"))
        reqs.append(_repeat(sid, 1, 4, 0, 8,
            {"backgroundColor": MED_BLUE,
             "textFormat": {"foregroundColor": WHITE, "fontSize": 10,
                            "fontFamily": "Arial", "bold": False}},
            "userEnteredFormat(backgroundColor,textFormat)"))
        for ri in [r for r in [games_hdr, how_hdr, guide_hdr] if r is not None]:
            reqs.append(_repeat(sid, ri, ri+1, 0, 8,
                {"backgroundColor": NAVY,
                 "textFormat": {"bold": True, "fontSize": 11,
                                "foregroundColor": WHITE, "fontFamily": "Arial"}},
                "userEnteredFormat(backgroundColor,textFormat)"))
        for ri in [r for r in [games_col_hdr,
                                how_hdr+1 if how_hdr else None,
                                guide_hdr+1 if guide_hdr else None] if r is not None]:
            reqs.append(_repeat(sid, ri, ri+1, 0, 8,
                {"backgroundColor": MED_BLUE,
                 "textFormat": {"bold": True, "foregroundColor": WHITE,
                                "fontSize": 10, "fontFamily": "Arial"},
                 "horizontalAlignment": "CENTER"},
                "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)"))
        if games_data_start is not None and games_data_end is not None:
            for ri in range(games_data_start, games_data_end):
                bg = LIGHT if (ri - games_data_start) % 2 == 1 else WHITE_BG
                reqs.append(_repeat(sid, ri, ri+1, 0, 8,
                    {"backgroundColor": bg}, "userEnteredFormat(backgroundColor)"))
            reqs.append({"addConditionalFormatRule": {"rule": {
                "ranges": [_range(sid, games_data_start, games_data_end, 3, 4)],
                "booleanRule": {
                    "condition": {"type": "TEXT_CONTAINS",
                                  "values": [{"userEnteredValue": "Yes"}]},
                    "format": {"backgroundColor": GREEN_BG,
                               "textFormat": {"foregroundColor": GREEN_FG, "bold": True}},
                }}, "index": 0}})
            reqs.append({"addConditionalFormatRule": {"rule": {
                "ranges": [_range(sid, games_data_start, games_data_end, 3, 4)],
                "booleanRule": {
                    "condition": {"type": "TEXT_CONTAINS",
                                  "values": [{"userEnteredValue": "Pending"}]},
                    "format": {"backgroundColor": AMBER_BG,
                               "textFormat": {"foregroundColor": AMBER_FG, "bold": True}},
                }}, "index": 1}})
        reqs.append({"updateSheetProperties": {
            "properties": {"sheetId": sid,
                           "tabColorStyle": {"rgbColor": _rgb(23, 37, 63)}},
            "fields": "tabColorStyle"}})

        sh.batch_update({"requests": reqs})
        sh.batch_update({"requests": [{"autoResizeDimensions": {
            "dimensions": {"sheetId": sid, "dimension": "COLUMNS",
                           "startIndex": 0, "endIndex": 8}}}]})
        logger.info("NBA Overview dashboard updated")
    except Exception as e:
        logger.warning("Dashboard update failed (non-fatal): %s", e)


def list_saved_games() -> List[Dict]:
    """Return list of saved game tabs."""
    try:
        gc = _get_client()
        sh = gc.open_by_key(_get_sheet_id())
        games = []
        for ws in sh.worksheets():
            t = ws.title
            if "@" not in t or "_" not in t:
                continue
            try:
                matchup, date = t.rsplit("_", 1)
                away, home = matchup.split("@", 1)
                games.append({"tab_name": t, "away": away, "home": home,
                              "game_date": date, "sheet_id": ws.id})
            except Exception:
                continue
        return sorted(games, key=lambda g: g["game_date"], reverse=True)
    except Exception as e:
        logger.warning("list_saved_games failed: %s", e)
        return []


def read_game_tab(tab_name: str) -> Dict[str, pd.DataFrame]:
    """Read a saved game tab, return dict of DataFrames by section."""
    gc = _get_client()
    sh = gc.open_by_key(_get_sheet_id())
    ws = sh.worksheet(tab_name)
    all_vals = ws.get_all_values()

    sections: Dict[str, List] = {}
    current = None
    current_rows: List = []
    HEADERS = {"METADATA", "TEAM PROJECTIONS", "GAME LINES", "PLAYER PROPS"}

    for row in all_vals:
        cell0 = row[0].strip().upper() if row else ""
        if not any(c.strip() for c in row):
            if current and current_rows:
                sections[current] = current_rows
                current_rows = []; current = None
            continue
        if cell0 in HEADERS and not any(row[1:]):
            if current and current_rows:
                sections[current] = current_rows
            current = cell0; current_rows = []
        elif current:
            current_rows.append(row)

    if current and current_rows:
        sections[current] = current_rows

    result = {}
    for key, raw in sections.items():
        if not raw:
            continue
        header = raw[0]; data = raw[1:]
        df = pd.DataFrame(data, columns=[str(c).strip() for c in header])
        result[key.lower().replace(" ", "_")] = df
    return result


def sync_actuals(tab_name: str, db_path: str) -> Dict[str, int]:
    """Pull actual stats from warehouse and fill Actual Result / Hit/Miss columns."""
    import duckdb

    gc = _get_client()
    sh = gc.open_by_key(_get_sheet_id())
    ws = sh.worksheet(tab_name)
    all_vals = ws.get_all_values()

    # Parse date from tab name
    try:
        matchup, date_str = tab_name.rsplit("_", 1)
        away, home = matchup.split("@", 1)
    except Exception as e:
        raise ValueError(f"Cannot parse tab name '{tab_name}': {e}")

    con = duckdb.connect(db_path, read_only=True)
    try:
        game_row = con.execute("""
            SELECT GAME_ID, TEAM_ABBREVIATION
            FROM clean.player_game_stats
            WHERE CAST(GAME_DATE AS VARCHAR) LIKE ?
            LIMIT 1
        """, [f"{date_str}%"]).fetchone()

        if not game_row:
            raise ValueError(
                f"No game data found for {date_str}. "
                "Run the data pipeline first to ingest actuals."
            )
        game_id = game_row[0]

        player_actuals = con.execute("""
            SELECT PLAYER_NAME, PTS, REB, AST, FG3M, STL, BLK, TOV
            FROM clean.player_game_stats
            WHERE GAME_ID = ?
        """, [game_id]).df()

        team_actuals = con.execute("""
            SELECT TEAM_ABBREVIATION, PTS
            FROM clean.team_game_stats
            WHERE GAME_ID = ?
        """, [game_id]).df() if "clean.team_game_stats" in [
            t[0] for t in con.execute(
                "SELECT table_schema||'.'||table_name FROM information_schema.tables"
            ).fetchall()
        ] else pd.DataFrame()
    finally:
        con.close()

    STAT_COL_MAP = {
        "Points": "PTS", "Rebounds": "REB", "Assists": "AST",
        "3-Pt Made": "FG3M", "Steals": "STL", "Blocks": "BLK",
        "Turnovers": "TOV",
    }

    player_lookup: Dict[str, Dict[str, float]] = {}
    for _, row in player_actuals.iterrows():
        name = str(row["PLAYER_NAME"]).strip()
        player_lookup[name] = {
            label: float(row.get(col, 0) or 0)
            for label, col in STAT_COL_MAP.items()
        }
        # Combos
        pts = float(row.get("PTS", 0) or 0)
        reb = float(row.get("REB", 0) or 0)
        ast = float(row.get("AST", 0) or 0)
        player_lookup[name].update({
            "Pts+Reb+Ast": pts+reb+ast, "Pts+Reb": pts+reb,
            "Pts+Ast": pts+ast, "Reb+Ast": reb+ast,
        })

    updates = []
    players_updated = teams_updated = 0
    in_props = False
    props_header_done = False

    ACTUAL_COL = 15   # 1-indexed column O
    HIT_COL    = 16   # 1-indexed column P
    T_ACTUAL_COL = 9  # Team Actual PTS col

    in_teams = False
    teams_header_done = False

    for i, row in enumerate(all_vals):
        cell0 = row[0].strip().upper() if row else ""
        if not any(c.strip() for c in row):
            in_props = False; props_header_done = False
            in_teams = False; teams_header_done = False
            continue
        if cell0 == "PLAYER PROPS" and not any(c for c in row[1:] if c):
            in_props = True; props_header_done = False; continue
        if cell0 == "TEAM PROJECTIONS" and not any(c for c in row[1:] if c):
            in_teams = True; teams_header_done = False; continue

        if in_props:
            if not props_header_done:
                props_header_done = True; continue
            if len(row) < 5:
                continue
            player_name = row[0].strip()
            stat_label  = row[3].strip()
            line_val    = row[5].strip()
            actuals = player_lookup.get(player_name, {})
            actual = actuals.get(stat_label)
            if actual is not None:
                updates.append((i+1, ACTUAL_COL, actual))
                try:
                    hit = "Hit" if actual >= float(line_val) else "Miss"
                except (ValueError, TypeError):
                    hit = ""
                updates.append((i+1, HIT_COL, hit))
                players_updated += 1

        if in_teams:
            if not teams_header_done:
                teams_header_done = True; continue
            if len(row) < 2:
                continue
            abbr = row[0].strip().upper()
            if not team_actuals.empty:
                match = team_actuals[team_actuals["TEAM_ABBREVIATION"].str.upper() == abbr]
                if not match.empty:
                    updates.append((i+1, T_ACTUAL_COL, float(match.iloc[0]["PTS"])))
                    teams_updated += 1

    if updates:
        cell_list = []
        for r, c, v in updates:
            cell = ws.cell(r, c)
            cell.value = v
            cell_list.append(cell)
        ws.update_cells(cell_list)

    logger.info("NBA actuals synced: %d player rows, %d team rows", players_updated, teams_updated)
    return {"players_updated": players_updated, "teams_updated": teams_updated}
