"""
Shared engine state for NBA Projection App.
Loaded once per session via st.cache_resource.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

_PAGES_DIR = Path(__file__).resolve().parent
_ROOT      = _PAGES_DIR.parent
for _p in [str(_ROOT), str(_PAGES_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nba_engine import (
    NBAProjectionEngine, ProjectionResult, TeamProjection, PlayerProjection,
    NBAPricingEngine, N_SIMS,
    LG_PTS, LG_REB, LG_AST, LG_FG3M, LG_STL, LG_BLK, LG_TOV,
)

DB_PATH = os.getenv(
    "NBA_DB_PATH",
    str(_ROOT / "data" / "analytics_database" / "nba_warehouse.duckdb"),
)
_AUTOSAVE_PATH = _ROOT / "data" / "session_autosave.json"

# ---------------------------------------------------------------------------
# Team color / name maps
# ---------------------------------------------------------------------------
TEAM_COLORS = {
    "ATL": "#e03a3e", "BOS": "#007a33", "BKN": "#000000", "CHA": "#1d1160",
    "CHI": "#ce1141", "CLE": "#860038", "DAL": "#00538c", "DEN": "#0e2240",
    "DET": "#c8102e", "GSW": "#1d428a", "HOU": "#ce1141", "IND": "#002d62",
    "LAC": "#c8102e", "LAL": "#552583", "MEM": "#5d76a9", "MIA": "#98002e",
    "MIL": "#00471b", "MIN": "#0c2340", "NOP": "#0c2340", "NYK": "#f58426",
    "OKC": "#007ac1", "ORL": "#0077c0", "PHI": "#006bb6", "PHX": "#1d1160",
    "POR": "#e03a3e", "SAC": "#5a2d81", "SAS": "#c4ced4", "TOR": "#ce1141",
    "UTA": "#002b5c", "WAS": "#002b5c",
}

TEAM_NAMES = {
    "ATL": "Hawks",      "BOS": "Celtics",    "BKN": "Nets",
    "CHA": "Hornets",    "CHI": "Bulls",      "CLE": "Cavaliers",
    "DAL": "Mavericks",  "DEN": "Nuggets",    "DET": "Pistons",
    "GSW": "Warriors",   "HOU": "Rockets",    "IND": "Pacers",
    "LAC": "Clippers",   "LAL": "Lakers",     "MEM": "Grizzlies",
    "MIA": "Heat",       "MIL": "Bucks",      "MIN": "Timberwolves",
    "NOP": "Pelicans",   "NYK": "Knicks",     "OKC": "Thunder",
    "ORL": "Magic",      "PHI": "76ers",      "PHX": "Suns",
    "POR": "Blazers",    "SAC": "Kings",      "SAS": "Spurs",
    "TOR": "Raptors",    "UTA": "Jazz",       "WAS": "Wizards",
}

TEAM_FULL_NAMES = {k: f"{k} {v}" for k, v in TEAM_NAMES.items()}

def team_color(abbr: str) -> str:
    return TEAM_COLORS.get(str(abbr).upper(), "#475569")

def team_name(abbr: str) -> str:
    return TEAM_FULL_NAMES.get(str(abbr).upper(), str(abbr))

# ---------------------------------------------------------------------------
# Rating override definitions (player-level)
# ---------------------------------------------------------------------------
PLAYER_RATING_DEFS = {
    "minutes_override": {
        "label": "Minutes",
        "help": "Override projected minutes. Model default is based on recent EWM.",
        "min": 0.0, "max": 48.0, "step": 0.5, "fmt": "{:.1f}",
        "positions": ["G", "F", "C", "UNK"],
    },
    "pts_pm_override": {
        "label": "Pts/min rate",
        "help": "Points per minute rate. League avg ~0.52.",
        "min": 0.0, "max": 2.0, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["G", "F", "C", "UNK"],
    },
    "reb_pm_override": {
        "label": "Reb/min rate",
        "help": "Rebounds per minute rate.",
        "min": 0.0, "max": 1.5, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["G", "F", "C", "UNK"],
    },
    "ast_pm_override": {
        "label": "Ast/min rate",
        "help": "Assists per minute rate.",
        "min": 0.0, "max": 1.0, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["G", "F", "C", "UNK"],
    },
    "fg3_pct_override": {
        "label": "3PT%",
        "help": "3-point shooting percentage. Affects distribution shape.",
        "min": 0.0, "max": 0.65, "step": 0.01, "fmt": "{:.3f}",
        "positions": ["G", "F", "C", "UNK"],
    },
}

# ---------------------------------------------------------------------------
# DB bootstrap
# ---------------------------------------------------------------------------
def _db_is_valid() -> bool:
    p = Path(DB_PATH)
    if not p.exists() or p.stat().st_size < 4096:
        return False
    try:
        import duckdb
        con = duckdb.connect(str(p), read_only=True)
        n = con.execute("SELECT COUNT(*) FROM clean.player_game_stats").fetchone()[0]
        con.close()
        return n > 0
    except Exception:
        return False


def _ensure_db() -> None:
    if _db_is_valid():
        return
    bootstrap = _ROOT / "scripts" / "bootstrap_db.py"
    if not bootstrap.exists():
        st.error("Database not found and bootstrap script is missing.")
        st.stop()
    with st.spinner("Building database from scratch — first load only, ~2-5 minutes…"):
        result = subprocess.run(
            [sys.executable, str(bootstrap)],
            capture_output=True, text=True,
        )
    if result.returncode != 0:
        st.error(f"Database build failed.\n\n```\n{result.stderr[-2000:]}\n```")
        st.stop()


_ensure_db()

# ---------------------------------------------------------------------------
# Engine cache
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading NBA projection engine…")
def get_engine() -> NBAProjectionEngine:
    engine = NBAProjectionEngine(db_path=DB_PATH)
    engine.load()
    engine.fit()
    return engine


# ---------------------------------------------------------------------------
# Autosave / autorestore
# ---------------------------------------------------------------------------
def _autosave() -> None:
    try:
        payload = {
            "selected_game":     st.session_state.get("selected_game"),
            "depth_charts":      st.session_state.get("depth_charts", {}),
            "hold_pct":          st.session_state.get("hold_pct", 0.05),
            "season_filter":     st.session_state.get("season_filter"),
            "version": 1,
        }
        _AUTOSAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _AUTOSAVE_PATH.write_text(json.dumps(payload, default=str), encoding="utf-8")
    except Exception:
        pass


def _autorestore() -> bool:
    if st.session_state.get("_autorestore_done"):
        return False
    st.session_state["_autorestore_done"] = True
    if not _AUTOSAVE_PATH.exists():
        return False
    try:
        payload = json.loads(_AUTOSAVE_PATH.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            return False
        restored = False
        if payload.get("selected_game"):
            st.session_state["selected_game"] = payload["selected_game"]
            restored = True
        if payload.get("depth_charts"):
            st.session_state["depth_charts"] = payload["depth_charts"]
            restored = True
        if "hold_pct" in payload:
            st.session_state["hold_pct"] = float(payload["hold_pct"])
        if payload.get("season_filter"):
            st.session_state["season_filter"] = payload["season_filter"]
        stale = [k for k in st.session_state
                 if k.startswith(("pr_num_", "tr_num_", "hold_num_"))]
        for k in stale:
            del st.session_state[k]
        if restored:
            st.session_state["_run_after_load"] = True
            st.session_state["last_result"] = None
        return restored
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session() -> None:
    defaults = {
        "selected_game":  None,
        "last_result":    None,
        "depth_charts":   {},   # {team_abbr: {player_id: {active, injury_rating, overrides}}}
        "hold_pct":       0.05,
        "season_filter":  "2025-26",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    _autorestore()


# ---------------------------------------------------------------------------
# Depth chart helpers
# ---------------------------------------------------------------------------
def get_depth_chart(team_abbr: str) -> Dict:
    if team_abbr not in st.session_state.depth_charts:
        st.session_state.depth_charts[team_abbr] = {}
    return st.session_state.depth_charts[team_abbr]


def set_player_override(team_abbr: str, player_id: int, key: str, value) -> None:
    dc = get_depth_chart(team_abbr)
    if player_id not in dc:
        dc[player_id] = {}
    dc[player_id][key] = value
    _autosave()


def set_player_rating(team_abbr: str, player_id: int, rating_key: str, value: float) -> None:
    dc = get_depth_chart(team_abbr)
    if player_id not in dc:
        dc[player_id] = {}
    if "rating_overrides" not in dc[player_id]:
        dc[player_id]["rating_overrides"] = {}
    dc[player_id]["rating_overrides"][rating_key] = value
    _autosave()


def build_player_overrides() -> Dict[int, Dict]:
    """Flatten depth_charts into {player_id: override_dict} for the engine."""
    merged: Dict[int, Dict] = {}
    for team_dc in st.session_state.depth_charts.values():
        for pid, settings in team_dc.items():
            entry: Dict = {}
            if "active" in settings:
                entry["active"] = settings["active"]
            if "injury_rating" in settings:
                entry["injury_rating"] = float(settings["injury_rating"])
            if "minutes_override" in settings:
                entry["minutes_override"] = float(settings["minutes_override"])
            for rk, rv in settings.get("rating_overrides", {}).items():
                entry[rk] = rv
            if entry:
                merged[int(pid)] = entry
    return merged


def run_projection(engine: NBAProjectionEngine, game: Dict) -> Optional[ProjectionResult]:
    home = str(game.get("home_team_abbr", game.get("home_team", "")))
    away = str(game.get("away_team_abbr", game.get("away_team", "")))
    if not home or not away:
        return None
    result = engine.project(
        home_team_abbr=home,
        away_team_abbr=away,
        game_date=str(game.get("game_date", "")),
        player_overrides=build_player_overrides() or None,
    )
    st.session_state.last_result = result
    st.session_state.selected_game = game
    _autosave()
    return result


def render_update_btn(engine: NBAProjectionEngine, key: str = "upd") -> bool:
    if st.button("🔄 Update Projection", key=key, type="secondary", width="stretch"):
        game = st.session_state.get("selected_game")
        if game:
            with st.spinner("Updating…"):
                run_projection(engine, game)
            st.rerun()
        return True
    return False


# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------
SHARED_CSS = """
<style>
  .main .block-container { padding-top: 1rem; max-width: 1800px; }
  .pll-card {
    background: rgba(30,41,59,.55);
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 8px;
    text-align: center;
  }
  .pll-card-label { font-size: .72rem; color: #64748b; text-transform: uppercase; letter-spacing:.06em; }
  .pll-card-value { font-size: 1.5rem; font-weight: 700; color: #f1f5f9; }
  .pll-card-sub   { font-size: .78rem; color: #94a3b8; }
  .note-text { font-size: .75rem; color: #64748b; }
  .odds-fav  { color: #f87171; }
  .odds-dog  { color: #34d399; }
  .odds-even { color: #94a3b8; }
</style>
"""

def card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="pll-card-sub">{sub}</div>' if sub else ""
    return (f'<div class="pll-card">'
            f'<div class="pll-card-label">{label}</div>'
            f'<div class="pll-card-value">{value}</div>'
            f'{sub_html}</div>')

def fmt_odds(odds_str: str) -> str:
    try:
        v = int(str(odds_str).replace("+", ""))
        cls = "odds-fav" if v < 0 else ("odds-dog" if v > 0 else "odds-even")
    except Exception:
        cls = "odds-even"
    return f'<span class="{cls}">{odds_str}</span>'

def pos_badge(pos: str) -> str:
    colors = {"G": "#1d428a", "F": "#007a33", "C": "#ce1141", "UNK": "#475569"}
    c = colors.get(str(pos).upper(), "#475569")
    return (f'<span style="background:{c};color:#fff;border-radius:3px;'
            f'padding:1px 5px;font-size:.72rem;font-weight:700;">{pos}</span>')
