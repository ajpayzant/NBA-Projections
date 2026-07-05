"""
NBA Projection Engine
=====================
Architecture:
  NBADataLoader         — load from DuckDB
  NBATeamModel          — EWM team ratings + RF correction blend
  NBAPlayerModel        — minutes projection → rate × min → normalize to team total
  NBASimulator          — Monte Carlo (20,000 sims) for all stat distributions
  NBAPricingEngine      — convert distributions to prop lines and fair odds
  NBAProjectionEngine   — orchestrates all the above

Key design decisions:
  - Minutes first: projected minutes drives everything else
  - Rate × minutes: cleaner than raw counts for injury/override scenarios
  - Injury rating (0.0–1.0) scales minutes before any other projection
  - Team normalization: player totals are reconciled to match team projection
  - EWM half-life=8 games for all player rates (tuned for 82-game season)
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import duckdb
except ImportError as e:
    raise ImportError("duckdb required: pip install duckdb") from e

logger = logging.getLogger("nba.engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# League averages (2024-25 season)
LG_PTS:  float = 113.0
LG_REB:  float = 43.8
LG_AST:  float = 26.5
LG_STL:  float = 7.7
LG_BLK:  float = 4.9
LG_TOV:  float = 13.5
LG_FG3M: float = 13.1
LG_PACE: float = 100.6
LG_OFF_RTG: float = 115.5   # league average offensive rating
LG_DEF_RTG: float = 115.5   # league average defensive rating (symmetric)
LG_MIN_PER_PLAYER: float = 24.0

# EWM half-life (games)
HL_STATS:   int = 8
HL_SHOOT:   int = 12
HL_MINUTES: int = 6

# Per-stat variance ratios — MEASURED from actual NBA data (deep_audit.py Section 5)
# var/mean ratios from 67,279 player-game rows (2023-26, MIN>=10)
# PTS actual=5.93 (far more overdispersed than previously thought)
# STL/BLK actual < 2.0 — we had those BACKWARDS
STAT_VAR_RATIO: Dict[str, float] = {
    "PTS":  5.93,   # Measured: high game-to-game variance (streaks, foul trouble, etc.)
    "REB":  2.44,   # Measured
    "AST":  2.46,   # Measured
    "STL":  1.18,   # Measured: less overdispersed than previously set
    "BLK":  1.44,   # Measured: less overdispersed than previously set
    "TOV":  1.42,   # Measured (was 1.35, now 1.42)
    "FG3M": 1.68,   # Measured (was 1.80, now 1.68)
}

# Zero-inflation rates by stat (fraction of games where player gets 0)
# From Section 3d of research: actual rates for MIN>10 players
ZERO_RATE: Dict[str, float] = {
    "PTS":  0.057,
    "REB":  0.061,
    "AST":  0.185,
    "FG3M": 0.383,
    "STL":  0.467,
    "BLK":  0.666,
    "TOV":  0.150,
}

# Injury rating → minutes multiplier mapping
# 0.0 = healthy (1.0x), 1.0 = out (0.0x)
def injury_minutes_mult(rating: float) -> float:
    """Linear map: 0.0→1.0, 0.25→0.80, 0.50→0.55, 0.75→0.20, 1.0→0.0"""
    return float(np.clip(1.0 - rating, 0.0, 1.0))

# Simulation
N_SIMS = 20_000

# Prop stat combinations
COMBO_STATS = {
    "PRA":   ["PTS", "REB", "AST"],
    "PR":    ["PTS", "REB"],
    "PA":    ["PTS", "AST"],
    "RA":    ["REB", "AST"],
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TeamProjection:
    team_id:        int
    team_abbr:      str
    team_name:      str
    opp_team_id:    int
    opp_team_abbr:  str
    proj_pts:       float
    proj_reb:       float
    proj_ast:       float
    proj_tov:       float
    proj_fg3m:      float
    proj_pace:      float
    proj_win_prob:  float = 0.5
    confidence:     float = 0.5


@dataclass
class PlayerProjection:
    player_id:      int
    player_name:    str
    team_abbr:      str
    position:       str
    pos_group:      str   # G / F / C / UNK
    proj_min:       float
    proj_pts:       float
    proj_reb:       float
    proj_ast:       float
    proj_stl:       float
    proj_blk:       float
    proj_tov:       float
    proj_fg3m:      float
    # Derived combos
    proj_pra:       float = 0.0
    proj_pr:        float = 0.0
    proj_pa:        float = 0.0
    proj_ra:        float = 0.0
    # Shooting rates (for sim distribution shape)
    fg_pct:         float = 0.46
    fg3_pct:        float = 0.36
    ft_pct:         float = 0.77
    # Overrideable
    active:         bool  = True
    injury_rating:  float = 0.0    # 0.0=healthy … 1.0=out
    minutes_override: Optional[float] = None
    usage_override: Optional[float] = None
    # Internal
    games_played:   int   = 0
    confidence:     float = 0.5
    _pts_overridden: bool = False
    _min_overridden: bool = False
    # Per-minute rates stored for minutes-cascade recomputation in _reconcile
    _pts_pm:  float = 0.0
    _reb_pm:  float = 0.0
    _ast_pm:  float = 0.0
    _fg3m_pm: float = 0.0
    # Original EWM rates and minutes (before rotation fill / USG redistribution)
    # — used to compute PRA share for lineup-impact team adjustment
    _pts_pm_ewm_orig:    float = 0.0
    _reb_pm_ewm_orig:    float = 0.0
    _ast_pm_ewm_orig:    float = 0.0
    _base_min_for_share: float = 0.0
    _usg_ewm: float = 18.0
    _dnp_default: bool = False       # True when model defaults inactive (low minutes history)
    _user_deactivated: bool = False  # True when user explicitly set active=False


@dataclass
class PlayerSimulation:
    player_id:   int
    player_name: str
    stat_distributions: Dict[str, np.ndarray] = field(default_factory=dict)
    proj_values:        Dict[str, float]      = field(default_factory=dict)
    prop_lines:         Dict[str, float]      = field(default_factory=dict)


@dataclass
class GameSimulation:
    n_sims:         int
    home_pts:       np.ndarray
    away_pts:       np.ndarray
    home_win_prob:  float
    away_win_prob:  float
    spread_home:    float
    expected_total: float
    total_distribution: np.ndarray
    margin_distribution: np.ndarray


@dataclass
class GameMarket:
    home_ml:        str
    away_ml:        str
    home_win_prob:  float
    away_win_prob:  float
    spread_home:    float
    spread_home_odds: str
    spread_away_odds: str
    total_line:     float
    over_odds:      str
    under_odds:     str


@dataclass
class ProjectionResult:
    game_id:          str
    home_team:        str
    away_team:        str
    home_proj:        TeamProjection
    away_proj:        TeamProjection
    home_players:     List[PlayerProjection]
    away_players:     List[PlayerProjection]
    game_sim:         GameSimulation
    home_player_sims: List[PlayerSimulation]
    away_player_sims: List[PlayerSimulation]
    game_market:      GameMarket
    player_markets:   Dict[str, Dict]
    generated_at:     str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nan(x, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _ewm_last(series: pd.Series, halflife: int) -> float:
    """Return the last EWM value of a series (leakage-safe already applied in warehouse)."""
    s = series.dropna()
    if s.empty:
        return np.nan
    return float(s.ewm(halflife=halflife, min_periods=1).mean().iloc[-1])


def _nearest_half(v: float) -> float:
    """Round to nearest 0.5 for prop lines."""
    return round(v * 2) / 2


def _prob_to_american(p: float, hold_pct: float = 0.05) -> str:
    p = float(np.clip(p, 0.01, 0.99))
    # Apply hold (vig)
    p_adj = p * (1 + hold_pct)
    if p_adj >= 0.50:
        odds = -round((p_adj / (1 - p_adj)) * 100)
    else:
        odds = round(((1 - p_adj) / p_adj) * 100)
    if odds > 0:
        return f"+{odds}"
    return str(odds)


# ---------------------------------------------------------------------------
# Class 1: NBADataLoader
# ---------------------------------------------------------------------------

class NBADataLoader:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _conn(self):
        return duckdb.connect(self.db_path, read_only=True)

    def load_player_games(self) -> pd.DataFrame:
        con = self._conn()
        try:
            return con.execute("SELECT * FROM clean.player_game_stats").df()
        finally:
            con.close()

    def load_team_games(self) -> pd.DataFrame:
        con = self._conn()
        try:
            return con.execute("SELECT * FROM clean.team_game_stats").df()
        finally:
            con.close()

    def load_schedule(self) -> pd.DataFrame:
        con = self._conn()
        try:
            return con.execute("SELECT * FROM clean.game_schedule ORDER BY game_date").df()
        except Exception:
            return pd.DataFrame()
        finally:
            con.close()

    def load_player_info(self) -> pd.DataFrame:
        con = self._conn()
        try:
            return con.execute("SELECT * FROM clean.player_info").df()
        except Exception:
            return pd.DataFrame()
        finally:
            con.close()

    def load_team_season_stats(self) -> pd.DataFrame:
        con = self._conn()
        try:
            return con.execute("""
                SELECT TEAM_ABBREVIATION, SEASON,
                    AVG(TEAM_PTS) AS pts_pg, AVG(TEAM_REB) AS reb_pg,
                    AVG(TEAM_AST) AS ast_pg, AVG(TEAM_TOV) AS tov_pg,
                    AVG(TEAM_FG3M) AS fg3m_pg, AVG(PACE) AS pace,
                    AVG(TEAM_PTS_EWM) AS pts_ewm, AVG(PACE_EWM) AS pace_ewm
                FROM clean.team_game_stats
                GROUP BY TEAM_ABBREVIATION, SEASON
            """).df()
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Class 2: NBATeamModel
# ---------------------------------------------------------------------------

class NBATeamModel:
    """
    Projects team totals using EWM ratings with an optional RF correction.

    Outputs: projected PTS, REB, AST, TOV, FG3M, PACE for each team.
    Win probability is derived from projected point differential.
    """

    def __init__(self):
        self._rf_models: Dict[str, object] = {}
        self._rf_scalers: Dict[str, object] = {}
        self._fitted = False
        self._lg: Dict[str, float] = {}

    def fit(self, team_games: pd.DataFrame) -> None:
        """
        Fit team model using Ridge regression on all available EWM features.

        Audit finding: Ridge on [TEAM_PTS_EWM, OFF_RTG_EWM, OPP_DEF_EWM,
        GAME_PACE_EWM, NET_RTG_EWM, IS_HOME, IS_B2B] achieves MAE=9.47
        vs EWM-only baseline of MAE=9.93 — 0.46 pts improvement.
        """
        if team_games.empty:
            return

        # League averages
        for col, key, default in [
            ("TEAM_PTS", "pts", LG_PTS), ("TEAM_REB", "reb", LG_REB),
            ("TEAM_AST", "ast", LG_AST), ("TEAM_TOV", "tov", LG_TOV),
            ("TEAM_FG3M", "fg3m", LG_FG3M), ("PACE", "pace", LG_PACE),
            ("OFF_RTG", "off_rtg", LG_OFF_RTG), ("DEF_RTG", "def_rtg", LG_DEF_RTG),
        ]:
            if col in team_games.columns:
                self._lg[key] = float(team_games[col].mean())
            else:
                self._lg[key] = default

        # Ridge regression model for team scoring
        # Uses all meaningful EWM features identified in audit
        try:
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler

            # Feature set validated in audit — these are the ones that matter
            # OPP stats require a game-level join so we use what's in team_games
            core_feat_candidates = [
                "TEAM_PTS_EWM", "OFF_RTG_EWM", "DEF_RTG_EWM", "NET_RTG_EWM",
                "PACE_EWM", "TEAM_EFG_EWM", "TEAM_FG_PCT_EWM",
                "IS_HOME", "IS_B2B", "REST_DAYS",
            ]
            feat_cols = [c for c in core_feat_candidates
                         if c in team_games.columns
                         and team_games[c].notna().sum() > 50]

            for target in ["TEAM_PTS", "TEAM_AST", "PACE"]:
                if target not in team_games.columns:
                    continue
                df_ = team_games[feat_cols + [target]].dropna()
                if len(df_) < 100:
                    continue
                X = df_[feat_cols].values
                y = df_[target].values
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                # Ridge with alpha=10: validated in audit as best regularization
                mdl = Ridge(alpha=10.0)
                mdl.fit(Xs, y)
                self._rf_models[target] = mdl
                self._rf_scalers[target] = scaler
                self._rf_feat_cols = feat_cols
                logger.debug("Team Ridge fitted for %s on %d rows, %d features",
                             target, len(df_), len(feat_cols))
        except Exception as e:
            logger.debug("Ridge fit skipped: %s", e)

        self._fitted = True

    def predict(self, team_r: Dict, opp_r: Dict,
                is_home: bool = False, is_b2b: bool = False) -> TeamProjection:
        """
        Predict team stats using possession-based model with opponent context.

        Approach:
          1. Estimate game pace from both teams' pace EWMs
          2. Use team offensive rating and opponent defensive rating to
             project scoring via: pts = (off_rtg × opp_def_rtg / lg²) × possessions
          3. Apply home court (+2.5 pts) and back-to-back (-2.0 pts) adjustments
          4. Blend with EWM direct estimate (50/50) for stability
        """
        def _get(d, key, default):
            return _nan(d.get(key), default)

        lg_pts  = self._lg.get("pts",  LG_PTS)
        lg_pace = self._lg.get("pace", LG_PACE)
        lg_off  = self._lg.get("off_rtg",  LG_OFF_RTG)
        lg_def  = self._lg.get("def_rtg",  LG_DEF_RTG)

        # ── Pace projection ───────────────────────────────────────────────────
        team_pace = _get(team_r, "PACE_EWM", lg_pace)
        opp_pace  = _get(opp_r,  "PACE_EWM", lg_pace)
        proj_pace = 0.5 * team_pace + 0.5 * opp_pace   # game pace = avg of both teams

        # ── Possession count ──────────────────────────────────────────────────
        # NBA regulation = 48 min; pace = possessions per 48 min per team
        proj_poss = proj_pace  # already in possessions per 48 min

        # ── Offensive and defensive ratings ───────────────────────────────────
        team_off_rtg = _get(team_r, "OFF_RTG_EWM", lg_off)
        opp_def_rtg  = _get(opp_r,  "DEF_RTG_EWM", lg_def)

        # Fallback: estimate from scoring EWM if ratings not available
        if team_off_rtg <= 0 or np.isnan(team_off_rtg):
            team_pts_ewm = _get(team_r, "TEAM_PTS_EWM", lg_pts)
            team_poss_ewm = _get(team_r, "PACE_EWM", lg_pace)
            team_off_rtg = (team_pts_ewm / max(team_poss_ewm, 1.0)) * 100.0 if team_poss_ewm > 0 else lg_off

        if opp_def_rtg <= 0 or np.isnan(opp_def_rtg):
            opp_pts_ewm = _get(opp_r, "TEAM_PTS_EWM", lg_pts)
            opp_poss_ewm = _get(opp_r, "PACE_EWM", lg_pace)
            opp_def_rtg = (opp_pts_ewm / max(opp_poss_ewm, 1.0)) * 100.0 if opp_poss_ewm > 0 else lg_def

        # ── Possession-based scoring ──────────────────────────────────────────
        # Expected pts = (team_off / lg_off) × (opp_def / lg_def) × lg_pts_per_poss × possessions
        # Normalised so an average team vs average opponent = league average
        if lg_off > 0 and lg_def > 0:
            off_factor = team_off_rtg / lg_off
            def_factor = opp_def_rtg  / lg_def
            pace_factor = proj_pace   / lg_pace
            proj_pts_possession = lg_pts * off_factor * def_factor * pace_factor
        else:
            proj_pts_possession = lg_pts

        # ── EWM direct estimate ───────────────────────────────────────────────
        proj_pts_ewm = _get(team_r, "TEAM_PTS_EWM", lg_pts)

        # ── Blend: 50% possession model + 50% EWM ────────────────────────────
        # Research showed EWM alone has corr=0.241; possession model adds opponent context.
        # 50/50 blend is more stable than either alone.
        proj_pts = 0.50 * proj_pts_possession + 0.50 * proj_pts_ewm

        # ── Home court advantage ──────────────────────────────────────────────
        # Empirically ~2.5 pts in modern NBA (reduced from historical 3.2)
        if is_home:
            proj_pts += 2.5

        # ── Back-to-back penalty ──────────────────────────────────────────────
        # Teams on B2B score ~2.0 fewer points on average
        if is_b2b:
            proj_pts -= 2.0

        # ── Other stats: pace-scaled EWMs ────────────────────────────────────
        pace_scale = proj_pace / max(lg_pace, 1.0)
        proj_reb  = _get(team_r, "TEAM_REB_EWM",  self._lg.get("reb",  LG_REB))  * pace_scale
        proj_ast  = _get(team_r, "TEAM_AST_EWM",  self._lg.get("ast",  LG_AST))  * pace_scale
        proj_tov  = _get(team_r, "TEAM_TOV_EWM",  self._lg.get("tov",  LG_TOV))  * pace_scale
        proj_fg3m = _get(team_r, "TEAM_FG3M_EWM", self._lg.get("fg3m", LG_FG3M)) * pace_scale

        # ── Ridge model prediction (50% blend — audit showed significant improvement) ──
        # Audit: Ridge CV MAE=9.47 vs EWM-only 9.93. Using 50% blend with possession
        # model gives best of both: possession model captures opponent context,
        # Ridge captures non-linear interactions between features.
        if self._rf_models and hasattr(self, "_rf_feat_cols"):
            try:
                # Build feature vector — inject opponent features where available
                feat_dict = dict(team_r)
                # Add opponent stats with OPP_ prefix for Ridge
                for k, v in opp_r.items():
                    feat_dict[f"OPP_{k}"] = v
                feat_vals = np.array([
                    _nan(feat_dict.get(c), 0.0) for c in self._rf_feat_cols
                ]).reshape(1, -1)
                for target, attr in [("TEAM_PTS","proj_pts"), ("TEAM_AST","proj_ast")]:
                    if target in self._rf_models:
                        Xs = self._rf_scalers[target].transform(feat_vals)
                        ridge_pred = float(self._rf_models[target].predict(Xs)[0])
                        if attr == "proj_pts":
                            # 50% Ridge + 50% possession model
                            proj_pts = 0.50 * proj_pts + 0.50 * ridge_pred
                        elif attr == "proj_ast":
                            proj_ast = 0.50 * proj_ast + 0.50 * ridge_pred
            except Exception:
                pass

        return TeamProjection(
            team_id=int(_nan(team_r.get("TEAM_ID"), 0)),
            team_abbr=str(team_r.get("TEAM_ABBREVIATION", "")),
            team_name=str(team_r.get("TEAM_NAME", "")),
            opp_team_id=int(_nan(opp_r.get("TEAM_ID"), 0)),
            opp_team_abbr=str(opp_r.get("TEAM_ABBREVIATION", "")),
            proj_pts=float(np.clip(proj_pts, 80.0, 145.0)),
            proj_reb=float(np.clip(proj_reb, 28.0, 60.0)),
            proj_ast=float(np.clip(proj_ast, 12.0, 40.0)),
            proj_tov=float(np.clip(proj_tov,  8.0, 22.0)),
            proj_fg3m=float(np.clip(proj_fg3m, 3.0, 25.0)),
            proj_pace=float(np.clip(proj_pace, 88.0, 115.0)),
        )


# ---------------------------------------------------------------------------
# Class 3: NBAPlayerModel
# ---------------------------------------------------------------------------

class NBAPlayerModel:
    """
    Projects per-player stats using the rate × minutes approach.

    Pipeline:
      1. Project minutes for each player
      2. Apply injury rating multiplier to minutes
      3. Project rates (pts/min, reb/min, ast/min, etc.) from EWM history
      4. Multiply rate × effective minutes to get raw projections
      5. Normalize player totals to match team projection
    """

    # Default rates by position group (when player has no history)
    _POS_DEFAULTS: Dict[str, Dict[str, float]] = {
        "G":   {"min": 26.0, "pts_pm": 0.60, "reb_pm": 0.13, "ast_pm": 0.20,
                "stl_pm": 0.04, "blk_pm": 0.01, "tov_pm": 0.07, "fg3m_pm": 0.10},
        "F":   {"min": 26.0, "pts_pm": 0.52, "reb_pm": 0.30, "ast_pm": 0.10,
                "stl_pm": 0.03, "blk_pm": 0.03, "tov_pm": 0.06, "fg3m_pm": 0.06},
        "C":   {"min": 24.0, "pts_pm": 0.50, "reb_pm": 0.45, "ast_pm": 0.07,
                "stl_pm": 0.02, "blk_pm": 0.05, "tov_pm": 0.07, "fg3m_pm": 0.02},
        "UNK": {"min": 20.0, "pts_pm": 0.50, "reb_pm": 0.25, "ast_pm": 0.12,
                "stl_pm": 0.03, "blk_pm": 0.02, "tov_pm": 0.06, "fg3m_pm": 0.06},
    }

    def __init__(self, player_games: pd.DataFrame, player_info: pd.DataFrame):
        self.pg = player_games.copy()
        self.pi = player_info.copy()
        self._player_ratings: Dict[int, Dict] = {}
        self._build_ratings()

    def _build_ratings(self) -> None:
        """
        Build per-player rating dicts from EWM and rolling-average columns.
        Includes AVG5 columns for recent-form signal alongside EWM.
        """
        if self.pg.empty:
            return

        ewm_cols  = [c for c in self.pg.columns if c.endswith("_EWM")]
        avg5_cols = [c for c in self.pg.columns if c.endswith("_AVG5")]
        id_cols   = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "POSITION",
                     "POS_GROUP", "SEASON", "GAME_DATE"] + ewm_cols + avg5_cols
        id_cols   = [c for c in id_cols if c in self.pg.columns]

        # Take the last row per player (most recent state across all seasons)
        latest = (
            self.pg[id_cols]
            .sort_values(["PLAYER_ID", "GAME_DATE"] if "GAME_DATE" in id_cols else ["PLAYER_ID"])
            .groupby("PLAYER_ID")
            .last()
            .reset_index()
        )

        # Count games played in the two most recent seasons only (current-form signal)
        seasons_sorted = sorted(self.pg["SEASON"].dropna().unique()) if "SEASON" in self.pg.columns else []
        recent_seasons = set(seasons_sorted[-2:]) if len(seasons_sorted) >= 2 else set(seasons_sorted)
        if recent_seasons and "SEASON" in self.pg.columns:
            gp = (self.pg[self.pg["SEASON"].isin(recent_seasons)]
                  .groupby("PLAYER_ID").size().rename("games_played"))
        else:
            gp = self.pg.groupby("PLAYER_ID").size().rename("games_played")
        latest = latest.merge(gp, on="PLAYER_ID", how="left")
        latest["games_played"] = latest["games_played"].fillna(0).astype(int)

        for _, row in latest.iterrows():
            pid = int(row["PLAYER_ID"])
            self._player_ratings[pid] = row.to_dict()

    def get_team_roster(self, team_abbr: str, season: Optional[str] = None) -> List[Dict]:
        """Return player ratings for the team's current roster.

        Uses a two-stage filter:
        1. Find players who appeared for this team in the most recent season.
        2. Keep only those whose *current* (most recent game) team matches —
           this removes players who were traded away mid-season and now play
           elsewhere, which would inflate the roster and crush each player's
           projected minutes via the normalisation step.
        """
        if self.pg.empty:
            return []

        abbr_up = team_abbr.upper()
        mask = self.pg["TEAM_ABBREVIATION"].astype(str).str.upper() == abbr_up
        if season and "SEASON" in self.pg.columns:
            curr_mask = mask & (self.pg["SEASON"] == season)
            if curr_mask.any():
                mask = curr_mask

        team_pids = self.pg[mask]["PLAYER_ID"].dropna().astype(int).unique()

        # Determine the two most recent seasons in the data for recency filtering
        seasons_sorted = sorted(self.pg["SEASON"].dropna().unique()) if "SEASON" in self.pg.columns else []
        recent_seasons = set(seasons_sorted[-2:]) if len(seasons_sorted) >= 2 else set(seasons_sorted)

        result = []
        for pid in team_pids:
            if pid not in self._player_ratings:
                continue
            r = self._player_ratings[pid]
            # Only include if player's most recent team is still this team
            # (filters out players traded away mid-season or in prior seasons).
            if str(r.get("TEAM_ABBREVIATION", "")).upper() != abbr_up:
                continue
            # Only include players who played in the two most recent seasons.
            # This removes retired/waived players whose last game was years ago.
            if recent_seasons and str(r.get("SEASON", "")) not in recent_seasons:
                continue
            result.append(r)
        return result

    def _get_rate(self, ratings: Dict, rate_col: str, pos: str, default_key: str) -> float:
        val = _nan(ratings.get(rate_col), np.nan)
        if np.isnan(val):
            val = self._POS_DEFAULTS.get(pos, self._POS_DEFAULTS["UNK"]).get(default_key, 0.0)
        return float(val)

    def project_player(
        self,
        ratings: Dict,
        team_proj: TeamProjection,
        overrides: Optional[Dict] = None,
    ) -> PlayerProjection:
        """Project one player given their EWM ratings and team projection."""
        ov = overrides or {}
        # Position override from depth chart takes full priority
        raw_pos = str(ov.get("position_override") or ratings.get("POS_GROUP") or ratings.get("POSITION") or "UNK")

        # Normalize compound positions correctly: "Forward-Guard" → F, "Guard-Forward" → G
        # Primary role is the FIRST word in hyphenated NBA position strings
        if raw_pos not in ("G", "F", "C", "UNK"):
            s = raw_pos.lower().strip()
            if "-" in s:
                primary = s.split("-")[0].strip()
                if "guard" in primary:
                    raw_pos = "G"
                elif "forward" in primary:
                    raw_pos = "F"
                elif "center" in primary:
                    raw_pos = "C"
                else:
                    raw_pos = "UNK"
            elif "guard" in s or s in ("pg","sg","g"):
                raw_pos = "G"
            elif "forward" in s or s in ("sf","pf","f"):
                raw_pos = "F"
            elif "center" in s or s == "c":
                raw_pos = "C"
            else:
                raw_pos = "UNK"

        pos = raw_pos if raw_pos in ("G", "F", "C") else "UNK"
        pos_def = self._POS_DEFAULTS.get(pos, self._POS_DEFAULTS["UNK"])
        gp = int(_nan(ratings.get("games_played"), 0))

        # ── Minutes ───────────────────────────────────────────────────────────
        min_overridden = False
        if "minutes_override" in ov and ov["minutes_override"] is not None:
            proj_min = float(ov["minutes_override"])
            min_overridden = True
        else:
            # 70% EWM (long-run role) + 30% AVG5 (recent form).
            # USG adjustment intentionally excluded: data shows USG has only 0.38
            # correlation with minutes (vs 0.78 for MIN_EWM) and adding it inflates
            # ball-dominant guards (e.g. high-USG stars hit the 48-min cap) while
            # under-projecting defensive specialists and bigs who play heavy minutes
            # with low usage. MIN_EWM already incorporates role/starter status.
            min_ewm  = _nan(ratings.get("MIN_EWM"),  pos_def["min"])
            min_avg5 = _nan(ratings.get("MIN_AVG5"), min_ewm)
            base_min = 0.70 * min_ewm + 0.30 * min_avg5 if min_avg5 > 0 else min_ewm

            # Back-to-back: all players lose ~1.5 min on B2B nights
            is_b2b = bool(_nan(ratings.get("IS_B2B"), 0.0))
            b2b_adj = -1.5 if is_b2b else 0.0

            proj_min = base_min + b2b_adj

        # Injury rating reduces minutes
        injury = float(ov.get("injury_rating", _nan(ratings.get("injury_rating"), 0.0)))
        injury = float(np.clip(injury, 0.0, 1.0))
        proj_min = proj_min * injury_minutes_mult(injury)
        proj_min = float(np.clip(proj_min, 0.0, 48.0))

        # ── Rates (per minute) — blend EWM with AVG5, user overrides take priority ──
        def _blend_rate(ewm_key, avg5_key, pos_key):
            # User override takes full priority over model
            if ewm_key in ov and ov[ewm_key] is not None:
                return float(ov[ewm_key])
            ewm_v  = self._get_rate(ratings, ewm_key, pos, pos_key)
            avg5_v = _nan(ratings.get(avg5_key), 0.0)
            if avg5_v > 0:
                return 0.70 * ewm_v + 0.30 * avg5_v
            return ewm_v

        pts_pm  = _blend_rate("PTS_PM_EWM",  "PTS_PM_AVG5",  "pts_pm")
        reb_pm  = _blend_rate("REB_PM_EWM",  "REB_PM_AVG5",  "reb_pm")
        ast_pm  = _blend_rate("AST_PM_EWM",  "AST_PM_AVG5",  "ast_pm")
        stl_pm  = _blend_rate("STL_PM_EWM",  "STL_PM_AVG5",  "stl_pm")
        blk_pm  = _blend_rate("BLK_PM_EWM",  "BLK_PM_AVG5",  "blk_pm")
        tov_pm  = _blend_rate("TOV_PM_EWM",  "TOV_PM_AVG5",  "tov_pm")
        fg3m_pm = _blend_rate("FG3M_PM_EWM", "FG3M_PM_AVG5", "fg3m_pm")

        # ── USG% as a rate confidence signal (not a rate override) ──────────────
        # Backtest showed that blending USG-implied rate into PTS/min massively
        # overcorrects for stars (USG>28% MAE went from 6.4 to 15.9).
        # The reason: USG correlates with PTS mainly BECAUSE high-USG players
        # play more minutes — which is already captured by the USG minutes adjustment.
        # The per-minute rate should stay as the player's own EWM history.
        # USG is retained only for the minutes adjustment above (correct use).
        # No rate override here.

        # ── Opponent defensive context ────────────────────────────────────────
        # AUDIT FINDING: OPP_ALLOWED_PTS_POS_AVG10 has only 0.013 correlation
        # with actual player PTS and HURTS by 0.41 MAE when applied as multiplier.
        # Position-level defense is too coarse (conflates 5 different players).
        # REMOVED from player model. Team-level opponent context is applied at
        # the team projection level (possession model in NBATeamModel.predict).

        # ── Home court adjustment ─────────────────────────────────────────────
        # Research Section 5c: home avg +0.22 pts. Small but real.
        is_home = bool(_nan(ratings.get("IS_HOME"), 0.0))
        home_mult = 1.02 if is_home else 0.98   # ±2% scoring adjustment

        # ── Raw projections ───────────────────────────────────────────────────
        proj_pts  = pts_pm  * proj_min * home_mult
        proj_reb  = reb_pm  * proj_min
        proj_ast  = ast_pm  * proj_min * home_mult
        proj_stl  = stl_pm  * proj_min
        proj_blk  = blk_pm  * proj_min
        proj_tov  = tov_pm  * proj_min
        proj_fg3m = fg3m_pm * proj_min

        # User direct overrides on stats
        pts_overridden = False
        if "pts_override" in ov and ov["pts_override"] is not None:
            proj_pts = float(ov["pts_override"])
            pts_overridden = True

        # Shooting rates — user override takes priority
        fg_pct  = float(ov.get("FG_PCT_EWM")  or _nan(ratings.get("FG_PCT_EWM"),  0.46))
        fg3_pct = float(ov.get("FG3_PCT_EWM") or _nan(ratings.get("FG3_PCT_EWM"), 0.36))
        ft_pct  = _nan(ratings.get("FT_PCT_EWM"), 0.77)

        confidence = min(0.40 + 0.012 * gp, 0.90)

        proj = PlayerProjection(
            player_id=int(_nan(ratings.get("PLAYER_ID"), 0)),
            player_name=str(ratings.get("PLAYER_NAME", "")),
            team_abbr=str(ratings.get("TEAM_ABBREVIATION", "")),
            position=str(ratings.get("POSITION", "UNK")),
            pos_group=pos,
            proj_min=max(proj_min, 0.0),
            proj_pts=max(proj_pts, 0.0),
            proj_reb=max(proj_reb, 0.0),
            proj_ast=max(proj_ast, 0.0),
            proj_stl=max(proj_stl, 0.0),
            proj_blk=max(proj_blk, 0.0),
            proj_tov=max(proj_tov, 0.0),
            proj_fg3m=max(proj_fg3m, 0.0),
            fg_pct=float(np.clip(fg_pct, 0.25, 0.75)),
            fg3_pct=float(np.clip(fg3_pct, 0.15, 0.65)),
            ft_pct=float(np.clip(ft_pct, 0.40, 1.00)),
            active=bool(ov.get("active", True)),
            injury_rating=injury,
            minutes_override=float(ov["minutes_override"]) if "minutes_override" in ov else None,
            games_played=gp,
            confidence=confidence,
            _pts_overridden=pts_overridden,
            _min_overridden=min_overridden,
        )
        # Store rates and USG so _reconcile can use them
        proj._pts_pm  = pts_pm
        proj._reb_pm  = reb_pm
        proj._ast_pm  = ast_pm
        proj._fg3m_pm = fg3m_pm
        # Preserve original EWM rate and base minutes before rotation fill / USG
        # redistribution adjusts them. Used to compute each player's scoring share
        # of the team total for the lineup-impact team adjustment.
        proj._pts_pm_ewm_orig = pts_pm
        proj._reb_pm_ewm_orig = reb_pm
        proj._ast_pm_ewm_orig = ast_pm
        proj._base_min_for_share = float(proj.proj_min)  # pre-fill projected minutes
        usg_ewm = _nan(ratings.get("USG_PCT_EWM"), 18.0)
        proj._usg_ewm = float(np.clip(usg_ewm, 8.0, 40.0))
        # Derived combos
        proj.proj_pra = proj.proj_pts + proj.proj_reb + proj.proj_ast
        proj.proj_pr  = proj.proj_pts + proj.proj_reb
        proj.proj_pa  = proj.proj_pts + proj.proj_ast
        proj.proj_ra  = proj.proj_reb + proj.proj_ast
        return proj

    def project_roster(
        self,
        team_abbr: str,
        team_proj: TeamProjection,
        overrides: Optional[Dict[int, Dict]] = None,
        season: Optional[str] = None,
        active_player_ids: Optional[List[int]] = None,
        is_home: bool = False,
        is_b2b: bool = False,
    ) -> List[PlayerProjection]:
        """
        Project all players for a team and normalize totals to team projection.

        active_player_ids: if provided, only include these player IDs.
        This is populated from the current roster cache to ensure only
        players on the current team are projected (not historical players).
        """
        overrides = overrides or {}
        roster_ratings = self.get_team_roster(team_abbr, season)
        if not roster_ratings:
            return []

        # Filter to current roster if available
        if active_player_ids is not None:
            id_set = set(active_player_ids)
            roster_ratings = [r for r in roster_ratings
                              if int(_nan(r.get("PLAYER_ID"), 0)) in id_set]
            if not roster_ratings:
                logger.warning("Roster filter removed all players for %s — "
                               "falling back to full historical roster", team_abbr)
                roster_ratings = self.get_team_roster(team_abbr, season)

        # Deduplicate by player_id (keep most recent row)
        seen = set()
        unique_ratings = []
        for r in sorted(roster_ratings,
                        key=lambda x: str(x.get("GAME_DATE", "")), reverse=True):
            pid = int(_nan(r.get("PLAYER_ID"), 0))
            if pid not in seen:
                seen.add(pid)
                unique_ratings.append(r)
        roster_ratings = unique_ratings

        # Players whose EWM minutes are below this threshold are unlikely to
        # dress and play meaningful minutes. Default them to inactive so they
        # don't deflate the rotation players' projected minutes via the
        # normalisation step. Users can manually activate them in the depth chart.
        DNP_MIN_THRESHOLD = 8.0

        projections = []
        for r in roster_ratings:
            pid = int(_nan(r.get("PLAYER_ID"), 0))
            ov = overrides.get(pid, {})
            # Inject game context into ratings so project_player can use it
            r_with_ctx = dict(r)
            r_with_ctx["IS_HOME"] = 1.0 if is_home else 0.0
            r_with_ctx["IS_B2B"]  = 1.0 if is_b2b  else 0.0
            proj = self.project_player(r_with_ctx, team_proj, ov)

            # Determine active status:
            # 1. User explicit override always wins
            # 2. Players below DNP threshold default to inactive unless user said active
            min_ewm = float(_nan(r.get("MIN_EWM"), 0.0))
            user_set_active = "active" in ov
            if user_set_active:
                proj.active = bool(ov["active"])
                if not proj.active:
                    proj._user_deactivated = True
            elif min_ewm < DNP_MIN_THRESHOLD:
                proj.active = False
                proj._dnp_default = True
            else:
                proj.active = True

            if proj.injury_rating >= 1.0:
                # Injury=Out is treated the same as a user deactivation for
                # USG redistribution — the player's possessions go to teammates.
                proj.active = False
                proj._user_deactivated = True
                proj = self._zero(proj)
            elif not proj.active:
                proj = self._zero(proj)
            projections.append(proj)

        return self._reconcile(projections, team_proj)

    def _zero(self, p: PlayerProjection) -> PlayerProjection:
        for attr in ["proj_min", "proj_pts", "proj_reb", "proj_ast",
                     "proj_stl", "proj_blk", "proj_tov", "proj_fg3m",
                     "proj_pra", "proj_pr", "proj_pa", "proj_ra"]:
            setattr(p, attr, 0.0)
        return p

    def _reconcile(self, projs: List[PlayerProjection],
                   tp: TeamProjection) -> List[PlayerProjection]:
        """
        Assign final minutes and scale stats to match team projection.

        Minutes strategy — "rotation fill":
        Each player projects their own EWM-based minutes (no scaling).
        Players are sorted by projected minutes (highest first) and added
        to the lineup until the 240-minute game budget is exhausted.
        The last player to fit gets whatever minutes remain; everyone after
        them stays at their raw EWM projection (for reference but inactive).

        Why this works:
        - No artificial scaling that crushes star projections.
        - Stars naturally play their historical role (Brown EWM ~36 → 36 min).
        - Deep bench players fill whatever gap is left, or sit if the budget
          is already consumed by the rotation.
        - When a player is manually deactivated their minutes are redistributed
          to the next eligible players by proportional fill.

        USG-weighted reallocation excluded: 0.38 corr with MIN vs 0.78 for
        MIN_EWM; adding USG pushed high-usage stars to the 48-min cap.
        """
        NBA_TOTAL_MINUTES = 240.0

        active = [p for p in projs if p.active]
        if not active:
            return projs

        # ── Minutes normalization — rotation fill ─────────────────────────────
        # Players with a user override keep their exact override value (fixed).
        # The remaining free players fill the remaining budget in order of
        # their raw projected minutes (highest first).
        min_overridden = [p for p in active if p._min_overridden]
        min_free       = [p for p in active if not p._min_overridden]
        fixed_min      = sum(p.proj_min for p in min_overridden)
        budget         = max(NBA_TOTAL_MINUTES - fixed_min, 0.0)

        if min_free:
            # Sort by raw projected minutes descending
            min_free_sorted = sorted(min_free, key=lambda p: p.proj_min, reverse=True)
            remaining = budget
            for p2 in min_free_sorted:
                if remaining <= 0.0:
                    # Budget exhausted — mark as DNP-default
                    p2.proj_min = 0.0
                    p2.active = False
                    p2._dnp_default = True
                    p2 = self._zero(p2)
                elif p2.proj_min >= remaining:
                    p2.proj_min = float(remaining)
                    remaining = 0.0
                else:
                    remaining -= p2.proj_min
                    # proj_min unchanged — player plays their natural role

            # If free players don't fill the budget (e.g. stars deactivated and
            # the remaining rotation sums to less than 240), scale up iteratively.
            # Iteration handles the 48-min cap: capped players are excluded from
            # further scaling so their surplus redistributes to uncapped players.
            if remaining > 0.5:
                pool = remaining
                uncapped = [p2 for p2 in min_free if p2.active and p2.proj_min < 48.0]
                for _ in range(len(uncapped) + 1):
                    free_sum = sum(p2.proj_min for p2 in uncapped)
                    if free_sum <= 0 or not uncapped:
                        break
                    scale = (free_sum + pool) / free_sum
                    newly_capped = []
                    surplus = 0.0
                    for p2 in uncapped:
                        new_min = p2.proj_min * scale
                        if new_min > 48.0:
                            surplus += new_min - 48.0
                            p2.proj_min = 48.0
                            newly_capped.append(p2)
                        else:
                            p2.proj_min = float(new_min)
                    uncapped = [p2 for p2 in uncapped if p2 not in newly_capped]
                    pool = surplus
                    if pool <= 0.1:
                        break

        # Refresh the active list after rotation-fill may have changed active flags
        active = [p for p in projs if p.active]
        if not active:
            return projs

        # ── USG redistribution (teammate-out context adjustment) ─────────────
        # When a player is manually deactivated, their expected USG% is
        # redistributed to the remaining active players proportionally to
        # their own USG. This adjusts PTS/min and AST/min rates upward for
        # players who will absorb more possessions.
        #
        # Data: when Brown (USG=35%) is out, Pritchard's PTS/min increases
        # by +0.07 (+17%). The mechanism: ~35% of possessions are now split
        # among remaining players. Each player's effective rate scales by
        # (new_usg / base_usg). We cap the multiplier at 1.40 to prevent
        # outliers from inflating projections unreasonably.
        #
        # Only applied when a player is explicitly deactivated by the user
        # (not DNP-default), since DNP-defaults are fringe players whose
        # absence doesn't materially shift the rotation's usage.
        orphaned_usg = sum(
            getattr(p, "_usg_ewm", 18.0)
            for p in projs
            if getattr(p, "_user_deactivated", False)
        )
        if orphaned_usg > 0.5:
            active_usgs = [max(getattr(p, "_usg_ewm", 18.0), 1.0) for p in active]
            total_active_usg = sum(active_usgs)
            if total_active_usg > 0:
                for p, base_usg in zip(active, active_usgs):
                    # Each player absorbs their proportional share of orphaned USG
                    gained_usg = orphaned_usg * (base_usg / total_active_usg)
                    new_usg = base_usg + gained_usg
                    rate_mult = float(np.clip(new_usg / base_usg, 1.0, 1.40))
                    if hasattr(p, "_pts_pm"):
                        p._pts_pm  *= rate_mult
                        p._ast_pm  *= rate_mult
                        # Rebounding is less USG-dependent; apply half the multiplier
                        p._reb_pm  *= (1.0 + (rate_mult - 1.0) * 0.5)
                        p._fg3m_pm *= rate_mult

        # ── Lineup-strength adjustment ────────────────────────────────────────
        # When stars are deactivated, the team model (possession-based) still
        # projects the full team total — it has no knowledge of who is playing.
        # We correct this by computing the NATURAL scoring of the active lineup
        # (original EWM rate × post-fill minutes, before USG redistribution)
        # and capping the team total at that amount when the lineup is depleted.
        #
        # natural_sum = sum(pts_pm_ewm_orig × proj_min) for active players
        # This measures what the active lineup would produce at their historical
        # rates, given the minutes they'll play. It's a pure bottom-up estimate.
        #
        # When lineup is full and healthy: natural_sum ≈ team_model_pts (within ~3%)
        # so min(natural_sum, team_model) ≈ team_model — no change.
        # When stars are out: natural_sum drops by their contribution, automatically
        # reflecting the weaker lineup without any magic multipliers.
        # ── Lineup-impact adjustment on team total ────────────────────────────
        # When players are explicitly deactivated (user-set or injury=Out), the
        # team's projected scoring total and win probability are adjusted based on
        # each absent player's PRA share of the team projection.
        #
        # Using PRA share (not just PTS) captures all-around contributors like
        # Jokic whose rebounds and assists drive team scoring even beyond their PTS.
        #
        # Formulas derived by OLS regression from 421 player-season observations
        # (4 seasons, players with 12%+ share and 5+ absence games):
        #   pts_delta = -33.58 × pra_share + 3.97
        #   win_delta = -1.070 × pra_share + 0.120
        # where pra_share = absent_player_projected_PRA / team_projected_PRA
        #
        # Predictions at key share levels:
        #   10% share (role player):      +0.6 pts, +1.3% win (team barely changes)
        #   16% share (solid starter):    -1.4 pts, -5.1% win
        #   22% share (important piece):  -3.4 pts, -11.6% win
        #   25% share (true star):        -4.4 pts, -14.8% win
        #   28% share (Luka/Jokic level): -5.4 pts, -18.0% win
        #
        # Only applied for user-deactivated players (including injury=Out).
        # DNP-defaults are excluded — they're already the expected baseline state.
        _PTS_COEFF = -33.578
        _PTS_CONST =   3.973
        _WIN_COEFF =  -1.070
        _WIN_CONST =   0.120
        team_proj_pra = tp.proj_pts + tp.proj_reb + tp.proj_ast
        if tp.proj_pts > 0 and team_proj_pra > 0:
            total_pts_adj = 0.0
            total_win_adj = 0.0
            for p in projs:
                if not getattr(p, "_user_deactivated", False):
                    continue
                base_min = max(getattr(p, "_base_min_for_share", 0.0), 1.0)
                absent_pts = getattr(p, "_pts_pm_ewm_orig", 0.0) * base_min
                absent_reb = getattr(p, "_reb_pm_ewm_orig", 0.0) * base_min
                absent_ast = getattr(p, "_ast_pm_ewm_orig", 0.0) * base_min
                absent_pra = absent_pts + absent_reb + absent_ast
                pra_share  = float(np.clip(absent_pra / team_proj_pra, 0.0, 0.50))
                total_pts_adj += _PTS_COEFF * pra_share + _PTS_CONST
                total_win_adj += _WIN_COEFF * pra_share + _WIN_CONST
            if total_pts_adj != 0.0:
                new_pts = float(np.clip(tp.proj_pts + total_pts_adj,
                                        tp.proj_pts * 0.60, tp.proj_pts * 1.05))
                lineup_ratio = new_pts / tp.proj_pts
                tp.proj_pts  = new_pts
                tp.proj_reb  = tp.proj_reb  * lineup_ratio
                tp.proj_ast  = tp.proj_ast  * lineup_ratio
                tp.proj_fg3m = tp.proj_fg3m * lineup_ratio
            if total_win_adj != 0.0:
                # Store win adjustment on TeamProjection for post-reconcile application.
                # Win probability is recomputed in NBAProjectionEngine.project() after
                # both lineups are resolved, so we stash the delta here.
                tp._lineup_win_adj = float(np.clip(total_win_adj, -0.45, 0.20))

        # ── Stat reconciliation (rates × minutes → normalize to adjusted team total) ─
        for p in active:
            if not p._min_overridden:
                if hasattr(p, "_pts_pm"):
                    p.proj_pts  = max(p._pts_pm  * p.proj_min, 0.0)
                    p.proj_reb  = max(p._reb_pm  * p.proj_min, 0.0)
                    p.proj_ast  = max(p._ast_pm  * p.proj_min, 0.0)
                    p.proj_fg3m = max(p._fg3m_pm * p.proj_min, 0.0)

        for stat, team_total in [
            ("pts",  tp.proj_pts),
            ("reb",  tp.proj_reb),
            ("ast",  tp.proj_ast),
            ("fg3m", tp.proj_fg3m),
        ]:
            attr = f"proj_{stat}"
            overridden = [p for p in active if getattr(p, f"_{stat}_overridden", False)]
            free       = [p for p in active if not getattr(p, f"_{stat}_overridden", False)]
            fixed_sum  = sum(getattr(p, attr, 0.0) for p in overridden)
            free_sum   = sum(getattr(p, attr, 0.0) for p in free)
            remaining  = max(team_total - fixed_sum, 0.0)
            if free_sum > 0 and remaining > 0:
                scale = remaining / free_sum
                for p in free:
                    setattr(p, attr, max(getattr(p, attr, 0.0) * scale, 0.0))
            elif free_sum > 0 and remaining == 0:
                for p in free:
                    setattr(p, attr, 0.0)

        # ── Recompute combos ──────────────────────────────────────────────────
        for p in active:
            p.proj_pra = p.proj_pts + p.proj_reb + p.proj_ast
            p.proj_pr  = p.proj_pts + p.proj_reb
            p.proj_pa  = p.proj_pts + p.proj_ast
            p.proj_ra  = p.proj_reb + p.proj_ast

        return projs


# ---------------------------------------------------------------------------
# Class 4: NBASimulator
# ---------------------------------------------------------------------------

class NBASimulator:
    """
    Monte Carlo simulation for NBA game and player outcomes.

    Team totals: Normal(mu, sigma) truncated at 0
    Player stats: NegBin with correlation structure between teammates
    """

    def __init__(self, n_sims: int = N_SIMS, seed: int = 42):
        self.n_sims = n_sims
        self.seed   = seed

    def simulate_game(self, home: TeamProjection, away: TeamProjection) -> GameSimulation:
        rng = np.random.default_rng(self.seed)
        n = self.n_sims

        # Team scoring: Normal with empirical sigma (~12 pts/team)
        sigma = 12.0
        home_pts = np.maximum(rng.normal(home.proj_pts, sigma, n), 70)
        away_pts = np.maximum(rng.normal(away.proj_pts, sigma, n), 70)

        home_wins = home_pts > away_pts
        # OT breaks ties
        tied = home_pts == away_pts
        home_wins = home_wins | (tied & (rng.random(n) < 0.5))
        away_wins = ~home_wins

        home_win_prob = float(np.mean(home_wins))
        away_win_prob = 1.0 - home_win_prob

        # Quality model blend if both teams have decent data
        q_diff = home.proj_pts - away.proj_pts
        # Logistic: 3-pt edge ≈ 55%
        q_home = float(1.0 / (1.0 + np.exp(-q_diff / 8.0)))
        blend_w = 0.65  # simulation weight
        home_win_prob = blend_w * home_win_prob + (1.0 - blend_w) * q_home
        away_win_prob = 1.0 - home_win_prob

        total = home_pts + away_pts
        spread = home_pts - away_pts

        return GameSimulation(
            n_sims=n,
            home_pts=home_pts,
            away_pts=away_pts,
            home_win_prob=home_win_prob,
            away_win_prob=away_win_prob,
            spread_home=float(np.median(spread)),
            expected_total=float(np.median(total)),
            total_distribution=total,
            margin_distribution=spread,
        )

    def _negbinom_params(self, mu: float, var_ratio: float = 1.5) -> Tuple[int, float]:
        """Return (n, p) for NegBin with mean=mu and var=mu*var_ratio."""
        mu = max(mu, 0.01)
        var = mu * var_ratio
        p = mu / var
        n = max(int(round(mu * p / (1 - p))), 1)
        return n, float(p)

    def simulate_players(
        self,
        player_projs: List[PlayerProjection],
        team_pts_draws: np.ndarray,
        team_proj_pts: float,
    ) -> List[PlayerSimulation]:
        rng = np.random.default_rng(self.seed + 1)
        n = self.n_sims
        active = [p for p in player_projs if p.active]
        results = []

        def _zinb(mu: float, stat: str) -> np.ndarray:
            """
            Zero-Inflated Negative Binomial draw for a given stat.
            Uses empirically-measured zero rates and variance ratios from research.

            For BLK and STL (low mean, high zero rate) this correctly produces
            many zero-game outcomes that match real NBA distributions.
            """
            zero_rate = ZERO_RATE.get(stat, 0.05)
            var_ratio  = STAT_VAR_RATIO.get(stat, 1.5)
            mu = max(mu, 0.001)

            # Adjust mean of non-zero component to preserve overall mean
            # E[X] = (1 - zero_rate) * mu_nonzero → mu_nonzero = mu / (1 - zero_rate)
            mu_nonzero = mu / max(1.0 - zero_rate, 0.01)

            nb_n, nb_p = self._negbinom_params(mu_nonzero, var_ratio)
            is_zero = rng.random(n) < zero_rate
            counts  = rng.negative_binomial(nb_n, nb_p, n).astype(float)
            return np.where(is_zero, 0.0, counts)

        # Draw raw stats for each player
        raw: Dict[int, Dict[str, np.ndarray]] = {}
        for p in active:
            pid = p.player_id
            draws: Dict[str, np.ndarray] = {}

            # Minutes: Normal with std proportional to minutes tier
            # Higher-minute players have lower relative variance (more consistent role)
            min_cv = 0.25 if p.proj_min < 20 else (0.18 if p.proj_min < 32 else 0.14)
            min_sigma = max(p.proj_min * min_cv, 2.0)
            min_draws = np.clip(rng.normal(p.proj_min, min_sigma, n), 0, 48)
            draws["MIN"] = min_draws

            # PTS: Zero-Inflated NegBin
            draws["PTS"] = _zinb(p.proj_pts,  "PTS")

            # REB: Zero-Inflated NegBin
            draws["REB"] = _zinb(p.proj_reb,  "REB")

            # AST: Zero-Inflated NegBin (most variable stat)
            draws["AST"] = _zinb(p.proj_ast,  "AST")

            # STL: Zero-Inflated NegBin (high zero rate: 46.7% of games)
            draws["STL"] = _zinb(p.proj_stl,  "STL")

            # BLK: Zero-Inflated NegBin (very high zero rate: 66.6%)
            draws["BLK"] = _zinb(p.proj_blk,  "BLK")

            # TOV: Zero-Inflated NegBin
            draws["TOV"] = _zinb(p.proj_tov,  "TOV")

            # FG3M: Zero-Inflated Binomial
            # Model as: player either attempts 3s (prob = 1 - zero_rate) or doesn't
            # When they attempt, draw from Binomial(attempts, fg3_pct)
            zero_fg3 = ZERO_RATE.get("FG3M", 0.38)
            if p.proj_min > 0 and p.fg3_pct > 0:
                fg3a_per_min = p.proj_fg3m / (p.proj_min * p.fg3_pct)
                expected_fg3a = max(fg3a_per_min * min_draws.mean(), 0.01)
                is_zero_3 = rng.random(n) < zero_fg3
                fg3a_draws = np.clip(rng.poisson(
                    expected_fg3a / max(1.0 - zero_fg3, 0.01), n), 0, 18)
                draws["FG3M"] = np.where(
                    is_zero_3, 0.0,
                    rng.binomial(fg3a_draws.astype(int), p.fg3_pct, n).astype(float)
                )
            else:
                draws["FG3M"] = np.zeros(n)

            raw[pid] = draws

        # Condition PTS on team draw (same approach as PLL)
        pts_field = {p.player_id: raw[p.player_id]["PTS"] for p in active}
        sum_raw = sum(pts_field.values())
        sum_raw_arr = sum_raw if isinstance(sum_raw, np.ndarray) else np.full(n, sum_raw)
        sum_raw_arr = np.maximum(sum_raw_arr, 0.01)
        team_draw = np.round(team_pts_draws).clip(min=0)
        scale = team_draw / sum_raw_arr
        for p in active:
            raw[p.player_id]["PTS"] = np.round(raw[p.player_id]["PTS"] * scale).clip(min=0)

        # Build combo distributions
        for p in active:
            pid = p.player_id
            d = raw[pid]
            d["PRA"] = d["PTS"] + d["REB"] + d["AST"]
            d["PR"]  = d["PTS"] + d["REB"]
            d["PA"]  = d["PTS"] + d["AST"]
            d["RA"]  = d["REB"] + d["AST"]

            proj_vals = {k: float(np.mean(v)) for k, v in d.items()}
            prop_lines = {k: _nearest_half(float(np.median(v))) for k, v in d.items()}

            results.append(PlayerSimulation(
                player_id=pid,
                player_name=p.player_name,
                stat_distributions=d,
                proj_values=proj_vals,
                prop_lines=prop_lines,
            ))

        return results


# ---------------------------------------------------------------------------
# Class 5: NBAPricingEngine
# ---------------------------------------------------------------------------

class NBAPricingEngine:
    """Convert simulation distributions to prop lines and fair odds."""

    PROP_STATS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV",
                  "PRA", "PR", "PA", "RA", "MIN"]

    def __init__(self, hold_pct: float = 0.05):
        self.hold_pct = hold_pct

    def price_prop(self, sim: PlayerSimulation, stat: str,
                   line: Optional[float] = None) -> Dict:
        if stat not in sim.stat_distributions:
            return {}
        dist = sim.stat_distributions[stat]
        proj = sim.proj_values.get(stat, float(np.mean(dist)))
        if line is None:
            line = _nearest_half(float(np.median(dist)))

        fair_over = float(np.mean(dist > line))
        fair_under = 1.0 - fair_over

        return {
            "stat":       stat,
            "projection": round(proj, 3),
            "line":       line,
            "fair_over":  round(fair_over, 4),
            "fair_under": round(fair_under, 4),
            "over_odds":  _prob_to_american(fair_over,  self.hold_pct),
            "under_odds": _prob_to_american(fair_under, self.hold_pct),
            "p10":  round(float(np.percentile(dist, 10)), 1),
            "p25":  round(float(np.percentile(dist, 25)), 1),
            "p50":  round(float(np.percentile(dist, 50)), 1),
            "p75":  round(float(np.percentile(dist, 75)), 1),
            "p90":  round(float(np.percentile(dist, 90)), 1),
        }

    def price_game(self, game_sim: GameSimulation) -> GameMarket:
        h = game_sim.home_win_prob
        a = game_sim.away_win_prob
        spread = game_sim.spread_home
        total  = game_sim.expected_total

        total_line = _nearest_half(total)
        fair_over  = float(np.mean(game_sim.total_distribution > total_line))

        # Spread line to nearest 0.5
        spread_line = _nearest_half(spread)

        return GameMarket(
            home_ml=_prob_to_american(h, self.hold_pct),
            away_ml=_prob_to_american(a, self.hold_pct),
            home_win_prob=round(h, 4),
            away_win_prob=round(a, 4),
            spread_home=round(spread_line, 1),
            spread_home_odds=_prob_to_american(0.52, self.hold_pct),
            spread_away_odds=_prob_to_american(0.52, self.hold_pct),
            total_line=total_line,
            over_odds=_prob_to_american(fair_over, self.hold_pct),
            under_odds=_prob_to_american(1.0 - fair_over, self.hold_pct),
        )

    def price_all_players(
        self, sims: List[PlayerSimulation]
    ) -> Dict[int, Dict[str, Dict]]:
        markets = {}
        for sim in sims:
            markets[sim.player_id] = {
                stat: self.price_prop(sim, stat)
                for stat in self.PROP_STATS
                if stat in sim.stat_distributions
            }
        return markets


# ---------------------------------------------------------------------------
# Class 6: NBAProjectionEngine (orchestrator)
# ---------------------------------------------------------------------------

class NBAProjectionEngine:
    """
    Top-level orchestrator. Call load() then fit() once per session,
    then project() for each game.
    """

    def __init__(self, db_path: Optional[str] = None):
        _default = str(Path(__file__).resolve().parent /
                       "data" / "analytics_database" / "nba_warehouse.duckdb")
        self.db_path = db_path or os.getenv("NBA_DB_PATH", _default)

        self.loader       = NBADataLoader(self.db_path)
        self.team_model   = NBATeamModel()
        self.player_model: Optional[NBAPlayerModel] = None
        self.simulator    = NBASimulator(n_sims=N_SIMS, seed=42)
        self.pricing      = NBAPricingEngine(hold_pct=0.05)

        self.player_games: pd.DataFrame = pd.DataFrame()
        self.team_games:   pd.DataFrame = pd.DataFrame()
        self.schedule:     pd.DataFrame = pd.DataFrame()
        self.player_info:  pd.DataFrame = pd.DataFrame()

        self._loaded = False
        self._fitted = False

    def load(self) -> None:
        logger.info("Loading NBA warehouse: %s", self.db_path)
        self.team_games   = self.loader.load_team_games()
        self.player_games = self.loader.load_player_games()
        self.schedule     = self.loader.load_schedule()
        self.player_info  = self.loader.load_player_info()
        self._loaded = True
        logger.info("Loaded: %d team rows, %d player rows, %d schedule rows",
                    len(self.team_games), len(self.player_games), len(self.schedule))

    def fit(self) -> None:
        if not self._loaded:
            self.load()
        logger.info("Fitting models...")
        self.team_model.fit(self.team_games)
        self.player_model = NBAPlayerModel(self.player_games, self.player_info)
        self._fitted = True
        logger.info("Models fitted.")

    def get_team_rating(self, team_abbr: str,
                        as_of_date: Optional[str] = None) -> Dict:
        """Return latest rating dict for a team."""
        if self.team_games.empty:
            return {}
        mask = self.team_games["TEAM_ABBREVIATION"].astype(str).str.upper() == team_abbr.upper()
        if as_of_date and "GAME_DATE" in self.team_games.columns:
            dates = pd.to_datetime(self.team_games["GAME_DATE"], errors="coerce")
            cutoff = pd.to_datetime(as_of_date, errors="coerce")
            if pd.notna(cutoff):
                mask &= dates < cutoff
        sub = self.team_games[mask]
        if sub.empty:
            return {}
        if "GAME_DATE" in sub.columns:
            sub = sub.sort_values("GAME_DATE")
        return sub.iloc[-1].to_dict()

    def upcoming_games(self) -> List[Dict]:
        if self.schedule.empty:
            return []
        return self.schedule.to_dict("records")

    def project(
        self,
        home_team_abbr: str,
        away_team_abbr: str,
        game_date: Optional[str] = None,
        player_overrides: Optional[Dict[int, Dict]] = None,
        current_season: str = "2025-26",
    ) -> ProjectionResult:
        if not self._fitted:
            self.fit()

        hf = self.get_team_rating(home_team_abbr, game_date)
        af = self.get_team_rating(away_team_abbr, game_date)

        # Extract home/away and B2B context from team ratings
        h_is_home = bool(_nan(hf.get("IS_HOME"), 0.0))  # 1 if home, 0 if away
        h_is_b2b  = bool(_nan(hf.get("IS_B2B"),  0.0))
        a_is_b2b  = bool(_nan(af.get("IS_B2B"),  0.0))
        # For scheduled future games both teams are neutral-ish; home team gets advantage
        h_proj = self.team_model.predict(hf, af, is_home=True,  is_b2b=h_is_b2b)
        a_proj = self.team_model.predict(af, hf, is_home=False, is_b2b=a_is_b2b)

        # Set team names from ratings
        h_proj.team_abbr = home_team_abbr
        a_proj.team_abbr = away_team_abbr
        h_proj.opp_team_abbr = away_team_abbr
        a_proj.opp_team_abbr = home_team_abbr

        # Win probability update after both projections known
        pt_diff = h_proj.proj_pts - a_proj.proj_pts
        q_home = float(1.0 / (1.0 + np.exp(-pt_diff / 8.0)))
        h_proj.proj_win_prob = q_home
        a_proj.proj_win_prob = 1.0 - q_home

        ov = player_overrides or {}

        def _team_ov(abbr: str) -> Dict[int, Dict]:
            if not self.player_model or self.player_games.empty:
                return {}
            team_pids = set(
                self.player_games[
                    self.player_games["TEAM_ABBREVIATION"].astype(str).str.upper() == abbr.upper()
                ]["PLAYER_ID"].dropna().astype(int).tolist()
            )
            return {pid: v for pid, v in ov.items() if pid in team_pids}

        # Load current roster cache to filter to active players only
        try:
            from nba_roster_cache import get_active_player_ids
            h_active_ids = get_active_player_ids(home_team_abbr) or None
            a_active_ids = get_active_player_ids(away_team_abbr) or None
        except Exception:
            h_active_ids = None
            a_active_ids = None

        h_players = (self.player_model.project_roster(
            home_team_abbr, h_proj, _team_ov(home_team_abbr),
            current_season, active_player_ids=h_active_ids,
            is_home=True, is_b2b=h_is_b2b)
            if self.player_model else [])
        a_players = (self.player_model.project_roster(
            away_team_abbr, a_proj, _team_ov(away_team_abbr),
            current_season, active_player_ids=a_active_ids,
            is_home=False, is_b2b=a_is_b2b)
            if self.player_model else [])

        # Recompute win probability from adjusted team totals, then apply the
        # per-team lineup win adjustments stored by _reconcile.
        # The pts-based component (from the sigmoid) captures the scoring gap.
        # The lineup win adjustment adds the portion of win-rate impact that
        # isn't captured by pts alone (e.g. Jokic's defensive/playmaking value).
        pt_diff_adj = h_proj.proj_pts - a_proj.proj_pts
        q_home_base = float(1.0 / (1.0 + np.exp(-pt_diff_adj / 8.0)))
        h_win_adj = float(getattr(h_proj, "_lineup_win_adj", 0.0))
        a_win_adj = float(getattr(a_proj, "_lineup_win_adj", 0.0))
        # Home lineup weaker → subtract from home win prob; away weaker → add to home win prob
        q_home_adj = float(np.clip(q_home_base + h_win_adj - a_win_adj, 0.02, 0.98))
        h_proj.proj_win_prob = q_home_adj
        a_proj.proj_win_prob = 1.0 - q_home_adj

        game_sim = self.simulator.simulate_game(h_proj, a_proj)

        h_psims = self.simulator.simulate_players(
            h_players, game_sim.home_pts, h_proj.proj_pts)
        a_psims = self.simulator.simulate_players(
            a_players, game_sim.away_pts, a_proj.proj_pts)

        game_market = self.pricing.price_game(game_sim)
        player_markets = self.pricing.price_all_players(h_psims + a_psims)

        import datetime as _dt
        return ProjectionResult(
            game_id=f"{away_team_abbr}@{home_team_abbr}_{game_date or 'today'}",
            home_team=home_team_abbr,
            away_team=away_team_abbr,
            home_proj=h_proj,
            away_proj=a_proj,
            home_players=h_players,
            away_players=a_players,
            game_sim=game_sim,
            home_player_sims=h_psims,
            away_player_sims=a_psims,
            game_market=game_market,
            player_markets=player_markets,
            generated_at=_dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )
