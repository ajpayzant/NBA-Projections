"""Pregame features for every player-game (history and pending games alike).

Everything is an exponentially weighted (EW) running sum over EARLIER games (ew.pregame), so
a row never sees the game it describes.

League-relative: the NBA's scoring rose ~5% per possession and its 3-point volume ~15% from
2018-19 to 2025-26. Every rate is therefore measured against the league at the time the
game was played (an EW league environment, pregame), and turned back into a raw rate with
the environment on the night being projected. A rate of 1.10 means 10% above the league.

Who counts: a player is "available" for a team-game if he is in its box score and either
played or sat by coach's decision. Minutes features average over available games, zeros
included, so a player who often sits shows low expected minutes.

Per player:  EW minutes (short / long), minutes as a starter and off the bench, start rate,
             league-relative per-minute rates (shrunk toward his position group's level,
             which comes from 2018-19, before every test season), usage
Per team:    league-relative pace, offence and defence per possession, what it allows of
             each stat per possession, rest
Absences:    for each team-game, the usual minutes and usage of regulars not available
             tonight (a regular = available in any of the team's last 10 games)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import ew

HL_MIN = (4, 15)          # minutes: recent rotation, and the longer view
HL_ROLE = 6               # minutes as a starter / off the bench (tuned: 6 beat 10 and 20)
HL_RATE = 25              # per-minute production
HL_TEAM = 20              # team pace, defence, strength
HL_LEAGUE = 30            # league environment, in game days
GAP = 15                  # a summer counts as this many extra games
STATS = ["pts", "reb", "ast", "fg3m", "fga", "fta", "tov", "oreb", "dreb", "stl", "blk", "usg"]
K_RATE = 120              # minutes of position-group production in every player's prior
POS_GROUP = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}
PRIOR_SEASONS = (2018,)   # position-group levels come only from these seasons
ALLOW = ["pts", "reb", "ast", "fg3m"]


ONC_COLS = ["sec_on", "on_fga", "on_fta", "on_tov", "on_fgm", "on_pts", "on_reb", "on_opp_pts", "on_poss"]


def load_oncourt() -> pd.DataFrame:
    """Per player-game on-court totals from pbp.py (empty if not built)."""
    files = sorted(C.LAKE.glob("onc_[0-9][0-9][0-9][0-9].parquet"))
    return pd.concat([pd.read_parquet(f, columns=["game_id", "team", "player"] + ONC_COLS) for f in files],
                     ignore_index=True) if files else pd.DataFrame()


def available(pg: pd.DataFrame) -> pd.Series:
    """Played, or sat by coach's decision (not out injured)."""
    return ~pg.dnp | pg.dnp_reason.str.upper().str.contains("DECISION", na=False)


def prep(pg: pd.DataFrame, tg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pg = pg.copy()
    pg["date"] = pd.to_datetime(pg.date)
    pg["avail"] = available(pg)
    if "pending" not in pg:
        pg["pending"] = False
    pg = pg[pg.avail | pg.pending].copy()
    pg["min"] = pg["min"].fillna(0.0).where(~pg.dnp, 0.0)
    for s in STATS[:-1]:
        pg[s] = pg[s].fillna(0.0).where(~pg.dnp, 0.0)
    pg["usg"] = pg.fga + 0.44 * pg.fta + pg.tov
    pg["grp"] = pg.pos.map(POS_GROUP).fillna("F")
    onc = load_oncourt()
    if len(onc):
        pg = pg.merge(onc, on=["game_id", "team", "player"], how="left")
    for c in ONC_COLS:
        pg[c] = pg.get(c, pd.Series(0.0, index=pg.index)).fillna(0.0)
    pg["on_usg"] = pg.on_fga + 0.44 * pg.on_fta + pg.on_tov
    pg["on_tm_fgm"] = (pg.on_fgm - pg.fgm.fillna(0)).clip(lower=0)      # teammates' made shots while on
    pg = pg.sort_values(["date", "game_id"]).reset_index(drop=True)
    tg = tg.copy()
    tg["date"] = pd.to_datetime(tg.date)
    tg["reb"] = tg.oreb + tg.dreb       # ESPN's team total adds team rebounds in 2018-21
    if "pending" not in tg:
        tg["pending"] = False
    tg = tg.sort_values(["date", "game_id"]).reset_index(drop=True)
    return pg, tg


# ---------------------------------------------------------------- league environment

def _daily_env(day: pd.Series, num: pd.DataFrame, den: pd.Series) -> pd.DataFrame:
    """League rate of each column of num per unit of den, as of the day before each date (EW
    over game days)."""
    d = num.assign(den=den.to_numpy(), day=day.to_numpy()).groupby("day").sum().sort_index()
    d = d.assign(key="L", season=0)
    e = ew.pregame(d.reset_index(), "key", list(num.columns) + ["den"], HL_LEAGUE, 0)
    e.index = d.index
    rate = e[list(num.columns)].div(e.den.where(e.den > 0), axis=0)
    # the very first days have no history: use the first week's own rate
    return rate.bfill()


def league_env(pg: pd.DataFrame, tg: pd.DataFrame) -> dict[str, pd.DataFrame]:
    played = pg[(pg["min"] > 0) & ~pg.pending]
    per_min = _daily_env(played.date, played[STATS], played["min"])                     # per player-minute
    done = tg[~tg.pending]
    team_min = 48 + 5 * (done.periods.fillna(4) - 4)
    done = done.assign(miss=done.fga - done.fgm)
    per_poss = _daily_env(done.date, done[["pts", "reb", "ast", "fg3m", "miss", "fgm"]], done.poss)   # per possession
    pace = _daily_env(done.date, done[["poss"]], team_min).rename(columns={"poss": "pace"}) * 48
    return {"per_min": per_min, "per_poss": per_poss, "pace": pace}


def _on(env: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """The environment on each date (the last value at or before it; pending dates take the latest)."""
    idx = env.index.searchsorted(dates.to_numpy(), side="right") - 1
    return env.iloc[np.clip(idx, 0, len(env) - 1)].set_axis(dates.index)


# ---------------------------------------------------------------- players

def player_features(pg: pd.DataFrame, env: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=pg.index)
    L = _on(env["per_min"], pg.date)                     # league per-minute rates on each date
    tmp = pg[["player", "season", "min", "starter"]].copy()
    tmp["one"] = 1.0
    for name, hl in zip(("s", "l"), HL_MIN):
        e = ew.pregame(tmp, "player", ["min", "one"], hl, GAP)
        out[f"ew_min_{name}"] = e["min"] / e.one.where(e.one > 0)
    out["gp_w"] = ew.pregame(tmp, "player", ["one"], HL_MIN[1], GAP).one
    for role, flag in (("start", tmp.starter), ("bench", ~tmp.starter)):
        r = tmp.assign(m=tmp["min"].where(flag, 0.0), n=flag.astype(float))
        e = ew.pregame(r, "player", ["m", "n"], HL_ROLE, GAP)
        out[f"ew_min_{role}"] = e.m / e.n.where(e.n > 0.2)
    e = ew.pregame(tmp, "player", ["starter", "one"], HL_ROLE, GAP)
    out["start_rate"] = e.starter / e.one.where(e.one > 0)
    # league-relative production: each game's output divided by the league rate that night
    rel = pd.DataFrame({s: pg[s] / L[s].clip(1e-6) for s in STATS}, index=pg.index)
    rel["min"], rel["player"], rel["season"] = pg["min"], pg.player, pg.season
    e = ew.pregame(rel, "player", ["min"] + STATS, HL_RATE, GAP)
    base = pg[pg.season.isin(PRIOR_SEASONS) & (pg["min"] > 0)]
    lvl = base.groupby("grp")[STATS].sum().div(base.groupby("grp")["min"].sum(), axis=0) / \
        (base[STATS].sum() / base["min"].sum())
    for s in STATS:
        prior = pg.grp.map(lvl[s]).fillna(1.0)
        out[f"rel_{s}"] = (e[s] + K_RATE * prior) / (e["min"] + K_RATE)
        out[f"r_{s}"] = out[f"rel_{s}"] * L[s]            # per minute, in tonight's league
    out["hist_min"] = e["min"]
    # play-by-play shares: of team usage events, of rebounds, of teammates' baskets, while he was
    # on the court (pace- and minutes-free measures of his role); shrunk toward the group level
    sh = pg[["player", "season", "usg", "reb", "ast", "pts", "on_usg", "on_reb", "on_tm_fgm"]].copy()
    e = ew.pregame(sh, "player", ["usg", "reb", "ast", "pts", "on_usg", "on_reb", "on_tm_fgm"], HL_RATE, GAP)
    b = pg[pg.season.isin(PRIOR_SEASONS) & (pg.on_usg > 0)].groupby("grp")[["usg", "reb", "ast", "on_usg", "on_reb", "on_tm_fgm"]].sum()
    gl = {"usg": b.usg / b.on_usg, "reb": b.reb / b.on_reb, "ast": b.ast / b.on_tm_fgm}
    kden = {"usg": 150.0, "reb": 150.0, "ast": 100.0}       # prior weight, in opportunities
    for s, den in (("usg", "on_usg"), ("reb", "on_reb"), ("ast", "on_tm_fgm")):
        prior = pg.grp.map(gl[s]).fillna(gl[s].mean())
        out[f"sh_{s}"] = (e[s] + kden[s] * prior) / (e[den] + kden[s])
    out["pts_per_usg"] = (e.pts + 50 * 1.1) / (e.usg + 50)     # scoring efficiency per usage event
    return out


# ---------------------------------------------------------------- teams

def team_features(tg: pd.DataFrame, env: dict) -> pd.DataFrame:
    """League-relative pace, offence, defence and what each team allows, pregame."""
    t = tg[["game_id", "team", "opp", "season", "date", "pts", "opp_pts", "poss", "periods", "pending", "fga", "fgm"]].copy()
    o = tg[["game_id", "team", "reb", "ast", "fg3m", "poss", "fga", "fgm"]].rename(
        columns={"team": "opp", "reb": "o_reb", "ast": "o_ast", "fg3m": "o_fg3m", "poss": "o_poss", "fga": "o_fga", "fgm": "o_fgm"})
    t = t.merge(o, on=["game_id", "opp"], how="left")
    P, pace = _on(env["per_poss"], t.date), _on(env["pace"], t.date)["pace"]
    t["mins"] = 48 + 5 * (t.periods.fillna(4) - 4)
    t["x_poss"] = t.mins / 48 * pace                      # possessions an average team would have had
    t["x_pts"] = t.poss * P.pts                           # points an average offence would score on them
    t["x_opp_pts"] = t.o_poss * P.pts
    for s in ALLOW:                                        # what an average defence would have allowed
        t[f"x_o_{s}"] = t.o_poss * P[s]
    # own misses and made shots, and the opponent's misses (rebound / assist opportunities)
    t["miss"], t["fgm_"] = t.fga - t.fgm, t.fgm
    t["x_miss"], t["x_fgm"] = t.poss * P.miss, t.poss * P.fgm
    t["x_o_miss"] = t.o_poss * P.miss
    t["o_pts"] = t.opp_pts
    t["o_miss"] = t.o_fga - t.o_fgm
    cols = ["poss", "x_poss", "pts", "x_pts", "opp_pts", "x_opp_pts", "miss", "x_miss", "fgm_", "x_fgm", "o_miss", "x_o_miss"]         + [f"o_{s}" for s in ALLOW] + [f"x_o_{s}" for s in ALLOW]
    e = ew.pregame(t, "team", cols, HL_TEAM, GAP)
    out = pd.DataFrame(index=t.index)

    def shrunk(num: str, exp: str, games: float = 20) -> pd.Series:
        """Actual / league-expected, with `games` games of exactly league-average in the prior."""
        k = games * t[exp].mean()
        return (e[num] + k) / (e[exp] + k)
    out["pace_rel"] = shrunk("poss", "x_poss")
    out["ortg_rel"] = shrunk("pts", "x_pts")                # points per possession vs the league
    out["drtg_rel"] = shrunk("opp_pts", "x_opp_pts")        # allowed per possession (lower is better)
    for s in ALLOW:
        out[f"allow_{s}"] = shrunk(f"o_{s}", f"x_o_{s}")
    out["miss_rel"] = shrunk("miss", "x_miss")              # how often this team misses, vs the league
    out["fgm_rel"] = shrunk("fgm_", "x_fgm")                # how often it scores a basket
    out["allow_miss_rel"] = shrunk("o_miss", "x_o_miss")    # how often its opponents miss
    t["rest"] = t.groupby("team").date.diff().dt.days.clip(upper=4).fillna(4)
    out["rest"] = t.rest
    out["lg_ppp"], out["lg_pace"] = P.pts.to_numpy(), pace.to_numpy()
    out["lg_miss"], out["lg_fgm"] = P.miss.to_numpy(), P.fgm.to_numpy()
    return pd.concat([t[["game_id", "team", "opp"]], out], axis=1)


# ---------------------------------------------------------------- opponent defence by position

def position_defence(pg: pd.DataFrame, env: dict) -> pd.DataFrame:
    """What each defence allows to opposing guards / forwards / centres, per minute, vs the
    league (pregame EW). One row per (game, defending team, position group)."""
    L = _on(env["per_min"], pg.date)
    x = pg[(pg["min"] > 0) & ~pg.pending].copy()
    for s in ALLOW:
        x[f"x_{s}"] = x["min"] * L.loc[x.index, s]
    g = x.groupby(["game_id", "date", "season", "opp", "grp"])[ALLOW + [f"x_{s}" for s in ALLOW]].sum().reset_index()
    g["key"] = g.opp + "|" + g.grp
    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
    e = ew.pregame(g, "key", ALLOW + [f"x_{s}" for s in ALLOW], HL_TEAM, GAP)
    out = g[["game_id", "opp", "grp"]].copy()
    for s in ALLOW:
        k = 20 * g[f"x_{s}"].mean()
        out[f"pos_allow_{s}"] = (e[s] + k) / (e[f"x_{s}"] + k)
    return out


# ---------------------------------------------------------------- absences

def absences(pg: pd.DataFrame, pf: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: the usual minutes and usage-minutes of regulars who are not available."""
    d = pd.concat([pg[["game_id", "date", "team", "player"]], pf[["ew_min_l", "rel_usg"]]], axis=1)
    games = d.drop_duplicates(["game_id", "team"]).sort_values("date")
    # each player's teams over time, so a traded player stops counting as missing for his old team
    hist = {p_: (g.date.to_numpy(), g.team.to_numpy()) for p_, g in d.sort_values("date").groupby("player")}

    def team_before(p_, day):
        dates, teams = hist[p_]
        i = np.searchsorted(dates, day) - 1
        return teams[i] if i >= 0 else None
    rows = []
    for team, g in games.groupby("team", sort=False):
        ids = g.game_id.tolist()
        members = d[d.team == team].groupby("game_id").player.apply(set).to_dict()
        state = d[d.team == team].set_index(["game_id", "player"])[["ew_min_l", "rel_usg"]]
        last_seen: dict[int, tuple] = {}   # player -> (index of his last available game, ew_min_l, rel_usg)
        for k, gid in enumerate(ids):
            here = members.get(gid, set())
            miss_min = miss_usg = 0.0
            ids = []
            day = g.date.iloc[k]
            for p_, (k0, m, u) in last_seen.items():
                if p_ not in here and k - k0 <= 10 and np.isfinite(m) and team_before(p_, day) == team:
                    miss_min += m
                    miss_usg += m * (u if np.isfinite(u) else 1.0)
                    if m >= 12:                      # a rotation player: his with/without effects count
                        ids.append(int(p_))
            rows.append({"game_id": gid, "team": team, "miss_min": miss_min, "miss_usg": miss_usg, "missing_ids": ids})
            for p_ in here:
                m, u = state.loc[(gid, p_)].to_numpy() if (gid, p_) in state.index else (np.nan, np.nan)
                last_seen[p_] = (k, m, u)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- all together

def build(pg: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    pg, tg = prep(pg, tg)
    env = league_env(pg, tg)
    pf = player_features(pg, env)
    tf = team_features(tg, env)
    ab = absences(pg, pf)
    f = pd.concat([pg, pf], axis=1)
    f = f.merge(tf, on=["game_id", "team", "opp"], how="left")
    keep = ["pace_rel", "ortg_rel", "drtg_rel", "rest", "miss_rel", "allow_miss_rel"] + [f"allow_{s}" for s in ALLOW]
    o = tf[["game_id", "team"] + keep].rename(columns={"team": "opp", **{k: f"opp_{k}" for k in keep}})
    f = f.merge(o, on=["game_id", "opp"], how="left")
    f = f.merge(ab, on=["game_id", "team"], how="left")
    f = f.merge(position_defence(pg, env), on=["game_id", "opp", "grp"], how="left")
    # schedule density: a third game in four nights (team level)
    td = tg[["game_id", "team", "date"]].sort_values("date")
    td["g3in4"] = td.groupby("team").date.transform(lambda x: (x - x.shift(2)).dt.days.le(3)).astype(float)
    f = f.merge(td[["game_id", "team", "g3in4"]], on=["game_id", "team"], how="left")
    # player-specific: how his usage share moves when each absent rotation player is off the court
    import wowy
    states = wowy.pair_states(pg[["game_id", "date"]].drop_duplicates("game_id"))
    miss = f.missing_ids.apply(lambda x: x if isinstance(x, list) else [])
    f["wowy_usg"] = wowy.wowy_feature(f, miss, states)
    # and how many more minutes he has played in earlier games those teammates missed
    f["wowy_min"] = wowy.wowy_feature(f, miss, wowy.minutes_states(pg[~pg.pending]))
    # team-ratings view of the game: possessions x league points per possession x offence x
    # the other side's defence, for this team and for the opponent
    poss = f.lg_pace * f.pace_rel * f.opp_pace_rel
    f["team_pts_rt"] = poss * f.lg_ppp * f.ortg_rel * f.opp_drtg_rel
    f["opp_pts_rt"] = poss * f.lg_ppp * f.opp_ortg_rel * f.drtg_rel
    home = np.where(f.is_home, 1.0, -1.0) * (~f.neutral.astype(bool))
    f["proj_margin"] = f.team_pts_rt - f.opp_pts_rt + 2.0 * home
    # tonight's opportunities per minute on the court, from both teams' tendencies:
    #   rebounds = misses by both sides; teammates' baskets = his team's made shots;
    #   usage events = his team's possessions (FGA + 0.44 FTA + TOV ~ possessions)
    f["opp_poss_min"] = poss / 48
    miss_tm = f.lg_miss * f.miss_rel * f.opp_allow_miss_rel        # his team's misses (its shooting x their defence)
    miss_op = f.lg_miss * f.opp_miss_rel * f.allow_miss_rel        # the opponent's misses
    f["reb_opps_min"] = poss * (miss_tm + miss_op) / 48
    f["fgm_opps_min"] = poss * f.lg_fgm * f.fgm_rel * f.opp_drtg_rel / 48
    return f
