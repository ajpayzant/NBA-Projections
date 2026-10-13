"""Pregame features for every player-game (history and pending games alike).

Everything is an exponentially weighted (EW) running sum over EARLIER games (ew.pregame), so
a row never sees the game it describes.

Who counts: a player is "available" for a team-game if he is in its box score and either
played or sat by coach's decision (ESPN lists inactive players with an injury reason, or not at
all). Minutes features average over available games, zeros included, so a player who often
sits shows low expected minutes.

Per player:  EW minutes (short / long), minutes as a starter and off the bench, start rate,
             per-minute rates for each stat (shrunk toward his position's league rate), usage
Per team:    pace (possessions / 48), what it allows per possession for each stat, net rating,
             rest and back-to-backs
Absences:    for each team-game, the usual minutes and usage of regulars not available
             tonight (a regular = available in any of the team's last 10 games), which the
             model hands to the players who are available
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import ew

HL_MIN = (4, 15)          # minutes: recent rotation, and the longer view
HL_ROLE = 10              # minutes as a starter / off the bench
HL_RATE = 25              # per-minute production
HL_TEAM = 20              # team pace, defence, strength
GAP = 15                  # a summer counts as this many extra games
STATS = ["pts", "reb", "ast", "fg3m", "fga", "fta", "tov", "oreb", "dreb", "stl", "blk"]
K_RATE = 120              # minutes of league-average production in every player's prior
POS_GROUP = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}


def available(pg: pd.DataFrame) -> pd.Series:
    """Played, or sat by coach's decision (not out injured)."""
    return ~pg.dnp | pg.dnp_reason.str.upper().str.contains("DECISION", na=False)


def prep(pg: pd.DataFrame, tg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pg = pg.copy()
    pg["date"] = pd.to_datetime(pg.date)
    pg["avail"] = available(pg)
    pg = pg[pg.avail | pg.get("pending", False)].copy()
    pg["min"] = pg["min"].fillna(0.0).where(~pg.dnp, 0.0)
    for s in STATS:
        pg[s] = pg[s].fillna(0.0).where(~pg.dnp, 0.0)
    pg["grp"] = pg.pos.map(POS_GROUP).fillna("F")
    pg = pg.sort_values(["date", "game_id"]).reset_index(drop=True)
    tg = tg.copy()
    tg["date"] = pd.to_datetime(tg.date)
    tg = tg.sort_values(["date", "game_id"]).reset_index(drop=True)
    return pg, tg


def player_features(pg: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=pg.index)
    tmp = pg[["player", "season", "min", "starter"] + STATS].copy()
    tmp["one"] = 1.0
    tmp["usg"] = tmp.fga + 0.44 * tmp.fta + tmp.tov
    for name, hl in zip(("s", "l"), HL_MIN):
        e = ew.pregame(tmp, "player", ["min", "one"], hl, GAP)
        out[f"ew_min_{name}"] = e["min"] / e.one.where(e.one > 0)
    out["gp_w"] = ew.pregame(tmp, "player", ["one"], HL_MIN[1], GAP).one   # weighted games of history
    for role, flag in (("start", tmp.starter), ("bench", ~tmp.starter)):
        r = tmp.assign(m=tmp["min"].where(flag, 0.0), n=flag.astype(float))
        e = ew.pregame(r, "player", ["m", "n"], HL_ROLE, GAP)
        out[f"ew_min_{role}"] = e.m / e.n.where(e.n > 0.2)
    e = ew.pregame(tmp, "player", ["starter", "one"], HL_ROLE, GAP)
    out["start_rate"] = e.starter / e.one.where(e.one > 0)
    # per-minute rates, shrunk toward the league rate of his position group
    e = ew.pregame(tmp, "player", ["min", "usg"] + STATS, HL_RATE, GAP)
    league = pg[pg["min"] > 0].groupby("grp")[["min"] + STATS].sum()
    for s in STATS + ["usg"]:
        lg = (league[s] / league["min"]) if s != "usg" else \
            (league.fga + 0.44 * league.fta + league.tov) / league["min"]
        prior = pg.grp.map(lg)
        out[f"r_{s}"] = (e[s] + K_RATE * prior) / (e["min"] + K_RATE)
    out["hist_min"] = e["min"]
    return out


def team_features(tg: pd.DataFrame) -> pd.DataFrame:
    """Pace, per-possession defence, strength, rest: pregame, per team-game."""
    t = tg[["game_id", "team", "opp", "season", "date", "pts", "opp_pts", "poss", "periods"]].copy()
    opp = tg[["game_id", "team", "reb", "ast", "fg3m", "oreb", "dreb", "poss", "fga", "fta", "tov"]].rename(
        columns=lambda c: c if c in ("game_id",) else f"o_{c}").rename(columns={"o_team": "opp"})
    t = t.merge(opp, on=["game_id", "opp"], how="left")
    t["min48"] = 48 + 5 * (t.periods.fillna(4) - 4)
    t["one"] = 1.0
    t["mins"] = t.min48
    # what the team allows (the opponent's box) per opponent possession
    sums = ["poss", "mins", "pts", "opp_pts", "o_reb", "o_ast", "o_fg3m", "o_poss", "one"]
    e = ew.pregame(t, "team", sums, HL_TEAM, GAP)
    lg = t[["opp_pts", "o_reb", "o_ast", "o_fg3m", "o_poss"]].sum()
    out = pd.DataFrame(index=t.index)
    out["pace"] = 48 * (e.poss + 20 * 100) / (e.mins + 20 * 48)            # possessions per 48 minutes
    out["net"] = 100 * (e.pts - e.opp_pts) / (e.poss + 100)                 # net rating, shrunk
    lg_eff = 100 * t.pts.sum() / t.poss.sum()
    out["ortg"] = 100 * (e.pts + lg_eff) / (e.poss + 100)                    # points per 100 possessions, shrunk
    out["drtg"] = 100 * (e.opp_pts + lg_eff) / (e.o_poss + 100)              # allowed per 100
    lg_rate = {"pts": lg.opp_pts / lg.o_poss, "reb": lg.o_reb / lg.o_poss, "ast": lg.o_ast / lg.o_poss,
               "fg3m": lg.o_fg3m / lg.o_poss}
    for s, col in (("pts", "opp_pts"), ("reb", "o_reb"), ("ast", "o_ast"), ("fg3m", "o_fg3m")):
        out[f"allow_{s}"] = ((e[col] + 15 * lg_rate[s] * 100) / (e.o_poss + 15 * 100)) / lg_rate[s]
    t["rest"] = t.groupby("team").date.diff().dt.days.clip(upper=4).fillna(4)
    out["rest"] = t.rest
    out = pd.concat([t[["game_id", "team", "opp"]], out], axis=1)
    return out


def absences(pg: pd.DataFrame, pf: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: the usual minutes and usage-minutes of regulars who are not available."""
    d = pd.concat([pg[["game_id", "date", "team", "player"]], pf[["ew_min_l", "r_usg"]]], axis=1)
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
        state = d[d.team == team].set_index(["game_id", "player"])[["ew_min_l", "r_usg"]]
        last_seen: dict[int, tuple] = {}   # player -> (index of his last available game, ew_min_l, r_usg)
        for k, gid in enumerate(ids):
            here = members.get(gid, set())
            miss_min = miss_usg = 0.0
            day = g.date.iloc[k]
            for p_, (k0, m, u) in last_seen.items():
                if p_ not in here and k - k0 <= 10 and np.isfinite(m) and team_before(p_, day) == team:
                    miss_min += m
                    miss_usg += m * (u if np.isfinite(u) else 0.0)
            rows.append({"game_id": gid, "team": team, "miss_min": miss_min, "miss_usg": miss_usg})
            for p_ in here:
                m, u = state.loc[(gid, p_)].to_numpy() if (gid, p_) in state.index else (np.nan, np.nan)
                last_seen[p_] = (k, m, u)
    return pd.DataFrame(rows)


def build(pg: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    pg, tg = prep(pg, tg)
    pf = player_features(pg)
    tf = team_features(tg)
    ab = absences(pg, pf)
    f = pd.concat([pg, pf], axis=1)
    f = f.merge(tf, on=["game_id", "team", "opp"], how="left")
    o = tf[["game_id", "team", "pace", "net", "ortg", "drtg", "allow_pts", "allow_reb", "allow_ast", "allow_fg3m", "rest"]].rename(
        columns={"team": "opp", "pace": "opp_pace", "net": "opp_net", "ortg": "opp_ortg", "drtg": "opp_drtg", "allow_pts": "opp_allow_pts",
                 "allow_reb": "opp_allow_reb", "allow_ast": "opp_allow_ast", "allow_fg3m": "opp_allow_fg3m",
                 "rest": "opp_rest"})
    f = f.merge(o, on=["game_id", "opp"], how="left")
    f = f.merge(ab, on=["game_id", "team"], how="left")
    # the absent regulars' minutes as a share of the team's usual 240
    f["miss_share"] = f.miss_min / 240
    # projected margin from the two teams' net ratings and pace (home court ~ +2)
    f["proj_margin"] = (f.net - f.opp_net) * (f.pace + f.opp_pace) / 200 + np.where(f.is_home, 2.0, -2.0) * (~f.neutral)
    # team-ratings view of this team's points: game possessions x (its offence + their defence) / 2
    f["team_pts_rt"] = (f.pace + f.opp_pace) / 2 * (f.ortg + f.opp_drtg) / 200
    return f
