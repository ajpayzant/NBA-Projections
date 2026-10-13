"""With / without: how each player's usage changes when a particular teammate is off the court.

From the play-by-play stints (pbp.py): for every pair of teammates (A, B) in every game, A's
usage events and his team's usage events in stretches where both were on the court, and in
stretches where A was on and B was not. Accumulated over EARLIER games only (EW by pair),
shrunk toward A's normal share, so that

    ratio(A, B) = A's usage share without B on the court / with B on the court

For tonight: the sum of log ratio(A, B) over the regulars B who are out, a player-specific
version of "the usage freed by absent teammates".
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

import config as C

HL_PAIR = 40         # pair-games
K_PAIR = 25.0        # usage events of prior (at A's normal share) on each side


def pair_games() -> pd.DataFrame:
    """One row per (game, A, B): A's and the team's usage events with B on, and with B off."""
    rows = []
    for f in sorted(C.LAKE.glob("stints_[0-9][0-9][0-9][0-9].parquet")):
        st = pd.read_parquet(f, columns=["game_id", "team", "t0", "t1", "lineup", "fga", "fta", "tov", "pu"])
        st["U"] = st.fga + 0.44 * st.fta + st.tov
        for (gid, team), g in st.groupby(["game_id", "team"], sort=False):
            tot_u = defaultdict(float)     # A -> (A usage, team usage) while A on
            tot_U = defaultdict(float)
            with_u = defaultdict(float)    # (A, B) -> while both on
            with_U = defaultdict(float)
            for lineup, U, pu in zip(g.lineup, g.U, g.pu):
                five = lineup.split("-")
                u = json.loads(pu) if pu else {}
                for a in five:
                    ua = u.get(a, 0.0)
                    tot_u[a] += ua
                    tot_U[a] += U
                    for b in five:
                        if b != a:
                            with_u[(a, b)] += ua
                            with_U[(a, b)] += U
            players = list(tot_U)
            for a in players:
                for b in players:
                    if a == b:
                        continue
                    wu, wU = with_u.get((a, b), 0.0), with_U.get((a, b), 0.0)
                    rows.append((gid, team, int(a), int(b), wu, wU, tot_u[a] - wu, tot_U[a] - wU))
    return pd.DataFrame(rows, columns=["game_id", "team", "a", "b", "w_u", "w_U", "o_u", "o_U"])


def pair_states(pg_dates: pd.DataFrame) -> dict:
    """{(a, b): (dates, log ratio)} : the pair's state going into each later game."""
    pgm = pair_games()
    pgm = pgm.merge(pg_dates, on="game_id", how="left").sort_values(["date", "game_id"])
    decay = 0.5 ** (1 / HL_PAIR)
    out = {}
    for (a, b), g in pgm.groupby(["a", "b"], sort=False):
        s = np.zeros(4)
        dates, vals = [], []
        for d, x in zip(g.date.to_numpy(), g[["w_u", "w_U", "o_u", "o_U"]].to_numpy()):
            s = s * decay + x
            share = (s[0] + s[2] + 1.0) / (s[1] + s[3] + 5.0)        # A's overall share, lightly shrunk
            w = (s[0] + K_PAIR * share) / (s[1] + K_PAIR)
            o = (s[2] + K_PAIR * share) / (s[3] + K_PAIR)
            dates.append(d)
            vals.append(np.log(o / w))
        out[(a, b)] = (np.array(dates), np.array(vals))
    return out


def wowy_feature(rows: pd.DataFrame, missing: pd.Series, states: dict) -> pd.Series:
    """For each player-game row (player, date, its team-game's missing regulars): the sum of
    log ratio(player, B) over the missing B, using each pair's state from before that date."""
    out = np.zeros(len(rows))
    for i, (a, day, miss) in enumerate(zip(rows.player.to_numpy(), rows.date.to_numpy(), missing.to_numpy())):
        if not miss:
            continue
        tot = 0.0
        for b in miss:
            st = states.get((int(a), int(b)))
            if st is None:
                continue
            k = np.searchsorted(st[0], day) - 1          # strictly before tonight
            if k >= 0:
                tot += st[1][k]
        out[i] = tot
    return pd.Series(out, index=rows.index)


HL_MIN_PAIR = 30     # team games
K_MIN = 3.0          # games of "no change" in the prior of each without-B estimate


def minutes_states(pg: pd.DataFrame) -> dict:
    """{(a, b): (dates, minutes delta)}: how many more minutes A plays in games his regular
    teammate B misses than in games B plays, going into each later game (EW, shrunk)."""
    d = pg[["game_id", "date", "team", "player", "min"]].sort_values(["date", "game_id"])
    decay = 0.5 ** (1 / HL_MIN_PAIR)
    acc: dict = {}                 # (a, b) -> [sum_min_with, n_with, sum_min_without, n_without]
    out = defaultdict(lambda: ([], []))
    for team, g in d.groupby("team", sort=False):
        last_seen: dict[int, int] = {}
        for k, (gid, gg) in enumerate(g.groupby("game_id", sort=False)):
            avail = dict(zip(gg.player, gg["min"]))
            day = gg.date.iloc[0]
            regulars = [b for b, k0 in last_seen.items() if k - k0 <= 10]
            for a, m in avail.items():
                for b in regulars:
                    if b == a:
                        continue
                    s = acc.setdefault((a, b), [0.0, 0.0, 0.0, 0.0])
                    s[:] = [x * decay for x in s]
                    if b in avail:
                        s[0] += m; s[1] += 1
                    else:
                        s[2] += m; s[3] += 1
                    if s[1] > 0.5:
                        delta = (s[2] / s[3] - s[0] / s[1]) * s[3] / (s[3] + K_MIN) if s[3] > 0 else 0.0
                        dl, vl = out[(a, b)]
                        dl.append(day); vl.append(delta)
            for p_ in avail:
                last_seen[p_] = k
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in out.items()}
