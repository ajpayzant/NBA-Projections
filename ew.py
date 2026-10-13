"""Exponentially weighted running sums: the one way history enters the model.

Every rate in the model is (weighted events) / (weighted ice time) over a player's or team's
earlier games, where a game `n` games ago carries weight 0.5 ** (n / half_life). A summer
counts as `season_gap` extra games, so last season fades a little faster than last month.

`pregame` returns, for every row, the sums over that key's EARLIER rows only, so a backtest
never sees the game it is predicting. `latest` returns the state going into each key's next
game, for live projections.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clock(df: pd.DataFrame, key: str, season_gap: float) -> np.ndarray:
    n = df.groupby(key, sort=False).cumcount().to_numpy(float)
    return n + season_gap * (df["season"].to_numpy(float) - df["season"].min())


def pregame(df: pd.DataFrame, key: str, cols: list[str], half_life: float,
            season_gap: float = 0.0) -> pd.DataFrame:
    """EW sums of `cols` over each key's earlier rows. `df` must be in date order; rows of a
    key need not be contiguous."""
    df = df.sort_values(key, kind="stable")
    keys = df[key].to_numpy()
    clock = _clock(df, key, season_gap)
    x = np.nan_to_num(df[cols].to_numpy(float))
    have = (~df[cols].isna()).to_numpy()  # missing on-ice data: neither events nor weight
    out = np.zeros_like(x)
    s = np.zeros(len(cols))
    for i in range(1, len(df)):
        if keys[i] != keys[i - 1]:
            s[:] = 0.0
            continue
        s = (s + x[i - 1] * have[i - 1]) * 0.5 ** ((clock[i] - clock[i - 1]) / half_life)
        out[i] = s
    return pd.DataFrame(out, index=df.index, columns=cols).sort_index()


def latest(df: pd.DataFrame, key: str, cols: list[str], half_life: float,
           season_gap: float = 0.0, next_season: int | None = None) -> pd.DataFrame:
    """EW sums going into each key's next game (one game after its last row, plus a summer
    if `next_season` is later than the last row's season). Indexed by key."""
    pre = pregame(df, key, cols, half_life, season_gap)
    last = df.groupby(key, sort=False).tail(1).index
    x = np.nan_to_num(df.loc[last, cols].to_numpy(float)) * (~df.loc[last, cols].isna()).to_numpy()
    step = np.ones(len(last))
    if next_season is not None:
        step += season_gap * np.maximum(0, next_season - df.loc[last, "season"].to_numpy(float))
    s = (pre.loc[last].to_numpy() + x) * (0.5 ** (step / half_life))[:, None]
    return pd.DataFrame(s, index=df.loc[last, key].to_numpy(), columns=cols)
