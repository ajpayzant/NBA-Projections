"""Walk-forward backtest: each season projected with models fitted only on earlier seasons.

    python backtest.py            # 2021-22 .. 2025-26 -> data/backtest/backtest.parquet + report

Lineups: who was available and who started each night, which NBA injury reports and starting
lineups mostly reveal before tip-off (late scratches are the gap with live use). Every
feature is pregame (EW sums over earlier games only).

Baselines: his season-to-date and last-10 averages over games he played.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import features
import model
import parse

TEST = (2021, 2022, 2023, 2024, 2025)
FIRST_TRAIN = 2019
OUT = C.DATA / "backtest"
LINES = {"pts": [9.5, 14.5, 19.5, 24.5, 29.5], "reb": [3.5, 5.5, 7.5, 9.5], "ast": [2.5, 4.5, 6.5, 8.5],
         "fg3m": [0.5, 1.5, 2.5, 3.5], "pra": [19.5, 29.5, 39.5], "pr": [14.5, 24.5], "pa": [14.5, 24.5], "ra": [5.5, 9.5]}


def baselines(d: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date and last-10 averages over games played, as of the game before."""
    played = d[d["min"] > 0].sort_values(["date", "game_id"])
    out = pd.DataFrame(index=played.index)
    for s in ["min", "pts", "reb", "ast", "fg3m"]:
        g = played.groupby(["player", "season"])[s]
        out[f"s2d_{s}"] = g.transform(lambda x: x.shift().expanding().mean())
        out[f"l10_{s}"] = played.groupby("player")[s].transform(lambda x: x.shift().rolling(10, min_periods=3).mean())
    return out.reindex(d.index)


def _results(g: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """Final margin and total (home minus away) for the games in g."""
    h = tg[tg.is_home].set_index("game_id")
    return pd.DataFrame({"margin": (h.pts - h.opp_pts).reindex(g.game_id).to_numpy(),
                         "total": (h.pts + h.opp_pts).reindex(g.game_id).to_numpy()}, index=g.index)


def run() -> pd.DataFrame:
    pg, tg = parse.load("player_games"), parse.load("team_games")
    f = features.build(pg, tg)
    d = model.prepare(f[f.season_type.isin(C.SEASON_TYPES)])
    d = pd.concat([d, baselines(d)], axis=1)
    parts, cards, games = [], [], []
    for s in TEST:
        tr, te = d[(d.season < s) & (d.season >= FIRST_TRAIN)], d[d.season == s].copy()
        co_min = model.fit_minutes(tr)
        mh_tr, _ = model.predict_minutes(tr, co_min)
        trp = tr["min"] > 0          # the stat models learn from games he played (props are void otherwise)
        co_st = model.fit_stats(tr[trp], mh_tr[trp])
        p_tr = model.predict_stats(tr, mh_tr, co_st)
        alpha = model.fit_alpha(tr[tr["min"] > 0], p_tr[tr["min"] > 0])
        _, pp_tr = model.predict_minutes(tr, co_min)
        plain_tr = model.predict_stats(tr, mh_tr, co_st, boost=False)      # not memorised on these seasons
        g_tr = model.game_table(tr, plain_tr.pts_hat * pp_tr)
        res = _results(g_tr, tg)
        co_sc = model.fit_scores(g_tr, res.margin.to_numpy(), res.total.to_numpy())
        te["min_hat"], te["p_play"] = model.predict_minutes(te, co_min)
        p = model.predict_stats(te, te.min_hat, co_st)
        te = pd.concat([te, p], axis=1)
        for st_, lines in LINES.items():
            for ln in lines:
                te[f"p_{st_}_{ln}"] = model.prob_over(te[f"{st_}_hat"], alpha[st_], ln)
        te["fit_season"] = s
        parts.append(te)
        plain_te = model.predict_stats(te, te.min_hat, co_st, boost=False)
        g_te = model.game_table(te, plain_te.pts_hat * te.p_play)
        games.append(pd.concat([g_te, model.predict_scores(g_te, co_sc), _results(g_te, tg)], axis=1))
        cards.append({"season": s, **{f"alpha_{k}": v for k, v in alpha.items()},
                      "sd_margin": co_sc["margin"]["sd"], "sd_total": co_sc["total"]["sd"],
                      "w_margin": co_sc["margin"]["coef"], "w_total": co_sc["total"]["coef"]})
        print(f"{s}: fitted on {sorted(tr.season.unique())}, {len(te):,} test rows", flush=True)
    bt = pd.concat(parts, ignore_index=True)
    for c, parts_ in model.COMBOS.items():
        bt[c] = sum(bt[x] for x in parts_)
    OUT.mkdir(parents=True, exist_ok=True)
    bt.to_parquet(OUT / "backtest.parquet", index=False)
    pd.concat(games, ignore_index=True).to_parquet(OUT / "games.parquet", index=False)
    pd.DataFrame(cards).to_parquet(OUT / "fits.parquet", index=False)
    return bt


def report(bt: pd.DataFrame | None = None) -> None:
    bt = bt if bt is not None else pd.read_parquet(OUT / "backtest.parquet")
    reg = bt[bt.season_type == 2]
    pd.set_option("display.width", 220)
    print(f"\n{len(reg):,} available player-games, {reg.game_id.nunique():,} games (regular season 2021-26)")
    m = reg[reg.s2d_min.notna() & reg.l10_min.notna() & (reg["min"] > 0)]
    print(f"minutes MAE (games he played): model {np.abs(m['min'] - m.min_hat).mean():.2f} | season avg {np.abs(m['min'] - m.s2d_min).mean():.2f} | "
          f"last 10 {np.abs(m['min'] - m.l10_min).mean():.2f} | corr {m['min'].corr(m.min_hat):.3f}")
    played = reg[(reg["min"] > 0) & reg.s2d_pts.notna() & reg.l10_pts.notna()]
    rows = []
    for s in ["pts", "reb", "ast", "fg3m"]:
        y = played[s]
        rows.append({"stat": s, "avg": y.mean(), "corr": y.corr(played[f"{s}_hat"]),
                     "MAE model": np.abs(y - played[f"{s}_hat"]).mean(), "MAE season avg": np.abs(y - played[f"s2d_{s}"]).mean(),
                     "MAE last 10": np.abs(y - played[f"l10_{s}"]).mean(), "bias": (y - played[f"{s}_hat"]).mean()})
    for c in model.COMBOS:
        y = played[c]
        rows.append({"stat": c, "avg": y.mean(), "corr": y.corr(played[f"{c}_hat"]),
                     "MAE model": np.abs(y - played[f"{c}_hat"]).mean(), "bias": (y - played[f"{c}_hat"]).mean()})
    print("\nplayers who played (props are void otherwise):")
    print(pd.DataFrame(rows).set_index("stat").round(3).to_string())
    print("\ncalibration, P(over) on common lines: predicted -> actual by fifth of predicted probability")
    pl = reg[reg["min"] > 0]
    for s, lines in LINES.items():
        for ln in lines:
            p, y = pl[f"p_{s}_{ln}"], (pl[s] > ln).astype(float)
            b = pd.qcut(p.rank(method="first"), 5, labels=False)
            t = pd.DataFrame({"p": p, "y": y, "b": b}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"))
            print(f"  {s:4s} o{ln:<5g} mean {p.mean():.3f}->{y.mean():.3f} | " + "  ".join(f"{r.p:.2f}->{r.y:.2f}" for r in t.itertuples()))




def report_games() -> None:
    """Moneyline / spread / total: the fitted blend vs the player-sum and net-rating views alone."""
    from scipy import stats as S
    g = pd.read_parquet(OUT / "games.parquet")
    g = g[g.season_type == 2].dropna(subset=["margin"])
    fits = pd.read_parquet(OUT / "fits.parquet").set_index("season")
    sd = g.season.map(fits.sd_margin)
    y = (g.margin > 0).astype(float)
    ll = lambda p: float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))
    views = {"blend (model)": g.margin_hat, "players only": g.pl_margin + 2 * g.home_court,
             "team ratings only": g.rt_margin + 2 * g.home_court}
    print(f"\n{len(g):,} regular-season games 2021-26 (home teams won {y.mean():.3f})")
    for k, m in views.items():
        p = S.norm.cdf(m / sd)
        print(f"  {k:18s} winner {np.mean((p >= .5) == y):.3f} | ML log loss {ll(p):.4f} | margin MAE "
              f"{np.abs(g.margin - m).mean():.2f} (corr {np.corrcoef(g.margin, m)[0, 1]:.3f})")
    print(f"  total: blend MAE {np.abs(g.total - g.total_hat).mean():.2f} (corr {np.corrcoef(g.total, g.total_hat)[0, 1]:.3f}, "
          f"bias {(g.total - g.total_hat).mean():+.2f}) | players-only corr {np.corrcoef(g.total, g.pl_total)[0, 1]:.3f} | "
          f"ratings-only corr {np.corrcoef(g.total, g.rt_total)[0, 1]:.3f} | season-average MAE "
          f"{np.abs(g.total - g.groupby('season').total.transform('mean')).mean():.2f}")
    print("  margin blend weights by season (home court, players, ratings):",
          [list(np.round(w, 2)) for w in fits.w_margin])


if __name__ == "__main__":
    report(run())
    report_games()
