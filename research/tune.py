"""One-at-a-time tuning of the feature half-lives and priors on the walk-forward backtest.
python -m research.tune  -> research/tune_results.csv"""
import sys

import numpy as np
import pandas as pd

import backtest
import features
from scipy import stats as S

BASE = dict(HL_MIN=(4, 15), HL_ROLE=10, HL_RATE=25, HL_TEAM=20, K_RATE=120, GAP=15)
TRIALS = [("base", {})] + [(f"{k}={v}", {k: v}) for k, v in [
    ("HL_RATE", 15), ("HL_RATE", 40), ("K_RATE", 60), ("K_RATE", 240), ("HL_MIN", (3, 10)), ("HL_MIN", (6, 25)),
    ("HL_ROLE", 6), ("HL_ROLE", 20), ("GAP", 5), ("GAP", 30), ("HL_TEAM", 12), ("HL_TEAM", 35)]]


def score(bt: pd.DataFrame) -> dict:
    r = bt[(bt.season_type == 2) & (bt["min"] > 0)]
    out = {f"mae_{s}": float(np.abs(r[s] - r[f"{s}_hat"]).mean()) for s in ("pts", "reb", "ast", "fg3m", "pra")}
    out["mae_min"] = float(np.abs(r["min"] - r.min_hat).mean())
    g = pd.read_parquet(backtest.OUT / "games.parquet")
    g = g[g.season_type == 2].dropna(subset=["margin"])
    y = (g.margin > 0).astype(float)
    p = np.clip(g.p_home, 1e-6, 1 - 1e-6)
    out["ml_logloss"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    out["total_mae"] = float(np.abs(g.total - g.total_hat).mean())
    return out


rows = []
for name, change in TRIALS:
    for k, v in {**BASE, **change}.items():
        setattr(features, k, v)
    bt = backtest.run()
    rows.append({"trial": name, **score(bt)})
    pd.DataFrame(rows).to_csv("research/tune_results.csv", index=False)
    print(rows[-1], flush=True)
for k, v in BASE.items():
    setattr(features, k, v)
