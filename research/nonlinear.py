"""Does a gradient-boosted model find what the projection misses?  python -m research.nonlinear

LightGBM is trained on 2021-24 backtest rows to predict each stat's miss (actual / projected,
via a Poisson objective with the projection as the base) from every pregame feature, and scored
on 2024-26. If it improves little, the model's structure is not leaving interactions behind.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

import backtest

FEATS = ["ew_min_s", "ew_min_l", "ew_min_start", "ew_min_bench", "start_rate", "starter", "is_home", "rest", "opp_rest",
         "pace_rel", "opp_pace_rel", "ortg_rel", "drtg_rel", "opp_ortg_rel", "opp_drtg_rel", "miss_min", "miss_usg",
         "miss_usg_share", "absorb", "wowy_usg", "wowy_min", "blowout", "g3in4", "gp_w", "hist_min",
         "rel_pts", "rel_reb", "rel_ast", "rel_fg3m", "rel_usg", "rel_fga", "rel_fta", "rel_oreb", "rel_dreb",
         "sh_usg", "sh_reb", "sh_ast", "pts_per_usg", "opp_allow_pts", "opp_allow_reb", "opp_allow_ast", "opp_allow_fg3m",
         "pos_allow_pts", "pos_allow_reb", "pos_allow_ast", "min_hat", "proj_margin"]

b = pd.read_parquet(backtest.OUT / "backtest.parquet")
b = b[(b.season_type == 2) & (b["min"] > 0)].copy()
b["starter"], b["is_home"] = b.starter.astype(float), b.is_home.astype(float)
X = b[[c for c in FEATS if c in b]].astype(float)
tr, te = (b.fit_season <= 2023).to_numpy(), (b.fit_season >= 2024).to_numpy()
print(f"train {tr.sum():,} rows (2021-24), test {te.sum():,} rows (2024-26), {X.shape[1]} features")
for s in ["min", "pts", "reb", "ast", "fg3m"]:
    hat = b["min_hat"] if s == "min" else b[f"{s}_hat"]
    y = b[s]
    ds = lgb.Dataset(X[tr], y[tr], init_score=np.log(hat[tr].clip(0.05)))
    m = lgb.train({"objective": "poisson", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 400,
                   "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10, "verbose": -1},
                  ds, num_boost_round=300)
    new = np.exp(m.predict(X[te], raw_score=True) + np.log(hat[te].clip(0.05)))
    mae0, mae1 = np.abs(y[te] - hat[te]).mean(), np.abs(y[te] - new).mean()
    imp = pd.Series(m.feature_importance("gain"), index=X.columns).nlargest(4)
    print(f"{s:5s} MAE {mae0:.4f} -> {mae1:.4f} ({100 * (mae1 / mae0 - 1):+.2f}%) | most used: {', '.join(imp.index)}")
