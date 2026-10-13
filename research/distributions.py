"""Which outcome distribution prices the lines best, stat by stat?  python -m research.distributions

For each stat and combo, the variance model var = c0 + c1 mu + c2 mu^2 is fitted on 2021-24
backtest rows and every candidate family is scored on 2024-26 by log loss over the over/under
of the common lines (the probabilities that props are priced with):
    nb_lin    negative binomial, var = c1 mu (the current model)
    nb_quad   negative binomial, var = c1 mu + c2 mu^2
    nb_full   negative binomial, var = c0 + c1 mu + c2 mu^2
    normal    normal with that full variance and a continuity correction
"""
import numpy as np
import pandas as pd
from scipy import stats

import backtest
import model

b = pd.read_parquet(backtest.OUT / "backtest.parquet")
b = b[(b.season_type == 2) & (b["min"] > 0)].copy()
tr, te = b[b.fit_season <= 2023], b[b.fit_season >= 2024]


def fit(y, mu, cols):
    X = np.column_stack([{"c0": np.ones_like(mu), "c1": mu, "c2": mu ** 2}[c] for c in cols])
    return dict(zip(cols, np.linalg.lstsq(X, (y - mu) ** 2, rcond=None)[0]))


def p_over(mu, var, line, fam):
    k = np.floor(line)
    if fam == "normal":
        return stats.norm.sf((k + 0.5 - mu) / np.sqrt(np.maximum(var, 1e-3)))
    over = var > mu * 1.0001
    n = mu ** 2 / np.where(over, var - mu, 1.0)
    return np.where(over, stats.nbinom.sf(k, n, n / (n + mu)), stats.poisson.sf(k, mu))


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


FAMS = {"nb_lin": ("nb", ["c1"]), "nb_quad": ("nb", ["c1", "c2"]), "nb_full": ("nb", ["c0", "c1", "c2"]),
        "normal": ("normal", ["c0", "c1", "c2"])}
rows = []
for s, lines in backtest.LINES.items():
    mu_tr, mu_te = tr[f"{s}_hat"].to_numpy(), te[f"{s}_hat"].to_numpy()
    r = {"stat": s}
    for name, (fam, cols) in FAMS.items():
        c = fit(tr[s].to_numpy(), mu_tr, cols)
        var = sum(c.get(k, 0) * {"c0": 1, "c1": mu_te, "c2": mu_te ** 2}[k] for k in c)
        var = np.maximum(var, 0.3 * mu_te)
        r[name] = np.mean([ll(p_over(mu_te, var, ln, fam), (te[s] > ln).to_numpy(float)) for ln in lines])
    rows.append(r)
R = pd.DataFrame(rows).set_index("stat")
R["best"] = R.idxmin(axis=1)
pd.set_option("display.width", 200)
print("mean log loss over the common lines, 2024-26 (lower is better):")
print(R.round(5).to_string())
