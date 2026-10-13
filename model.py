"""The model: minutes, then per-minute production in tonight's context, then distributions.

1. Minutes (OLS, starters and bench fitted separately): recent and long-run minutes, minutes
   in tonight's role, start rate, the absent regulars' minutes he usually absorbs (in
   proportion to his own minutes), blowout risk (|projected margin|), back-to-backs, playoffs.
   A team's projected minutes are then scaled to add up to 240 plus expected overtime.
2. Stats (Poisson GLM with offset log(minutes x his per-minute rate)): game pace vs league,
   what the opponent allows of that stat, home, rest, the usage freed by absent teammates,
   playoffs, and a 'spread' term (log of his base vs a typical base) that corrects the
   history-only base where it is too regressed or not enough.
3. Distributions: negative binomial for each stat and each combo (PRA, P+R, P+A, R+A), with
   the overdispersion fitted on the training games (method of moments).
4. Team score: the summed player points, recalibrated, with normal margin and total errors.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import config as C

STAT_COLS = ["pts", "reb", "ast", "fg3m"]
COMBOS = {"pra": ["pts", "reb", "ast"], "pr": ["pts", "reb"], "pa": ["pts", "ast"], "ra": ["reb", "ast"]}
TYPICAL = {"pts": 0.45, "reb": 0.18, "ast": 0.10, "fg3m": 0.045}   # per-minute rates of a typical player
EXP_MIN = 240 + 25 * 0.06    # regulation + expected overtime (about 6% of games, 25 player-minutes each)
CEILING = 40.0               # above this, a player takes almost none of a team's spare minutes


# ---------------------------------------------------------------- shared prep

def prepare(f: pd.DataFrame) -> pd.DataFrame:
    """Derived columns used by both the minutes and the stat models (needs tonight's
    availability and starters, i.e. the rows of available players)."""
    d = f.copy()
    d["ew_min_s"] = d.ew_min_s.fillna(d.ew_min_l)
    for c in ("ew_min_s", "ew_min_l"):
        d[c] = d[c].fillna(pd.Series(np.where(d.starter, 28.0, 12.0), index=d.index))
    d["role_min"] = np.where(d.starter, d.ew_min_start, d.ew_min_bench)
    d["role_min"] = d.role_min.fillna(d.ew_min_l)
    d["start_rate"] = d.start_rate.fillna(d.starter.astype(float))
    key = [d.game_id, d.team]
    usual = d.groupby(key).ew_min_l.transform("sum")
    d["absorb"] = d.miss_min.fillna(0) * d.ew_min_l / usual.where(usual > 0)
    usg_min = (d.ew_min_l * d.r_usg).groupby(key).transform("sum")
    # usage freed by absent teammates: how much of the team's usual usage is missing, and how
    # that interacts with his own usage (high-usage players absorb more, but not without limit)
    d["miss_usg_share"] = np.log1p(d.miss_usg.fillna(0) / usg_min.where(usg_min > 0)).fillna(0)
    d["rel_usg"] = np.log((d.r_usg / d.r_usg.mean()).clip(0.3, 3))
    d["usg_boost"] = d.miss_usg_share * d.rel_usg
    d["blowout"] = d.proj_margin.abs().fillna(0)
    d["b2b"] = (d.rest == 1).astype(float)
    d["opp_b2b"] = (d.opp_rest == 1).astype(float)
    d["po"] = (d.season_type == 3).astype(float)
    lg_pace = d.pace.mean()
    d["pace_g"] = np.log(((d.pace + d.opp_pace) / 2 / lg_pace).fillna(1.0))
    return d


# ---------------------------------------------------------------- minutes

def _min_X(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"const": 1.0, "ew_min_s": d.ew_min_s, "ew_min_l": d.ew_min_l, "role_min": d.role_min,
                         "start_rate": d.start_rate, "absorb": d.absorb.fillna(0), "blowout": d.blowout,
                         "b2b": d.b2b, "po": d.po, "po_min": d.po * d.ew_min_l,
                         "ew_min_l2": d.ew_min_l ** 2 / 40}, index=d.index)


def _play_X(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"const": 1.0, "ew_min_s": d.ew_min_s, "ew_min_l": d.ew_min_l,
                         "absorb": d.absorb.fillna(0), "blowout": d.blowout}, index=d.index)


def fit_minutes(d: pd.DataFrame) -> dict:
    """Minutes IF he plays (OLS on games he played, starters and bench separately), and the
    chance he plays at all (logistic: end-of-bench players often sit by coach's decision)."""
    co = {}
    played = d["min"] > 0
    for role, m in (("start", d.starter), ("bench", ~d.starter)):
        k = m & played
        X, y = _min_X(d[k]), d.loc[k, "min"]
        co[role] = dict(zip(X.columns, map(float, sm.OLS(y, X).fit().params)))
    b = d[~d.starter]
    lg = sm.Logit(played[~d.starter].astype(float), _play_X(b)).fit(disp=0)
    co["play"] = dict(zip(_play_X(b).columns, map(float, lg.params)))
    m_if, p_play = predict_minutes(d, co, normalise=False)
    resid = (d["min"] - m_if)[played]
    co["sd"] = {"start": float(resid[d.starter].std()), "bench": float(resid[~d.starter].std())}
    return co


def predict_minutes(d: pd.DataFrame, co: dict, normalise: bool = True) -> tuple[pd.Series, pd.Series]:
    """(minutes if he plays, chance he plays). Normalised so the team's EXPECTED minutes,
    chance x minutes, add up to 240 plus expected overtime."""
    m_if = pd.Series(np.nan, index=d.index)
    for role, m in (("start", d.starter), ("bench", ~d.starter)):
        X = _min_X(d[m])
        m_if[m] = X.to_numpy() @ np.array([co[role][c] for c in X.columns])
    m_if = m_if.clip(1, 46)
    Xp = _play_X(d)
    p_play = pd.Series(1 / (1 + np.exp(-(Xp.to_numpy() @ np.array([co["play"][c] for c in Xp.columns])))), index=d.index)
    p_play = p_play.where(~d.starter, 1.0)
    if normalise:
        # make the team's expected minutes add up to 240 + overtime. Extra minutes go to
        # players with room below a ceiling (coaches cap stars at ~36 even when teammates are
        # out and play the deep bench more); a surplus comes off in proportion to minutes.
        key = [d.game_id, d.team]
        gap = EXP_MIN - (m_if * p_play).groupby(key).transform("sum")
        room = (CEILING - m_if).clip(lower=0) * p_play
        add = gap.clip(lower=0) * room / room.groupby(key).transform("sum").where(lambda x: x > 0) / p_play
        cut = gap.clip(upper=0) * m_if / (m_if * p_play).groupby(key).transform("sum")
        m_if = (m_if + add.fillna(0) + cut.fillna(0)).clip(1, 46)
    return m_if, p_play


# ---------------------------------------------------------------- stats

def _stat_X(d: pd.DataFrame, s: str, base: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame({"const": 1.0, "pace": d.pace_g, "home": d.is_home.astype(float), "b2b": d.b2b,
                      "opp_b2b": d.opp_b2b, "miss_usg": d.miss_usg_share, "usg_boost": d.usg_boost, "po": d.po,
                      "spread": np.log((d[f"r_{s}"].clip(1e-4)) / TYPICAL[s])}, index=d.index)
    X["spread2"] = X.spread ** 2       # production flattens at the extremes (a log-linear fit over-reaches)
    X["min_lvl"] = np.log((base / d[f"r_{s}"].clip(1e-4)).clip(1, 48) / 24)   # minutes level: per-minute output is not constant across roles
    if s in ("pts", "reb", "ast", "fg3m"):
        X["opp"] = np.log(d[f"opp_allow_{s}"].clip(0.7, 1.4)).fillna(0)
    return X


def _top_X(mu: np.ndarray, knots: list[float]) -> np.ndarray:
    """Hinge terms above the knots of log(projection): the calibration bends only the top."""
    lm = np.log(np.clip(mu, 1e-3, None))
    return np.column_stack([np.maximum(0, lm - k) for k in knots])


def fit_stats(d: pd.DataFrame, min_hat: pd.Series) -> dict:
    co = {}
    for s in STAT_COLS:
        base = (min_hat * d[f"r_{s}"]).clip(1e-3)
        X = _stat_X(d, s, base)
        m = sm.GLM(d[s], X, family=sm.families.Poisson(), offset=np.log(base)).fit()
        co[s] = dict(zip(X.columns, map(float, m.params)))
        # top-end calibration: the very highest projections over-reach (seen in and out of
        # sample), so a correction above the 90th / 99th percentile of the fitted means
        mu = base * np.exp(X.to_numpy() @ m.params.to_numpy())
        knots = [float(np.log(np.quantile(mu, q))) for q in (0.90, 0.99)]
        H = _top_X(mu.to_numpy(), knots)
        c = sm.GLM(d[s].to_numpy(), H, family=sm.families.Poisson(), offset=np.log(mu.to_numpy())).fit()
        co[s]["_top"] = {"knots": knots, "coef": list(map(float, c.params))}
    return co


def predict_stats(d: pd.DataFrame, min_hat: pd.Series, co: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    for s in STAT_COLS:
        base = (min_hat * d[f"r_{s}"]).clip(1e-3)
        X = _stat_X(d, s, base)
        mu = base * np.exp(X.to_numpy() @ np.array([co[s][c] for c in X.columns]))
        top = co[s].get("_top")
        if top:
            mu = mu * np.exp(_top_X(mu.to_numpy(), top["knots"]) @ np.array(top["coef"]))
        out[f"{s}_hat"] = mu
    for c, parts in COMBOS.items():
        out[f"{c}_hat"] = sum(out[f"{p}_hat"] for p in parts)
    return out


# ---------------------------------------------------------------- distributions

def _var_fit(y, mu) -> tuple[float, float]:
    """Variance as c1 * mu + c2 * mu^2, fitted by least squares on the squared misses: one
    overdispersion for everyone made the spread too wide for high scorers and too narrow
    for low ones."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    X = np.column_stack([mu, mu ** 2])
    c1, c2 = np.linalg.lstsq(X, (y - mu) ** 2, rcond=None)[0]
    return float(max(c1, 0.5)), float(max(c2, 0.0))


def fit_alpha(d: pd.DataFrame, p: pd.DataFrame) -> dict:
    a = {s: _var_fit(d[s], p[f"{s}_hat"]) for s in STAT_COLS}
    for c, parts in COMBOS.items():
        a[c] = _var_fit(sum(d[x] for x in parts), p[f"{c}_hat"])
    return a


def prob_over(mu, var_coef, line: float) -> np.ndarray:
    """P(X > line) for a count with mean mu and variance c1 mu + c2 mu^2: a negative binomial
    when that is above the mean, a Poisson otherwise (line x.5)."""
    mu = np.clip(np.asarray(mu, float), 1e-6, None)
    c1, c2 = var_coef
    var = c1 * mu + c2 * mu ** 2
    k = np.floor(line)
    over = var > mu * 1.0001
    out = stats.poisson.sf(k, mu)
    n = mu ** 2 / np.where(over, var - mu, 1.0)
    nb = stats.nbinom.sf(k, n, n / (n + mu))
    return np.where(over, nb, out)


# ---------------------------------------------------------------- team scores

def game_table(d: pd.DataFrame, exp_pts: pd.Series) -> pd.DataFrame:
    """One row per game (home side): both views of each team's points.
    exp_pts = projected points x chance he plays, summed over the team's available players."""
    t = d.assign(e=exp_pts).groupby(["game_id", "team"]).agg(
        sum_hat=("e", "sum"), rt=("team_pts_rt", "first"), is_home=("is_home", "first"),
        season=("season", "first"), season_type=("season_type", "first"), neutral=("neutral", "first")).reset_index()
    h, a = t[t.is_home].set_index("game_id"), t[~t.is_home].set_index("game_id")
    j = h.index.intersection(a.index)
    return pd.DataFrame({"game_id": j, "season": h.season.loc[j].to_numpy(), "season_type": h.season_type.loc[j].to_numpy(),
                         "home_court": (~h.neutral.loc[j].astype(bool)).astype(float).to_numpy(),
                         "pl_margin": (h.sum_hat.loc[j] - a.sum_hat.loc[j]).to_numpy(),
                         "rt_margin": (h.rt.loc[j] - a.rt.loc[j]).to_numpy(),
                         "pl_total": (h.sum_hat.loc[j] + a.sum_hat.loc[j]).to_numpy(),
                         "rt_total": (h.rt.loc[j] + a.rt.loc[j]).to_numpy()})


def _score_X(g: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "margin":
        return np.column_stack([g.home_court, g.pl_margin, g.rt_margin])
    return np.column_stack([np.ones(len(g)), g.pl_total, g.rt_total])


def fit_scores(g: pd.DataFrame, margin: np.ndarray, total: np.ndarray) -> dict:
    """Margin and total as fitted blends of the player-sum and team-ratings views, with normal
    errors whose spreads come from the training games."""
    out = {}
    ok = np.isfinite(margin) & np.isfinite(total) & np.isfinite(g.rt_total.to_numpy())
    for kind, y in (("margin", margin), ("total", total)):
        X = _score_X(g, kind)
        b = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        out[kind] = {"coef": list(map(float, b)), "sd": float(np.std(y[ok] - X[ok] @ b))}
    return out


def predict_scores(g: pd.DataFrame, co: dict) -> pd.DataFrame:
    m = _score_X(g, "margin") @ np.array(co["margin"]["coef"])
    t = _score_X(g, "total") @ np.array(co["total"]["coef"])
    return pd.DataFrame({"margin_hat": m, "total_hat": t, "p_home": stats.norm.cdf(m / co["margin"]["sd"])}, index=g.index)


def save(co: dict, name: str = "coefs.json") -> None:
    C.MODELS.mkdir(parents=True, exist_ok=True)
    (C.MODELS / name).write_text(json.dumps(co, indent=2))


def load(name: str = "coefs.json") -> dict:
    return json.loads((C.MODELS / name).read_text())
