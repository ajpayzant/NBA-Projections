"""The model: minutes, then per-minute production in tonight's context, then distributions.

1. Minutes (OLS, starters and bench fitted separately): recent and long-run minutes, minutes
   in tonight's role, start rate, the absent regulars' minutes he usually absorbs (in
   proportion to his own minutes), blowout risk (|projected margin|), back-to-backs, playoffs.
   A team's projected minutes are then scaled to add up to 240 plus expected overtime.
2. Stats (Poisson GLM with offset log(minutes x his per-minute rate)): game pace vs league,
   what the opponent allows of that stat, home, rest, the usage freed by absent teammates,
   playoffs, and a 'spread' term (log of his league-relative rate) that corrects the
   history-only base where it is too regressed or not enough.
3. Distributions: negative binomial for each stat and each combo (PRA, P+R, P+A, R+A), with
   the overdispersion fitted on the training games (method of moments).
4. Boosted corrections: a regularised LightGBM, fitted on the training seasons only, corrects
   what the transparent models systematically miss (minutes curvature, short histories).
5. Team score: a fitted blend of the summed (unboosted) player points and team ratings, with
   normal margin and total errors.

Everything is league-relative (features.py), so scoring and 3-point drift across eras do not
bias the projections.
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
    d["lrel_usg"] = np.log(d.rel_usg.clip(0.3, 3))          # his usage vs the league, log
    d["usg_boost"] = d.miss_usg_share * d.lrel_usg
    d["blowout"] = d.proj_margin.abs().fillna(0)
    d["b2b"] = (d.rest == 1).astype(float)
    d["opp_b2b"] = (d.opp_rest == 1).astype(float)
    d["po"] = (d.season_type == 3).astype(float)
    d["pace_g"] = np.log((d.pace_rel * d.opp_pace_rel).fillna(1.0))   # game pace vs the league, log
    return d


# ---------------------------------------------------------------- boosted corrections
# After the transparent models, a regularised LightGBM learns what they systematically miss
# (curvature in recent minutes, players with little history, interactions), fitted on the
# training seasons only. research/nonlinear.py measured the headroom: minutes -3.3%, points -0.9%.

BOOST_FEATS = ["ew_min_s", "ew_min_l", "ew_min_start", "ew_min_bench", "start_rate", "starter", "is_home", "rest",
               "opp_rest", "pace_rel", "opp_pace_rel", "ortg_rel", "drtg_rel", "opp_ortg_rel", "opp_drtg_rel", "miss_min",
               "miss_usg", "miss_usg_share", "absorb", "wowy_usg", "wowy_min", "blowout", "g3in4", "gp_w", "hist_min",
               "rel_pts", "rel_reb", "rel_ast", "rel_fg3m", "rel_usg", "rel_fga", "rel_fta", "rel_oreb", "rel_dreb",
               "sh_usg", "sh_reb", "sh_ast", "pts_per_usg", "opp_allow_pts", "opp_allow_reb", "opp_allow_ast",
               "opp_allow_fg3m", "pos_allow_pts", "pos_allow_reb", "pos_allow_ast", "proj_margin", "po"]
BOOST_PARAMS = {"learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 400, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10, "verbose": -1, "seed": 7}


def _bX(d: pd.DataFrame, base: pd.Series) -> pd.DataFrame:
    X = d[[c for c in BOOST_FEATS if c in d]].astype(float).copy()
    X["base"] = base.astype(float)
    return X


def fit_boost(d: pd.DataFrame, y: pd.Series, base: pd.Series, objective: str, rounds: int = 300):
    import lightgbm as lgb
    X = _bX(d, base)
    init = np.log(base.clip(0.05)) if objective == "poisson" else base
    ds = lgb.Dataset(X, y.astype(float), init_score=init.to_numpy())
    return lgb.train({**BOOST_PARAMS, "objective": objective}, ds, num_boost_round=rounds)


def apply_boost(booster, d: pd.DataFrame, base: pd.Series, objective: str) -> pd.Series:
    raw = booster.predict(_bX(d, base), raw_score=True)
    out = base * np.exp(raw) if objective == "poisson" else base + raw
    return pd.Series(np.asarray(out), index=d.index)


# ---------------------------------------------------------------- minutes

def _min_X(d: pd.DataFrame) -> pd.DataFrame:
    ab = d.absorb.fillna(0)
    return pd.DataFrame({"const": 1.0, "ew_min_s": d.ew_min_s, "ew_min_l": d.ew_min_l, "role_min": d.role_min,
                         "start_rate": d.start_rate, "absorb": ab,
                         # who absorbs an absent teammate's minutes is not simply proportional:
                         # high-usage players and players with a lot of room take different shares
                         "absorb_usg": ab * d.lrel_usg, "absorb_min": ab * d.ew_min_l / 30, "absorb2": np.sqrt(ab),
                         "wowy_min": d.wowy_min.fillna(0).clip(-15, 15),   # his own history without them
                         "blowout": d.blowout, "g3in4": d.g3in4.fillna(0) * d.ew_min_l / 30,
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
    m_lin, _ = predict_minutes(d, {**co, "boost": None}, normalise=False)
    co["boost"] = fit_boost(d[played], d.loc[played, "min"], m_lin[played], "regression")
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
    if co.get("boost") is not None:
        m_if = apply_boost(co["boost"], d, m_if, "regression")
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
                      "spread": np.log(d[f"rel_{s}"].clip(0.05, 20))}, index=d.index)
    X["spread2"] = X.spread ** 2       # production flattens at the extremes (a log-linear fit over-reaches)
    X["min_lvl"] = np.log((base / d[f"r_{s}"].clip(1e-4)).clip(1, 48) / 24)   # minutes level: per-minute output is not constant across roles
    if s in ("pts", "reb", "ast", "fg3m"):
        X["opp"] = np.log(d[f"opp_allow_{s}"].clip(0.7, 1.4)).fillna(0)
        if s in ("reb", "ast"):   # what it allows to HIS position group, beyond its overall level
            X["opp_pos"] = (np.log(d[f"pos_allow_{s}"].clip(0.6, 1.6)) - X["opp"]).fillna(0)
    X["g3in4"] = d.g3in4.fillna(0)
    # play-by-play view: minutes x tonight's opportunities per minute x his share of them,
    # entered as its log ratio to the per-minute base so the fit decides how much it adds
    mins = base / d[f"r_{s}"].clip(1e-4)
    alt = {"reb": mins * d.reb_opps_min * d.sh_reb,
           "ast": mins * d.fgm_opps_min * d.sh_ast,
           "pts": mins * d.opp_poss_min * d.sh_usg * d.pts_per_usg}.get(s)
    if alt is not None:
        X["alt"] = np.log((alt / base).clip(0.25, 4)).fillna(0)
    if s in ("pts", "ast", "fg3m"):     # his own with/without-teammate usage response (play-by-play)
        X["wowy"] = d.wowy_usg.fillna(0).clip(-1, 1)
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
        mu = mu * np.exp(H @ c.params)
        co[s]["_boost"] = fit_boost(d, d[s], pd.Series(np.asarray(mu), index=d.index), "poisson")
    return co


def predict_stats(d: pd.DataFrame, min_hat: pd.Series, co: dict, boost: bool = True) -> pd.DataFrame:
    """boost=False gives the transparent models' means, for the team-total blend: boosted means
    are partly memorised on the training seasons, so a blend fitted on them over-trusts them."""
    out = pd.DataFrame(index=d.index)
    for s in STAT_COLS:
        base = (min_hat * d[f"r_{s}"]).clip(1e-3)
        X = _stat_X(d, s, base)
        mu = base * np.exp(X.to_numpy() @ np.array([co[s][c] for c in X.columns]))
        top = co[s].get("_top")
        if top:
            mu = mu * np.exp(_top_X(mu.to_numpy(), top["knots"]) @ np.array(top["coef"]))
        if boost and co[s].get("_boost") is not None:
            mu = apply_boost(co[s]["_boost"], d, pd.Series(np.asarray(mu), index=d.index), "poisson")
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
    return float(max(c1, 0.5)), float(max(c2, -0.5 * c1 / max(float(np.quantile(mu, 0.99)), 1.0)))


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


# ---------------------------------------------------------------- double- and triple-doubles

# Points, rebounds and assists move together (a big-minutes night lifts all three), so treating
# them as independent understates both: 7.1% predicted vs 8.2% actual double-doubles, 0.31% vs
# 0.51% triple-doubles (backtest 2021-26). A logistic calibration of the independent probability,
# fitted on the walk-forward backtest's out-of-sample projections, corrects it (held-out 2024-26
# log loss: DD 0.1859 -> 0.1842, TD 0.0208 -> 0.0197).
DD_CAL = {"dd": (0.0016, 0.8671), "td": (0.0571, 0.8315)}


def double_triple(p_pts, p_reb, p_ast) -> tuple[np.ndarray, np.ndarray]:
    """P(double-double), P(triple-double) from each stat's P(10+), calibrated."""
    p, r, a = (np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6) for x in (p_pts, p_reb, p_ast))
    ind = {"dd": p * r + p * a + r * a - 2 * p * r * a, "td": p * r * a}
    out = []
    for k in ("dd", "td"):
        q = np.clip(ind[k], 1e-6, 1 - 1e-6)
        c0, c1 = DD_CAL[k]
        out.append(1 / (1 + np.exp(-(c0 + c1 * np.log(q / (1 - q))))))
    return out[0], out[1]
