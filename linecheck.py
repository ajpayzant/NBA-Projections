"""Price one betting line: the Edges tab's "Check a line", and every price edges.py compares.

Each market uses the same distribution the rest of the tool prices with:
    player props   the count model around the projection (model.prob_over), for a game he plays
    game lines     normal margin and total around the projections, on whole-number scores
                   (P(score = k) = the normal's mass between k - 0.5 and k + 0.5)
Whole-number lines can push: the push is refunded, so the edge is measured on the
probability of winning given no push, and the EV counts the push as money back.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

import model


def to_decimal(american: float) -> float:
    a = float(american)
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def prop(mean: float, var_coef, line: float, over: bool) -> tuple[float, float]:
    """(P(win), P(push)) for an over / under on a player stat."""
    gt = float(model.prob_over([mean], var_coef, line)[0])             # P(X > line)
    if float(line).is_integer():
        ge = float(model.prob_over([mean], var_coef, line - 0.5)[0])  # P(X >= line)
        return (gt if over else 1 - ge), ge - gt
    return (gt if over else 1 - gt), 0.0


def _split(mu: float, sd: float, line: float) -> tuple[float, float]:
    """(P(score > line), P(score < line)) for a whole-number score ~ normal(mu, sd)."""
    above = float(stats.norm.sf((np.floor(line) + 0.5 - mu) / sd))
    below = float(stats.norm.cdf((np.ceil(line) - 0.5 - mu) / sd))
    return above, below


def game(margin: float, total: float, sd_margin: float, sd_total: float, market: str, home_side: bool,
         line: float = 0.0, over: bool = True) -> tuple[float, float]:
    """(P(win), P(push)) for a moneyline, spread (line = this team's handicap, e.g. -4.5) or total.
    margin = projected home minus away."""
    if market == "Total":
        a, b = _split(total, sd_total, line)
        return (a if over else b), 1 - a - b
    m = margin if home_side else -margin                  # this team's projected margin
    if market == "Moneyline":                             # no ties in basketball
        return float(stats.norm.cdf(m / sd_margin)), 0.0
    a, b = _split(m, sd_margin, -line)                    # covers when its margin beats -line
    return a, 1 - a - b


def evaluate(p_win: float, p_push: float, odds: float, other_odds: float | None = None) -> dict:
    """Our probability against the book's for one side at its price.
    book_p is the book's no-vig probability when the other side's price is given, else the
    raw implied probability (which still includes the book's margin)."""
    dec = to_decimal(odds)
    p = p_win / max(1 - p_push, 1e-9)                    # given no push
    implied = 1 / dec
    if other_odds is not None:
        o = 1 / to_decimal(other_odds)
        book_p, novig = implied / (implied + o), True
    else:
        book_p, novig = implied, False
    ev = 100 * (p_win * (dec - 1) - (1 - p_win - p_push))  # push = stake back
    return {"p": p, "push": p_push, "book_p": book_p, "novig": novig, "edge": p - book_p, "ev": ev, "dec": dec}
