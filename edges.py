"""Compare sportsbook prices (odds.py) with the model's probabilities, for the Edges tab.

Nothing here changes a projection: it reads the same projections the other tabs show and
prices each book's line with the model's own distributions (linecheck.py)
    player props     the count model around his projection, for a game he plays (books void
                     a prop when he doesn't)
    double / triple  the calibrated double- and triple-double probabilities
    game lines       normal margin and total around the projected score

For each bet: the book's price, its no-vig probability (from that book's two sides; books
post double- and triple-doubles as Yes only, so for those it is the implied probability with
the margin left in), the market consensus, the model's probability, the edge (model - book)
and the expected value of a $1 bet at that book's price.
"""
from __future__ import annotations

import difflib

import numpy as np
import pandas as pd

import linecheck
from live import _surname, key

PROPS = {"player_points": ("PTS", "pts"), "player_rebounds": ("REB", "reb"), "player_assists": ("AST", "ast"),
         "player_threes": ("3PM", "fg3m"), "player_points_rebounds_assists": ("PRA", "pra"),
         "player_points_rebounds": ("P+R", "pr"), "player_points_assists": ("P+A", "pa"),
         "player_rebounds_assists": ("R+A", "ra")}
YES_NO = {"player_double_double": ("Double-double", "p_dd"), "player_triple_double": ("Triple-double", "p_td")}
GAME = ("h2h", "spreads", "totals")
KIND = {"h2h": "Moneyline", "spreads": "Spread", "totals": "Total", **{k: v[0] for k, v in PROPS.items()},
        **{k: v[0] for k, v in YES_NO.items()}}
WORDS = {"PTS": "points", "REB": "rebounds", "AST": "assists", "3PM": "threes", "PRA": "pts + reb + ast",
         "P+R": "pts + reb", "P+A": "pts + ast", "R+A": "reb + ast"}


def decimal(price: pd.Series) -> pd.Series:
    price = price.astype(float)
    return np.where(price > 0, 1 + price / 100, 1 + 100 / price.abs())


def _devig(o: pd.DataFrame) -> pd.Series:
    """No-vig probability of each outcome from its book's two sides of the same bet."""
    imp = 1 / o.dec
    grp = [o.event_id, o.book, o.market, o.player.fillna(""), o.pair]
    tot = imp.groupby(grp).transform("sum")
    n = imp.groupby(grp).transform("size")
    return (imp / tot).where(n == 2)


def nick(name: str) -> str:
    """A team's nickname, the part every source agrees on ('LA Clippers' / 'Los Angeles Clippers')."""
    n = str(name)
    return key("Trail Blazers") if n.endswith("Trail Blazers") else key(n.split()[-1])


def match_games(odds: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Attach our game_id by team nicknames."""
    gid = {nick(h) + "|" + nick(a): g for h, a, g in zip(games.home_name, games.away_name, games.game_id)}
    return odds.assign(game_id=[gid.get(nick(h) + "|" + nick(a)) for h, a in zip(odds.home, odds.away)])


def _plain(name: str) -> str:
    """Name key without suffixes (Jr., III), which books and ESPN don't always agree on."""
    return key(" ".join(p for p in str(name).replace(".", " ").split()
                        if key(p) not in {"jr", "sr", "ii", "iii", "iv"}))


def _same_first(a: str, b: str) -> bool:
    """First names that are the same person spelled two ways: same initial (Nic / Nicolas,
    Herb / Herbert) or a close transliteration."""
    fa, fb = key(str(a).split()[0]), key(str(b).split()[0])
    return fa[:1] == fb[:1] or difflib.SequenceMatcher(None, fa, fb).ratio() >= 0.75


def compare(odds: pd.DataFrame, games: pd.DataFrame, proj: pd.DataFrame, lines: pd.DataFrame, alpha: dict,
            now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """One row per (bet, book) with the model's probability. Also returns match counts.
    proj: player projections (live.project); lines: game lines (live.game_lines)."""
    now = now or pd.Timestamp.now(tz="UTC")
    o = match_games(odds, games)
    o = o[o.game_id.notna() & (pd.to_datetime(o.commence_time, utc=True) > now)].copy()   # pregame only
    o["game_id"] = o.game_id.astype(int)
    o = o[o.game_id.isin(proj.game_id.unique())]
    stats = {"prices": len(o)}
    if o.empty:
        return pd.DataFrame(), stats
    o["dec"] = decimal(o.price)
    # the two sides of a bet: over/under at one point, yes/no, or two teams (spreads by |point|)
    o["pair"] = np.where(o.market == "spreads", o.point.abs(), o.point.fillna(0.0))
    o["novig"] = _devig(o)
    yes_only = o.market.isin(list(YES_NO)) & o.novig.isna()
    o["book_p"] = o.novig.where(~yes_only, 1 / o.dec)
    o["one_sided"] = yes_only
    for c in ("p_model", "p_push", "proj", "proj_own", "proj_opp", "pid"):
        o[c] = np.nan
    o["team"] = None
    gm = games.set_index("game_id")
    gl = lines.set_index("game_id") if len(lines) else pd.DataFrame()

    # ---- game lines
    for gid, idx in o[o.market.isin(GAME)].groupby("game_id").groups.items():
        if gid not in gl.index:
            continue
        g, ln = gm.loc[gid], gl.loc[gid]
        pts = {True: (ln.total_hat + ln.margin_hat) / 2, False: (ln.total_hat - ln.margin_hat) / 2}
        for i in idx:
            r = o.loc[i]
            home = nick(r["name"]) == nick(g.home_name)
            mk = {"h2h": "Moneyline", "spreads": "Spread", "totals": "Total"}[r.market]
            win, push = linecheck.game(ln.margin_hat, ln.total_hat, ln.sd_margin, ln.sd_total, mk, home,
                                       0.0 if r.market == "h2h" else float(r.point), str(r["name"]).lower() == "over")
            o.at[i, "p_model"], o.at[i, "p_push"] = win / max(1 - push, 1e-9), push
            if r.market == "totals":
                o.at[i, "proj"] = ln.total_hat
            else:
                o.at[i, "proj_own"], o.at[i, "proj_opp"] = pts[home], pts[not home]

    # ---- player props
    pr = proj.assign(k=proj["name"].map(_plain), sur=proj["name"].map(_surname))
    look = {(gid, k): row for gid, k, row in zip(pr.game_id, pr.k, pr.itertuples(index=False))}
    once = pr.groupby(["game_id", "sur"]).sur.transform("size") == 1   # a surname unique in that game
    by_sur = {(gid, s): row for gid, s, row, u in zip(pr.game_id, pr.sur, pr.itertuples(index=False), once) if u}
    sk = o[o.market.isin(list(PROPS) + list(YES_NO))]
    matched = 0
    for i, r in sk.iterrows():
        row = look.get((r.game_id, _plain(r.player)))
        if row is None:
            row = by_sur.get((r.game_id, _surname(r.player)))
            if row is not None and not _same_first(row.name, r.player):
                row = None
        if row is None:
            continue
        side = str(r["name"]).lower()
        if r.market in YES_NO:
            p = float(getattr(row, YES_NO[r.market][1]))
            o.at[i, "p_model"] = p if side == "yes" else 1 - p
        else:
            if r.point is None or not np.isfinite(r.point):
                continue
            s = PROPS[r.market][1]
            win, push = linecheck.prop(getattr(row, f"{s}_hat"), alpha[s], float(r.point), side == "over")
            o.at[i, "p_model"], o.at[i, "p_push"] = win / max(1 - push, 1e-9), push
            o.at[i, "proj"] = getattr(row, f"{s}_hat")
        o.at[i, "team"] = row.team
        o.at[i, "pid"] = row.player
        matched += 1
    stats["player props"], stats["player props matched"] = len(sk), matched

    o = o[o.p_model.notna()].copy()
    o["p_push"] = o.p_push.fillna(0.0)
    o["kind"] = o.market.map(KIND)
    o["bet"] = [_label(r) for r in o.itertuples(index=False)]
    o["game"] = [f"{gm.loc[g].away} @ {gm.loc[g].home}" for g in o.game_id]
    o["date"] = [gm.loc[g].date for g in o.game_id]
    o["market_p"] = o.groupby(["game_id", "market", "bet"]).book_p.transform("mean")
    o["books"] = o.groupby(["game_id", "market", "bet"]).book.transform("nunique")
    o["edge"] = o.p_model - o.book_p          # vs this book, so Edge = Our % - Book % as shown
    # EV per $1, a push refunded: p_model is given no push, so scale back to the full probability
    o["ev"] =o.p_model * (1 - o.p_push) * (o.dec - 1) - (1 - o.p_model) * (1 - o.p_push)
    abbr = {nick(n): a for n, a in zip(games.home_name, games.home)} | \
        {nick(n): a for n, a in zip(games.away_name, games.away)}
    o["subject"] = [r.player if r.market not in GAME
                    else ("Game total" if r.market == "totals" else abbr.get(nick(r.name), r.name))
                    for r in o.itertuples(index=False)]
    o["line"] = [_line(r) for r in o.itertuples(index=False)]
    o["proj_text"] = [_proj_text(r) for r in o.itertuples(index=False)]
    return o, stats


def _line(r) -> str:
    """'Over 24.5 points', 'Double-double', 'Moneyline', 'Spread -4.5', 'Under 228.5 points'."""
    pt = f"{r.point:g}" if r.point is not None and np.isfinite(r.point) else ""
    if r.market == "h2h":
        return "Moneyline"
    if r.market == "spreads":
        return f"Spread {'+' if r.point > 0 else ''}{pt}"
    if r.market in YES_NO:
        return f"{YES_NO[r.market][0]}: {r.name}"
    side = "Over" if str(r.name).lower() == "over" else "Under"
    return f"{side} {pt} {'points' if r.market == 'totals' else WORDS[KIND[r.market]]}"


def _proj_text(r) -> str:
    """The model's own number for the bet: '24.3 points', '227.1 total', '114.2–109.8'."""
    if r.market in ("h2h", "spreads"):
        return f"{r.proj_own:.1f}–{r.proj_opp:.1f}" if np.isfinite(r.proj_own) else ""
    if r.market in YES_NO or not np.isfinite(r.proj):
        return ""
    if r.market == "totals":
        return f"{r.proj:.1f} total"
    return f"{r.proj:.{2 if r.market == 'player_threes' else 1}f} {WORDS[KIND[r.market]]}"


def _label(r) -> str:
    pt = f"{r.point:g}" if r.point is not None and np.isfinite(r.point) else ""
    if r.market == "h2h":
        return f"{r.name} ML"
    if r.market == "spreads":
        return f"{r.name} {'+' if r.point > 0 else ''}{pt}"
    if r.market == "totals":
        return f"{r.name} {pt}"
    if r.market in YES_NO:
        return f"{r.player} {KIND[r.market]} {r.name}"
    side = "o" if str(r.name).lower() == "over" else "u"
    return f"{r.player} {side}{pt} {KIND[r.market]}"
