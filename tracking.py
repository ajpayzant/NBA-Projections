"""Results tracking for the Results tab: the model's final projections vs the betting lines.

    python tracking.py snapshot                 # store today's final projections (no API credits)
    python tracking.py grade                    # grade every finished day: opening + closing lines
    python tracking.py backtest 2025-10-21 2026-04-12 [max_credits]
                                                # the walk-forward backtest's projections vs
                                                # historical lines, day by day (cached per day)

Snapshot keeps the last projection of each game before it starts (a later refresh replaces
games not yet started). Grade pulls, from The Odds API's historical data, the opening (10 AM ET)
and closing (10 minutes before tip-off) prices of DraftKings, FanDuel, BetMGM, Caesars and
Fanatics for every game line and prop, prices each with the stored projection exactly as the
Edges tab does, and grades it with the final box score. About 200 credits per game (10 prop
markets x 10 credits x 2 snapshots) plus the game lines.

Only aggregates are kept in the results files (bets, wins, profit and log-loss sums by date,
opening / closing price, market, side and edge bucket): no individual prices, since these files
are published with the app's data. Profit is per 1-unit bet at the best of the five books' prices.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import config as C
import edges
import live
import model
import odds
import parse

TRACK = C.DATA / "tracking"
HIST_DIR = C.DATA / "odds_hist"      # raw historical prices per day (local only, never published)
RESULTS_LIVE = TRACK / "results_live.parquet"
RESULTS_BACKTEST = TRACK / "results_backtest.parquet"
HIST = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba"
OPEN_ET = time(10, 0)
CLOSE_BEFORE = timedelta(minutes=10)
BUCKETS = [(-1.0, 0.0, "< 0"), (0.0, 0.02, "0-2%"), (0.02, 0.05, "2-5%"), (0.05, 0.08, "5-8%"),
           (0.08, 0.12, "8-12%"), (0.12, 1.0, "12%+")]
PRIMARY = {"Over", "Yes", "Favourite", "Favourite -"}   # one side per bet, for accuracy (log loss)
ET = ZoneInfo("America/New_York")
STAT = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast", "player_threes": "fg3m",
        "player_points_rebounds_assists": "pra", "player_points_rebounds": "pr", "player_points_assists": "pa",
        "player_rebounds_assists": "ra"}
# team names for past games (the lake keeps ESPN abbreviations); games match on the nickname
NAMES = {"ATL": "Atlanta Hawks", "BKN": "Brooklyn Nets", "BOS": "Boston Celtics", "CHA": "Charlotte Hornets",
         "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
         "DET": "Detroit Pistons", "GS": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
         "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
         "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NO": "New Orleans Pelicans",
         "NY": "New York Knicks", "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers",
         "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SA": "San Antonio Spurs",
         "TOR": "Toronto Raptors", "UTAH": "Utah Jazz", "WSH": "Washington Wizards"}
CREDITS = {"used": 0}


def _today() -> str:
    return datetime.now(ET).date().isoformat()


def _files(day: str) -> dict:
    return {k: TRACK / f"snap_{day}_{k}.parquet" for k in ("games", "players", "lines")} | \
        {"meta": TRACK / f"snap_{day}_meta.json", "done": TRACK / f"graded_{day}.done"}


PLAYER_COLS = ["game_id", "team", "opp", "player", "name", "min_hat", "p_play"] + \
    [f"{s}_hat" for s in ("pts", "reb", "ast", "fg3m", "pra", "pr", "pa", "ra")] + ["p_dd", "p_td"]
LINE_COLS = ["game_id", "margin_hat", "total_hat", "sd_margin", "sd_total"]


# ---------------------------------------------------------------- snapshot

def snapshot(state: dict | None = None, day: str | None = None, verbose: bool = True) -> None:
    """Store the day's projections; games that have started keep their earlier snapshot."""
    state = state or live.load_state()
    day = day or _today()
    games = state["games"][state["games"].date == day]
    if games.empty:
        if verbose:
            print(f"tracking: no games on {day}")
        return
    p = live.project(state, day, live.lineups(state, day))
    gl = live.game_lines(state, p)
    f = _files(day)
    TRACK.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    started = set(games.game_id[pd.to_datetime(games.start_utc, utc=True) <= now])
    for k, new in (("games", games), ("players", p[PLAYER_COLS]), ("lines", gl[LINE_COLS])):
        if f[k].exists() and started:   # keep what was projected before those games began
            old = pd.read_parquet(f[k])
            new = pd.concat([old[old.game_id.isin(started)], new[~new.game_id.isin(started)]], ignore_index=True)
        new.to_parquet(f[k], index=False)
    f["meta"].write_text(json.dumps({"alpha": state["coefs"]["alpha"], "built_at": state["built_at"],
                                     "snapshot_at": now.isoformat(timespec="minutes")}))
    if verbose:
        print(f"tracking: snapshot {day}: {len(games)} games, {len(p)} players ({len(started)} already started, kept)")


# ---------------------------------------------------------------- historical odds

def _hist(path: str, key: str, **params) -> dict:
    r = requests.get(HIST + path, params={"apiKey": key, "bookmakers": ",".join(odds.BOOKS),
                                          "oddsFormat": "american", **params}, timeout=60)
    r.raise_for_status()
    CREDITS["used"] += int(r.headers.get("x-requests-last") or 0)
    return r.json()


def _open_stamp(day: str) -> str:
    t = datetime.combine(datetime.fromisoformat(day).date(), OPEN_ET, ET).astimezone(timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _odds_for(day: str, games: pd.DataFrame, key: str) -> dict[str, pd.DataFrame]:
    """{'open': rows, 'close': rows} in odds.py's row format, for the day's games. Cached in
    HIST_DIR, so a day is only ever paid for once."""
    cache = {k: HIST_DIR / f"{day}_{k}.parquet" for k in ("open", "close")}
    if all(p.exists() for p in cache.values()):
        return {k: pd.read_parquet(p) for k, p in cache.items()}
    stamp = _open_stamp(day)
    out = {"open": [], "close": []}
    events = _hist("/events", key, date=stamp).get("data", [])
    names = {edges.nick(h) + "|" + edges.nick(a) for h, a in zip(games.home_name, games.away_name)}
    events = [e for e in events if edges.nick(e["home_team"]) + "|" + edges.nick(e["away_team"]) in names]
    snap = _hist("/odds", key, date=stamp, markets=",".join(odds.GAME_MARKETS)).get("data", [])
    out["open"] += [r for ev in snap for r in odds._rows(ev, stamp)]
    for start, grp in pd.Series(events, dtype=object).groupby([e["commence_time"] for e in events]):
        close = (pd.Timestamp(start) - CLOSE_BEFORE).strftime("%Y-%m-%dT%H:%M:%SZ")
        ids = {e["id"] for e in grp}
        snap = _hist("/odds", key, date=close, markets=",".join(odds.GAME_MARKETS)).get("data", [])
        out["close"] += [r for ev in snap if ev["id"] in ids for r in odds._rows(ev, close)]
        for e in grp:
            for tag, when in (("open", stamp), ("close", close)):
                if when > close and tag == "open":   # an early tip-off before 10 AM: no opening snapshot
                    continue
                try:
                    ev = _hist(f"/events/{e['id']}/odds", key, date=when, markets=",".join(odds.PROP_MARKETS)).get("data")
                except requests.HTTPError:
                    continue
                if ev:
                    out[tag] += odds._rows(ev, when)
    res = {k: pd.DataFrame(v) for k, v in out.items()}
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    for k, d in res.items():
        d.to_parquet(cache[k], index=False)
    return res


# ---------------------------------------------------------------- bets and outcomes

def _bets(o: pd.DataFrame, games: pd.DataFrame, pl: pd.DataFrame, gl: pd.DataFrame, alpha: dict) -> pd.DataFrame:
    """One row per bet side: the model's probability, the books' consensus probability, best price."""
    if o.empty:
        return pd.DataFrame()
    o = o[o.book.isin(odds.BOOKS.values())]
    df, _ = edges.compare(o, games, pl, gl, alpha, now=pd.Timestamp("1970-01-01", tz="UTC"))
    if df.empty:
        return df
    df = df[df.book_p.notna()]
    keys = ["game_id", "market", "name", "player", "point"]
    b = df.groupby(keys, dropna=False).agg(p_model=("p_model", "first"), p_mkt=("book_p", "mean"), dec=("dec", "max"),
                                           kind=("kind", "first"), pid=("pid", "first")).reset_index()
    b["side"] = np.where(b.market.isin(["totals"]) | b.market.str.startswith("player_"), b.name,
                         np.where(b.p_mkt >= 0.5, "Favourite", "Underdog"))
    sp = b.market == "spreads"
    b.loc[sp, "side"] = np.where(b.point[sp] < 0, "Favourite -", "Underdog +")
    return b


def _box(seasons: list[int]) -> tuple[dict, pd.DataFrame]:
    tg = parse.load("team_games", seasons)
    score = {(g, t): (p, o) for g, t, p, o in zip(tg.game_id, tg.team, tg.pts, tg.opp_pts)}
    pg = parse.load("player_games", seasons)
    pg = pg[pg["min"] > 0].copy()
    pg = pg.assign(pra=pg.pts + pg.reb + pg.ast, pr=pg.pts + pg.reb, pa=pg.pts + pg.ast, ra=pg.reb + pg.ast)
    tens = (pg[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1)   # how books settle them
    pg["dd"], pg["td"] = (tens >= 2).astype(float), (tens >= 3).astype(float)
    return score, pg.set_index(["game_id", "player"])


def _outcomes(b: pd.DataFrame, games: pd.DataFrame) -> pd.Series:
    """1 = won, 0 = lost, NaN = push / void (did not play) / not found."""
    seasons = sorted({C.season_of(str(d)[:10]) for d in games.date})
    score, pg = _box(seasons)
    gm = games.set_index("game_id")
    out = []
    for r in b.itertuples(index=False):
        g = gm.loc[r.game_id] if r.game_id in gm.index else None
        if r.market in edges.GAME:
            if g is None or (r.game_id, g.home) not in score:
                out.append(np.nan)
                continue
            hp, ap = score[(r.game_id, g.home)]
            if r.market == "totals":
                t = hp + ap
                out.append(np.nan if t == r.point else float((t > r.point) == (r.name == "Over")))
                continue
            home = edges.nick(r.name) == edges.nick(g.home_name)
            margin = (hp - ap) if home else (ap - hp)
            v = margin if r.market == "h2h" else margin + r.point
            out.append(np.nan if v == 0 else float(v > 0))
            continue
        k = (r.game_id, int(r.pid)) if pd.notna(r.pid) else None
        if k is None or k not in pg.index:
            out.append(np.nan)
            continue
        if r.market in edges.YES_NO:
            v = pg.at[k, "dd" if r.market == "player_double_double" else "td"]
            out.append(float(v == 1) if r.name == "Yes" else float(v == 0))
            continue
        v = pg.at[k, STAT[r.market]]
        out.append(np.nan if v == r.point else float((v > r.point) == (r.name == "Over")))
    return pd.Series(out, index=b.index)


def aggregate(b: pd.DataFrame, source: str) -> pd.DataFrame:
    """Sums by date, snapshot, market, side and edge bucket (no individual prices)."""
    b = b.dropna(subset=["won", "p_model", "p_mkt", "dec"]).copy()
    b["edge"] = b.p_model - b.p_mkt
    b["bucket"] = pd.cut(b.edge, [x[0] for x in BUCKETS] + [1.0], labels=[x[2] for x in BUCKETS], right=False)
    b["pnl"] = np.where(b.won == 1, b.dec - 1, -1.0)
    pm, pk = np.clip(b.p_model, 1e-6, 1 - 1e-6), np.clip(b.p_mkt, 1e-6, 1 - 1e-6)
    b["ll_model"] = -(b.won * np.log(pm) + (1 - b.won) * np.log(1 - pm))
    b["ll_mkt"] = -(b.won * np.log(pk) + (1 - b.won) * np.log(1 - pk))
    b["primary"] = b.side.isin(PRIMARY)
    a = b.groupby(["date", "snap", "kind", "side", "bucket", "primary"], observed=True).agg(
        bets=("won", "size"), wins=("won", "sum"), pnl=("pnl", "sum"), ll_model=("ll_model", "sum"),
        ll_mkt=("ll_mkt", "sum")).reset_index()
    a["bucket"] = a.bucket.astype(str)
    return a.assign(source=source)


def _grade_day(day: str, games: pd.DataFrame, pl: pd.DataFrame, gl: pd.DataFrame, alpha: dict, key: str) -> pd.DataFrame:
    """Every bet side of the day, priced at the open and the close, with its result."""
    parts = []
    for snap, rows in _odds_for(day, games, key).items():
        b = _bets(rows, games, pl, gl, alpha)
        if len(b):
            b["won"] = _outcomes(b, games)
            parts.append(b.assign(snap=snap, date=day))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def grade(day: str | None = None, verbose: bool = True) -> None:
    """Grade every snapshot day before today that is not graded yet (or just `day`)."""
    key = odds.api_key()
    if not key:
        if verbose:
            print("tracking: no ODDS_API_KEY, grading skipped")
        return
    days = [day] if day else sorted({f.name[5:15] for f in TRACK.glob("snap_*_games.parquet")})
    for d in days:
        f = _files(d)
        if f["done"].exists() or d >= _today() or not f["games"].exists():
            continue
        games, pl, gl = (pd.read_parquet(f[k]) for k in ("games", "players", "lines"))
        alpha = json.loads(f["meta"].read_text())["alpha"]
        b = _grade_day(d, games, pl, gl, alpha, key)
        if b.empty:
            continue
        agg = aggregate(b, "live")
        old = pd.read_parquet(RESULTS_LIVE) if RESULTS_LIVE.exists() else pd.DataFrame()
        if len(old):
            old = old[old.date != d]
        pd.concat([old, agg], ignore_index=True).to_parquet(RESULTS_LIVE, index=False)
        f["done"].write_text(datetime.now(timezone.utc).isoformat(timespec="minutes"))
        if verbose:
            print(f"tracking: graded {d}: {int(agg.bets.sum()):,} bet sides, profit {agg.pnl.sum():+.1f} units")


# ---------------------------------------------------------------- historical backtest

def _backtest_day(day: str, B: pd.DataFrame, G: pd.DataFrame, fits: pd.DataFrame) -> tuple:
    """games, players, lines and alpha for one past day, from the walk-forward backtest."""
    pl = B[B.date == day].copy()
    season = int(pl.fit_season.iloc[0])
    fit = fits.loc[season]
    alpha = {s: list(fit[f"alpha_{s}"]) for s in ("pts", "reb", "ast", "fg3m", "pra", "pr", "pa", "ra")}
    p10 = {s: model.prob_over(pl[f"{s}_hat"], alpha[s], 9.5) for s in ("pts", "reb", "ast")}
    pl["p_dd"], pl["p_td"] = model.double_triple(p10["pts"], p10["reb"], p10["ast"])
    home = pl[pl.is_home].drop_duplicates("game_id").set_index("game_id")
    games = pd.DataFrame({"game_id": home.index, "date": day, "home": home.team.to_numpy(), "away": home.opp.to_numpy()})
    games = games[games.home.isin(NAMES) & games.away.isin(NAMES)]
    games["home_name"], games["away_name"] = games.home.map(NAMES), games.away.map(NAMES)
    gl = G[G.game_id.isin(games.game_id)].assign(sd_margin=fit.sd_margin, sd_total=fit.sd_total)
    return games, pl, gl[LINE_COLS], alpha


def backtest(start: str, end: str, max_credits: int = 50_000, verbose: bool = True) -> None:
    """Grade the backtest's projections against historical lines for regular-season days in
    [start, end], stopping before max_credits are spent. Writes RESULTS_BACKTEST from every
    day graded so far (cached days are free)."""
    key = odds.api_key()
    B = pd.read_parquet(C.DATA / "backtest" / "backtest.parquet")
    B = B[B.season_type == 2].copy()
    B["date"] = pd.to_datetime(B.date).dt.strftime("%Y-%m-%d")
    G = pd.read_parquet(C.DATA / "backtest" / "games.parquet")
    fits = pd.read_parquet(C.DATA / "backtest" / "fits.parquet").set_index("season")
    days = sorted(d for d in B.date.unique() if start <= d <= end)
    bets_file = HIST_DIR / "bets.parquet"
    have = pd.read_parquet(bets_file) if bets_file.exists() else pd.DataFrame()
    CREDITS["used"] = 0
    for d in days:
        cached = all((HIST_DIR / f"{d}_{k}.parquet").exists() for k in ("open", "close"))
        if not cached and CREDITS["used"] >= max_credits:
            if verbose:
                print(f"tracking: credit budget reached before {d}")
            break
        games, pl, gl, alpha = _backtest_day(d, B, G, fits)
        b = _grade_day(d, games, pl, gl, alpha, key)
        if len(have):
            have = have[have.date != d]
        have = pd.concat([have, b], ignore_index=True)
        if verbose:
            print(f"tracking: {d}: {len(b):,} bet sides ({'cached' if cached else 'fetched'}, "
                  f"{CREDITS['used']:,} credits so far)", flush=True)
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    have.to_parquet(bets_file, index=False)          # individual bets: local only
    TRACK.mkdir(parents=True, exist_ok=True)
    aggregate(have, "backtest").to_parquet(RESULTS_BACKTEST, index=False)
    if verbose:
        print(f"tracking: backtest results from {len(have):,} bet sides -> {RESULTS_BACKTEST.name}")


if __name__ == "__main__":
    try:
        cmd = sys.argv[1]
        if cmd == "backtest":
            backtest(sys.argv[2], sys.argv[3], *(int(x) for x in sys.argv[4:5]))
        else:
            {"snapshot": snapshot, "grade": grade}[cmd]()
    except Exception as e:   # never fail the refresh over tracking
        print(f"tracking: {sys.argv[1:]} failed: {e!r}", file=sys.stderr)
        if "backtest" in sys.argv:
            raise
