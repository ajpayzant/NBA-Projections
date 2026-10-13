"""The live slate: upcoming games -> rosters and lineups -> every player's projection.

    python live.py               # fetch finished games, rebuild tables, refit, cache the slate
    python live.py --no-fetch    # refit and re-cache from the lake as it is

The heavy part (pregame features for every roster player, the model fit) is cached in
data/live/state.pkl. The lineup is applied on top at view time (`project`): who is available,
who starts, minutes overrides. Absences, minute redistribution and with/without effects are
recomputed from it, so an injury report, a Rotowire update or a manual edit takes effect at
once, without a refit.

Lineup for a team, in order: a manual override; Rotowire's starting five (Confirmed or
Expected) and its may-not-play list; ESPN's injury statuses; the team's last game.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import config as C
import features
import fetch
import model
import parse
import pbp

LIVE = C.DATA / "live"
HORIZON_DAYS = 3
FIRST_TRAIN = 2019
ET = ZoneInfo("America/New_York")
LINES = {"pts": [x + 0.5 for x in range(4, 40)], "reb": [x + 0.5 for x in range(1, 16)],
         "ast": [x + 0.5 for x in range(0, 14)], "fg3m": [x + 0.5 for x in range(0, 8)],
         "pra": [x + 0.5 for x in range(8, 60)], "pr": [x + 0.5 for x in range(6, 50)],
         "pa": [x + 0.5 for x in range(6, 50)], "ra": [x + 0.5 for x in range(2, 25)]}
DRESS = 14              # players kept available per team (see lineups)
SPARE = "Outside the rotation"   # left out by DRESS: not an absence (no minutes to hand out)
NEW_PLAYER_MIN = 10.0   # usual minutes assumed for a player with no NBA history, for that ranking
STALE_DAYS = 30         # a last game older than this (the offseason) does not set the starters
# chance he plays, from an injury tag (ESPN status or Rotowire's tag)
PLAY_PROB = {"out": 0.0, "doubtful": 0.2, "doubt": 0.2, "questionable": 0.5, "ques": 0.5, "gtd": 0.6,
             "game time decision": 0.6, "day-to-day": 0.75, "probable": 0.9, "prob": 0.9}


def key(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def _surname(name: str) -> str:
    parts = [p for p in str(name).replace(".", " ").split() if key(p) not in {"jr", "sr", "ii", "iii", "iv"}]
    return key(parts[-1]) if parts else ""


# ---------------------------------------------------------------- sources

def schedule(start: date | None = None, days: int = HORIZON_DAYS) -> pd.DataFrame:
    """Games not yet final from `start` (ET) for `days` days, regular season, play-in, playoffs
    and (before opening night) preseason."""
    start = start or datetime.now(ET).date()
    rows = []
    for k in range(days):
        d = start + timedelta(days=k)
        sb = fetch.get_json(f"{C.ESPN}/scoreboard", {"dates": d.strftime("%Y%m%d"), "limit": 50})
        for e in sb.get("events", []):
            comp = e["competitions"][0]
            if e["status"]["type"].get("completed"):
                continue
            side = {c["homeAway"]: c["team"] for c in comp["competitors"]}
            rows.append({"game_id": int(e["id"]), "date": d.isoformat(), "start_utc": e["date"],
                         "season_type": e["season"]["type"], "neutral": bool(comp.get("neutralSite")),
                         "home": side["home"]["abbreviation"], "away": side["away"]["abbreviation"],
                         "home_id": side["home"]["id"], "away_id": side["away"]["id"],
                         "home_name": side["home"]["displayName"], "away_name": side["away"]["displayName"]})
    g = pd.DataFrame(rows)
    if len(g) and (g.season_type != 1).any():   # once real games are on the horizon, skip exhibition ones
        g = g[g.season_type != 1]
    return g.reset_index(drop=True)


def roster(team_id: str, team: str) -> pd.DataFrame:
    """ESPN roster: id, name, position, jersey, headshot, age, injury status."""
    try:
        r = fetch.get_json(f"{C.ESPN}/teams/{team_id}/roster")
    except Exception:
        return pd.DataFrame(columns=["player", "name", "pos", "team"])
    rows = []
    for a in r.get("athletes", []):
        inj = a.get("injuries") or []
        rows.append({"player": int(a["id"]), "name": a.get("displayName"), "pos": (a.get("position") or {}).get("abbreviation"),
                     "jersey": a.get("jersey"), "headshot": (a.get("headshot") or {}).get("href"), "age": a.get("age"),
                     "injury": inj[0].get("status") if inj else "", "team": team})
    return pd.DataFrame(rows)


def injuries() -> pd.DataFrame:
    """ESPN's league-wide injury list: player, team, status and the latest note."""
    try:
        d = fetch.get_json(f"{C.ESPN}/injuries")
    except Exception:
        return pd.DataFrame(columns=["player", "status", "note"])
    rows = []
    for t in d.get("injuries", []):
        for i in t.get("injuries", []):
            a = i.get("athlete") or {}
            if a.get("id") or a.get("links"):
                pid = a.get("id") or re.search(r"/id/(\d+)", json.dumps(a.get("links", ""))).group(1)
                rows.append({"player": int(pid), "status": i.get("status"), "note": i.get("shortComment", ""),
                             "updated": i.get("date")})
    return pd.DataFrame(rows)


def rotowire(day: str | None = None) -> dict[str, dict]:
    """Rotowire's lineups page for `day` (it publishes today and tomorrow): per team, the lineup
    status (Confirmed / Expected), the starting five and the may-not-play list (name, injury
    tag, play likelihood). Empty for any other date, or if the page is for another date."""
    day = day or datetime.now(ET).date().isoformat()
    today = datetime.now(ET).date()
    which = {today.isoformat(): "", (today + timedelta(days=1)).isoformat(): "?date=tomorrow"}.get(day)
    if which is None:
        return {}
    try:
        html = requests.get(C.ROTOWIRE_LINEUPS + which, headers={"User-Agent": C.USER_AGENT},
                            timeout=C.REQUEST_TIMEOUT).text
    except Exception:
        return {}
    shown = re.search(r"Starting lineups for ([A-Z][a-z]+ \d{1,2}, \d{4})", html)
    if not shown or datetime.strptime(shown.group(1), "%B %d, %Y").date().isoformat() != day:
        return {}
    html = re.sub(r"<style.*?</style>|<script.*?</script>", "", html, flags=re.S)
    out = {}
    for block in re.findall(r'<div class="lineup is-nba[^"]*".*?(?=<div class="lineup is-nba|$)', html, re.S):
        abbrs = re.findall(r'lineup__abbr">([A-Z]{2,4})<', block)
        lists = re.findall(r'<ul class="lineup__list is-(visit|home)">(.*?)</ul>', block, re.S)
        for side, body in lists:
            team = abbrs[0] if side == "visit" else (abbrs[1] if len(abbrs) > 1 else None)
            if not team:
                continue
            status = re.search(r'lineup__status[^"]*">.*?</div>\s*([^<]+)<', body, re.S)
            starters_part, _, mnp = body.partition("MAY NOT PLAY")
            starters = re.findall(r'lineup__pos[^>]*>\s*([A-Z]+)\s*</div>\s*<a title="([^"]+)"', starters_part)
            maybe = []
            for li in re.findall(r'<li class="lineup__player([^"]*)"[^>]*>(.*?)</li>', mnp, re.S):
                cls, inner = li
                nm = re.search(r'<a title="([^"]+)"', inner)
                tag = re.search(r'lineup__inj">([^<]*)<', inner)
                pct = re.search(r"is-pct-play-(\d+)", cls)
                if nm:
                    maybe.append({"name": nm.group(1), "tag": (tag.group(1) if tag else "").strip(),
                                  "pct": int(pct.group(1)) / 100 if pct else None})
            out[ESPN_ABBR.get(team, team)] = {"status": (status.group(1).strip() if status else ""),
                                              "starters": [{"pos": p, "name": n} for p, n in starters], "maybe": maybe}
    return out


ESPN_ABBR = {"GSW": "GS", "NYK": "NY", "NOP": "NO", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH", "PHO": "PHX", "BKN": "BKN"}


def match(name: str, pool: pd.DataFrame) -> int | None:
    """ESPN id for a name from another source: full name, then a surname unique on the roster
    with the same first initial."""
    k = key(name)
    hit = pool[pool.name.map(key) == k]
    if len(hit) == 1:
        return int(hit.player.iloc[0])
    sur = pool[pool.name.map(_surname) == _surname(name)]
    sur = sur[sur.name.map(lambda n: key(n)[:1]) == k[:1]]
    return int(sur.player.iloc[0]) if len(sur) == 1 else None


# ---------------------------------------------------------------- lineups

def lineups(state: dict, day: str, use_sources: bool = True) -> dict[str, dict]:
    """Per team playing on `day`: {player: chance he plays}, starters, minutes overrides, source."""
    games = state["games"][state["games"].date == day]
    teams = sorted(set(games.home) | set(games.away))
    rw = rotowire(day) if use_sources else {}
    inj = injuries() if use_sources else pd.DataFrame(columns=["player", "status", "note"])
    status = dict(zip(inj.player, inj.status)) if len(inj) else {}
    out = {}
    for t in teams:
        r = state["rosters"].get(t, pd.DataFrame(columns=["player", "name", "injury"]))
        avail, why = {}, {}
        for p, inj_s in zip(r.player, r.injury):
            st = (status.get(p) or inj_s or "").lower()
            avail[int(p)] = PLAY_PROB.get(st, 1.0)
            why[int(p)] = st.title() if st else ""
        last = state["last_starters"].get(t, [])
        stale = (pd.Timestamp(day) - pd.Timestamp(state.get("last_game", {}).get(t, day))).days > STALE_DAYS
        starters, source = ([], "usual minutes") if stale else ([p for p in last if avail.get(p, 0) >= 0.5][:5], "last game")
        named = set()
        info = rw.get(t)
        if info:
            ids = [match(s["name"], r) for s in info["starters"]]
            ids = [i for i in ids if i is not None]
            if len(ids) >= 4:
                starters, source = ids[:5], f"Rotowire {info['status'] or 'lineup'}".strip()
                for i in ids:
                    avail[i] = max(avail.get(i, 0.0), 0.9)
            named |= set(ids)
            for m in info["maybe"]:
                i = match(m["name"], r)
                if i is None:
                    continue
                named.add(i)
                prob = PLAY_PROB.get(m["tag"].lower(), m["pct"] if m["pct"] is not None else 0.5)
                if m["pct"] is not None:
                    prob = min(prob, m["pct"]) if m["tag"] else m["pct"]
                avail[i], why[i] = prob, m["tag"] or why.get(i, "")
        # fill to five starters by usual minutes if a source gave fewer
        ew = state["usual_min"].get(t, {})
        for p in sorted(avail, key=lambda p: -ew.get(p, 0)):
            if len(starters) >= 5:
                break
            if p not in starters and avail[p] >= 0.5:
                starters.append(p)
        # a team dresses about 13 (the backtest's players: median 13, at most 15), but the ESPN
        # roster lists everyone under contract, two-ways and camp invites included. Beyond
        # DRESS, the players with the least usual minutes are left out (an override can add them)
        cand = [p for p, a in avail.items() if a >= 0.5]
        if len(cand) > DRESS:
            keep = set(starters) | named
            rest = sorted((p for p in cand if p not in keep), key=lambda p: -ew.get(p, NEW_PLAYER_MIN))
            spare = rest[max(DRESS - len(keep & set(cand)), 0):]
            for p in spare:
                avail[p], why[p] = 0.0, SPARE
        out[t] = {"avail": avail, "why": why, "starters": starters, "minutes": {}, "source": source,
                  "rotowire": info["status"] if info else ""}
    for k, v in load_overrides().items():
        d, t = k.split(":")
        if d == day and t in out:
            o = out[t]
            o["avail"].update({int(p): float(x) for p, x in v.get("avail", {}).items()})
            o["starters"] = [int(p) for p in v.get("starters", o["starters"])]
            o["minutes"] = {int(p): float(x) for p, x in v.get("minutes", {}).items()}
            o["source"] = "manual override"
    return out


OVERRIDE_FILE = C.OVERRIDES / "lineups.json"


def load_overrides() -> dict:
    return json.loads(OVERRIDE_FILE.read_text()) if OVERRIDE_FILE.exists() else {}


def save_override(day: str, team: str, lineup: dict | None) -> None:
    o = load_overrides()
    k = f"{day}:{team}"
    if lineup is None:
        o.pop(k, None)
    else:
        o[k] = {"avail": {str(p): x for p, x in lineup["avail"].items()}, "starters": lineup["starters"],
                "minutes": {str(p): x for p, x in lineup.get("minutes", {}).items()}}
    C.OVERRIDES.mkdir(parents=True, exist_ok=True)
    OVERRIDE_FILE.write_text(json.dumps(o, indent=1))


# ---------------------------------------------------------------- projection

def _absences(rows: pd.DataFrame, state: dict, spare: dict[str, set] | None = None) -> pd.DataFrame:
    """Tonight's absence features for the players who will play, from the team's regulars.
    spare: per team, players left out only to keep the team to a dressed list (not absent)."""
    rows = rows.copy()
    spare = spare or {}
    pairs_u, pairs_m = state["pairs_usg"], state["pairs_min"]
    for c in ("miss_min", "miss_usg", "wowy_usg", "wowy_min"):
        rows[c] = 0.0
    rows["missing_ids"] = [[] for _ in range(len(rows))]
    for (gid, team), idx in rows.groupby(["game_id", "team"]).groups.items():
        here = set(rows.loc[idx, "player"]) | spare.get(team, set())
        regs = state["regulars"].get(team, {})
        missing = {p: v for p, v in regs.items() if p not in here}
        mm = sum(v[0] for v in missing.values())
        mu = sum(v[0] * v[1] for v in missing.values())
        ids = [p for p, v in missing.items() if v[0] >= 12]
        rows.loc[idx, "miss_min"], rows.loc[idx, "miss_usg"] = mm, mu
        for i in idx:
            a = int(rows.at[i, "player"])
            rows.at[i, "missing_ids"] = ids
            rows.at[i, "wowy_usg"] = sum(pairs_u.get((a, b), 0.0) for b in ids)
            rows.at[i, "wowy_min"] = sum(pairs_m.get((a, b), 0.0) for b in ids)
    return rows


def project(state: dict, day: str, lu: dict[str, dict]) -> pd.DataFrame:
    """Every player expected to play on `day`: minutes and each stat, with prop probabilities."""
    s = state["slate"][state["slate"].date == pd.Timestamp(day)].copy()
    if s.empty:
        return s
    s["p_avail"] = [lu.get(t, {}).get("avail", {}).get(int(p), 0.0) for t, p in zip(s.team, s.player)]
    s = s[s.p_avail >= 0.5].copy()                              # projected as playing
    s["starter"] = [int(p) in lu.get(t, {}).get("starters", []) for t, p in zip(s.team, s.player)]
    spare = {t: {p for p, w in l.get("why", {}).items() if w == SPARE and l["avail"].get(p, 0) < 0.5}
             for t, l in lu.items()}
    s = _absences(s, state, spare)
    d = model.prepare(s)
    co = state["coefs"]
    d["min_hat"], d["p_play"] = model.predict_minutes(d, co["min"])
    d = _apply_minutes(d, lu)
    p = model.predict_stats(d, d.min_hat, co["stats"])
    d = pd.concat([d, p], axis=1)
    plain = model.predict_stats(d, d.min_hat, co["stats"], boost=False)
    d["pts_plain"] = plain.pts_hat
    probs = {f"p_{st_}_{ln}": model.prob_over(d[f"{st_}_hat"], co["alpha"][st_], ln) for st_ in LINES for ln in LINES[st_]}
    probs["p_dd"], probs["p_td"] = model.double_triple(probs["p_pts_9.5"], probs["p_reb_9.5"], probs["p_ast_9.5"])
    return pd.concat([d, pd.DataFrame(probs, index=d.index)], axis=1)


def _apply_minutes(d: pd.DataFrame, lu: dict) -> pd.DataFrame:
    """Manual minutes: fixed for those players, the rest of the team rescaled to fill the 240."""
    for t, l in lu.items():
        fixed = {p: m for p, m in l.get("minutes", {}).items()}
        if not fixed:
            continue
        idx = d.index[d.team == t]
        is_fixed = d.loc[idx, "player"].isin(list(fixed))
        for i in idx[is_fixed]:
            d.at[i, "min_hat"] = fixed[int(d.at[i, "player"])]
        free = idx[~is_fixed]
        target = model.EXP_MIN - (d.loc[idx[is_fixed], "min_hat"] * d.loc[idx[is_fixed], "p_play"]).sum()
        have = (d.loc[free, "min_hat"] * d.loc[free, "p_play"]).sum()
        if have > 0 and target > 0:
            d.loc[free, "min_hat"] = (d.loc[free, "min_hat"] * target / have).clip(1, 46)
    return d


def game_lines(state: dict, proj: pd.DataFrame) -> pd.DataFrame:
    """Moneyline, spread and total per game from the projections (the backtested blend)."""
    if proj.empty:
        return pd.DataFrame()
    g = model.game_table(proj, proj.pts_plain * proj.p_play)
    out = pd.concat([g, model.predict_scores(g, state["coefs"]["scores"])], axis=1)
    games = state["games"].set_index("game_id")
    out["home"] = out.game_id.map(games.home)
    out["away"] = out.game_id.map(games.away)
    out["sd_margin"] = state["coefs"]["scores"]["margin"]["sd"]
    out["sd_total"] = state["coefs"]["scores"]["total"]["sd"]
    return out


# ---------------------------------------------------------------- build

def refresh_lake(verbose: bool = True) -> None:
    fetch.fetch_season(C.CURRENT_SEASON, verbose)
    if (C.RAW / str(C.CURRENT_SEASON) / "schedule.json").exists():
        parse.build_season(C.CURRENT_SEASON, verbose)
        if list((C.RAW / str(C.CURRENT_SEASON)).glob("*.json.gz")):
            pbp.build_season(C.CURRENT_SEASON, verbose)


def _pending(games: pd.DataFrame, rosters: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    prow, trow = [], []
    for g in games.itertuples():
        for team, opp, home in ((g.home, g.away, True), (g.away, g.home, False)):
            base = {"game_id": g.game_id, "season": C.CURRENT_SEASON, "season_type": 2 if g.season_type == 1 else g.season_type,
                    "date": g.date, "start_utc": g.start_utc, "neutral": g.neutral, "team": team, "opp": opp,
                    "is_home": home, "pending": True}
            trow.append({**base, "pts": np.nan, "opp_pts": np.nan, "poss": np.nan, "periods": 4,
                         **{c: np.nan for c in ("fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "reb", "oreb", "dreb", "ast", "stl",
                                                 "blk", "tov", "pf")}})
            for p in rosters.get(team, pd.DataFrame()).itertuples():
                prow.append({**base, "player": int(p.player), "name": p.name, "pos": p.pos, "starter": False,
                             "dnp": False, "dnp_reason": "", "min": np.nan})
    return pd.DataFrame(prow), pd.DataFrame(trow)


def build_state(fetch_new: bool = True, verbose: bool = True) -> dict:
    if fetch_new:
        refresh_lake(verbose)
    pg, tg = parse.load("player_games"), parse.load("team_games")
    pg, tg = pg[pg.season_type.isin(C.SEASON_TYPES)], tg[tg.season_type.isin(C.SEASON_TYPES)]
    games = schedule()
    teams = sorted(set(games.home) | set(games.away)) if len(games) else []
    ids = dict(zip(games.home, games.home_id)) | dict(zip(games.away, games.away_id))
    rosters = {t: roster(ids[t], t) for t in teams}
    ppend, tpend = _pending(games, rosters) if len(games) else (pd.DataFrame(), pd.DataFrame())
    f = features.build(pd.concat([pg, ppend], ignore_index=True), pd.concat([tg, tpend], ignore_index=True))
    hist = model.prepare(f[~f.pending & (f.season >= FIRST_TRAIN)])
    co_min = model.fit_minutes(hist)
    mh, pp = model.predict_minutes(hist, co_min)
    played = hist["min"] > 0
    co_st = model.fit_stats(hist[played], mh[played])
    p = model.predict_stats(hist, mh, co_st)
    alpha = model.fit_alpha(hist[played], p[played])
    plain = model.predict_stats(hist, mh, co_st, boost=False)
    g = model.game_table(hist, plain.pts_hat * pp)
    h = tg[tg.is_home].set_index("game_id")
    res = pd.DataFrame({"margin": (h.pts - h.opp_pts).reindex(g.game_id).to_numpy(),
                        "total": (h.pts + h.opp_pts).reindex(g.game_id).to_numpy()})
    co_sc = model.fit_scores(g, res.margin.to_numpy(), res.total.to_numpy())
    slate = f[f.pending].reset_index(drop=True)
    # each team's regulars going into tonight (available in any of its last 10 games): usual
    # minutes and usage, from the pending rows (their states as of tonight)
    regulars, usual_min, last_starters, last_game = {}, {}, {}, {}
    done = f[~f.pending]
    for t in teams:
        tgm = done[done.team == t].drop_duplicates("game_id").sort_values("date").game_id.tail(10)
        recent = set(done[done.game_id.isin(tgm) & (done.team == t)].player)
        sl = slate[(slate.team == t)].drop_duplicates("player")
        regulars[t] = {int(r.player): (float(r.ew_min_l), float(r.rel_usg)) for r in sl.itertuples()
                       if int(r.player) in recent and np.isfinite(r.ew_min_l)}
        usual_min[t] = {int(r.player): float(r.ew_min_l) for r in sl.itertuples() if np.isfinite(r.ew_min_l)}
        last = done[(done.team == t)].sort_values("date")
        if len(last):
            lg = last[last.game_id == last.game_id.iloc[-1]]
            last_starters[t] = [int(x) for x in lg[lg.starter].player]
            last_game[t] = str(pd.Timestamp(lg.date.iloc[0]).date())
    # with/without pair effects as of now (the latest state of every pair)
    states_u, states_m = features.LAST_PAIRS["usg"], features.LAST_PAIRS["min"]
    state = {"built_at": datetime.now(ET).isoformat(timespec="minutes"), "games": games, "rosters": rosters,
             "slate": slate, "coefs": {"min": co_min, "stats": co_st, "alpha": alpha, "scores": co_sc},
             "regulars": regulars, "usual_min": usual_min, "last_starters": last_starters, "last_game": last_game,
             "pairs_usg": {k: float(v[1][-1]) for k, v in states_u.items() if len(v[1])},
             "pairs_min": {k: float(v[1][-1]) for k, v in states_m.items() if len(v[1])},
             "history_games": done[["game_id", "date", "season", "season_type", "team", "opp", "is_home", "player", "name",
                                    "pos", "starter", "min", "pts", "reb", "ast", "fg3m", "fga", "fta", "tov", "stl", "blk",
                                    "oreb", "dreb"]][done.season >= C.CURRENT_SEASON - 2]}
    LIVE.mkdir(parents=True, exist_ok=True)
    with open(LIVE / "state.pkl", "wb") as fh:
        pickle.dump(state, fh)
    if verbose:
        print(f"state built: {len(games)} games, {len(slate)} roster players, fitted on {len(hist):,} player-games")
    return state


def load_state() -> dict | None:
    p = LIVE / "state.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    state = build_state(fetch_new=not args.no_fetch)
    for day in sorted(state["games"].date.unique())[:1]:
        lu = lineups(state, day)
        proj = project(state, day, lu)
        cols = ["name", "team", "opp", "starter", "min_hat", "pts_hat", "reb_hat", "ast_hat", "fg3m_hat", "pra_hat"]
        print(f"\n{day}: top projected points")
        print(proj.sort_values("pts_hat", ascending=False)[cols].head(12).round(1).to_string(index=False))
        for t, l in lu.items():
            print(f"  {t}: starters from {l['source']}, {sum(v >= 0.5 for v in l['avail'].values())} available")


if __name__ == "__main__":
    main()
