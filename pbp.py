"""Play-by-play -> who was on the court, and what happened while each player was out there.

    python pbp.py                 # every season -> data/lake/oncourt_<season>.parquet (+ stints)

ESPN's play-by-play logs substitutions ("X enters the game for Y", participants [in, out]) but
not the five on the court at the start of each period. The five start as the box score's
starters; every substitution swaps one player; players simply carry over between periods
(ESPN logs period-break changes as substitutions at the start of the next period). Checked
against the box score's minutes.

Per player-game, totals while he was on the court:
    sec_on                 seconds on the court
    on_fga, on_fta, on_tov team shot attempts, free throws, turnovers (his team)
    on_fgm, on_pts         team made shots and points (for assist and scoring shares)
    on_reb                 rebounds by either team (the rebounds he could have taken)
    on_opp_pts             points allowed
    on_poss                team possessions (FGA - OREB + TOV + 0.44 FTA, while on)
from which features.py builds TRUE usage (his share of team usage events while on the court),
rebound rate (his share of available rebounds), assist rate (his share of teammates' baskets).

stints_<season> keeps every stretch of game with the same ten players (both teams' fives, the
seconds, points and usage events), for with/without-teammate effects.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict

import pandas as pd

import config as C


def _clock(p: dict) -> float:
    """Seconds elapsed in the game at this play."""
    per = p["period"]["number"]
    mm, _, ss = str(p["clock"]["displayValue"]).partition(":")
    left = float(mm) * 60 + float(ss or 0) if ss else float(mm)
    length = 720 if per <= 4 else 300
    start = 720 * min(per - 1, 4) + 300 * max(per - 5, 0)
    return start + length - left


def game_onc(d: dict) -> tuple[list[dict], list[dict], dict]:
    comp = d["header"]["competitions"][0]
    gid = int(d["header"]["id"])
    abbr = {t["team"]["id"]: t["team"]["abbreviation"] for t in comp["competitors"]}
    on: dict[str, set] = {tid: set() for tid in abbr}
    for side in d["boxscore"].get("players", []):
        tid = side["team"]["id"]
        for a in side["statistics"][0].get("athletes", []) if side.get("statistics") else []:
            if a.get("starter") and (a.get("athlete") or {}).get("id"):
                on[tid].add(a["athlete"]["id"])
    tids = list(abbr)
    other = {tids[0]: tids[1], tids[1]: tids[0]}
    acc = defaultdict(lambda: defaultdict(float))           # (tid, player) -> counters
    stints, cur = [], None
    t_prev, bad = 0.0, 0

    def open_stint(t):
        return {"t0": t, "lineup": {tid: tuple(sorted(on[tid])) for tid in tids}, "pu": defaultdict(float),
                **{f"{k}_{tid}": 0.0 for tid in tids for k in ("pts", "fga", "fta", "tov", "oreb")}}

    # ESPN appends late-entered or corrected plays with higher sequence numbers, so sequence order
    # is not game order: sort by game time, then sequence within the same moment
    plays = sorted(d.get("plays", []), key=lambda x: (_clock(x), int(x.get("sequenceNumber", 0))))
    team_of = {}                                             # player -> team id (from the box score)
    for side in d["boxscore"].get("players", []):
        for a in side["statistics"][0].get("athletes", []) if side.get("statistics") else []:
            if (a.get("athlete") or {}).get("id"):
                team_of[a["athlete"]["id"]] = side["team"]["id"]
    period_start = {}                                       # period -> {tid: set of five}
    by_period = defaultdict(list)
    for p in plays:
        by_period[p["period"]["number"]].append(p)
    prev_end = {tid: set(on[tid]) for tid in tids}
    for per in sorted(by_period):
        first, subbed_in = {tid: [] for tid in tids}, set()
        for p in by_period[per]:
            parts = [x["athlete"]["id"] for x in p.get("participants", []) if x.get("athlete")]
            if p["type"]["text"] == "Substitution" and len(parts) >= 2:
                inn, out = parts[0], parts[1]
                tid = team_of.get(out) or (p.get("team") or {}).get("id")
                if out not in subbed_in and tid in first and out not in first[tid]:
                    first[tid].append(out)
                subbed_in.add(inn)
                continue
            for pl in parts:                                 # anyone who does something was on the court
                tid = team_of.get(pl)
                if tid in first and pl not in subbed_in and pl not in first[tid]:
                    first[tid].append(pl)
        start = {}
        for tid in tids:
            five = first[tid][:5]
            for pl in sorted(prev_end[tid]):                 # fill from who ended the last period
                if len(five) >= 5:
                    break
                if pl not in five and pl not in subbed_in:
                    five.append(pl)
            start[tid] = set(five)
        period_start[per] = start
        # end-of-period lineup for the next period's fill: replay this period's substitutions
        cur_on = {tid: set(start[tid]) for tid in tids}
        for p in by_period[per]:
            parts = [x["athlete"]["id"] for x in p.get("participants", []) if x.get("athlete")]
            if p["type"]["text"] == "Substitution" and len(parts) >= 2:
                tid = team_of.get(parts[1]) or (p.get("team") or {}).get("id")
                if tid in cur_on:
                    cur_on[tid].discard(parts[1])
                    cur_on[tid].add(parts[0])
        prev_end = cur_on
    if 1 in period_start:
        on = {tid: set(period_start[1][tid]) for tid in tids}
    cur = open_stint(0.0)
    period = 1
    for p in plays:
        if p["period"]["number"] != period and p["period"]["number"] in period_start:
            period = p["period"]["number"]
            t_new = _clock(p)
            dt = max(0.0, t_new - t_prev)
            for tid in tids:
                for pl in on[tid]:
                    acc[(tid, pl)]["sec_on"] += dt
            t_prev = t_new
            cur["t1"] = t_new
            stints.append(cur)
            on = {tid: set(period_start[period][tid]) for tid in tids}
            cur = open_stint(t_new)
        t = _clock(p)
        dt = max(0.0, t - t_prev)
        for tid in tids:
            for pl in on[tid]:
                acc[(tid, pl)]["sec_on"] += dt
        t_prev = t
        typ = p["type"]["text"]
        tid = (p.get("team") or {}).get("id")
        parts = [x["athlete"]["id"] for x in p.get("participants", []) if x.get("athlete")]
        if typ == "Substitution" and len(parts) >= 2:
            tid = team_of.get(parts[1]) or tid
            if tid not in on:
                continue
            cur["t1"] = t
            stints.append(cur)
            inn, out = parts[0], parts[1]
            if out in on[tid]:
                on[tid].discard(out)
            else:
                bad += 1
            on[tid].add(inn)
            cur = open_stint(t)
            continue
        if tid not in on:
            continue
        opp = other[tid]
        ev = {}
        pa = p.get("pointsAttempted") or 0
        if typ.startswith("Free Throw"):
            ev["fta"] = 1
        elif p.get("shootingPlay") and pa in (2, 3):
            ev["fga"] = 1
            if p.get("scoringPlay"):
                ev["fgm"] = 1
        if "Turnover" in typ:
            ev["tov"] = 1
        if typ == "Offensive Rebound":
            ev["oreb"] = 1
        if "Rebound" in typ:
            for side in tids:
                for pl in on[side]:
                    acc[(side, pl)]["on_reb"] += 1
        if p.get("scoringPlay"):
            ev["pts"] = p.get("scoreValue") or 0
        if parts and any(k in ev for k in ("fga", "fta", "tov")):   # his own usage event in this stint
            cur["pu"][parts[0]] += ev.get("fga", 0) + 0.44 * ev.get("fta", 0) + ev.get("tov", 0)
        for k, v in ev.items():
            for pl in on[tid]:
                acc[(tid, pl)][f"on_{k}"] += v
            if k == "pts":
                for pl in on[opp]:
                    acc[(opp, pl)]["on_opp_pts"] += v
            if f"{k}_{tid}" in cur:
                cur[f"{k}_{tid}"] += v
    cur["t1"] = t_prev
    stints.append(cur)
    rows = []
    for (tid, pl), c in acc.items():
        rows.append({"game_id": gid, "team": abbr[tid], "player": int(pl), **c})
    st_rows = []
    for i, s in enumerate(stints):
        if s["t1"] <= s["t0"]:
            continue
        for tid in tids:
            st_rows.append({"game_id": gid, "team": abbr[tid], "t0": s["t0"], "t1": s["t1"],
                            "lineup": "-".join(s["lineup"][tid]), "opp_lineup": "-".join(s["lineup"][other[tid]]),
                            **{k: s[f"{k}_{tid}"] for k in ("pts", "fga", "fta", "tov", "oreb")},
                            "opp_pts": s[f"pts_{other[tid]}"], "seg": i,
                            "pu": json.dumps({k: round(v, 2) for k, v in s["pu"].items() if k in s["lineup"][tid]})})
    return rows, st_rows, {"game_id": gid, "bad_subs": bad}


def build_season(season: int, verbose: bool = True) -> pd.DataFrame:
    folder = C.RAW / str(season)
    sched = {g["gid"] for g in json.loads((folder / "schedule.json").read_text())}
    R, S, Q = [], [], []
    for f in sorted(folder.glob("*.json.gz")):
        if f.name.split(".")[0] not in sched:
            continue
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                r, s, q = game_onc(json.load(fh))
        except Exception:
            continue
        R += r; S += s; Q.append(q)
    oc = pd.DataFrame(R)
    oc["on_poss"] = oc.on_fga.fillna(0) - oc.on_oreb.fillna(0) + oc.on_tov.fillna(0) + 0.44 * oc.on_fta.fillna(0)
    oc = oc.fillna(0.0)
    oc.to_parquet(C.LAKE / f"onc_{season}.parquet", index=False)
    pd.DataFrame(S).to_parquet(C.LAKE / f"stints_{season}.parquet", index=False)
    if verbose:
        q = pd.DataFrame(Q)
        print(f"{season}: {len(Q)} games, {len(oc):,} player rows, {len(S):,} stint rows, "
              f"games with unmatched subs {(q.bad_subs > 0).mean():.1%}", flush=True)
    return oc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, action="append")
    args = ap.parse_args()
    for yr in args.season or [*C.HISTORY_SEASONS]:
        build_season(yr)
