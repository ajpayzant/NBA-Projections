"""ESPN game summaries (data/raw) -> lake tables (data/lake), one file per season and table.

    python parse.py                 # every season with raw files
    python parse.py --season 2025

    player_games_<season>  one row per player per game he was listed for (DNPs included, with
                           the reason): minutes, points, shooting, rebounds, assists, ...
    team_games_<season>    one row per team per game: box totals, final score, overtimes,
                           possessions (FGA - OREB + TOV + 0.44 FTA)
    injuries_<season>      the injury list ESPN attached to each game (who was out, and why)
"""
from __future__ import annotations

import argparse
import gzip
import json

import pandas as pd

import config as C

PLAYER_KEYS = {"minutes": "min", "points": "pts", "rebounds": "reb", "assists": "ast", "turnovers": "tov",
               "steals": "stl", "blocks": "blk", "offensiveRebounds": "oreb", "defensiveRebounds": "dreb",
               "fouls": "pf", "plusMinus": "pm"}
PAIRS = {"fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
         "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("fg3m", "fg3a"),
         "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta")}
TEAM_KEYS = {"totalRebounds": "reb", "offensiveRebounds": "oreb", "defensiveRebounds": "dreb", "assists": "ast",
             "steals": "stl", "blocks": "blk", "totalTurnovers": "tov", "fouls": "pf", "fastBreakPoints": "fb_pts",
             "pointsInPaint": "paint_pts", "turnoverPoints": "tov_pts"}


def _num(v) -> float:
    try:
        v = str(v).replace("+", "")
        if ":" in v:   # "36:12"
            m, s = v.split(":")
            return int(m) + int(s) / 60
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


def parse_game(d: dict, sched: dict) -> tuple[list, list, list]:
    comp = d["header"]["competitions"][0]
    gid = d["header"]["id"]
    sides = {t["team"]["id"]: t for t in comp["competitors"]}
    abbr = {tid: t["team"]["abbreviation"] for tid, t in sides.items()}
    base = {"game_id": int(gid), "season": C.season_of(sched["date"]), "season_type": sched["type"],
            "date": sched["date"], "start_utc": comp["date"], "neutral": bool(comp.get("neutralSite"))}
    teams, players, injuries = [], [], []
    tstats = {t["team"]["id"]: {s["name"]: s["displayValue"] for s in t["statistics"]} for t in d["boxscore"]["teams"]}
    for tid, t in sides.items():
        oid = next(o for o in sides if o != tid)
        s = tstats.get(tid, {})
        row = {**base, "team": abbr[tid], "opp": abbr[oid], "is_home": t["homeAway"] == "home",
               "pts": _num(t.get("score")), "opp_pts": _num(sides[oid].get("score")),
               "periods": len(t.get("linescores", [])), "won": bool(t.get("winner"))}
        for k, (a, b) in PAIRS.items():
            m, n = (s.get(k, "-").split("-") + ["", ""])[:2]
            row[a], row[b] = _num(m), _num(n)
        row |= {v: _num(s.get(k)) for k, v in TEAM_KEYS.items()}
        teams.append(row)
    for side in d["boxscore"].get("players", []):
        tid = side["team"]["id"]
        oid = next(o for o in sides if o != tid)
        st = side["statistics"][0] if side.get("statistics") else {"keys": [], "athletes": []}
        keys = st.get("keys") or []
        for a in st.get("athletes", []):
            ath = a.get("athlete") or {}
            if not ath.get("id"):   # an occasional placeholder entry with no player behind it
                continue
            row = {**base, "team": abbr[tid], "opp": abbr[oid], "is_home": sides[tid]["homeAway"] == "home",
                   "player": int(ath["id"]), "name": ath.get("displayName"),
                   "pos": (ath.get("position") or {}).get("abbreviation"), "starter": bool(a.get("starter")),
                   "dnp": bool(a.get("didNotPlay")) or not a.get("stats"), "dnp_reason": a.get("reason") or ""}
            vals = dict(zip(keys, a.get("stats") or []))
            for k, v in PLAYER_KEYS.items():
                row[v] = _num(vals.get(k))
            for k, (m_, n_) in PAIRS.items():
                m, n = (str(vals.get(k, "-")).split("-") + ["", ""])[:2]
                row[m_], row[n_] = _num(m), _num(n)
            players.append(row)
    for t in d.get("injuries", []):
        team = t.get("team", {}).get("abbreviation")
        for i in t.get("injuries", []):
            injuries.append({**base, "team": team, "player": int(i["athlete"]["id"]) if i.get("athlete", {}).get("id") else None,
                             "name": i.get("athlete", {}).get("displayName"), "status": i.get("status"),
                             "detail": (i.get("details") or {}).get("type") or i.get("type", {}).get("description")})
    return teams, players, injuries


def build_season(season: int, verbose: bool = True) -> dict[str, pd.DataFrame]:
    folder = C.RAW / str(season)
    sched = {g["gid"]: g for g in json.loads((folder / "schedule.json").read_text())}
    T, P, I = [], [], []
    bad = 0
    for f in sorted(folder.glob("*.json.gz")):
        gid = f.name.split(".")[0]
        if gid not in sched:
            continue
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                t, p, i = parse_game(json.load(fh), sched[gid])
        except Exception:
            bad += 1
            continue
        T += t; P += p; I += i
    out = {"team_games": pd.DataFrame(T), "player_games": pd.DataFrame(P), "injuries": pd.DataFrame(I)}
    tg = out["team_games"]
    if len(tg):
        tg["poss"] = tg.fga - tg.oreb + tg.tov + 0.44 * tg.fta
    C.LAKE.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        v.to_parquet(C.LAKE / f"{k}_{season}.parquet", index=False)
    if verbose:
        print(f"{season}: {len(tg)} team-games, {len(out['player_games'])} player rows, {len(out['injuries'])} injury rows"
              + (f", {bad} unreadable" if bad else ""), flush=True)
    return out


def load(name: str, seasons: list[int] | None = None) -> pd.DataFrame:
    files = sorted(C.LAKE.glob(f"{name}_[0-9][0-9][0-9][0-9].parquet"))
    if seasons:
        files = [f for f in files if int(f.stem[-4:]) in seasons]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, action="append")
    args = ap.parse_args()
    for yr in args.season or sorted(int(p.name) for p in C.RAW.iterdir() if p.is_dir() and (p / "schedule.json").exists()):
        build_season(yr)
