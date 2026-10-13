"""NBA player game projections: Slate · Game · Lineups & minutes · Player · How it works.

    run_app.bat            # http://localhost:8680
"""
from __future__ import annotations

import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

import config as C
import live

st.set_page_config(page_title="NBA Player Projections", layout="wide")
PROJ, ACT, MUTED = "#2a78d6", "#eb6834", "#8a8984"   # same palette as the NHL tool
STATS = [("PTS", "pts"), ("REB", "reb"), ("AST", "ast"), ("3PM", "fg3m"), ("PRA", "pra"), ("P+R", "pr"), ("P+A", "pa"),
         ("R+A", "ra")]


# ---------------------------------------------------------------- data

@st.cache_resource
def _state(mtime: float) -> dict | None:
    return live.load_state()


def state() -> dict | None:
    p = live.LIVE / "state.pkl"
    return _state(p.stat().st_mtime) if p.exists() else None


@st.cache_data(ttl=900, show_spinner="Reading Rotowire lineups and ESPN injuries...")
def _sources(day: str, built_at: str) -> dict:
    lu = live.lineups(state(), day, use_sources=True)
    return json.loads(json.dumps({t: {**v, "avail": {str(k): x for k, x in v["avail"].items()},
                                      "why": {str(k): x for k, x in v["why"].items()}} for t, v in lu.items()}))


def lineups(day: str) -> dict:
    """Rotowire / ESPN lineups (cached 15 min) with today's manual overrides on top."""
    raw = _sources(day, state()["built_at"])
    lu = {t: {**v, "avail": {int(k): x for k, x in v["avail"].items()}, "why": {int(k): x for k, x in v["why"].items()},
              "starters": [int(x) for x in v["starters"]], "minutes": {int(k): x for k, x in v.get("minutes", {}).items()}}
          for t, v in raw.items()}
    for k, v in live.load_overrides().items():
        d, t = k.split(":")
        if d == day and t in lu:
            lu[t]["avail"].update({int(p): float(x) for p, x in v.get("avail", {}).items()})
            lu[t]["starters"] = [int(p) for p in v.get("starters", lu[t]["starters"])]
            lu[t]["minutes"] = {int(p): float(x) for p, x in v.get("minutes", {}).items()}
            lu[t]["source"] = "manual override"
    return lu


@st.cache_data(show_spinner="Projecting...", max_entries=16)
def _project(day: str, built_at: str, lu_key: str) -> pd.DataFrame:
    return live.project(state(), day, lineups(day))


def projections(day: str) -> pd.DataFrame:
    lu = lineups(day)
    return _project(day, state()["built_at"], json.dumps(lu, sort_keys=True, default=str))


def games_on(day: str) -> pd.DataFrame:
    return state()["games"][state()["games"].date == day]


def roster_info(team: str) -> pd.DataFrame:
    r = state()["rosters"].get(team)
    return r if r is not None else pd.DataFrame(columns=["player", "name", "pos", "jersey", "headshot", "age", "injury"])


def when(ts) -> str:
    try:
        return pd.Timestamp(ts).tz_convert("America/New_York").strftime("%a %b %d · %I:%M %p ET")
    except Exception:
        return str(ts or "")


def no_state():
    st.warning("No projections cached yet. Run `python live.py` (or refresh.bat) first.")
    st.stop()


def american(p) -> str:
    """Fair (no-vig) American odds for probability p: -150 means risk 150 to win 100."""
    if p is None or not np.isfinite(p) or p <= 0 or p >= 1:
        return "—"
    p = min(max(p, 0.005), 0.995)
    o = -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p
    o = 5 * round(o / 5)
    return "EVEN" if abs(o) <= 100 else f"{o:+d}"


def odds_mode() -> bool:
    return st.session_state.get("prob_view", "%") == "Fair odds"


def odds_toggle():
    st.sidebar.segmented_control("Probabilities", ["%", "Fair odds"], default="%", key="prob_view",
                                 help="Fair odds = the no-vig American price implied by the model's probability.")


def pick_day(s: dict) -> str:
    days = sorted(s["games"].date.unique())
    if not days:
        st.info("No upcoming games in the next few days.")
        st.stop()
    return st.sidebar.selectbox("Date", days, key="day")


def pick_game(day: str):
    g = games_on(day)
    lab = {r.game_id: f"{r.away} @ {r.home}" for r in g.itertuples()}
    gid = st.sidebar.selectbox("Game", list(lab), format_func=lab.get, key=f"game_{day}")
    return g[g.game_id == gid].iloc[0]


def fair_line(row, s: str) -> float:
    """The x.5 line closest to a coin flip for this player and stat."""
    lines = live.LINES[s]
    return min(lines, key=lambda ln: abs(row[f"p_{s}_{ln}"] - 0.5))


# ---------------------------------------------------------------- tables

# prop columns, in display order: (header, probability column). Shown as a % bar, or as the
# fair (no-vig) American price for the over when the sidebar is set to Fair odds (as in the NHL tool)
PROP_COLS = [*[(f"o{x:g} PTS", f"p_pts_{x}") for x in (9.5, 14.5, 19.5, 24.5, 29.5)],
             *[(f"o{x:g} REB", f"p_reb_{x}") for x in (4.5, 7.5, 9.5, 11.5)],
             *[(f"o{x:g} AST", f"p_ast_{x}") for x in (3.5, 5.5, 7.5, 9.5)],
             *[(f"o{x:g} 3PM", f"p_fg3m_{x}") for x in (1.5, 2.5, 3.5)],
             ("Double-double", "p_dd"), ("Triple-double", "p_td")]
PROP_NAMES = [h for h, _ in PROP_COLS]


def player_table(p: pd.DataFrame, with_team: bool = True) -> pd.DataFrame:
    t = p.sort_values(["team", "starter", "min_hat"], ascending=[True, False, False])
    out = pd.DataFrame({
        "Team": t.team, "Player": t["name"], "Pos": t.pos.fillna(""), "Start": np.where(t.starter, "S", ""),
        "MIN": t.min_hat, "PTS": t.pts_hat, "REB": t.reb_hat, "AST": t.ast_hat, "3PM": t.fg3m_hat, "PRA": t.pra_hat,
        **{h: t[c] for h, c in PROP_COLS}})
    return out if with_team else out.drop(columns=["Team"])


def show(t: pd.DataFrame, height="auto"):
    t = t.copy()
    pcols = [c for c in t.columns if c in PROP_NAMES]
    if odds_mode():  # same columns, shown as the fair price for the over
        for c in pcols:
            t[c] = t[c].map(american)
        pcols = []
    else:
        t[pcols] = (t[pcols] * 100).round()
    cfg = {c: st.column_config.NumberColumn(format="%.1f", width="small") for c in ("MIN", "PTS", "REB", "AST", "PRA")}
    cfg["3PM"] = st.column_config.NumberColumn(format="%.2f", width="small")
    cfg |= {c: st.column_config.TextColumn(width="small", pinned=c == "Team") for c in ("Team", "Pos", "Start")}
    cfg["Player"] = st.column_config.TextColumn("Player", pinned=True)
    cfg |= {c: st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100, width="small") for c in pcols}
    st.dataframe(t, hide_index=True, column_config=cfg, width="stretch", height=height)


# ---------------------------------------------------------------- game lines and the board

def game_lines(day: str) -> pd.DataFrame:
    p = projections(day)
    return live.game_lines(state(), p) if len(p) else pd.DataFrame()


BOARD_CSS = """<style>
.gb{font-family:"Source Sans Pro","Segoe UI",sans-serif;max-width:860px;border:1px solid #e5e4df;border-radius:10px;padding:10px 12px 8px;
    background:#fff;color:#0b0b0b}
.gb .bar{display:flex;height:6px;border-radius:3px;overflow:hidden;margin:2px 0 4px}
.gb .bar span{display:block}
.gb .barlab{display:flex;justify-content:space-between;font-size:11px;color:#52514e;margin-bottom:6px}
.gb table{width:100%;border-collapse:separate;border-spacing:4px 4px}
.gb th{font-size:11px;font-weight:600;color:#8a8984;text-align:center;padding:0 4px;text-transform:uppercase;letter-spacing:.03em}
.gb th.t{text-align:left}
.gb td{text-align:center;border-radius:7px;padding:4px 6px;background:#f6f6f4;line-height:1.15}
.gb td.t{text-align:left;background:none;padding-left:0;white-space:nowrap}
.gb td.fav{background:rgba(42,120,214,.10)}
.gb .o{font-weight:700;font-size:15px}.gb .p{font-size:11px;color:#52514e}
.gb .tm{font-weight:700;font-size:15px}.gb .xg{font-size:11px;color:#52514e}
.gb.sm{padding:8px 9px 6px}.gb.sm .o,.gb.sm .tm{font-size:13px}.gb.sm td{padding:3px 4px}
.gb .ttl{font-size:12px;font-weight:600;color:#52514e;margin-bottom:2px}
.gb .ar{display:inline-flex;flex-direction:column;margin-left:5px;vertical-align:middle;gap:2px}
.gb .ar b{display:block;font-size:8px;line-height:8px;height:8px;font-weight:400;color:#8a8984;cursor:pointer;
    user-select:none;padding:0 2px}
.gb .ar b:hover{color:#2a78d6}
</style>"""


def _cell(p: float, fav: bool) -> str:
    return f"<td class='{'fav' if fav else ''}'><div class='o'>{american(p)}</div><div class='p'>{p:.0%}</div></td>"


def _step_cell(bid: str, key: str, label: str, p: float, fav: bool, title: str) -> str:
    return (f"<td id='{bid}-{key}' class='{'fav' if fav else ''}'><div class='o'><span class='tl'>{label}</span> "
            f"<span class='to'>{american(p)}</span><span class='ar' data-g='{key[0]}' title='{title}'>"
            f"<b data-d='1'>&#9650;</b><b data-d='-1'>&#9660;</b></span></div><div class='p'>{p:.0%}</div></td>")


def _step_rows(pairs: list) -> list:
    return [[la, american(pa), f"{pa:.0%}", bool(fa), lb, american(pb), f"{pb:.0%}", bool(fb)]
            for la, pa, fa, lb, pb, fb in pairs]


def _step_script(bid: str, grp: str, keys: tuple[str, str], rows: list, start: int) -> str:
    """Steps one market of a board through its alternate lines in the browser (no rerun)."""
    return f"""<script>(function(){{
const L={json.dumps(rows)}, id="{bid}"; let i={start};
const b=document.getElementById(id); if(!b) return;
const a=document.getElementById(id+"-{keys[0]}"), c=document.getElementById(id+"-{keys[1]}");
const put=(td,l,od,pc,fav)=>{{td.querySelector(".tl").textContent=l;td.querySelector(".to").textContent=od;
  td.querySelector(".p").textContent=pc;td.classList.toggle("fav",fav);}};
b.querySelectorAll(".ar[data-g='{grp}'] b").forEach(x=>x.addEventListener("click",()=>{{
  i=Math.max(0,Math.min(L.length-1,i+Number(x.dataset.d)));const r=L[i];
  put(a,r[0],r[1],r[2],r[3]);put(c,r[4],r[5],r[6],r[7]);}}));
}})();</script>"""


def board_html(g, team_pts: dict, small: bool = False, title: str = "") -> str:
    """Sportsbook-style card: win-probability bar, then away / home rows with fair moneyline,
    spread and total. Margin and total are normal around the projections (the backtested
    error spreads); the arrows step the spread and the total through alternate lines."""
    bid = f"gb{g.game_id}{'s' if small else ''}"
    ph = g.p_home
    pa = 1 - ph
    m, sdm, t, sdt = g.margin_hat, g.sd_margin, g.total_hat, g.sd_total
    home_fav = m >= 0
    fav_m = abs(m)
    # spreads: the favourite -k (covers if he wins by more than k), x.5 lines around the fair one
    k0 = max(0.5, np.floor(fav_m) + 0.5)
    ks = [k for k in np.arange(k0 - 6, k0 + 7, 1.0) if k > 0]
    cover = {k: float(stats.norm.sf((k - fav_m) / sdm)) for k in ks}
    start_k = min(ks, key=lambda k: abs(cover[k] - 0.5))
    t0 = np.floor(t) + 0.5
    ts = list(np.arange(t0 - 12, t0 + 13, 1.0))
    over = {x: float(stats.norm.sf((x - t) / sdt)) for x in ts}
    start_t = min(ts, key=lambda x: abs(over[x] - 0.5))
    rows = ""
    for side, team, p_win in (("away", g.away, pa), ("home", g.home, ph)):
        fav = (side == "home") == home_fav
        c = cover[start_k]
        sp = _step_cell(bid, "sf", f"-{start_k:g}", c, c >= 0.5, "Alternate spreads") if fav else \
            _step_cell(bid, "sd", f"+{start_k:g}", 1 - c, 1 - c > 0.5, "Alternate spreads")
        o = over[start_t]
        tot = _step_cell(bid, "to", f"o{start_t:g}", o, o >= 0.5, "Alternate totals") if side == "away" else \
            _step_cell(bid, "tu", f"u{start_t:g}", 1 - o, o < 0.5, "Alternate totals")
        rows += (f"<tr><td class='t'><span class='tm'>{team}</span> <span class='xg'>{team_pts[team]:.1f} pts</span></td>"
                 f"{_cell(p_win, p_win >= 0.5)}{sp}{tot}</tr>")
    bar = (f"<div class='bar'><span style='width:{pa:.1%};background:#eb6834'></span>"
           f"<span style='width:{ph:.1%};background:#2a78d6'></span></div>"
           f"<div class='barlab'><span>{g.away} {pa:.0%}</span><span>win probability</span><span>{ph:.0%} {g.home}</span></div>")
    ttl = f"<div class='ttl'>{title}</div>" if title else ""
    return (f"<div class='gb{' sm' if small else ''}' id='{bid}'>{ttl}{bar}<table><tr><th class='t'></th><th>Moneyline</th>"
            f"<th>Spread</th><th>Total</th></tr>{rows}</table></div>"
            + _step_script(bid, "t", ("to", "tu"), _step_rows(
                [(f"o{x:g}", over[x], over[x] >= 0.5, f"u{x:g}", 1 - over[x], over[x] < 0.5) for x in ts]), ts.index(start_t))
            + _step_script(bid, "s", ("sf", "sd"), _step_rows(
                [(f"-{k:g}", cover[k], cover[k] >= 0.5, f"+{k:g}", 1 - cover[k], 1 - cover[k] > 0.5) for k in ks]),
                ks.index(start_k)))


def team_points(g) -> dict:
    """Each side's projected points from the blended margin and total."""
    return {g.home: (g.total_hat + g.margin_hat) / 2, g.away: (g.total_hat - g.margin_hat) / 2}


# ---------------------------------------------------------------- Slate

def page_slate():
    s = state() or no_state()
    day = pick_day(s)
    odds_toggle()
    st.title(f"Slate · {day}")
    p = projections(day)
    if p.empty:
        st.info("Nothing to project for this date.")
        return
    lu = lineups(day)
    with st.expander("Lineup sources and injuries"):
        rows = []
        for t, l in sorted(lu.items()):
            r = roster_info(t)
            nm = dict(zip(r.player, r.name))
            out = [f"{nm.get(pl, pl)} ({l['why'].get(pl) or 'out'})" for pl, x in l["avail"].items() if x < 0.5]
            q = [f"{nm.get(pl, pl)} ({l['why'].get(pl)})" for pl, x in l["avail"].items() if 0.5 <= x < 1 and l["why"].get(pl)]
            rows.append({"Team": t, "Starters from": l["source"], "Starters": ", ".join(nm.get(x, str(x)) for x in l["starters"]),
                         "Out": ", ".join(out), "Questionable / probable": ", ".join(q)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    gl = game_lines(day)
    with st.expander("Game lines (fair moneyline, spread, total)"):
        games = games_on(day).set_index("game_id")
        cards = [board_html(r, team_points(r), small=True, title=f"{r.away} @ {r.home} · {when(games.loc[r.game_id].start_utc)}")
                 for r in gl.itertuples()]
        for i in range(0, len(cards), 3):
            cols = st.columns(3)
            for c, html in zip(cols, cards[i:i + 3]):
                with c:
                    st.html(BOARD_CSS + html, unsafe_allow_javascript=True)
        st.caption("Fair = no-vig prices from the projected margin and total. Blue = the more likely side. "
                   "The arrows on the spread and total step through alternate lines.")
    c1, c2, c3 = st.columns([2, 2, 1])
    teams = c1.multiselect("Teams", sorted(p.team.unique()))
    pos = c2.multiselect("Position", ["G", "F", "C"])
    sort = c3.selectbox("Sort by", ["PTS", "REB", "AST", "3PM", "PRA", "MIN"] + PROP_NAMES)
    q = p[p.team.isin(teams)] if teams else p
    q = q[q.grp.isin(pos)] if pos else q
    t = player_table(q).sort_values(sort, ascending=False)
    st.caption(f"{p.game_id.nunique()} games · {len(t)} players projected to play · built {s['built_at']}")
    show(t, height=720)
    st.caption("Projections and probabilities are for a player who plays (books void a prop otherwise). "
               "Every player's full prop ladder, at any line, is on the Player page.")
    st.download_button("Download CSV", t.to_csv(index=False), f"nba_projections_{day}.csv", "text/csv")


# ---------------------------------------------------------------- Game

def team_panel(team: str, name: str, p: pd.DataFrame, l: dict, pts: float):
    tp = p[p.team == team]
    st.subheader(name)
    st.caption(f"Starters from {l['source']} · {len(tp)} projected to play")
    a, b, c = st.columns(3)
    a.metric("Points", f"{pts:.1f}", border=True)
    b.metric("Rebounds", f"{(tp.reb_hat * tp.p_play).sum():.1f}", border=True)
    c.metric("Assists", f"{(tp.ast_hat * tp.p_play).sum():.1f}", border=True)
    a, b, c = st.columns(3)
    a.metric("3-pointers", f"{(tp.fg3m_hat * tp.p_play).sum():.1f}", border=True)
    b.metric("Pace", f"{tp.lg_pace.iloc[0] * tp.pace_rel.iloc[0] * tp.opp_pace_rel.iloc[0]:.1f}", border=True,
             help="Projected possessions in the game")
    out = [roster_info(team).set_index("player").name.get(pl, pl) for pl, x in l["avail"].items() if x < 0.5]
    c.metric("Out", f"{len(out)}", border=True, help=", ".join(map(str, out)) or None)
    top = tp.nlargest(3, "pts_hat")
    st.caption("Top projected: " + " · ".join(f"**{n}** {v:.1f} pts" for n, v in zip(top["name"], top.pts_hat)))


def page_game():
    s = state() or no_state()
    day = pick_day(s)
    g = pick_game(day)
    odds_toggle()
    p = projections(day)
    p = p[p.game_id == g.game_id]
    lu = lineups(day)
    st.title(f"{g.away_name} @ {g.home_name}")
    st.caption(when(g.start_utc))
    gl = game_lines(day)
    row = gl[gl.game_id == g.game_id]
    pts = team_points(row.iloc[0]) if len(row) else {g.home: np.nan, g.away: np.nan}
    left, right = st.columns(2, gap="large")
    with left:
        team_panel(g.away, g.away_name, p, lu[g.away], pts[g.away])
    with right:
        team_panel(g.home, g.home_name, p, lu[g.home], pts[g.home])
    if len(row):
        st.html(BOARD_CSS + board_html(row.iloc[0], pts), unsafe_allow_javascript=True)
        st.caption("Fair (no-vig) prices from the projected margin and total. The arrows on the spread and total step "
                   "through alternate lines.")
    st.divider()
    c1, c2 = st.columns([2, 3])
    view = c1.segmented_control("Team", [g.away, g.home], default=g.away, key=f"gv_{g.game_id}") or g.away
    stat = c2.segmented_control("Chart", ["PTS", "REB", "AST", "3PM", "PRA", "MIN"], default="PTS", key="gstat") or "PTS"
    t = player_table(p[p.team == view], with_team=False)
    fit = lambda df: 35 * (len(df) + 1) + 3
    st.markdown("**Starters**")
    sx = t[t.Start == "S"]
    show(sx, height=fit(sx))
    st.markdown("**Bench**")
    bx = t[t.Start != "S"]
    show(bx, height=fit(bx))
    st.markdown(f"**Projected {stat}, {view}**")
    ch = t[["Player", "MIN", "PTS"]].assign(val=t[stat].to_numpy())
    bar = alt.Chart(ch).mark_bar(color=PROJ, cornerRadiusEnd=4, height=16).encode(
        y=alt.Y("Player:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=200)),
        x=alt.X("val:Q", title=f"Projected {stat}"),
        tooltip=["Player", alt.Tooltip("MIN:Q", format=".1f"), alt.Tooltip("val:Q", format=".1f", title=stat)])
    text = bar.mark_text(align="left", dx=4, fontSize=11, color="#52514e").encode(text=alt.Text("val:Q", format=".1f"))
    st.altair_chart((bar + text).properties(height=22 * len(ch) + 40), width="stretch")


# ---------------------------------------------------------------- Lineups & minutes

def page_lineups():
    s = state() or no_state()
    day = pick_day(s)
    teams = sorted(set(games_on(day).home) | set(games_on(day).away))
    team = st.sidebar.selectbox("Team", teams, key=f"lt_{day}")
    st.title(f"Lineups & minutes · {team}")
    lu = lineups(day)
    l = lu[team]
    p = projections(day)
    proj_min = dict(zip(p[p.team == team].player, p[p.team == team].min_hat))
    r = roster_info(team)
    usual = s["usual_min"].get(team, {})
    st.caption(f"Starters from **{l['source']}**. Edit who plays, who starts and anyone's minutes; Save updates every "
               "projection for this team (minutes for the others are rescaled to fill the 240).")
    ed = pd.DataFrame({
        "player": r.player.astype(int), "Player": r.name, "Pos": r.pos.fillna(""),
        "Status": [l["why"].get(int(x), "") for x in r.player],
        "Plays": [l["avail"].get(int(x), 0.0) >= 0.5 for x in r.player],
        "Starts": [int(x) in l["starters"] for x in r.player],
        "Minutes": [l["minutes"].get(int(x), np.nan) for x in r.player],
        "Projected": [proj_min.get(int(x), np.nan) for x in r.player],
        "Usual": [usual.get(int(x), np.nan) for x in r.player]}).sort_values(["Starts", "Usual"], ascending=False)
    out = st.data_editor(ed, hide_index=True, width="stretch", height=35 * (len(ed) + 1) + 3, key=f"ed_{day}_{team}",
                         disabled=["Player", "Pos", "Status", "Projected", "Usual"],
                         column_config={"player": None,
                                        "Plays": st.column_config.CheckboxColumn(help="Untick if he will not play"),
                                        "Starts": st.column_config.CheckboxColumn(help="Five starters"),
                                        "Minutes": st.column_config.NumberColumn(format="%.0f", min_value=0, max_value=48,
                                                                                help="Leave blank for the model's minutes"),
                                        "Projected": st.column_config.NumberColumn(format="%.1f", help="Model minutes now"),
                                        "Usual": st.column_config.NumberColumn(format="%.1f", help="His recent average")})
    a, b, _ = st.columns([1, 1, 4])
    n_start = int(out.Starts.sum())
    if a.button("Save", type="primary", disabled=n_start != 5, help=None if n_start == 5 else "Pick exactly five starters"):
        new = {"avail": {int(pl): (max(l["avail"].get(int(pl), 1.0), 0.9) if ok else 0.0) for pl, ok in zip(out.player, out.Plays)},
               "starters": [int(pl) for pl, x in zip(out.player, out.Starts) if x],
               "minutes": {int(pl): float(m) for pl, m in zip(out.player, out.Minutes) if pd.notna(m)}}
        live.save_override(day, team, new)
        _project.clear()
        st.rerun()
    if b.button("Reset to sources"):
        live.save_override(day, team, None)
        _project.clear()
        st.rerun()
    if n_start != 5:
        st.warning(f"{n_start} starters ticked; a lineup needs five.")


# ---------------------------------------------------------------- Player

LADDER = {"pts": 6, "reb": 4, "ast": 4, "fg3m": 3, "pra": 8, "pr": 6, "pa": 6, "ra": 4}


def page_player():
    s = state() or no_state()
    day = pick_day(s)
    odds_toggle()
    p = projections(day)
    if p.empty:
        st.stop()
    p = p.sort_values(["team", "pts_hat"], ascending=[True, False])
    lab = {i: f"{r.name} · {r.team}" for i, r in zip(p.index, p.itertuples())}
    keys = list(lab)
    i = st.sidebar.selectbox("Player", keys, index=keys.index(p.pts_hat.idxmax()), format_func=lab.get, key=f"pp_{day}")
    r = p.loc[i]
    h = s["history_games"]
    h = h[(h.player == r.player) & (h["min"] > 0)].sort_values("date")
    h = h.assign(pra=h.pts + h.reb + h.ast, pr=h.pts + h.reb, pa=h.pts + h.ast, ra=h.reb + h.ast)
    info = roster_info(r.team)
    me = info[info.player == r.player]
    me = me.iloc[0] if len(me) else None
    l = lineups(day)[r.team]

    c0, c1, c2 = st.columns([1, 4, 4])
    if me is not None and isinstance(me.get("headshot"), str):
        c0.image(me["headshot"], width=110)
    bits = [f"#{me['jersey']}" if me is not None and me.get("jersey") else None, r.team, r.pos,
            f"age {me['age']}" if me is not None and pd.notna(me.get("age")) else None]
    c1.title(r["name"])
    c1.markdown(" · ".join(str(b) for b in bits if b))
    status = l["why"].get(int(r.player), "")
    c2.markdown(f"**Tonight** {'vs' if r.is_home else '@'} {r.opp} · {day}  \n"
                f"**Role** {'starter' if r.starter else 'bench'} ({l['source']})"
                f"{'  \n**Status** ' + status if status else ''}  \n"
                f"**Teammates out** {len(r.missing_ids) if isinstance(r.missing_ids, list) else 0} rotation players")

    st.markdown("#### Tonight's projection")
    cols = st.columns(6)
    for c, (t, v) in zip(cols, [("Minutes", r.min_hat), ("Points", r.pts_hat), ("Rebounds", r.reb_hat),
                                ("Assists", r.ast_hat), ("3-pointers", r.fg3m_hat), ("PRA", r.pra_hat)]):
        c.metric(t, f"{v:.1f}", border=True)
    st.caption(f"Double-double **{r.p_dd:.1%}** (fair {american(r.p_dd)}) · Triple-double **{r.p_td:.1%}** "
               f"(fair {american(r.p_td)}) · for a game he plays")
    cur = h[h.season == C.CURRENT_SEASON]
    last = h[(h.season == C.CURRENT_SEASON - 1) & (h.season_type == 2)]
    wins = (("Last 10", h.tail(10)), ("This season", cur), ("Last season", last), ("Last 3 seasons", h[h.season_type == 2]))
    names = [("Minutes", "min", r.min_hat), ("Points", "pts", r.pts_hat), ("Rebounds", "reb", r.reb_hat),
             ("Assists", "ast", r.ast_hat), ("3-pointers", "fg3m", r.fg3m_hat), ("PRA", "pra", r.pra_hat)]
    comp = pd.DataFrame({"Tonight": [v for *_, v in names],
                         **{f"{wl} ({len(w)} gp)": [w[c].mean() if len(w) else np.nan for _, c, _ in names] for wl, w in wins}},
                        index=[n for n, *_ in names])
    st.dataframe(comp.map(lambda v: "—" if pd.isna(v) else f"{v:.1f}"), width="stretch")

    st.markdown("#### Prop lines and hit rates")
    stat_lab = st.segmented_control("Stat", [lab for lab, _ in STATS], default="PTS", key="pl_stat") or "PTS"
    sk = dict(STATS)[stat_lab]
    fl = fair_line(r, sk)
    lines = [ln for ln in live.LINES[sk] if abs(ln - fl) <= LADDER[sk]]
    windows = [("Last 5", h.tail(5)), ("Last 10", h.tail(10)), ("Last 20", h.tail(20)), ("This season", cur),
               ("Last season", last), (f"vs {r.opp}", h[h.opp == r.opp])]
    rows = []
    for ln in lines:
        pr = float(r[f"p_{sk}_{ln}"])
        row = {"Line": f"{stat_lab} o{ln:g}", "Model": pr, "Over": american(pr), "Under": american(1 - pr)}
        for wl, w in windows:
            row[wl] = (w[sk] > ln).mean() if len(w) else np.nan
        rows.append(row)
    hr = pd.DataFrame(rows)
    pc = [c for c in hr.columns if c not in ("Line", "Over", "Under")]
    if odds_mode():
        for c in pc:
            hr[c] = hr[c].map(american)
        cfg = {c: st.column_config.TextColumn(c, width="small") for c in pc}
    else:
        hr[pc] = (hr[pc] * 100).round()
        cfg = {c: st.column_config.ProgressColumn(c, format="%d%%", min_value=0, max_value=100) for c in pc}
    cfg |= {"Over": st.column_config.TextColumn("Fair over", width="small"),
            "Under": st.column_config.TextColumn("Fair under", width="small")}
    st.dataframe(hr, hide_index=True, width="stretch", column_config=cfg, height=35 * (len(hr) + 1) + 3)
    st.caption(f"Fair line, the one closest to a coin flip: **{stat_lab} {fl:g}**. Probabilities are for a game he plays. "
               "Games in each window: " + " · ".join(f"{wl} {len(w)}" for wl, w in windows) + ".")

    st.markdown("#### Game log")
    c1, c2, c3 = st.columns([3, 2, 1.3])
    lg_stat = c1.segmented_control("Stat", ["PTS", "REB", "AST", "3PM", "PRA", "MIN"], default=stat_lab if stat_lab in
                                   ("PTS", "REB", "AST", "3PM", "PRA") else "PTS", key="lg_stat") or "PTS"
    n = c2.segmented_control("Games", [10, 20, 40, 82], default=20, key="lg_n") or 20
    col = {"PTS": "pts", "REB": "reb", "AST": "ast", "3PM": "fg3m", "PRA": "pra", "MIN": "min"}[lg_stat]
    proj_v = float(r["min_hat"] if col == "min" else r[f"{col}_hat"])
    line = c3.number_input("Line", value=float(fair_line(r, col)) if col != "min" else float(round(proj_v) + 0.5),
                           step=1.0, key=f"lg_line_{col}")
    log = h.tail(n).copy()
    if len(log):
        log["value"] = log[col]
        log["game"] = pd.to_datetime(log.date).dt.strftime("%m-%d") + np.where(log.is_home, " v ", " @ ") + log.opp
        log["result"] = np.where(log.value > line, f"over {line:g}", f"under {line:g}")
        bars = alt.Chart(log).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("game:N", sort=None, title=None, axis=alt.Axis(labelAngle=-45)), y=alt.Y("value:Q", title=lg_stat),
            color=alt.Color("result:N", scale=alt.Scale(domain=[f"over {line:g}", f"under {line:g}"], range=[PROJ, "#c3c2b7"]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("game:N", title="Game"), alt.Tooltip("value:Q", title=lg_stat), alt.Tooltip("min:Q", title="MIN")])
        rule = alt.Chart(pd.DataFrame({"y": [line]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
        pr_ = alt.Chart(pd.DataFrame({"y": [proj_v]})).mark_rule(color=ACT, strokeWidth=2).encode(y="y:Q")
        st.altair_chart((bars + rule + pr_).properties(height=280), width="stretch")
        st.caption(f"Over {line:g} in {(log.value > line).mean():.0%} of his last {len(log)} games. "
                   f"Orange line: tonight's projection ({proj_v:.1f}). Dashed: the line.")
    else:
        st.info("No NBA games in the data for him yet.")
    t1, t2, t3 = st.tabs(["Recent games", "Season by season", "Splits"])
    with t1:
        rec = h.tail(15).iloc[::-1]
        st.dataframe(pd.DataFrame({"Date": pd.to_datetime(rec.date).dt.strftime("%Y-%m-%d"),
                                   "Opp": np.where(rec.is_home, "vs ", "@ ") + rec.opp, "Start": np.where(rec.starter, "S", ""),
                                   "MIN": rec["min"], "PTS": rec.pts, "REB": rec.reb, "AST": rec.ast, "3PM": rec.fg3m,
                                   "STL": rec.stl, "BLK": rec.blk, "TOV": rec.tov}), hide_index=True, width="stretch")
    with t2:
        reg = h[h.season_type == 2].groupby("season")
        ss = pd.DataFrame({"GP": reg.size(), "MIN": reg["min"].mean(), "PTS": reg.pts.mean(), "REB": reg.reb.mean(),
                           "AST": reg.ast.mean(), "3PM": reg.fg3m.mean(), "PRA": reg.pra.mean()})
        ss.index = [f"{y}-{str(y + 1)[2:]}" for y in ss.index]
        st.dataframe(ss.iloc[::-1].style.format("{:.1f}", subset=["MIN", "PTS", "REB", "AST", "3PM", "PRA"]), width="stretch")
    with t3:
        splits(h, r, s)


def splits(h: pd.DataFrame, r, s: dict):
    """Home / away, starter / bench, rest days and career vs tonight's opponent: per-game averages
    and how often he went over tonight's fair points line."""
    recent = h[h.season >= C.CURRENT_SEASON - 1]
    allg = s["history_games"].drop_duplicates(["team", "date"])[["team", "date"]].sort_values("date")
    allg["rest"] = allg.groupby("team").date.transform(lambda x: pd.to_datetime(x).diff().dt.days)
    rest = dict(zip(zip(allg.team, allg.date), allg.rest))
    recent = recent.assign(rest=[rest.get((t, d)) for t, d in zip(recent.team, recent.date)])
    fl = fair_line(r, "pts")
    line_col = f"o{fl:g} PTS"

    def split(df, by, label):
        g = df.groupby(by)
        out = pd.DataFrame({"GP": g.size(), "MIN": g["min"].mean(), "PTS": g.pts.mean(), "REB": g.reb.mean(),
                            "AST": g.ast.mean(), "3PM": g.fg3m.mean(), "PRA": g.pra.mean(),
                            line_col: g.pts.apply(lambda x: (x > fl).mean())})
        out.index = [f"{label}: {k}" for k in out.index]
        return out
    rest_lab = np.select([recent.rest == 1, recent.rest == 2], ["back-to-back", "1 day"], "2+ days")
    parts = [split(recent.assign(k=np.where(recent.is_home, "home", "away")), "k", "Venue"),
             split(recent.assign(k=np.where(recent.starter, "starter", "bench")), "k", "Role"),
             split(recent.assign(k=rest_lab), "k", "Rest")]
    vs = h[h.opp == r.opp]
    if len(vs):
        parts.append(split(vs.assign(k=r.opp), "k", "Career vs"))
    t = pd.concat([x for x in parts if len(x)]) if any(len(x) for x in parts) else pd.DataFrame()
    if t.empty:
        st.info("No games yet.")
        return
    st.dataframe(t, width="stretch", column_config={
        **{c: st.column_config.NumberColumn(format="%.1f") for c in ("MIN", "PTS", "REB", "AST", "PRA")},
        "3PM": st.column_config.NumberColumn(format="%.2f"),
        line_col: st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1,
                                                  help="How often he went over tonight's fair points line")})
    first = pd.to_datetime(s["history_games"].date).min()
    st.caption(f"Venue, role and rest splits cover last season and this season; 'Career vs' covers every game in the data "
               f"(since {first:%Y}). {line_col} = tonight's fair points line.")


# ---------------------------------------------------------------- How it works

def page_model():
    doc = C.ROOT / "HOW_IT_WORKS.md"
    st.markdown(doc.read_text(encoding="utf-8") if doc.exists() else "")


_start = os.environ.get("APP_PAGE", "slate")  # headless tests pick the page to open
PAGES = [("slate", page_slate, "Slate"), ("game", page_game, "Game"), ("lineups", page_lineups, "Lineups & minutes"),
         ("player", page_player, "Player"), ("model", page_model, "How it works")]
st.navigation([st.Page(f, title=t, url_path=k, default=_start == k) for k, f, t in PAGES]).run()
