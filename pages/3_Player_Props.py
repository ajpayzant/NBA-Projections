"""Page 3 — Player Props"""
from __future__ import annotations
import sys
from pathlib import Path

_PAGES_DIR = Path(__file__).resolve().parent
_ROOT      = _PAGES_DIR.parent
for _p in [str(_ROOT), str(_PAGES_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _engine_state import (
    SHARED_CSS, get_engine, init_session, team_name, team_color, fmt_odds,
)

st.set_page_config(page_title="Player Props · NBA", page_icon="🏀", layout="wide")
init_session()
st.markdown(SHARED_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
.prop-card {
    background:rgba(15,23,42,.6);
    border:1px solid rgba(148,163,184,.12);
    border-radius:8px;
    padding:12px 16px;
    margin-bottom:10px;
}
.prop-player { font-size:.95rem;font-weight:700;color:#f1f5f9; }
.prop-team   { font-size:.72rem;color:#94a3b8; }
.prop-stat   { font-size:.78rem;font-weight:700;color:#7dd3fc;
               text-transform:uppercase;letter-spacing:.05em; }
.prop-line   { font-size:1.4rem;font-weight:800;color:#f1f5f9; }
.prop-proj   { font-size:.82rem;color:#94a3b8; }
.edge-pos    { color:#34d399;font-weight:700; }
.edge-neg    { color:#f87171;font-weight:700; }
.edge-neu    { color:#94a3b8; }
.pct-bar-bg  { background:rgba(148,163,184,.15);border-radius:4px;height:6px;margin:4px 0; }
.pct-bar-fg  { border-radius:4px;height:6px; }
</style>""", unsafe_allow_html=True)

engine = get_engine()
result = st.session_state.get("last_result")
if result is None:
    st.warning("No projection loaded. Go to **Projections** first.")
    st.stop()

home_id = result.home_team
away_id = result.away_team

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    team_filter = st.multiselect(
        "Team", [team_name(away_id), team_name(home_id)],
        default=[team_name(away_id), team_name(home_id)], key="pp_team",
    )
    stat_filter = st.multiselect(
        "Stats",
        ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA","PR","PA","RA","MIN"],
        default=["PTS","REB","AST","FG3M","STL","BLK"],
        key="pp_stat",
    )
    pos_filter = st.multiselect(
        "Position", ["G","F","C","UNK"], default=["G","F","C"], key="pp_pos",
    )
    min_min = st.slider("Min projected minutes", 0.0, 40.0, 10.0, 1.0, key="pp_minmin")
    sort_by = st.selectbox("Sort players by",
                           ["Projected PTS","Projected MIN","Alphabetical"],
                           key="pp_sort")

    st.markdown("---")
    st.markdown("### Market comparison")
    mkt_player = st.text_input("Player name (partial)", key="pp_mkt_player")
    mkt_stat   = st.selectbox("Stat", ["PTS","REB","AST","FG3M","STL","BLK","PRA"],
                               key="pp_mkt_stat")
    mkt_line   = st.number_input("Market line", 0.0, 100.0, step=0.5, key="pp_mkt_line")
    mkt_odds   = st.number_input("Market over odds", -999, 999, -110, key="pp_mkt_odds")

    view_mode = st.radio("View mode", ["Cards", "Table"], key="pp_view", horizontal=True)

st.title("🎯 Player Props")
st.markdown(f"**{team_name(away_id)} @ {team_name(home_id)}**")

# ── Market edge banner ────────────────────────────────────────────────────────
if mkt_player and mkt_line > 0:
    all_players = {p.player_id: p for p in result.home_players + result.away_players}
    all_sims    = result.home_player_sims + result.away_player_sims
    for sim in all_sims:
        p = all_players.get(sim.player_id)
        if p and mkt_player.lower() in p.player_name.lower() and mkt_stat in sim.stat_distributions:
            dist     = sim.stat_distributions[mkt_stat]
            fair_p   = float(np.mean(dist > mkt_line))
            mkt_impl = (abs(mkt_odds)/(abs(mkt_odds)+100) if mkt_odds < 0
                        else 100/(mkt_odds+100))
            edge     = fair_p - mkt_impl
            ec       = "#34d399" if edge > 0.03 else ("#f87171" if edge < -0.03 else "#94a3b8")
            st.markdown(
                f'<div style="background:rgba(30,41,59,.7);border:1px solid {ec};'
                f'border-radius:8px;padding:12px 18px;margin-bottom:16px;">'
                f'<b style="color:#f1f5f9;">{p.player_name} — {mkt_stat} O{mkt_line}</b>&nbsp;&nbsp;'
                f'Model: <b>{fair_p*100:.1f}%</b> &nbsp;|&nbsp; '
                f'Market: <b>{mkt_impl*100:.1f}%</b> &nbsp;|&nbsp; '
                f'<span style="color:{ec};font-weight:700;">Edge: {edge*100:+.1f}%</span>'
                f'</div>', unsafe_allow_html=True,
            )
            break

# ── Collect and filter players ────────────────────────────────────────────────
all_players  = {p.player_id: p for p in result.home_players + result.away_players}
all_sims     = result.home_player_sims + result.away_player_sims
sim_map      = {s.player_id: s for s in all_sims}

filtered = []
for pid, p in all_players.items():
    if not p.active:
        continue
    if team_name(p.team_abbr) not in team_filter:
        continue
    if p.pos_group not in pos_filter:
        continue
    if p.proj_min < min_min:
        continue
    filtered.append(p)

if sort_by == "Projected PTS":
    filtered.sort(key=lambda x: -x.proj_pts)
elif sort_by == "Projected MIN":
    filtered.sort(key=lambda x: -x.proj_min)
else:
    filtered.sort(key=lambda x: x.player_name)

if not filtered:
    st.info("No players match current filters.")
    st.stop()

STAT_LABELS = {
    "PTS":"Points","REB":"Rebounds","AST":"Assists","FG3M":"3-Pt Made",
    "STL":"Steals","BLK":"Blocks","TOV":"Turnovers",
    "PRA":"Pts+Reb+Ast","PR":"Pts+Reb","PA":"Pts+Ast","RA":"Reb+Ast","MIN":"Minutes",
}

# ── TABLE VIEW ────────────────────────────────────────────────────────────────
if view_mode == "Table":
    rows = []
    for p in filtered:
        sim = sim_map.get(p.player_id)
        if sim is None:
            continue
        mkt_data = result.player_markets.get(p.player_id, {})
        for stat in stat_filter:
            pv = sim.proj_values.get(stat)
            if pv is None:
                continue
            m = mkt_data.get(stat, {})
            fair = m.get("fair_over", 0)
            rows.append({
                "Player":   p.player_name,
                "Team":     p.team_abbr,
                "Pos":      p.pos_group,
                "Min":      f"{p.proj_min:.0f}",
                "Stat":     STAT_LABELS.get(stat, stat),
                "Proj":     f"{pv:.2f}",
                "Line":     m.get("line", ""),
                "Over":     m.get("over_odds", ""),
                "Under":    m.get("under_odds", ""),
                "P(Over)":  f"{fair*100:.1f}%" if fair else "",
                "P10":      m.get("p10",""),
                "P25":      m.get("p25",""),
                "P50":      m.get("p50",""),
                "P75":      m.get("p75",""),
                "P90":      m.get("p90",""),
            })
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

# ── CARD VIEW ─────────────────────────────────────────────────────────────────
else:
    for p in filtered:
        sim = sim_map.get(p.player_id)
        if sim is None:
            continue
        mkt_data  = result.player_markets.get(p.player_id, {})
        pv        = sim.proj_values
        c_team    = team_color(p.team_abbr)

        # Player header
        st.markdown(
            f'<div style="border-left:4px solid {c_team};padding:3px 10px;margin:14px 0 6px;">'
            f'<span class="prop-player">{p.player_name}</span>&nbsp;&nbsp;'
            f'<span class="prop-team">{team_name(p.team_abbr)} · {p.pos_group} · {p.proj_min:.0f} min</span>'
            f'</div>', unsafe_allow_html=True,
        )

        # Stat grid — up to 4 per row
        stats_to_show = [s for s in stat_filter if s in pv]
        for row_start in range(0, len(stats_to_show), 4):
            chunk = stats_to_show[row_start:row_start+4]
            cols  = st.columns(len(chunk))
            for col, stat in zip(cols, chunk):
                m      = mkt_data.get(stat, {})
                proj_v = round(float(pv.get(stat, 0)), 2)
                line   = m.get("line", "")
                fair_o = float(m.get("fair_over", 0.5))
                fair_u = 1.0 - fair_o
                ov_odd = m.get("over_odds","")
                un_odd = m.get("under_odds","")
                p10    = m.get("p10",""); p50 = m.get("p50",""); p90 = m.get("p90","")

                # Edge color
                if line:
                    try:
                        edge = fair_o - 0.5  # vs pick-em
                        if fair_o > 0.55:
                            ec = "edge-pos"
                        elif fair_o < 0.45:
                            ec = "edge-neg"
                        else:
                            ec = "edge-neu"
                    except Exception:
                        ec = "edge-neu"
                else:
                    ec = "edge-neu"

                # Bar fill % (0–100 based on fair_over)
                bar_pct = int(fair_o * 100)
                bar_color = "#34d399" if fair_o > 0.55 else ("#f87171" if fair_o < 0.45 else "#94a3b8")

                with col:
                    st.markdown(f"""
<div class="prop-card">
  <div class="prop-stat">{STAT_LABELS.get(stat, stat)}</div>
  <div style="display:flex;align-items:baseline;gap:8px;margin:4px 0;">
    <span class="prop-line">{line if line else proj_v}</span>
    <span class="prop-proj">proj {proj_v}</span>
  </div>
  <div class="pct-bar-bg"><div class="pct-bar-fg" style="width:{bar_pct}%;background:{bar_color};"></div></div>
  <div style="display:flex;justify-content:space-between;margin-top:6px;">
    <div>
      <span style="font-size:.7rem;color:#64748b;">OVER</span><br>
      <span style="font-size:.82rem;color:#f1f5f9;">{ov_odd}</span>
      <span style="font-size:.72rem;color:{bar_color};margin-left:4px;">{fair_o*100:.0f}%</span>
    </div>
    <div style="text-align:right;">
      <span style="font-size:.7rem;color:#64748b;">UNDER</span><br>
      <span style="font-size:.82rem;color:#f1f5f9;">{un_odd}</span>
      <span style="font-size:.72rem;color:#94a3b8;margin-left:4px;">{fair_u*100:.0f}%</span>
    </div>
  </div>
  <div style="margin-top:6px;font-size:.7rem;color:#64748b;">
    P10: {p10} &nbsp; P50: {p50} &nbsp; P90: {p90}
  </div>
</div>""", unsafe_allow_html=True)

        # Milestone quick-hits for primary stats
        for stat in ["PTS", "REB", "AST"]:
            if stat not in sim.stat_distributions:
                continue
            dist = sim.stat_distributions[stat]
            proj = pv.get(stat, 0)
            thresholds = sorted({int(proj * 0.6), int(proj), int(proj * 1.3),
                                  int(proj * 1.6)} | {5,10,15,20,25,30} ,)
            thresholds = [t for t in thresholds if 1 <= t <= 60][:6]
            if not thresholds:
                continue
            mc = st.columns(len(thresholds))
            for col, thr in zip(mc, thresholds):
                p_hit = float(np.mean(dist >= thr))
                c_hit = "#34d399" if p_hit > 0.6 else ("#fbbf24" if p_hit > 0.4 else "#f87171")
                col.markdown(
                    f'<div style="text-align:center;background:rgba(15,23,42,.5);'
                    f'border-radius:6px;padding:4px;">'
                    f'<div style="font-size:.68rem;color:#64748b;">{stat} {thr}+</div>'
                    f'<div style="font-size:.88rem;font-weight:700;color:{c_hit};">{p_hit*100:.0f}%</div>'
                    f'</div>', unsafe_allow_html=True,
                )
            st.markdown("")

        st.markdown("")
