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

engine = get_engine()
result = st.session_state.get("last_result")
if result is None:
    st.warning("No projection loaded. Go to **Projections** first.")
    st.stop()

home_id = result.home_team
away_id = result.away_team

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    team_filter = st.multiselect(
        "Team", [team_name(away_id), team_name(home_id)],
        default=[team_name(away_id), team_name(home_id)], key="pp_team"
    )
    stat_filter = st.multiselect(
        "Stat",
        ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA","PR","PA","RA","MIN"],
        default=["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"],
        key="pp_stat",
    )
    pos_filter = st.multiselect("Position", ["G","F","C","UNK"],
                                default=["G","F","C"], key="pp_pos")
    min_proj   = st.slider("Min projected PTS", 0.0, 30.0, 5.0, 0.5, key="pp_minpts")
    view_mode  = st.radio("View", ["Table", "Expander"], key="pp_view", horizontal=True)

    st.markdown("---")
    st.markdown("### Market comparison")
    mkt_player = st.text_input("Player name (partial)", key="pp_mkt_player")
    mkt_stat   = st.selectbox("Stat", ["PTS","REB","AST","FG3M","STL","BLK","PRA"],
                               key="pp_mkt_stat")
    mkt_line   = st.number_input("Market line", min_value=0.0, max_value=100.0,
                                  step=0.5, key="pp_mkt_line")
    mkt_odds   = st.number_input("Market over odds (American)", min_value=-999,
                                  max_value=999, value=-110, key="pp_mkt_odds")

st.title("🎯 Player Props")
st.markdown(
    f"**{team_name(away_id)} @ {team_name(home_id)}** · "
    f"{str(st.session_state.get('selected_game', {}).get('game_date',''))[:10]}"
)

# ── Collect all active players ────────────────────────────────────────────────
all_players = {p.player_id: p for p in result.home_players + result.away_players}
all_sims    = result.home_player_sims + result.away_player_sims

filtered = []
for sim in all_sims:
    p = all_players.get(sim.player_id)
    if p is None or not p.active:
        continue
    if team_name(p.team_abbr) not in team_filter:
        continue
    if p.pos_group not in pos_filter:
        continue
    if p.proj_pts < min_proj:
        continue
    filtered.append((p, sim))

# ── Market edge calculation ───────────────────────────────────────────────────
mkt_edge_row = None
if mkt_player and mkt_line > 0:
    for p, sim in filtered:
        if mkt_player.lower() in p.player_name.lower() and mkt_stat in sim.stat_distributions:
            dist = sim.stat_distributions[mkt_stat]
            fair_p = float(np.mean(dist > mkt_line))
            if mkt_odds > 0:
                mkt_implied = 100 / (mkt_odds + 100)
            else:
                mkt_implied = abs(mkt_odds) / (abs(mkt_odds) + 100)
            edge = fair_p - mkt_implied
            mkt_edge_row = (p.player_name, mkt_stat, mkt_line, fair_p, mkt_implied, edge)
            break

if mkt_edge_row:
    name, stat, line, fair, impl, edge = mkt_edge_row
    ec = "#34d399" if edge > 0.02 else ("#f87171" if edge < -0.02 else "#94a3b8")
    st.markdown(
        f'<div style="background:rgba(30,41,59,.6);border:1px solid {ec};'
        f'border-radius:6px;padding:10px 16px;margin-bottom:12px;">'
        f'<b>{name} {stat} O{line}</b> · '
        f'Fair: {fair*100:.1f}% · Market: {impl*100:.1f}% · '
        f'<span style="color:{ec};font-weight:700;">Edge: {edge*100:+.1f}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Table view ────────────────────────────────────────────────────────────────
if view_mode == "Table":
    rows = []
    for p, sim in filtered:
        mkt = result.player_markets.get(p.player_id, {})
        pv  = sim.proj_values
        for stat in stat_filter:
            if stat not in pv:
                continue
            m = mkt.get(stat, {})
            proj_v = round(pv[stat], 2)
            rows.append({
                "Player":       p.player_name,
                "Team":         p.team_abbr,
                "Pos":          p.pos_group,
                "Stat":         stat,
                "Projection":   proj_v,
                "Line":         sim.prop_lines.get(stat, ""),
                "Over Odds":    m.get("over_odds", ""),
                "Under Odds":   m.get("under_odds", ""),
                "Fair P(Over)": f"{m.get('fair_over', 0)*100:.1f}%",
                "P25":          m.get("p25", ""),
                "P50":          m.get("p50", ""),
                "P75":          m.get("p75", ""),
            })

    if rows:
        df = pd.DataFrame(rows).sort_values(["Team","Pos","Player","Stat"])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No props match current filters.")

# ── Expander view ─────────────────────────────────────────────────────────────
else:
    for p, sim in sorted(filtered, key=lambda x: -x[0].proj_pts):
        mkt = result.player_markets.get(p.player_id, {})
        pv  = sim.proj_values
        label = (f"{p.player_name} · {p.team_abbr} · {p.pos_group} · "
                 f"{pv.get('MIN',0):.0f}min · {pv.get('PTS',0):.1f}pts")
        with st.expander(label):
            # Distribution chart for primary stat
            primary = "PTS" if "PTS" in sim.stat_distributions else list(sim.stat_distributions.keys())[0]
            dist = sim.stat_distributions[primary]
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=dist, nbinsx=30,
                                       marker_color=team_color(p.team_abbr), opacity=0.75))
            line_val = sim.prop_lines.get(primary, float(np.median(dist)))
            fig.add_vline(x=line_val, line_dash="dash", line_color="#fbbf24",
                          annotation_text=f"Line: {line_val}")
            fig.update_layout(
                title=f"{p.player_name} {primary} Distribution",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", height=220,
                margin=dict(t=30, b=10, l=10, r=10), showlegend=False,
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(gridcolor="rgba(148,163,184,.15)")
            st.plotly_chart(fig, use_container_width=True)

            # Prop lines table
            prop_rows = []
            for stat in stat_filter:
                if stat not in pv:
                    continue
                m = mkt.get(stat, {})
                prop_rows.append({
                    "Stat":         stat,
                    "Projection":   round(pv[stat], 2),
                    "Line":         sim.prop_lines.get(stat, ""),
                    "P(Over)":      f"{m.get('fair_over',0)*100:.1f}%",
                    "Over Odds":    m.get("over_odds", ""),
                    "Under Odds":   m.get("under_odds", ""),
                    "P10":          m.get("p10",""),
                    "P25":          m.get("p25",""),
                    "P50":          m.get("p50",""),
                    "P75":          m.get("p75",""),
                    "P90":          m.get("p90",""),
                })
            if prop_rows:
                st.dataframe(pd.DataFrame(prop_rows), use_container_width=True, hide_index=True)

            # Milestone probabilities
            for stat in ["PTS", "REB", "AST", "FG3M"]:
                if stat not in sim.stat_distributions:
                    continue
                d = sim.stat_distributions[stat]
                proj = pv.get(stat, 0)
                thresholds = [t for t in [5,10,15,20,25,30,35,40,50] if t <= proj * 2 + 5][:6]
                if thresholds:
                    m_cols = st.columns(len(thresholds))
                    for col, thr in zip(m_cols, thresholds):
                        p_hit = float(np.mean(d >= thr))
                        col.metric(f"{stat} {thr}+", f"{p_hit*100:.0f}%")
