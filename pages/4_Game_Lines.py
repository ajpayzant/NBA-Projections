"""Page 4 — Game Lines"""
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
    SHARED_CSS, get_engine, init_session, team_name, team_color,
    card, fmt_odds,
)

st.set_page_config(page_title="Game Lines · NBA", page_icon="🏀", layout="wide")
init_session()
st.markdown(SHARED_CSS, unsafe_allow_html=True)

engine = get_engine()
result = st.session_state.get("last_result")
if result is None:
    st.warning("No projection loaded. Go to **Projections** first.")
    st.stop()

home_id = result.home_team
away_id = result.away_team
h = result.home_proj
a = result.away_proj
gm = result.game_market
gs = result.game_sim

st.title("📈 Game Lines")
st.markdown(
    f"**{team_name(away_id)} @ {team_name(home_id)}** · "
    f"{str(st.session_state.get('selected_game', {}).get('game_date',''))[:10]}"
)

# ── Moneyline ─────────────────────────────────────────────────────────────────
st.markdown("### Moneyline")
ml1, ml2 = st.columns(2)
with ml1:
    st.markdown(card(
        f"{team_name(away_id)}",
        fmt_odds(gm.away_ml),
        f"Win prob: {gm.away_win_prob*100:.1f}% · Proj: {a.proj_pts:.1f} pts"
    ), unsafe_allow_html=True)
with ml2:
    st.markdown(card(
        f"{team_name(home_id)}",
        fmt_odds(gm.home_ml),
        f"Win prob: {gm.home_win_prob*100:.1f}% · Proj: {h.proj_pts:.1f} pts"
    ), unsafe_allow_html=True)

# ── Spread & Total ────────────────────────────────────────────────────────────
st.markdown("### Spread & Total")
sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    st.markdown(card(
        f"{team_name(away_id)} Spread",
        f"{-gm.spread_home:+.1f}",
        gm.spread_away_odds
    ), unsafe_allow_html=True)
with sc2:
    st.markdown(card(
        f"{team_name(home_id)} Spread",
        f"{gm.spread_home:+.1f}",
        gm.spread_home_odds
    ), unsafe_allow_html=True)
with sc3:
    fair_over = float(np.mean(gs.total_distribution > gm.total_line))
    st.markdown(card(
        "Total Over",
        f"{gm.total_line:.1f}",
        f"{gm.over_odds} · {fair_over*100:.1f}% fair"
    ), unsafe_allow_html=True)
with sc4:
    st.markdown(card(
        "Total Under",
        f"{gm.total_line:.1f}",
        f"{gm.under_odds} · {(1-fair_over)*100:.1f}% fair"
    ), unsafe_allow_html=True)

# ── Team totals ────────────────────────────────────────────────────────────────
st.markdown("### Team Totals")
tt1, tt2 = st.columns(2)
for col, pts_arr, abbr, proj_pts in [
    (tt1, gs.home_pts, home_id, h.proj_pts),
    (tt2, gs.away_pts, away_id, a.proj_pts),
]:
    with col:
        from nba_engine import _nearest_half
        tt_line = _nearest_half(float(np.median(pts_arr)))
        fair_tt_over  = float(np.mean(pts_arr > tt_line))
        fair_tt_under = 1.0 - fair_tt_over
        st.markdown(
            f"**{team_name(abbr)}** · Proj {proj_pts:.1f} pts · "
            f"TT line {tt_line:.1f}"
        )
        c1, c2 = st.columns(2)
        c1.metric("Over fair prob", f"{fair_tt_over*100:.1f}%")
        c2.metric("Under fair prob", f"{fair_tt_under*100:.1f}%")

# ── Margin distribution ────────────────────────────────────────────────────────
st.markdown("### Margin Distribution (Home − Away)")
fig_margin = go.Figure()
fig_margin.add_trace(go.Histogram(
    x=gs.margin_distribution, nbinsx=50,
    marker_color="#c9a227", opacity=0.7, name="Margin"
))
fig_margin.add_vline(x=0, line_dash="dash", line_color="#94a3b8",
                     annotation_text="Pick'em")
fig_margin.add_vline(x=gm.spread_home, line_dash="dot", line_color="#fbbf24",
                     annotation_text=f"Spread {gm.spread_home:+.1f}")
fig_margin.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0", height=300, showlegend=False,
    xaxis_title="Point Margin (Home)", yaxis_title="Simulations",
    margin=dict(t=20, b=20, l=10, r=10),
)
fig_margin.update_xaxes(showgrid=False)
fig_margin.update_yaxes(gridcolor="rgba(148,163,184,.15)")
st.plotly_chart(fig_margin, use_container_width=True)

# ── Alternate totals ──────────────────────────────────────────────────────────
st.markdown("### Alternate Totals")
total_dist = gs.total_distribution
base = round(float(np.median(total_dist)) / 2.5) * 2.5
alt_lines = [base - 5, base - 2.5, base, base + 2.5, base + 5]
alt_rows = []
for line in alt_lines:
    p_over  = float(np.mean(total_dist > line))
    p_under = 1.0 - p_over
    alt_rows.append({
        "Line": f"{line:.1f}",
        "Over fair prob":  f"{p_over*100:.1f}%",
        "Under fair prob": f"{p_under*100:.1f}%",
    })
st.dataframe(pd.DataFrame(alt_rows), use_container_width=True, hide_index=True)

# ── Score grid ────────────────────────────────────────────────────────────────
st.markdown("### Score Probability Grid (5-pt buckets)")
home_pts = np.round(gs.home_pts / 5) * 5
away_pts = np.round(gs.away_pts / 5) * 5
h_buckets = sorted(pd.Series(home_pts).value_counts().nlargest(8).index.tolist())
a_buckets = sorted(pd.Series(away_pts).value_counts().nlargest(8).index.tolist())

grid = np.zeros((len(h_buckets), len(a_buckets)))
n = len(home_pts)
for i, hb in enumerate(h_buckets):
    for j, ab in enumerate(a_buckets):
        grid[i, j] = float(np.sum((home_pts == hb) & (away_pts == ab))) / n

grid_df = pd.DataFrame(
    (grid * 100).round(1),
    index=[f"Home {int(b)}" for b in h_buckets],
    columns=[f"Away {int(b)}" for b in a_buckets],
)
st.dataframe(grid_df.style.background_gradient(cmap="Blues", axis=None),
             use_container_width=True)
