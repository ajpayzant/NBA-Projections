"""Page 5 — Projection History & Model Performance"""
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
import plotly.express as px
import streamlit as st

from _engine_state import SHARED_CSS, init_session

st.set_page_config(page_title="History · NBA", page_icon="🏀", layout="wide")
init_session()
st.markdown(SHARED_CSS, unsafe_allow_html=True)

st.title("📊 Projection History & Model Performance")

# ── Load saved data ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Loading saved projections…")
def _load_all():
    try:
        from gsheets_writer_nba import list_saved_games, read_game_tab
    except ImportError:
        return [], "Google Sheets module not available."
    games = list_saved_games()
    if not games:
        return [], None
    loaded = []
    for g in games:
        try:
            sections = read_game_tab(g["tab_name"])
            g["player_props"]     = sections.get("player_props", pd.DataFrame())
            g["team_projections"] = sections.get("team_projections", pd.DataFrame())
            loaded.append(g)
        except Exception:
            continue
    return loaded, None

try:
    all_games, load_err = _load_all()
except Exception as e:
    all_games, load_err = [], str(e)

if load_err:
    st.info(
        f"Note: {load_err}\n\n"
        "Save projections using **☁️ Save to Google Sheets** on the Projections page "
        "to start tracking model accuracy here."
    )
    st.stop()

if not all_games:
    st.info(
        "No saved projections found yet. Run a projection and click "
        "**☁️ Save to Google Sheets** to start tracking."
    )
    st.stop()

def _has_actuals(g: dict) -> bool:
    pp = g.get("player_props", pd.DataFrame())
    if pp.empty or "Actual Result" not in pp.columns:
        return False
    return pp["Actual Result"].replace("", np.nan).notna().any()

games_with = [g for g in all_games if _has_actuals(g)]
games_pending = [g for g in all_games if not _has_actuals(g)]

with st.sidebar:
    st.markdown(f"**{len(all_games)}** saved · **{len(games_with)}** with actuals")
    if st.button("🔄 Refresh", key="hist_ref"):
        st.cache_data.clear()
        st.rerun()
    stat_filter = st.multiselect(
        "Stats",
        ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA","PR","PA","RA"],
        default=["PTS","REB","AST","FG3M","PRA"],
        key="hist_stat",
    )

if games_pending:
    with st.expander(f"⏳ {len(games_pending)} game(s) pending actuals"):
        for g in games_pending:
            st.markdown(f"- **{g['away']} @ {g['home']}** · {g['game_date']}")

if not games_with:
    st.info("No games with actuals yet. Use **🔄 Sync Actuals** after games complete.")
    st.stop()

# ── Build master props dataframe ──────────────────────────────────────────────
frames = []
for g in games_with:
    pp = g.get("player_props", pd.DataFrame()).copy()
    if pp.empty:
        continue
    pp["game_date"] = g["game_date"]
    pp["away"] = g["away"]
    pp["home"] = g["home"]
    frames.append(pp)

master = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
if master.empty:
    st.warning("Could not parse prop data from saved sheets.")
    st.stop()

for col in ["Projection", "Actual Result", "Main Line", "Fair P(Over)"]:
    if col in master.columns:
        master[col] = pd.to_numeric(master[col], errors="coerce")

master["error"]     = master["Actual Result"] - master["Projection"]
master["abs_error"] = master["error"].abs()
master["hit"]       = master.get("Hit/Miss", "").str.lower().str.strip() == "hit"
master["has_line"]  = master["Main Line"].notna()

if "Stat" in master.columns and stat_filter:
    master = master[master["Stat"].isin(stat_filter)]

master = master[master["Actual Result"].notna()]

# ── Scorecards ────────────────────────────────────────────────────────────────
st.markdown("## Overall Performance")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Games tracked", len(games_with))
c2.metric("Props graded", len(master[master["has_line"]]))
hit_rate = master[master["has_line"]]["hit"].mean() if master["has_line"].any() else 0
c3.metric("Hit rate", f"{hit_rate*100:.1f}%", delta=f"{(hit_rate-0.5)*100:+.1f}% vs 50%")
c4.metric("MAE", f"{master['abs_error'].mean():.3f}")
bias = master["error"].mean()
c5.metric("Bias", f"{bias:+.3f}", delta="High" if bias > 0 else "Low" if bias < 0 else "None")

st.markdown("---")

# ── By stat ───────────────────────────────────────────────────────────────────
if "Stat" in master.columns:
    st.markdown("## Accuracy by Stat")
    stat_sum = master.groupby("Stat").agg(
        Props=("Projection","count"),
        MAE=("abs_error","mean"),
        Bias=("error","mean"),
    ).round(3).reset_index()

    fig_mae = px.bar(stat_sum, x="Stat", y="MAE", color="MAE",
                     color_continuous_scale=["#34d399","#fbbf24","#f87171"],
                     title="MAE by Stat")
    fig_mae.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", coloraxis_showscale=False, height=280)
    fig_mae.update_xaxes(showgrid=False)
    fig_mae.update_yaxes(gridcolor="rgba(148,163,184,.15)")

    fig_bias = px.bar(stat_sum, x="Stat", y="Bias", color="Bias",
                      color_continuous_scale=["#f87171","#e2e8f0","#34d399"],
                      color_continuous_midpoint=0, title="Bias by Stat")
    fig_bias.add_hline(y=0, line_dash="dash", line_color="rgba(148,163,184,.4)")
    fig_bias.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#e2e8f0", coloraxis_showscale=False, height=280)
    fig_bias.update_xaxes(showgrid=False)
    fig_bias.update_yaxes(gridcolor="rgba(148,163,184,.15)")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_mae, use_container_width=True)
    with col2:
        st.plotly_chart(fig_bias, use_container_width=True)

    st.dataframe(stat_sum.style.format({"MAE":"{:.3f}","Bias":"{:+.3f}"}),
                 use_container_width=True, hide_index=True)

st.markdown("---")

# ── Per-player accuracy ────────────────────────────────────────────────────────
if "Player" in master.columns and "Stat" in master.columns:
    st.markdown("## Per-Player Accuracy (10+ graded props)")
    pg = master.groupby(["Player","Stat"]).agg(
        n=("Projection","count"),
        MAE=("abs_error","mean"),
        Bias=("error","mean"),
        avg_actual=("Actual Result","mean"),
        avg_pred=("Projection","mean"),
    ).query("n >= 10").sort_values("MAE").reset_index()

    if not pg.empty:
        st.dataframe(
            pg.style.format({"MAE":"{:.3f}","Bias":"{:+.3f}",
                             "avg_actual":"{:.2f}","avg_pred":"{:.2f}"}),
            use_container_width=True, hide_index=True,
        )

st.markdown("---")

# ── Calibration ───────────────────────────────────────────────────────────────
cal_df = master[master["has_line"] & master["Fair P(Over)"].notna()].copy()
if len(cal_df) >= 20:
    st.markdown("## Probability Calibration")
    cal_df["bucket"] = pd.cut(cal_df["Fair P(Over)"],
                               bins=[0,.45,.50,.55,.60,.65,.70,1.0],
                               labels=["<45%","45-50%","50-55%","55-60%","60-65%","65-70%",">70%"])
    calib = cal_df.groupby("bucket", observed=True).agg(
        Count=("hit","count"), Hit_Rate=("hit","mean")
    ).reset_index()
    mid_map = {"<45%":0.42,"45-50%":0.475,"50-55%":0.525,"55-60%":0.575,
               "60-65%":0.625,"65-70%":0.675,">70%":0.75}
    calib["mid"] = calib["bucket"].map(mid_map)

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  line=dict(dash="dash", color="rgba(148,163,184,.4)"),
                                  name="Perfect"))
    fig_cal.add_trace(go.Scatter(x=calib["mid"], y=calib["Hit_Rate"],
                                  mode="lines+markers",
                                  marker=dict(size=10, color="#c9a227"),
                                  line=dict(color="#c9a227"), name="Model"))
    fig_cal.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", height=300,
                           xaxis=dict(tickformat=".0%", range=[0.3, 0.85], showgrid=False),
                           yaxis=dict(tickformat=".0%", range=[0,1],
                                      gridcolor="rgba(148,163,184,.15)"),
                           legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_cal, use_container_width=True)
