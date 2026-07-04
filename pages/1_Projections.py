"""Page 1 — Projections"""
from __future__ import annotations
import sys
from pathlib import Path

_PAGES_DIR = Path(__file__).resolve().parent
_ROOT      = _PAGES_DIR.parent
for _p in [str(_ROOT), str(_PAGES_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import io
import datetime as _dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _engine_state import (
    SHARED_CSS, get_engine, init_session, team_name, team_color,
    run_projection, render_update_btn, card, fmt_odds,
)

st.set_page_config(page_title="Projections · NBA", page_icon="🏀", layout="wide")
init_session()
st.markdown(SHARED_CSS, unsafe_allow_html=True)

engine = get_engine()

# ── Sidebar ──────────────────────────────────────────────────────────────────
ALL_TEAMS = [
    "ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET","GSW",
    "HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK",
    "OKC","ORL","PHI","PHX","POR","SAC","SAS","TOR","UTA","WAS",
]

with st.sidebar:
    st.markdown("### Game Selection")

    upcoming = engine.upcoming_games()

    if upcoming:
        mode = st.radio("Mode", ["Scheduled games", "Manual matchup"],
                        key="sel_mode_p1", horizontal=True)
    else:
        mode = "Manual matchup"
        st.info("Off-season — no scheduled games. Use manual matchup.")

    if mode == "Scheduled games" and upcoming:
        game_options = {}
        for g in upcoming:
            date = str(g.get("game_date", ""))[:10]
            label = f"{g.get('away_team_abbr','?')} @ {g.get('home_team_abbr','?')}"
            key = f"{date} — {label}"
            game_options[key] = g
        selected_key = st.selectbox("Select game", list(game_options.keys()),
                                    key="game_selector_p1")
        game = game_options[selected_key]
    else:
        st.markdown("**Away team**")
        away_pick = st.selectbox("Away", ALL_TEAMS,
                                 index=ALL_TEAMS.index("LAL"),
                                 key="manual_away_p1",
                                 label_visibility="collapsed")
        st.markdown("**Home team**")
        home_pick = st.selectbox("Home", ALL_TEAMS,
                                 index=ALL_TEAMS.index("BOS"),
                                 key="manual_home_p1",
                                 label_visibility="collapsed")
        game_date_pick = st.date_input("Game date",
                                       value=_dt.date.today(),
                                       key="manual_date_p1")
        game = {
            "away_team_abbr": away_pick,
            "home_team_abbr": home_pick,
            "away_team_id":   0,
            "home_team_id":   0,
            "game_date":      str(game_date_pick),
            "game_id":        f"{away_pick}@{home_pick}_{game_date_pick}",
            "status":         "Manual",
        }

    st.markdown("---")
    hold_pct = st.number_input("Market hold %", min_value=0.0, max_value=15.0,
                               value=float(st.session_state.get("hold_pct", 5.0)),
                               step=0.5, format="%.1f", key="hold_num_p1")
    st.session_state.hold_pct = hold_pct / 100.0
    engine.pricing.hold_pct = hold_pct / 100.0

    st.markdown("---")
    run_btn = st.button("▶ Run Projection", type="primary", width="stretch")
    render_update_btn(engine, key="upd_p1")

# ── Run projection ────────────────────────────────────────────────────────────
if run_btn or st.session_state.get("_run_after_load"):
    st.session_state.pop("_run_after_load", None)
    with st.spinner("Running projection…"):
        run_projection(engine, game)

result = st.session_state.get("last_result")
if result is None:
    st.info("Select a game and click **▶ Run Projection** to get started.")
    st.stop()

home_id = result.home_team
away_id = result.away_team
h = result.home_proj
a = result.away_proj
gm = result.game_market
gs = result.game_sim

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏀 Projections")
st.markdown(
    f"**{team_name(away_id)} @ {team_name(home_id)}** · "
    f"{str(game.get('game_date',''))[:10]}"
)

# ── Team summary cards ────────────────────────────────────────────────────────
st.markdown("### Team Projections")
cols = st.columns(2)
for col, proj, abbr in [(cols[0], a, away_id), (cols[1], h, home_id)]:
    with col:
        c = team_color(abbr)
        st.markdown(
            f'<div style="border-left:4px solid {c};padding:4px 12px;'
            f'margin-bottom:8px;">'
            f'<span style="font-size:1.1rem;font-weight:700;color:{c};">'
            f'{team_name(abbr)}</span></div>',
            unsafe_allow_html=True,
        )
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Proj PTS", f"{proj.proj_pts:.1f}")
        r2.metric("Proj REB", f"{proj.proj_reb:.1f}")
        r3.metric("Proj AST", f"{proj.proj_ast:.1f}")
        r4.metric("Proj 3PM", f"{proj.proj_fg3m:.1f}")
        r5.metric("Win Prob", f"{proj.proj_win_prob*100:.0f}%")

# ── Game lines summary ────────────────────────────────────────────────────────
st.markdown("### Game Lines")
lc1, lc2, lc3, lc4 = st.columns(4)
with lc1:
    st.markdown(card(f"{team_name(away_id)} ML", fmt_odds(gm.away_ml),
                     f"{gm.away_win_prob*100:.1f}%"), unsafe_allow_html=True)
with lc2:
    st.markdown(card(f"{team_name(home_id)} ML", fmt_odds(gm.home_ml),
                     f"{gm.home_win_prob*100:.1f}%"), unsafe_allow_html=True)
with lc3:
    st.markdown(card("Total (O/U)", f"{gm.total_line:.1f}",
                     f"O {gm.over_odds} / U {gm.under_odds}"), unsafe_allow_html=True)
with lc4:
    spread_label = f"{team_name(home_id)} {gm.spread_home:+.1f}"
    st.markdown(card("Spread", spread_label,
                     f"{gm.spread_home_odds} / {gm.spread_away_odds}"), unsafe_allow_html=True)

# ── Score distribution chart ──────────────────────────────────────────────────
st.markdown("### Score Distribution")
fc1, fc2 = st.columns(2)
for col, pts_arr, abbr in [(fc1, gs.home_pts, home_id), (fc2, gs.away_pts, away_id)]:
    with col:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=pts_arr, nbinsx=40,
                                   marker_color=team_color(abbr), opacity=0.75,
                                   name=team_name(abbr)))
        fig.update_layout(
            title=f"{team_name(abbr)} Score Distribution",
            xaxis_title="Points", yaxis_title="Simulations",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=280, margin=dict(t=40, b=20, l=20, r=10),
            showlegend=False,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(148,163,184,.15)")
        st.plotly_chart(fig, use_container_width=True)

# ── Player projections table ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Player Projections")

tab_away, tab_home = st.tabs([f"🏀 {team_name(away_id)}", f"🏀 {team_name(home_id)}"])

STAT_COLS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "PRA", "MIN"]

def _player_table(players, sims_list):
    sim_map = {s.player_id: s for s in sims_list}
    rows = []
    for p in sorted([x for x in players if x.active], key=lambda x: -x.proj_pts):
        sim = sim_map.get(p.player_id)
        row = {
            "Player":   p.player_name,
            "Pos":      p.pos_group,
            "Inj":      f"{p.injury_rating:.2f}" if p.injury_rating > 0 else "—",
            "Min":      f"{p.proj_min:.1f}",
            "PTS":      f"{p.proj_pts:.1f}",
            "REB":      f"{p.proj_reb:.1f}",
            "AST":      f"{p.proj_ast:.1f}",
            "3PM":      f"{p.proj_fg3m:.1f}",
            "STL":      f"{p.proj_stl:.1f}",
            "BLK":      f"{p.proj_blk:.1f}",
            "TOV":      f"{p.proj_tov:.1f}",
            "PRA":      f"{p.proj_pra:.1f}",
        }
        if sim:
            pv = sim.proj_values
            row["Line PTS"] = f"{sim.prop_lines.get('PTS','')}"
            row["Line REB"] = f"{sim.prop_lines.get('REB','')}"
            row["Line AST"] = f"{sim.prop_lines.get('AST','')}"
        rows.append(row)
    if not rows:
        st.caption("No active players.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_away:
    _player_table(result.away_players, result.away_player_sims)
with tab_home:
    _player_table(result.home_players, result.home_player_sims)

# ── Export + Save ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Export & Save")

_btn_dl, _btn_save, _btn_sync = st.columns([1, 1, 1], gap="small")

with _btn_dl:
    if st.button("📥 Download Excel", type="secondary", width="stretch"):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xl:
            meta = pd.DataFrame([
                ("Game", f"{team_name(away_id)} @ {team_name(home_id)}"),
                ("Date", str(game.get("game_date",""))[:10]),
                ("Generated", result.generated_at),
            ], columns=["Field", "Value"])
            meta.to_excel(xl, sheet_name="Metadata", index=False)
            lines_rows = [
                (f"{team_name(away_id)} ML", gm.away_ml, f"{gm.away_win_prob*100:.1f}%"),
                (f"{team_name(home_id)} ML", gm.home_ml, f"{gm.home_win_prob*100:.1f}%"),
                ("Total Over",  f"{gm.total_line}", gm.over_odds),
                ("Total Under", f"{gm.total_line}", gm.under_odds),
            ]
            pd.DataFrame(lines_rows, columns=["Market","Odds","Fair Prob"]).to_excel(
                xl, sheet_name="Game Lines", index=False)
            for abbr, players, sims in [
                (away_id, result.away_players, result.away_player_sims),
                (home_id, result.home_players, result.home_player_sims),
            ]:
                sim_map = {s.player_id: s for s in sims}
                prop_rows = []
                for p in [x for x in players if x.active]:
                    sim = sim_map.get(p.player_id)
                    if sim is None:
                        continue
                    pv = sim.proj_values
                    pm = sim.prop_lines
                    for stat in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA","PR","PA","RA"]:
                        if stat not in pv:
                            continue
                        mkt = result.player_markets.get(p.player_id, {}).get(stat, {})
                        prop_rows.append({
                            "Player": p.player_name, "Team": abbr, "Pos": p.pos_group,
                            "Stat": stat, "Projection": round(pv.get(stat, 0), 2),
                            "Line": pm.get(stat, ""),
                            "Over Odds": mkt.get("over_odds", ""),
                            "Under Odds": mkt.get("under_odds", ""),
                            "Fair P(Over)": round(mkt.get("fair_over", 0), 3),
                            "P10": round(mkt.get("p10", 0), 1),
                            "P50": round(mkt.get("p50", 0), 1),
                            "P90": round(mkt.get("p90", 0), 1),
                            "Actual Result": "", "Hit/Miss": "",
                        })
                if prop_rows:
                    pd.DataFrame(prop_rows).to_excel(
                        xl, sheet_name=f"Props {abbr}", index=False)
        buf.seek(0)
        fname = (f"NBA_{team_name(away_id).replace(' ','_')}_at_"
                 f"{team_name(home_id).replace(' ','_')}_"
                 f"{str(game.get('game_date',''))[:10]}.xlsx")
        st.download_button("⬇ Click to download", data=buf.getvalue(),
                           file_name=fname,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="dl_xlsx")

with _btn_save:
    if st.button("☁️ Save to Google Sheets", type="primary", width="stretch"):
        with st.spinner("Saving to Google Sheets..."):
            try:
                sys.path.insert(0, str(_ROOT))
                from gsheets_writer_nba import save_snapshot
                tab = save_snapshot(result, game, engine)
                st.success(f"Saved → NBA Projections · tab: {tab}")
            except Exception as e:
                st.error(f"Save failed: {e}")

with _btn_sync:
    if st.button("🔄 Sync Actuals", type="secondary", width="stretch"):
        with st.spinner("Syncing actuals from warehouse..."):
            try:
                from gsheets_writer_nba import sync_actuals, _tab_name
                import os as _os
                db_path = _os.getenv("NBA_DB_PATH", str(
                    _ROOT / "data" / "analytics_database" / "nba_warehouse.duckdb"))
                tab = _tab_name(game)
                counts = sync_actuals(tab, db_path)
                st.success(
                    f"Synced actuals: {counts['players_updated']} player rows, "
                    f"{counts['teams_updated']} team rows updated"
                )
            except ValueError as e:
                st.warning(str(e))
            except Exception as e:
                st.error(f"Sync failed: {e}")
