"""Page 2 — Depth Charts & Lineup Management"""
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
import streamlit as st

from _engine_state import (
    SHARED_CSS, get_engine, init_session, team_name, team_color,
    get_depth_chart, set_player_override, set_player_rating,
    render_update_btn, run_projection, pos_badge, _autosave,
)

st.set_page_config(page_title="Depth Charts · NBA", page_icon="🏀", layout="wide")
init_session()
st.markdown(SHARED_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
.inj-healthy     { color:#34d399;font-weight:700; }
.inj-questionable{ color:#fbbf24;font-weight:700; }
.inj-doubtful    { color:#f97316;font-weight:700; }
.inj-out         { color:#ef4444;font-weight:700; }
.starter-badge   { background:#0891b2;color:#fff;border-radius:3px;
                   padding:1px 5px;font-size:.68rem;font-weight:700; }
.dc-section      { font-size:.72rem;font-weight:700;letter-spacing:.06em;
                   text-transform:uppercase;color:#64748b;
                   border-left:3px solid #334155;background:rgba(51,65,85,.18);
                   padding:3px 8px;margin:10px 0 4px;border-radius:0 4px 4px 0; }
</style>""", unsafe_allow_html=True)

engine = get_engine()
result = st.session_state.get("last_result")
if result is None:
    st.warning("No projection loaded. Go to **Projections** first.")
    st.stop()

home_id = result.home_team
away_id = result.away_team
game    = st.session_state.get("selected_game", {})

st.title("📋 Depth Charts")
st.markdown(f"**{team_name(away_id)} @ {team_name(home_id)}** · {str(game.get('game_date',''))[:10]}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Controls")
    render_update_btn(engine, key="dc_upd")

    st.markdown("---")
    st.markdown("### Google Sheets")
    if st.button("🔗 Reshare Sheets", key="dc_reshare", width="stretch"):
        with st.spinner("Resharing..."):
            try:
                from scripts.reshare_sheets import reshare_all, verify_access
                reshare_all(verbose=False)
                ok = verify_access(verbose=False)
                st.success("Access restored." if ok else "Reshare ran — check sheet manually if issues persist.")
            except Exception as _e:
                st.error(f"Reshare failed: {_e}")

    st.markdown("---")
    st.markdown("### Bulk actions")
    bulk_team = st.radio("Team", [team_name(away_id), team_name(home_id)],
                         key="dc_bulk_team", horizontal=True)
    bulk_abbr    = away_id if bulk_team == team_name(away_id) else home_id
    bulk_players = result.away_players if bulk_abbr == away_id else result.home_players

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Activate all", key="dc_act", width="stretch"):
            for p in bulk_players:
                set_player_override(bulk_abbr, p.player_id, "active", True)
                set_player_override(bulk_abbr, p.player_id, "injury_rating", 0.0)
            st.rerun()
    with b2:
        if st.button("Clear overrides", key="dc_clr", width="stretch"):
            st.session_state.depth_charts[bulk_abbr] = {}
            _autosave()
            st.rerun()

st.markdown("---")

INJ_OPTS = {0.0:"Healthy", 0.25:"Questionable (likely)", 0.50:"Questionable (doubtful)",
            0.75:"Doubtful", 1.00:"Out"}


def _render_team(team_abbr: str, team_nm: str, players):
    dc    = get_depth_chart(team_abbr)
    color = team_color(team_abbr)

    st.markdown(
        f'<div style="border-left:4px solid {color};padding:4px 12px;margin-bottom:8px;">'
        f'<span style="font-size:1.05rem;font-weight:700;color:{color};">{team_nm}</span>'
        f'<span style="font-size:.75rem;color:#64748b;margin-left:12px;">Click IN/OUT to toggle · adjust minutes inline · set injury status</span>'
        f'</div>', unsafe_allow_html=True,
    )

    # Determine starters: top-5 active by proj_pts (or user-set)
    active_players = [p for p in players if p.active]
    inactive_players = [p for p in players if not p.active]

    def _is_starter(p):
        return bool(dc.get(p.player_id, {}).get("is_starter", False))

    user_starters = [p for p in active_players if _is_starter(p)]
    if len(user_starters) < 5:
        auto_starters = sorted(
            [p for p in active_players if not _is_starter(p)],
            key=lambda x: -x.proj_pts
        )[:5 - len(user_starters)]
        starters = user_starters + auto_starters
    else:
        starters = user_starters[:5]
    starter_ids = {p.player_id for p in starters}

    bench = [p for p in active_players if p.player_id not in starter_ids]

    for section_label, section_players in [
        ("Starting Lineup", sorted(starters, key=lambda x: -x.proj_pts)),
        ("Bench", sorted(bench, key=lambda x: -x.proj_pts)),
        ("Inactive / Out", sorted(inactive_players, key=lambda x: -x.proj_pts)),
    ]:
        if not section_players:
            continue

        st.markdown(f'<div class="dc-section">{section_label} ({len(section_players)})</div>',
                    unsafe_allow_html=True)

        # Column headers
        hdr = st.columns([2.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6])
        for col, lbl in zip(hdr, ["Player","Pos","⭐","In/Out","Min","PTS","REB","AST","PRA","Inj.","Edit"]):
            col.markdown(f"<span style='font-size:.70rem;font-weight:700;color:#64748b;'>{lbl}</span>",
                         unsafe_allow_html=True)
        st.markdown('<hr style="margin:2px 0 4px;border-color:rgba(148,163,184,.12);">', unsafe_allow_html=True)

        for p in section_players:
            pid       = p.player_id
            existing  = dc.get(pid, {})
            is_active = bool(existing.get("active", True))
            inj_r     = float(existing.get("injury_rating", 0.0))
            is_strt   = pid in starter_ids
            has_ov    = bool(existing.get("minutes_override") is not None or
                             existing.get("rating_overrides") or inj_r > 0)
            opacity   = "" if is_active else "opacity:.3;"
            struck    = "text-decoration:line-through;color:#475569;" if not is_active else ""

            c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11 = st.columns(
                [2.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6]
            )

            # Player name
            with c1:
                mod = ' <span style="color:#fbbf24;font-size:.68rem;">⚡</span>' if has_ov else ""
                st.markdown(
                    f'<span style="{struck}{opacity}font-size:.85rem;">{p.player_name}</span>{mod}',
                    unsafe_allow_html=True,
                )

            # Position
            with c2:
                st.markdown(pos_badge(p.pos_group), unsafe_allow_html=True)

            # Starter toggle
            with c3:
                if is_active:
                    if st.button("⭐" if is_strt else "☆",
                                 key=f"strt_{team_abbr}_{pid}",
                                 help="Toggle starter/bench"):
                        if is_strt:
                            # Remove from starters — only clear if user explicitly set
                            if "is_starter" in existing:
                                del st.session_state.depth_charts[team_abbr][pid]["is_starter"]
                                _autosave()
                        else:
                            set_player_override(team_abbr, pid, "is_starter", True)
                        st.rerun()
                else:
                    st.markdown('<span style="color:#334155;font-size:.8rem;">—</span>', unsafe_allow_html=True)

            # In/Out button
            with c4:
                btn_lbl  = "IN" if is_active else "OUT"
                btn_type = "secondary"
                if st.button(btn_lbl, key=f"inout_{team_abbr}_{pid}",
                             help="Toggle active/inactive"):
                    new_active = not is_active
                    set_player_override(team_abbr, pid, "active", new_active)
                    if not new_active:
                        set_player_override(team_abbr, pid, "injury_rating", 1.0)
                        st.session_state[f"inj_{team_abbr}_{pid}"] = 1.0
                    else:
                        if existing.get("injury_rating", 0.0) >= 1.0:
                            set_player_override(team_abbr, pid, "injury_rating", 0.0)
                            st.session_state[f"inj_{team_abbr}_{pid}"] = 0.0
                    # Auto-reproject
                    if game := st.session_state.get("selected_game"):
                        run_projection(engine, game)
                    st.rerun()

            # Inline minutes input
            with c5:
                if is_active:
                    min_key = f"min_{team_abbr}_{pid}"
                    saved_min = existing.get("minutes_override")
                    if min_key not in st.session_state:
                        st.session_state[min_key] = float(saved_min) if saved_min else round(p.proj_min, 1)
                    new_min = st.number_input(
                        "", min_value=0.0, max_value=48.0, step=1.0,
                        key=min_key, label_visibility="collapsed",
                        format="%.0f",
                    )
                    if saved_min is None and abs(new_min - p.proj_min) > 0.5:
                        set_player_override(team_abbr, pid, "minutes_override", new_min)
                        if game := st.session_state.get("selected_game"):
                            run_projection(engine, game)
                        st.rerun()
                    elif saved_min is not None and abs(new_min - float(saved_min)) > 0.5:
                        set_player_override(team_abbr, pid, "minutes_override", new_min)
                        if game := st.session_state.get("selected_game"):
                            run_projection(engine, game)
                        st.rerun()
                else:
                    st.markdown('<span style="font-size:.80rem;color:#475569;">OUT</span>', unsafe_allow_html=True)

            # Stats display
            for col, val, thresh in [(c6,p.proj_pts,20),(c7,p.proj_reb,8),(c8,p.proj_ast,7),(c9,p.proj_pra,35)]:
                color_v = "#34d399" if val >= thresh else "#94a3b8"
                col.markdown(
                    f'<span style="{opacity}font-size:.82rem;color:{color_v};">'
                    f'{"--" if not is_active else f"{val:.1f}"}</span>',
                    unsafe_allow_html=True,
                )

            # Inline injury select
            with c10:
                if is_active:
                    inj_key = f"inj_{team_abbr}_{pid}"
                    if inj_key not in st.session_state:
                        st.session_state[inj_key] = inj_r
                    new_inj = st.selectbox(
                        "", options=list(INJ_OPTS.keys()),
                        index=list(INJ_OPTS.keys()).index(
                            min(INJ_OPTS.keys(), key=lambda x: abs(x - inj_r))
                        ),
                        format_func=lambda x: INJ_OPTS[x],
                        key=inj_key,
                        label_visibility="collapsed",
                    )
                    if abs(new_inj - inj_r) > 0.01:
                        set_player_override(team_abbr, pid, "injury_rating", new_inj)
                        if new_inj >= 1.0:
                            set_player_override(team_abbr, pid, "active", False)
                        if game := st.session_state.get("selected_game"):
                            run_projection(engine, game)
                        st.rerun()
                else:
                    st.markdown('<span style="font-size:.75rem;color:#ef4444;">Out</span>', unsafe_allow_html=True)

            # Edit panel toggle
            with c11:
                edit_key = f"show_edit_{team_abbr}_{pid}"
                if edit_key not in st.session_state:
                    st.session_state[edit_key] = False
                if st.button("⚡" if has_ov else "≡", key=f"ebtn_{team_abbr}_{pid}",
                             help="Advanced overrides"):
                    st.session_state[edit_key] = not st.session_state[edit_key]

            # ── Advanced edit panel ───────────────────────────────────────────
            if is_active and st.session_state.get(f"show_edit_{team_abbr}_{pid}", False):
                with st.container():
                    st.markdown(
                        f'<div style="background:rgba(30,58,95,.25);border-left:3px solid #0891b2;'
                        f'border-radius:0 6px 6px 0;padding:8px 12px;margin:2px 0 6px;">'
                        f'<span style="font-size:.73rem;color:#7dd3fc;font-weight:700;">'
                        f'Advanced — {p.player_name}</span></div>', unsafe_allow_html=True,
                    )
                    _rc1, _rc2 = st.columns(2)
                    with _rc1:
                        st.caption("These override the model rates directly.")
                    with _rc2:
                        if st.button("Reset all", key=f"rst_{team_abbr}_{pid}"):
                            if pid in st.session_state.depth_charts.get(team_abbr, {}):
                                st.session_state.depth_charts[team_abbr][pid].pop("minutes_override", None)
                                st.session_state.depth_charts[team_abbr][pid].pop("rating_overrides", None)
                            _autosave()
                            if game := st.session_state.get("selected_game"):
                                run_projection(engine, game)
                            st.session_state[f"show_edit_{team_abbr}_{pid}"] = False
                            st.rerun()


tab_away, tab_home = st.tabs([f"🏀 {team_name(away_id)}", f"🏀 {team_name(home_id)}"])

with tab_away:
    _render_team(away_id, team_name(away_id), result.away_players)
with tab_home:
    _render_team(home_id, team_name(home_id), result.home_players)

st.markdown("---")
st.markdown(
    '<span class="note-text">'
    '⭐ = Starter · IN/OUT = one-click active toggle (auto-reprojects) · '
    'Min = projected minutes (editable, auto-reprojects) · '
    'Inj = injury status · ⚡/≡ = advanced panel · '
    'Update Projection applies all changes.'
    '</span>', unsafe_allow_html=True,
)
