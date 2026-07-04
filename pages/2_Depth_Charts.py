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
    render_update_btn, build_player_overrides, run_projection,
    pos_badge, PLAYER_RATING_DEFS, _autosave,
)

st.set_page_config(page_title="Depth Charts · NBA", page_icon="🏀", layout="wide")
init_session()
st.markdown(SHARED_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
.inj-healthy    { color: #34d399; font-weight: 700; }
.inj-questionable { color: #fbbf24; font-weight: 700; }
.inj-doubtful   { color: #f97316; font-weight: 700; }
.inj-out        { color: #ef4444; font-weight: 700; }
.modified-badge { font-size:.70rem; color:#fbbf24; font-weight:700; }
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
st.markdown(
    f"**{team_name(away_id)} @ {team_name(home_id)}** · "
    f"{str(game.get('game_date',''))[:10]}"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Controls")
    render_update_btn(engine, key="dc_upd")
    st.markdown("---")
    st.markdown("### Google Sheets")
    if st.button("🔗 Reshare Sheets", key="dc_reshare", width="stretch",
                 help="Re-grant service account access if Save to Sheets is failing"):
        with st.spinner("Resharing..."):
            try:
                import sys as _sys
                _sys.path.insert(0, str(_ROOT))
                from scripts.reshare_sheets import reshare_all, verify_access
                reshare_all(verbose=False)
                ok = verify_access(verbose=False)
                if ok:
                    st.success("Sheets access restored.")
                else:
                    st.error("Reshare ran but access still failing. "
                             "Reshare manually in Google Drive.")
            except Exception as _e:
                st.error(f"Reshare failed: {_e}")
    st.markdown("---")
    st.markdown("### Bulk actions")
    bulk_team = st.radio("Team", [team_name(away_id), team_name(home_id)],
                         key="dc_bulk_team", horizontal=True)
    bulk_abbr = away_id if bulk_team == team_name(away_id) else home_id
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
            stale = [k for k in st.session_state
                     if k.startswith(f"pr_num_{bulk_abbr}_")
                     or k.startswith(f"inj_{bulk_abbr}_")]
            for k in stale:
                del st.session_state[k]
            _autosave()
            st.rerun()

st.markdown("---")

# ── Injury rating labels ──────────────────────────────────────────────────────
INJ_LABELS = {
    0.00: ("Healthy",    "inj-healthy"),
    0.25: ("Questionable (likely)", "inj-questionable"),
    0.50: ("Questionable (doubtful)", "inj-questionable"),
    0.75: ("Doubtful",   "inj-doubtful"),
    1.00: ("Out",        "inj-out"),
}

def _inj_label(rating: float) -> tuple:
    r = round(float(rating) * 4) / 4  # snap to 0, 0.25, 0.5, 0.75, 1.0
    return INJ_LABELS.get(r, ("Unknown", "inj-out"))


def _render_team(team_abbr: str, team_nm: str, players):
    dc = get_depth_chart(team_abbr)
    c = team_color(team_abbr)
    st.markdown(
        f'<div style="border-left:4px solid {c};padding:4px 12px;margin-bottom:12px;">'
        f'<span style="font-size:1.05rem;font-weight:700;color:{c};">{team_nm}</span></div>',
        unsafe_allow_html=True,
    )

    sorted_players = sorted(
        players,
        key=lambda p: ({"G": 0, "F": 1, "C": 2, "UNK": 3}.get(p.pos_group, 3),
                       -p.proj_pts)
    )

    # Columns: Player | Pos | In/Out | Injury | Min | PTS | REB | AST | PRA | Edit
    hdr = st.columns([2.8, 0.55, 0.65, 0.75, 0.85, 0.85, 0.85, 0.85, 0.85, 0.7])
    for col, lbl in zip(hdr, ["Player", "Pos", "In/Out", "Injury", "Min",
                                "PTS", "REB", "AST", "PRA", ""]):
        col.markdown(f"<span style='font-size:.73rem;font-weight:700;color:#64748b;'>{lbl}</span>",
                     unsafe_allow_html=True)
    st.markdown('<hr style="margin:2px 0 6px;border-color:rgba(148,163,184,.15);">', unsafe_allow_html=True)

    for p in sorted_players:
        pid       = p.player_id
        existing  = dc.get(pid, {})
        is_active = bool(existing.get("active", True))
        inj_r     = float(existing.get("injury_rating", 0.0))
        has_ov    = bool(existing.get("rating_overrides") or
                         existing.get("minutes_override") is not None or
                         inj_r > 0)

        inj_lbl, inj_cls = _inj_label(inj_r)
        opacity = "" if is_active else "opacity:.35;"

        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = st.columns(
            [2.8, 0.55, 0.65, 0.75, 0.85, 0.85, 0.85, 0.85, 0.85, 0.7]
        )

        with c1:
            mod = f' <span class="modified-badge">⚡</span>' if has_ov else ""
            struck = "text-decoration:line-through;color:#475569;" if not is_active else ""
            st.markdown(
                f'<span style="{struck}{opacity}font-size:.88rem;">{p.player_name}</span>{mod}',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(pos_badge(p.pos_group), unsafe_allow_html=True)

        # ── In/Out toggle — single click to activate/deactivate ──────────────
        with c3:
            btn_color  = "#16a34a" if is_active else "#dc2626"
            btn_text   = "IN" if is_active else "OUT"
            btn_border = f"border:2px solid {btn_color};"
            btn_bg     = f"background:rgba({('22,163,74' if is_active else '220,38,38')},.15);"
            st.markdown(
                f'<div style="padding-top:2px;">',
                unsafe_allow_html=True,
            )
            if st.button(
                btn_text,
                key=f"inout_{team_abbr}_{pid}",
                help="Click to toggle player In/Out of lineup",
            ):
                new_active = not is_active
                set_player_override(team_abbr, pid, "active", new_active)
                # If setting Out, also set injury_rating to 1.0 visually
                if not new_active:
                    set_player_override(team_abbr, pid, "injury_rating", 1.0)
                    st.session_state[f"inj_{team_abbr}_{pid}"] = 1.0
                else:
                    # Restoring — clear Out injury rating
                    if existing.get("injury_rating", 0.0) >= 1.0:
                        set_player_override(team_abbr, pid, "injury_rating", 0.0)
                        st.session_state[f"inj_{team_abbr}_{pid}"] = 0.0
                st.rerun()

        with c4:
            st.markdown(
                f'<span class="{inj_cls}" style="font-size:.75rem;">'
                f'{"Out" if not is_active else inj_lbl}</span>',
                unsafe_allow_html=True,
            )
        with c5:
            st.markdown(
                f'<span style="{opacity}font-size:.82rem;color:#94a3b8;">'
                f'{"--" if not is_active else f"{p.proj_min:.1f}"}</span>',
                unsafe_allow_html=True,
            )
        for col, val in [(c6, p.proj_pts), (c7, p.proj_reb), (c8, p.proj_ast), (c9, p.proj_pra)]:
            color = "#34d399" if val >= 20 else "#94a3b8"
            col.markdown(
                f'<span style="{opacity}font-size:.82rem;color:{color};">'
                f'{"--" if not is_active else f"{val:.1f}"}</span>',
                unsafe_allow_html=True,
            )

        with c10:
            rating_key = f"show_edit_{team_abbr}_{pid}"
            if rating_key not in st.session_state:
                st.session_state[rating_key] = False
            btn_lbl = "⚡ Edit" if has_ov else "Edit"
            if st.button(btn_lbl, key=f"ebtn_{team_abbr}_{pid}", width="stretch"):
                st.session_state[rating_key] = not st.session_state[rating_key]

        # ── Edit panel ────────────────────────────────────────────────────────
        if st.session_state.get(f"show_edit_{team_abbr}_{pid}", False):
            with st.container():
                st.markdown(
                    f'<div style="background:rgba(30,58,95,.25);border-left:3px solid #0891b2;'
                    f'border-radius:0 6px 6px 0;padding:8px 12px;margin:2px 0 6px;">'
                    f'<span style="font-size:.75rem;color:#7dd3fc;font-weight:700;">'
                    f'Adjustments — {p.player_name}</span></div>',
                    unsafe_allow_html=True,
                )

                ep1, ep2 = st.columns(2)

                with ep1:
                    # Injury rating (Active/Inactive handled by In/Out button in row)
                    inj_key = f"inj_{team_abbr}_{pid}"
                    if inj_key not in st.session_state:
                        st.session_state[inj_key] = inj_r
                    new_inj = st.select_slider(
                        "Injury rating",
                        options=[0.0, 0.25, 0.50, 0.75, 1.0],
                        value=st.session_state[inj_key],
                        format_func=lambda x: {
                            0.0: "0.0 — Healthy",
                            0.25: "0.25 — Questionable (likely)",
                            0.50: "0.50 — Questionable (doubtful)",
                            0.75: "0.75 — Doubtful",
                            1.00: "1.0 — Out",
                        }[x],
                        key=inj_key,
                    )
                    if abs(new_inj - inj_r) > 0.01:
                        set_player_override(team_abbr, pid, "injury_rating", new_inj)

                with ep2:
                    # Minutes override
                    min_ov = existing.get("minutes_override")
                    min_wk = f"min_ov_{team_abbr}_{pid}"
                    if min_wk not in st.session_state:
                        st.session_state[min_wk] = float(min_ov) if min_ov is not None else p.proj_min
                    new_min = st.number_input(
                        "Minutes override (0=use model)",
                        min_value=0.0, max_value=48.0, step=0.5,
                        key=min_wk,
                        help="Set to 0 to use the model's projected minutes.",
                    )
                    if new_min > 0 and (min_ov is None or abs(new_min - float(min_ov)) > 0.1):
                        set_player_override(team_abbr, pid, "minutes_override", new_min)
                    elif new_min == 0 and min_ov is not None:
                        dc.get(pid, {}).pop("minutes_override", None)
                        _autosave()

                # Close / Reset row
                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("Reset", key=f"rst_{team_abbr}_{pid}"):
                        if pid in st.session_state.depth_charts.get(team_abbr, {}):
                            st.session_state.depth_charts[team_abbr][pid] = {}
                        _autosave()
                        st.session_state[f"show_edit_{team_abbr}_{pid}"] = False
                        if game := st.session_state.get("selected_game"):
                            run_projection(engine, game)
                        st.rerun()
                with rc2:
                    if st.button("Close", key=f"cls_{team_abbr}_{pid}"):
                        st.session_state[f"show_edit_{team_abbr}_{pid}"] = False
                        st.rerun()

    st.markdown("")


tab_away, tab_home = st.tabs([f"🏀 {team_name(away_id)}", f"🏀 {team_name(home_id)}"])

with tab_away:
    _render_team(away_id, team_name(away_id), result.away_players)
with tab_home:
    _render_team(home_id, team_name(home_id), result.home_players)

st.markdown("---")
st.markdown(
    '<span class="note-text">'
    'Injury rating: 0.0=Healthy · 0.25=Questionable (likely) · '
    '0.50=Questionable (doubtful) · 0.75=Doubtful · 1.0=Out. '
    'Minutes override of 0 uses the model default. '
    'Click Update Projection to apply changes.'
    '</span>',
    unsafe_allow_html=True,
)
