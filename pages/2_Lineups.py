"""Page 2 — Lineups"""
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

st.set_page_config(page_title="Lineups · NBA", page_icon="🏀", layout="wide")
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

st.title("📋 Lineups")
st.markdown(f"**{team_name(away_id)} @ {team_name(home_id)}** · {str(game.get('game_date',''))[:10]}")

# ── Baseline result (no overrides) for delta display ─────────────────────────
def _any_override():
    for team_dc in st.session_state.get("depth_charts", {}).values():
        for pid, s in team_dc.items():
            if s.get("rating_overrides") or s.get("minutes_override") is not None:
                return True
    return False

if _any_override():
    cache_key = f"nba_baseline_{home_id}_{away_id}_{game.get('game_date','')}"
    if (st.session_state.get("_nba_baseline_key") != cache_key or
            st.session_state.get("_baseline_result") is None):
        baseline = engine.project(
            home_team_abbr=home_id,
            away_team_abbr=away_id,
            game_date=str(game.get("game_date", "")),
        )
        st.session_state["_baseline_result"] = baseline
        st.session_state["_nba_baseline_key"] = cache_key
else:
    st.session_state["_baseline_result"] = None

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

ALL_POS = ["G", "F", "C", "UNK"]

# Injury tag labels matching standard NBA injury report designations
INJ_OPTS = {
    0.00: "Healthy",      # Will play, no limitation
    0.25: "Probable",     # Very likely to play, minor issue
    0.50: "Questionable", # 50/50, could go either way
    0.75: "Doubtful",     # Unlikely to play
    1.00: "Out",          # Will not play
}
INJ_COLORS = {
    0.00: "#34d399",  # green
    0.25: "#a3e635",  # light green
    0.50: "#fbbf24",  # amber
    0.75: "#f97316",  # orange
    1.00: "#ef4444",  # red
}


def _render_team(team_abbr: str, team_nm: str, players):
    dc    = get_depth_chart(team_abbr)
    color = team_color(team_abbr)

    # Resolve team projection for this team
    if team_abbr == home_id:
        team_proj = result.home_proj
    else:
        team_proj = result.away_proj

    # Baseline projection (no overrides) for delta display
    baseline_result = st.session_state.get("_baseline_result")
    baseline_map: dict = {}
    if baseline_result:
        for bp in (baseline_result.home_players + baseline_result.away_players):
            baseline_map[bp.player_id] = bp

    st.markdown(
        f'<div style="border-left:4px solid {color};padding:4px 12px;margin-bottom:8px;">'
        f'<span style="font-size:1.05rem;font-weight:700;color:{color};">{team_nm}</span>'
        f'<span style="font-size:.75rem;color:#64748b;margin-left:12px;">Click IN/OUT to toggle · adjust minutes inline · set injury status</span>'
        f'</div>', unsafe_allow_html=True,
    )

    # ── Team totals summary bar ───────────────────────────────────────────────
    tc1, tc2, tc3, tc4, tc5, tc6 = st.columns(6)
    for tc, label, val in [
        (tc1, "Proj PTS",  team_proj.proj_pts),
        (tc2, "Proj REB",  team_proj.proj_reb),
        (tc3, "Proj AST",  team_proj.proj_ast),
        (tc4, "Proj 3PM",  team_proj.proj_fg3m),
        (tc5, "Win Prob",  team_proj.proj_win_prob * 100),
        (tc6, "Pace",      team_proj.proj_pace),
    ]:
        fmt = f"{val:.0f}%" if label == "Win Prob" else f"{val:.1f}"
        tc.metric(label, fmt)
    st.markdown('<hr style="margin:4px 0 8px;border-color:rgba(148,163,184,.20);">', unsafe_allow_html=True)

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

    # ── Pending starter swap — show "who to replace" dropdown ────────────────
    pending_pid = st.session_state.get(f"pending_start_{team_abbr}")
    if pending_pid and pending_pid in {p.player_id for p in active_players}:
        incoming = next((p for p in active_players if p.player_id == pending_pid), None)
        if incoming:
            current_starters = [p for p in active_players if p.player_id in starter_ids]
            starter_names = {p.player_id: f"{p.player_name} ({p.proj_pts:.1f} pts)"
                             for p in current_starters}
            st.markdown(
                f'<div style="background:rgba(8,145,178,.15);border:1px solid #0891b2;'
                f'border-radius:6px;padding:8px 14px;margin-bottom:8px;">'
                f'<span style="color:#7dd3fc;font-weight:700;">Promote {incoming.player_name} to starter — '
                f'who comes out?</span></div>',
                unsafe_allow_html=True,
            )
            swap_cols = st.columns([2, 1, 1])
            with swap_cols[0]:
                bump_pid = st.selectbox(
                    "Replace",
                    options=list(starter_names.keys()),
                    format_func=lambda x: starter_names[x],
                    key=f"swap_select_{team_abbr}",
                    label_visibility="collapsed",
                )
            with swap_cols[1]:
                if st.button("Confirm swap", key=f"swap_ok_{team_abbr}", type="primary"):
                    # Demote bumped player
                    if bump_pid in st.session_state.depth_charts.get(team_abbr, {}):
                        st.session_state.depth_charts[team_abbr][bump_pid].pop("is_starter", None)
                    set_player_override(team_abbr, bump_pid, "is_starter", False)
                    # Promote incoming
                    set_player_override(team_abbr, pending_pid, "is_starter", True)
                    st.session_state.pop(f"pending_start_{team_abbr}", None)
                    st.rerun()
            with swap_cols[2]:
                if st.button("Cancel", key=f"swap_cancel_{team_abbr}"):
                    st.session_state.pop(f"pending_start_{team_abbr}", None)
                    st.rerun()

    for section_label, section_players in [
        ("Starting Lineup", sorted(starters, key=lambda x: -x.proj_pts)),
        ("Bench", sorted(bench, key=lambda x: -x.proj_pts)),
        ("Inactive / Out", sorted(inactive_players, key=lambda x: -x.proj_pts)),
    ]:
        if not section_players:
            continue

        st.markdown(f'<div class="dc-section">{section_label} ({len(section_players)})</div>',
                    unsafe_allow_html=True)

        # Column headers: Player|Pos|⭐|In/Out|Min|PTS|REB|AST|3PM|PRA|Inj.|Edit
        # 12 columns — added 3PM between AST and PRA
        COL_W = [2.2, 0.5, 0.6, 0.65, 0.75, 0.72, 0.72, 0.72, 0.72, 0.72, 0.8, 0.55]
        hdr = st.columns(COL_W)
        for col, lbl in zip(hdr, ["Player","Pos","⭐","In/Out","Min","PTS","REB","AST","3PM","PRA","Inj.","Edit"]):
            col.markdown(f"<span style='font-size:.68rem;font-weight:700;color:#64748b;'>{lbl}</span>",
                         unsafe_allow_html=True)
        st.markdown('<hr style="margin:2px 0 4px;border-color:rgba(148,163,184,.12);">', unsafe_allow_html=True)

        for p in section_players:
            pid       = p.player_id
            existing  = dc.get(pid, {})
            is_active = bool(existing.get("active", True))
            inj_r     = float(existing.get("injury_rating", 0.0))
            is_strt   = pid in starter_ids
            has_ov    = bool(existing.get("minutes_override") is not None or
                             existing.get("rating_overrides") or inj_r > 0 or
                             existing.get("position_override"))
            opacity   = "" if is_active else "opacity:.3;"
            struck    = "text-decoration:line-through;color:#475569;" if not is_active else ""

            # Baseline player for delta display
            bp = baseline_map.get(pid)

            c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12 = st.columns(COL_W)

            # Player name
            with c1:
                mod = ' <span style="color:#fbbf24;font-size:.68rem;">⚡</span>' if has_ov else ""
                st.markdown(
                    f'<span style="{struck}{opacity}font-size:.84rem;">{p.player_name}</span>{mod}',
                    unsafe_allow_html=True,
                )

            # Position — clickable to override
            with c2:
                pos_ov_key = f"pos_ov_{team_abbr}_{pid}"
                cur_pos = existing.get("position_override", p.pos_group)
                if pos_ov_key not in st.session_state:
                    st.session_state[pos_ov_key] = cur_pos
                new_pos = st.selectbox(
                    "", options=ALL_POS,
                    index=ALL_POS.index(cur_pos) if cur_pos in ALL_POS else 0,
                    key=pos_ov_key, label_visibility="collapsed",
                    help="Override player position — affects rate defaults for players with sparse history",
                )
                if new_pos != p.pos_group or "position_override" in existing:
                    if new_pos != p.pos_group:
                        set_player_override(team_abbr, pid, "position_override", new_pos)
                    elif "position_override" in existing and new_pos == p.pos_group:
                        st.session_state.depth_charts[team_abbr][pid].pop("position_override", None)
                        _autosave()

            # Starter toggle — when adding a starter, show dropdown to pick who gets bumped
            with c3:
                if is_active:
                    if st.button("⭐" if is_strt else "☆",
                                 key=f"strt_{team_abbr}_{pid}",
                                 help="Star = starter. Click bench player to promote; click starter to demote."):
                        if is_strt:
                            if pid in st.session_state.depth_charts.get(team_abbr, {}):
                                st.session_state.depth_charts[team_abbr][pid].pop("is_starter", None)
                            _autosave()
                            st.rerun()
                        else:
                            # If < 5 starters, just add. If 5 already, ask who to bump via session state flag.
                            current_starter_ids = {p2.player_id for p2 in active_players
                                                   if dc.get(p2.player_id, {}).get("is_starter")}
                            auto_ids = {p2.player_id for p2 in starters
                                        if p2.player_id not in current_starter_ids}
                            total_starters = current_starter_ids | auto_ids
                            if len(total_starters) >= 5:
                                # Store pending starter swap request
                                st.session_state[f"pending_start_{team_abbr}"] = pid
                            else:
                                set_player_override(team_abbr, pid, "is_starter", True)
                            st.rerun()
                else:
                    st.markdown('<span style="color:#334155;font-size:.8rem;">—</span>', unsafe_allow_html=True)

            # In/Out button
            with c4:
                if st.button("IN" if is_active else "OUT", key=f"inout_{team_abbr}_{pid}",
                             help="Toggle active/inactive — auto-reallocates minutes to remaining players"):
                    new_active = not is_active
                    set_player_override(team_abbr, pid, "active", new_active)
                    if not new_active:
                        set_player_override(team_abbr, pid, "injury_rating", 1.0)
                        st.session_state[f"inj_{team_abbr}_{pid}"] = 1.0
                    else:
                        if existing.get("injury_rating", 0.0) >= 1.0:
                            set_player_override(team_abbr, pid, "injury_rating", 0.0)
                            st.session_state[f"inj_{team_abbr}_{pid}"] = 0.0
                    if game := st.session_state.get("selected_game"):
                        run_projection(engine, game)
                    st.rerun()

            # Inline minutes input
            with c5:
                if is_active:
                    min_key   = f"min_{team_abbr}_{pid}"
                    saved_min = existing.get("minutes_override")
                    if min_key not in st.session_state:
                        st.session_state[min_key] = float(saved_min) if saved_min else round(p.proj_min, 1)
                    new_min = st.number_input(
                        "", min_value=0.0, max_value=48.0, step=0.5,
                        key=min_key, label_visibility="collapsed", format="%.1f",
                    )
                    # Show baseline below if overridden
                    if bp and abs(p.proj_min - bp.proj_min) > 0.3:
                        delta_c = "#34d399" if p.proj_min > bp.proj_min else "#f87171"
                        st.markdown(f'<div style="font-size:.60rem;color:{delta_c};margin-top:-4px;">was {bp.proj_min:.1f}</div>', unsafe_allow_html=True)
                    if saved_min is None and abs(new_min - p.proj_min) > 0.3:
                        set_player_override(team_abbr, pid, "minutes_override", new_min)
                        if game := st.session_state.get("selected_game"):
                            run_projection(engine, game)
                        st.rerun()
                    elif saved_min is not None and abs(new_min - float(saved_min)) > 0.3:
                        set_player_override(team_abbr, pid, "minutes_override", new_min)
                        if game := st.session_state.get("selected_game"):
                            run_projection(engine, game)
                        st.rerun()
                else:
                    st.markdown('<span style="font-size:.78rem;color:#475569;">OUT</span>', unsafe_allow_html=True)

            # Stats display with baseline delta below each value
            def _stat_cell(col, val, baseline_val, thresh, active):
                color_v = "#34d399" if val >= thresh else "#94a3b8"
                display = "--" if not active else f"{val:.1f}"
                col.markdown(
                    f'<span style="{opacity}font-size:.82rem;color:{color_v};">{display}</span>',
                    unsafe_allow_html=True,
                )
                if active and baseline_val is not None and abs(val - baseline_val) > 0.2:
                    delta_c = "#34d399" if val > baseline_val else "#f87171"
                    col.markdown(
                        f'<div style="font-size:.60rem;color:{delta_c};margin-top:-4px;">was {baseline_val:.1f}</div>',
                        unsafe_allow_html=True,
                    )

            _stat_cell(c6,  p.proj_pts,  bp.proj_pts  if bp else None, 20,  is_active)
            _stat_cell(c7,  p.proj_reb,  bp.proj_reb  if bp else None,  8,  is_active)
            _stat_cell(c8,  p.proj_ast,  bp.proj_ast  if bp else None,  7,  is_active)
            _stat_cell(c9,  p.proj_fg3m, bp.proj_fg3m if bp else None,  3,  is_active)
            _stat_cell(c10, p.proj_pra,  bp.proj_pra  if bp else None, 35,  is_active)

            # Inline injury status
            with c11:
                if is_active:
                    inj_key = f"inj_{team_abbr}_{pid}"
                    if inj_key not in st.session_state:
                        st.session_state[inj_key] = inj_r
                    closest = min(INJ_OPTS.keys(), key=lambda x: abs(x - inj_r))
                    new_inj = st.selectbox(
                        "", options=list(INJ_OPTS.keys()),
                        index=list(INJ_OPTS.keys()).index(closest),
                        format_func=lambda x: INJ_OPTS[x],
                        key=inj_key,
                        label_visibility="collapsed",
                        help="Healthy=100% play | Probable=~90% | Questionable=50/50 | Doubtful=~10% | Out=0%",
                    )
                    if abs(new_inj - inj_r) > 0.01:
                        set_player_override(team_abbr, pid, "injury_rating", new_inj)
                        if new_inj >= 1.0:
                            set_player_override(team_abbr, pid, "active", False)
                            st.session_state[f"inout_{team_abbr}_{pid}"] = False
                        if game := st.session_state.get("selected_game"):
                            run_projection(engine, game)
                        st.rerun()
                    # Show colored label
                    inj_color = INJ_COLORS.get(closest, "#94a3b8")
                    inj_label = INJ_OPTS.get(closest, "")
                    st.markdown(
                        f'<div style="font-size:.65rem;color:{inj_color};'
                        f'font-weight:700;margin-top:-4px;">{inj_label}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown('<span style="font-size:.75rem;color:#ef4444;font-weight:700;">Out</span>', unsafe_allow_html=True)

            # Edit panel toggle
            with c12:
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
                        f'Rate Overrides — {p.player_name}</span>'
                        f'<span style="font-size:.68rem;color:#64748b;margin-left:8px;">'
                        f'Model defaults shown — change to override projections</span>'
                        f'</div>', unsafe_allow_html=True,
                    )

                    rating_overrides = existing.get("rating_overrides", {})

                    # Rating definitions: key, label, help, min, max, step, model_default
                    _RATINGS = [
                        ("PTS_PM_EWM",  "Pts / Min",   "Points per minute rate. Model avg ~0.50.", 0.0, 2.0, 0.01,  p.proj_pts / max(p.proj_min, 1.0)),
                        ("REB_PM_EWM",  "Reb / Min",   "Rebounds per minute rate.",               0.0, 1.2, 0.01,  p.proj_reb / max(p.proj_min, 1.0)),
                        ("AST_PM_EWM",  "Ast / Min",   "Assists per minute rate.",                0.0, 0.8, 0.01,  p.proj_ast / max(p.proj_min, 1.0)),
                        ("FG3M_PM_EWM", "3PM / Min",   "3-pointers made per minute.",             0.0, 0.5, 0.005, p.proj_fg3m / max(p.proj_min, 1.0)),
                        ("FG3_PCT_EWM", "3PT %",       "3-point shooting percentage.",            0.0, 0.65,0.01,  p.fg3_pct),
                        ("FG_PCT_EWM",  "FG %",        "Field goal percentage.",                  0.25, 0.75,0.01, p.fg_pct),
                    ]

                    r1, r2, r3 = st.columns(3)
                    _rate_cols = [r1, r2, r3, r1, r2, r3]

                    _changed = False
                    for (rkey, rlabel, rhelp, rmin, rmax, rstep, rdefault), rcol in zip(_RATINGS, _rate_cols):
                        saved_rv = rating_overrides.get(rkey)
                        wk = f"rate_{team_abbr}_{pid}_{rkey}"
                        # Always reseed from saved or model default
                        display_val = float(saved_rv) if saved_rv is not None else round(float(rdefault), 4)
                        display_val = float(np.clip(display_val, rmin, rmax))
                        if wk not in st.session_state:
                            st.session_state[wk] = display_val
                        with rcol:
                            new_rv = st.number_input(
                                rlabel, min_value=rmin, max_value=rmax, step=rstep,
                                key=wk, help=rhelp, format="%.3f",
                            )
                            if saved_rv is None:
                                st.markdown(f'<div style="font-size:.62rem;color:#64748b;margin-top:-6px;">model: {rdefault:.3f}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div style="font-size:.62rem;color:#fbbf24;margin-top:-6px;">override</div>', unsafe_allow_html=True)
                            # Save if changed from model default
                            if abs(new_rv - display_val) > rstep * 0.1:
                                from _engine_state import set_player_rating
                                set_player_rating(team_abbr, pid, rkey, new_rv)
                                _changed = True
                            elif saved_rv is not None and abs(new_rv - float(saved_rv)) < rstep * 0.1:
                                pass  # unchanged saved value — fine

                    if _changed:
                        if game := st.session_state.get("selected_game"):
                            run_projection(engine, game)
                        st.rerun()

                    _rc_reset, _ = st.columns([1, 3])
                    with _rc_reset:
                        if st.button("Reset all rates", key=f"rst_{team_abbr}_{pid}"):
                            if pid in st.session_state.depth_charts.get(team_abbr, {}):
                                st.session_state.depth_charts[team_abbr][pid].pop("minutes_override", None)
                                st.session_state.depth_charts[team_abbr][pid].pop("rating_overrides", None)
                            # Clear widget state
                            for rkey, *_ in _RATINGS:
                                wk = f"rate_{team_abbr}_{pid}_{rkey}"
                                st.session_state.pop(wk, None)
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
