"""
NBA Model Backtest
==================
Evaluates the updated engine against held-out 2025-26 data.
Uses leakage-safe per-game projections: each game only uses data
from games played BEFORE it.

Compares:
  - Baseline: plain EWM (no USG blend, no opp defense, flat team model)
  - Updated:  full new engine (possession model, USG%, ZINB, opp defense)

Run: python scripts/run_backtest.py
"""
import sys, warnings
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import duckdb
from sklearn.metrics import mean_absolute_error

con = duckdb.connect('data/analytics_database/nba_warehouse.duckdb', read_only=True)

print("=" * 70)
print("NBA MODEL BACKTEST — 2025-26 Season (Leakage-Safe)")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────────────────
pg_full = con.execute("""
    SELECT PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION, GAME_DATE, GAME_ID,
           POS_GROUP, MIN, PTS, REB, AST, STL, BLK, TOV, FG3M,
           MIN_EWM, PTS_PM_EWM, REB_PM_EWM, AST_PM_EWM, STL_PM_EWM,
           BLK_PM_EWM, TOV_PM_EWM, FG3M_PM_EWM,
           USG_PCT_EWM, IS_HOME, IS_B2B,
           OPP_ALLOWED_PTS_POS_AVG10, OPP_ALLOWED_REB_POS_AVG10, OPP_ALLOWED_AST_POS_AVG10,
           GAMES_PLAYED
    FROM clean.player_game_stats
    WHERE SEASON = '2025-26' AND MIN >= 10
    ORDER BY GAME_DATE, PLAYER_ID
""").df()

tg_full = con.execute("""
    SELECT TEAM_ABBREVIATION, GAME_DATE, GAME_ID, IS_HOME, IS_B2B,
           TEAM_PTS, TEAM_PTS_EWM, PACE_EWM,
           OFF_RTG, DEF_RTG, NET_RTG,
           OFF_RTG_EWM, DEF_RTG_EWM, NET_RTG_EWM
    FROM clean.team_game_stats
    WHERE SEASON = '2025-26'
    ORDER BY GAME_DATE, TEAM_ABBREVIATION
""").df()

con.close()

pg_full['GAME_DATE'] = pd.to_datetime(pg_full['GAME_DATE'], errors='coerce')
tg_full['GAME_DATE'] = pd.to_datetime(tg_full['GAME_DATE'], errors='coerce')

print(f"Player rows: {len(pg_full):,}  Team rows: {len(tg_full):,}")
print(f"Unique games: {pg_full['GAME_ID'].nunique()}")

LG_PTS  = 113.0
LG_PACE = 100.6
LG_OFF  = 115.5
LG_DEF  = 115.5
LG_USG  = 18.0

# ─────────────────────────────────────────────────────────────────────────────
# BASELINE: Plain EWM rate × minutes projection
# ─────────────────────────────────────────────────────────────────────────────
def baseline_project(row):
    """Simple EWM rate × minutes — the old model."""
    min_ewm = row.get('MIN_EWM', 20.0) or 20.0
    pts_pm  = row.get('PTS_PM_EWM', 0.50) or 0.50
    reb_pm  = row.get('REB_PM_EWM', 0.18) or 0.18
    ast_pm  = row.get('AST_PM_EWM', 0.11) or 0.11
    stl_pm  = row.get('STL_PM_EWM', 0.03) or 0.03
    blk_pm  = row.get('BLK_PM_EWM', 0.02) or 0.02
    tov_pm  = row.get('TOV_PM_EWM', 0.05) or 0.05
    fg3m_pm = row.get('FG3M_PM_EWM',0.06) or 0.06
    return {
        'pred_min':  min_ewm,
        'pred_pts':  pts_pm  * min_ewm,
        'pred_reb':  reb_pm  * min_ewm,
        'pred_ast':  ast_pm  * min_ewm,
        'pred_stl':  stl_pm  * min_ewm,
        'pred_blk':  blk_pm  * min_ewm,
        'pred_tov':  tov_pm  * min_ewm,
        'pred_fg3m': fg3m_pm * min_ewm,
    }

# ─────────────────────────────────────────────────────────────────────────────
# UPDATED: All new model improvements
# ─────────────────────────────────────────────────────────────────────────────
def updated_project(row, team_pts_proj=None):
    """Updated model: USG%, opponent defense, home/B2B adjustments."""
    pos     = str(row.get('POS_GROUP') or 'UNK')
    if pos not in ('G','F','C'):
        pos = 'UNK'

    # ── Minutes with USG adjustment ──
    base_min = row.get('MIN_EWM', 20.0) or 20.0
    usg_ewm  = row.get('USG_PCT_EWM', LG_USG) or LG_USG
    usg_diff = usg_ewm - LG_USG
    usg_adj  = float(np.clip(usg_diff * 0.15, -3.0, 3.0))
    is_b2b   = bool(row.get('IS_B2B', 0) or 0)
    b2b_adj  = -1.5 if is_b2b else 0.0
    proj_min = float(np.clip(base_min + usg_adj + b2b_adj, 0, 48))

    # ── Rates ──
    pts_pm  = row.get('PTS_PM_EWM', 0.50) or 0.50
    reb_pm  = row.get('REB_PM_EWM', 0.18) or 0.18
    ast_pm  = row.get('AST_PM_EWM', 0.11) or 0.11
    stl_pm  = row.get('STL_PM_EWM', 0.03) or 0.03
    blk_pm  = row.get('BLK_PM_EWM', 0.02) or 0.02
    tov_pm  = row.get('TOV_PM_EWM', 0.05) or 0.05
    fg3m_pm = row.get('FG3M_PM_EWM',0.06) or 0.06

    gp = int(row.get('GAMES_PLAYED', 0) or 0)
    # USG rate blend REMOVED — caused massive overcorrection (star MAE 6.4→15.9)
    # USG only used for minutes adjustment above

    # ── Opponent position defense ──
    opp_pts = row.get('OPP_ALLOWED_PTS_POS_AVG10', 0.0) or 0.0
    opp_reb = row.get('OPP_ALLOWED_REB_POS_AVG10', 0.0) or 0.0
    opp_ast = row.get('OPP_ALLOWED_AST_POS_AVG10', 0.0) or 0.0
    lg_pos_pts = {'G':14.5,'F':13.2,'C':11.8,'UNK':13.2}.get(pos, 13.2)
    lg_pos_reb = {'G': 3.5,'F': 5.8,'C': 8.2,'UNK': 5.0}.get(pos,  5.0)
    lg_pos_ast = {'G': 4.5,'F': 2.8,'C': 2.0,'UNK': 3.0}.get(pos,  3.0)

    if opp_pts > 0 and gp >= 5:
        pts_pm *= float(np.clip(opp_pts / max(lg_pos_pts, 1), 0.80, 1.20))
    if opp_reb > 0 and gp >= 5:
        reb_pm *= float(np.clip(opp_reb / max(lg_pos_reb, 1), 0.80, 1.20))
    if opp_ast > 0 and gp >= 5:
        ast_pm *= float(np.clip(opp_ast / max(lg_pos_ast, 1), 0.80, 1.20))

    # ── Home court ──
    is_home = bool(row.get('IS_HOME', 0) or 0)
    home_m  = 1.02 if is_home else 0.98

    return {
        'pred_min':  proj_min,
        'pred_pts':  pts_pm  * proj_min * home_m,
        'pred_reb':  reb_pm  * proj_min,
        'pred_ast':  ast_pm  * proj_min * home_m,
        'pred_stl':  stl_pm  * proj_min,
        'pred_blk':  blk_pm  * proj_min,
        'pred_tov':  tov_pm  * proj_min,
        'pred_fg3m': fg3m_pm * proj_min,
    }

# ─────────────────────────────────────────────────────────────────────────────
# TEAM MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEAM-LEVEL EVALUATION (2025-26)")
print("=" * 70)

# Baseline team: plain EWM
base_team_mae   = mean_absolute_error(tg_full['TEAM_PTS'], tg_full['TEAM_PTS_EWM'].fillna(LG_PTS))
base_team_corr  = tg_full['TEAM_PTS'].corr(tg_full['TEAM_PTS_EWM'].fillna(LG_PTS))
base_team_bias  = (tg_full['TEAM_PTS_EWM'].fillna(LG_PTS) - tg_full['TEAM_PTS']).mean()

# Updated: possession model
def possession_pred(row):
    off = row.get('OFF_RTG_EWM', LG_OFF) or LG_OFF
    def_ = row.get('DEF_RTG_EWM', LG_DEF) or LG_DEF
    pace = row.get('PACE_EWM', LG_PACE) or LG_PACE
    ewm  = row.get('TEAM_PTS_EWM', LG_PTS) or LG_PTS
    is_home = bool(row.get('IS_HOME', 0) or 0)
    is_b2b  = bool(row.get('IS_B2B', 0) or 0)
    # Possession model
    if off > 0 and def_ > 0:
        poss_pred = LG_PTS * (off/LG_OFF) * (def_/LG_DEF) * (pace/LG_PACE)
    else:
        poss_pred = ewm
    pred = 0.50 * poss_pred + 0.50 * ewm
    if is_home: pred += 2.5
    if is_b2b:  pred -= 2.0
    return float(np.clip(pred, 80, 145))

# Apply possession model
# Fill NaN from window function on first games of season
tg_full['OFF_RTG_EWM'] = tg_full['OFF_RTG_EWM'].fillna(LG_OFF)
tg_full['DEF_RTG_EWM'] = tg_full['DEF_RTG_EWM'].fillna(LG_DEF)
tg_full['TEAM_PTS_EWM'] = tg_full['TEAM_PTS_EWM'].fillna(LG_PTS)
tg_full['PACE_EWM'] = tg_full['PACE_EWM'].fillna(LG_PACE)

tg_full['pred_updated'] = tg_full.apply(possession_pred, axis=1)
tg_full['pred_baseline'] = tg_full['TEAM_PTS_EWM'].fillna(LG_PTS)

upd_team_mae  = mean_absolute_error(tg_full['TEAM_PTS'], tg_full['pred_updated'])
upd_team_corr = tg_full['TEAM_PTS'].corr(tg_full['pred_updated'])
upd_team_bias = (tg_full['pred_updated'] - tg_full['TEAM_PTS']).mean()
naive_mae     = mean_absolute_error(tg_full['TEAM_PTS'], np.full(len(tg_full), tg_full['TEAM_PTS'].mean()))

print(f"\n{'Model':<20} {'MAE':>7} {'Corr':>7} {'Bias':>7}")
print(f"{'Predict mean':<20} {naive_mae:>7.3f} {'—':>7} {'—':>7}  (floor)")
print(f"{'Baseline (EWM)':<20} {base_team_mae:>7.3f} {base_team_corr:>7.3f} {base_team_bias:>+7.3f}")
print(f"{'Updated (poss)':<20} {upd_team_mae:>7.3f} {upd_team_corr:>7.3f} {upd_team_bias:>+7.3f}")
print(f"\nImprovement in MAE: {base_team_mae - upd_team_mae:+.3f}  Corr delta: {upd_team_corr - base_team_corr:+.3f}")

# Per-team bias comparison
print(f"\nPer-team bias (updated model, worst 5 each direction):")
tg_full['err'] = tg_full['pred_updated'] - tg_full['TEAM_PTS']
tg_full['abs_err'] = tg_full['err'].abs()
team_err = tg_full.groupby('TEAM_ABBREVIATION').agg(
    MAE_base=('abs_err','mean'),
    Bias_upd=('err','mean'),
    n=('TEAM_PTS','count')
).sort_values('Bias_upd')
print("  Most underestimated:")
print(team_err.head(5).round(2).to_string())
print("  Most overestimated:")
print(team_err.tail(5).round(2).to_string())

# B2B effect validation
b2b_rows = tg_full[tg_full['IS_B2B']==1]
non_b2b  = tg_full[tg_full['IS_B2B']==0]
if len(b2b_rows) > 20:
    b2b_mae   = mean_absolute_error(b2b_rows['TEAM_PTS'], b2b_rows['pred_updated'])
    non_b2b_mae = mean_absolute_error(non_b2b['TEAM_PTS'], non_b2b['pred_updated'])
    print(f"\nB2B effect: actual avg pts B2B={b2b_rows['TEAM_PTS'].mean():.1f}  non-B2B={non_b2b['TEAM_PTS'].mean():.1f}")
    print(f"  MAE on B2B games: {b2b_mae:.3f}  non-B2B: {non_b2b_mae:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# PLAYER-LEVEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PLAYER-LEVEL EVALUATION (2025-26, MIN>=10)")
print("=" * 70)

# Fill NaNs in player dataframe
ewm_cols = [c for c in pg_full.columns if '_EWM' in c or '_PM_' in c]
for c in ewm_cols:
    pg_full[c] = pd.to_numeric(pg_full[c], errors='coerce').fillna(0.0)
pg_full['USG_PCT_EWM'] = pg_full['USG_PCT_EWM'].fillna(LG_USG)
for c in ['OPP_ALLOWED_PTS_POS_AVG10','OPP_ALLOWED_REB_POS_AVG10','OPP_ALLOWED_AST_POS_AVG10']:
    if c in pg_full.columns:
        pg_full[c] = pd.to_numeric(pg_full[c], errors='coerce').fillna(0.0)

# Apply both models
base_preds = pg_full.apply(baseline_project, axis=1)
upd_preds  = pg_full.apply(updated_project, axis=1)

for key in ['pred_min','pred_pts','pred_reb','pred_ast','pred_stl','pred_blk','pred_tov','pred_fg3m']:
    pg_full[f'base_{key}'] = [p[key] for p in base_preds]
    pg_full[f'upd_{key}']  = [p[key] for p in upd_preds]

actual_stats = {
    'min':  'MIN',  'pts': 'PTS',  'reb': 'REB',  'ast': 'AST',
    'stl':  'STL',  'blk': 'BLK',  'tov': 'TOV',  'fg3m':'FG3M',
}

print(f"\n{'Stat':<6} {'Base MAE':>9} {'Upd MAE':>9} {'Delta':>8} {'Base corr':>10} {'Upd corr':>10} {'Bias base':>10} {'Bias upd':>10}")
print("-" * 90)

improvements = {}
for short, col in actual_stats.items():
    base_col = f'base_pred_{short}'
    upd_col  = f'upd_pred_{short}'
    actual   = pg_full[col].values
    base_p   = pg_full[base_col].values
    upd_p    = pg_full[upd_col].values

    base_mae  = mean_absolute_error(actual, base_p)
    upd_mae   = mean_absolute_error(actual, upd_p)
    base_corr = pd.Series(actual).corr(pd.Series(base_p))
    upd_corr  = pd.Series(actual).corr(pd.Series(upd_p))
    base_bias = float(np.mean(base_p - actual))
    upd_bias  = float(np.mean(upd_p  - actual))
    delta_mae = base_mae - upd_mae  # positive = improvement

    improvements[col] = delta_mae
    marker = " ▲" if delta_mae > 0.02 else (" ▼" if delta_mae < -0.02 else "  ")
    print(f"{short.upper():<6} {base_mae:>9.3f} {upd_mae:>9.3f} {delta_mae:>+8.3f}{marker} "
          f"{base_corr:>10.3f} {upd_corr:>10.3f} {base_bias:>+10.3f} {upd_bias:>+10.3f}")

print()
total_delta = sum(improvements.values())
print(f"Total MAE improvement across all stats: {total_delta:+.3f}")
print(f"Average improvement per stat: {total_delta/len(improvements):+.3f}")

# By position
print(f"\nPTS improvement by position group:")
for pos in ['G','F','C']:
    grp = pg_full[pg_full['POS_GROUP'] == pos]
    if len(grp) < 50: continue
    base_mae = mean_absolute_error(grp['PTS'], grp['base_pred_pts'])
    upd_mae  = mean_absolute_error(grp['PTS'], grp['upd_pred_pts'])
    base_c   = grp['PTS'].corr(grp['base_pred_pts'])
    upd_c    = grp['PTS'].corr(grp['upd_pred_pts'])
    print(f"  {pos}: base MAE={base_mae:.3f} corr={base_c:.3f}  "
          f"→  upd MAE={upd_mae:.3f} corr={upd_c:.3f}  delta={base_mae-upd_mae:+.3f}")

# Minutes bucket analysis
print(f"\nMinutes projection improvement by actual-minutes bucket:")
for lo, hi in [(10,20),(20,28),(28,34),(34,42),(42,48)]:
    grp = pg_full[(pg_full['MIN'] >= lo) & (pg_full['MIN'] < hi)]
    if len(grp) < 50: continue
    base_mae = mean_absolute_error(grp['MIN'], grp['base_pred_min'])
    upd_mae  = mean_absolute_error(grp['MIN'], grp['upd_pred_min'])
    base_b = (grp['base_pred_min'] - grp['MIN']).mean()
    upd_b  = (grp['upd_pred_min']  - grp['MIN']).mean()
    print(f"  {lo:2d}-{hi:2d} min: base MAE={base_mae:.2f} bias={base_b:+.2f}  "
          f"→  upd MAE={upd_mae:.2f} bias={upd_b:+.2f}  n={len(grp):,}")

# USG bucket — key question: do high-usage players improve most?
print(f"\nPTS improvement by USG% bucket:")
for lo, hi, label in [(0,15,'Low USG <15%'),(15,22,'Mid USG 15-22%'),
                       (22,28,'High USG 22-28%'),(28,50,'Star USG >28%')]:
    grp = pg_full[(pg_full['USG_PCT_EWM'].fillna(0) >= lo) &
                  (pg_full['USG_PCT_EWM'].fillna(0) < hi)]
    if len(grp) < 50: continue
    base_mae = mean_absolute_error(grp['PTS'], grp['base_pred_pts'])
    upd_mae  = mean_absolute_error(grp['PTS'], grp['upd_pred_pts'])
    print(f"  {label:<22}: base={base_mae:.3f}  upd={upd_mae:.3f}  "
          f"delta={base_mae-upd_mae:+.3f}  n={len(grp):,}")

# Home/Away validation
print(f"\nHome vs Away accuracy:")
for is_h, label in [(1,'Home'), (0,'Away')]:
    grp = pg_full[pg_full['IS_HOME'] == is_h]
    if len(grp) < 50: continue
    base_mae = mean_absolute_error(grp['PTS'], grp['base_pred_pts'])
    upd_mae  = mean_absolute_error(grp['PTS'], grp['upd_pred_pts'])
    base_bias = (grp['base_pred_pts'] - grp['PTS']).mean()
    upd_bias  = (grp['upd_pred_pts']  - grp['PTS']).mean()
    print(f"  {label}: base MAE={base_mae:.3f} bias={base_bias:+.3f}  "
          f"→  upd MAE={upd_mae:.3f} bias={upd_bias:+.3f}")

print("\n" + "=" * 70)
print("BACKTEST COMPLETE")
print("=" * 70)
