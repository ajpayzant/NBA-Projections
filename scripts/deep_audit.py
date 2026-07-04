"""
NBA Model Deep Audit
====================
Systematic investigation of:
  1. Data pipeline integrity — are EWM values actually leakage-safe and correct?
  2. Feature validity — do our features actually predict what we think?
  3. Model architecture — are we using the right approach for each target?
  4. Distribution assumptions — do our priors match actual NBA distributions?
  5. What's missing vs what's noise?
"""
import sys, warnings
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import duckdb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

con = duckdb.connect('data/analytics_database/nba_warehouse.duckdb', read_only=True)

print("=" * 70)
print("NBA MODEL DEEP AUDIT")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA INTEGRITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] DATA INTEGRITY CHECK")
print("-" * 50)

# 1a. Are EWM values actually leakage-safe? (should be NaN on first game)
first_games = con.execute("""
    SELECT COUNT(*) n_first_games,
           SUM(CASE WHEN MIN_EWM IS NULL OR MIN_EWM = 0 THEN 1 ELSE 0 END) null_ewm_first,
           SUM(CASE WHEN PTS_PM_EWM IS NULL OR PTS_PM_EWM = 0 THEN 1 ELSE 0 END) null_pts_ewm_first
    FROM (
        SELECT PLAYER_ID, MIN_EWM, PTS_PM_EWM,
               ROW_NUMBER() OVER (PARTITION BY PLAYER_ID ORDER BY GAME_DATE) rn
        FROM clean.player_game_stats WHERE MIN > 5
    ) WHERE rn = 1
""").fetchone()
print(f"  First-game EWM null/zero rate: MIN_EWM={first_games[1]}/{first_games[0]} ({first_games[1]/first_games[0]:.0%}), PTS_PM_EWM={first_games[2]}/{first_games[0]} ({first_games[2]/first_games[0]:.0%})")
print(f"  Expected: ~100% (first game has no prior history)")
if first_games[1]/first_games[0] < 0.5:
    print("  WARNING: First games should have null/zero EWM — data leakage possible!")
else:
    print("  OK: EWM leakage-safe on first games")

# 1b. Are EWM values actually exponentially weighted? Check a known player
luka = con.execute("""
    SELECT GAME_DATE, MIN, MIN_EWM, PTS, PTS_PM_EWM
    FROM clean.player_game_stats
    WHERE PLAYER_NAME LIKE '%Doncic%' AND SEASON='2025-26' AND MIN > 10
    ORDER BY GAME_DATE LIMIT 8
""").df()
print(f"\n  EWM check for Luka Doncic (2025-26):")
print(f"  {'Date':<12} {'MIN':>5} {'MIN_EWM':>8} {'PTS':>5} {'PTS_PM_EWM':>11}")
for _, r in luka.iterrows():
    print(f"  {str(r['GAME_DATE'])[:10]:<12} {r['MIN']:>5.1f} {r['MIN_EWM']:>8.1f} {r['PTS']:>5.1f} {r['PTS_PM_EWM']:>11.4f}")

# 1c. Do team stats look right?
team_sanity = con.execute("""
    SELECT AVG(TEAM_PTS) avg_pts, STDDEV(TEAM_PTS) std_pts,
           AVG(PACE) avg_pace, AVG(OFF_RTG) avg_off, AVG(DEF_RTG) avg_def,
           COUNT(*) n
    FROM clean.team_game_stats WHERE SEASON='2025-26'
""").fetchone()
print(f"\n  Team stats sanity (2025-26): avg_pts={team_sanity[0]:.1f} (expect 112-116), "
      f"std={team_sanity[1]:.1f} (expect 11-13), pace={team_sanity[2]:.1f} (expect 98-103)")
print(f"  OFF_RTG={team_sanity[3]:.1f}, DEF_RTG={team_sanity[4]:.1f} (both should be ~115)")

# 1d. OFF_RTG vs DEF_RTG symmetry check — league-wide they must equal
off_def_gap = abs((team_sanity[3] or 0) - (team_sanity[4] or 0))
print(f"  OFF/DEF gap: {off_def_gap:.2f} (should be ~0 — they're mirror stats)")
if off_def_gap > 5:
    print("  WARNING: Large gap suggests computation error in DEF_RTG")
else:
    print("  OK: OFF/DEF roughly symmetric")

# 1e. Check opponent context columns exist and are populated
opp_cols = con.execute("""
    SELECT
        AVG(CASE WHEN OPP_ALLOWED_PTS_POS_AVG10 IS NOT NULL AND OPP_ALLOWED_PTS_POS_AVG10 > 0 THEN 1.0 ELSE 0 END) pct_pts,
        AVG(CASE WHEN OPP_ALLOWED_REB_POS_AVG10 IS NOT NULL AND OPP_ALLOWED_REB_POS_AVG10 > 0 THEN 1.0 ELSE 0 END) pct_reb,
        AVG(CASE WHEN OPP_ALLOWED_AST_POS_AVG10 IS NOT NULL AND OPP_ALLOWED_AST_POS_AVG10 > 0 THEN 1.0 ELSE 0 END) pct_ast,
        AVG(CASE WHEN USG_PCT_EWM IS NOT NULL AND USG_PCT_EWM > 0 THEN 1.0 ELSE 0 END) pct_usg
    FROM clean.player_game_stats WHERE MIN > 10
""").fetchone()
print(f"\n  Opponent context columns populated (MIN>10): "
      f"OPP_PTS={opp_cols[0]:.0%}, OPP_REB={opp_cols[1]:.0%}, OPP_AST={opp_cols[2]:.0%}, USG={opp_cols[3]:.0%}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: WHAT ACTUALLY PREDICTS NBA GAME SCORING?
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] WHAT ACTUALLY PREDICTS TEAM SCORING?")
print("-" * 50)

# Load team data for proper analysis
tg = con.execute("""
    SELECT t.TEAM_ABBREVIATION, t.GAME_DATE, t.SEASON,
           t.TEAM_PTS, t.TEAM_PTS_EWM, t.PACE_EWM, t.PACE,
           t.OFF_RTG, t.DEF_RTG, t.OFF_RTG_EWM, t.DEF_RTG_EWM, t.NET_RTG_EWM,
           t.IS_HOME, t.IS_B2B, t.REST_DAYS,
           -- Opponent ratings
           o.OFF_RTG_EWM as OPP_OFF_EWM,
           o.DEF_RTG_EWM as OPP_DEF_EWM,
           o.PACE_EWM as OPP_PACE_EWM,
           o.NET_RTG_EWM as OPP_NET_EWM,
           -- Game pace (avg of both teams)
           (t.PACE_EWM + o.PACE_EWM) / 2 as GAME_PACE_EWM
    FROM clean.team_game_stats t
    JOIN clean.team_game_stats o
      ON t.GAME_ID = o.GAME_ID AND t.TEAM_ABBREVIATION != o.TEAM_ABBREVIATION
    WHERE t.SEASON >= '2022-23'
    ORDER BY t.GAME_DATE
""").df()

tg = tg.dropna(subset=['TEAM_PTS','TEAM_PTS_EWM'])
tg['OFF_RTG_EWM'] = tg['OFF_RTG_EWM'].fillna(115.5)
tg['OPP_DEF_EWM'] = tg['OPP_DEF_EWM'].fillna(115.5)
tg['GAME_PACE_EWM'] = tg['GAME_PACE_EWM'].fillna(100.6)

print(f"\n  Team data: {len(tg):,} team-game rows, {tg['GAME_DATE'].nunique()} unique dates")

# 2a. Individual feature correlations with actual scoring
print(f"\n  Feature correlations with actual TEAM_PTS:")
features_to_test = {
    'TEAM_PTS_EWM': 'Team scoring EWM',
    'OFF_RTG_EWM': 'Offensive rating EWM',
    'OPP_DEF_EWM': 'Opponent defensive rating EWM',
    'GAME_PACE_EWM': 'Game pace EWM (avg of both)',
    'NET_RTG_EWM': 'Net rating EWM',
    'OPP_NET_EWM': 'Opponent net rating EWM',
    'IS_HOME': 'Home court',
    'IS_B2B': 'Back-to-back',
    'REST_DAYS': 'Rest days',
}
for col, label in features_to_test.items():
    if col in tg.columns:
        c = tg['TEAM_PTS'].corr(tg[col].fillna(tg[col].mean()))
        print(f"    {col:<25} {c:>+7.4f}  {label}")

# 2b. Test possession model formula properly
print(f"\n  Possession model formula test:")
lg_pts  = tg['TEAM_PTS'].mean()
lg_off  = tg['OFF_RTG_EWM'].mean()
lg_def  = tg['OPP_DEF_EWM'].mean()
lg_pace = tg['GAME_PACE_EWM'].mean()

# Pure possession model
tg_valid = tg.dropna(subset=['OFF_RTG_EWM','OPP_DEF_EWM','GAME_PACE_EWM'])
poss_pred = lg_pts * (tg_valid['OFF_RTG_EWM']/lg_off) * (tg_valid['OPP_DEF_EWM']/lg_def) * (tg_valid['GAME_PACE_EWM']/lg_pace)
mae_poss = mean_absolute_error(tg_valid['TEAM_PTS'], poss_pred.clip(80,145))
corr_poss = tg_valid['TEAM_PTS'].corr(poss_pred)

# EWM only
mae_ewm = mean_absolute_error(tg_valid['TEAM_PTS'], tg_valid['TEAM_PTS_EWM'].fillna(lg_pts))
corr_ewm = tg_valid['TEAM_PTS'].corr(tg_valid['TEAM_PTS_EWM'])

# Best possible: ridge regression on all features
feat_cols = [c for c in ['TEAM_PTS_EWM','OFF_RTG_EWM','OPP_DEF_EWM','GAME_PACE_EWM',
                          'IS_HOME','IS_B2B','NET_RTG_EWM','OPP_NET_EWM'] if c in tg_valid.columns]
X = tg_valid[feat_cols].fillna(0).values
y = tg_valid['TEAM_PTS'].values
tscv = TimeSeriesSplit(n_splits=5)
ridge_scores = cross_val_score(Ridge(alpha=10), X, y, cv=tscv, scoring='neg_mean_absolute_error')

# Predict-mean baseline
mae_mean = mean_absolute_error(tg_valid['TEAM_PTS'], np.full(len(tg_valid), lg_pts))

print(f"    Predict mean always:        MAE={mae_mean:.3f}  corr=0.000  (floor reference)")
print(f"    EWM only:                   MAE={mae_ewm:.3f}  corr={corr_ewm:.3f}")
print(f"    Possession model:           MAE={mae_poss:.3f}  corr={corr_poss:.3f}")
print(f"    Ridge (all features) CV:    MAE={-ridge_scores.mean():.3f}  ±{ridge_scores.std():.3f}")
print(f"    Theoretical ceiling (~60% of std={tg['TEAM_PTS'].std():.1f}): MAE~{tg['TEAM_PTS'].std()*0.60:.1f}")
print(f"\n  FINDING: Best achievable with current features = {-ridge_scores.mean():.1f} MAE")
print(f"  Gap to theoretical ceiling: {-ridge_scores.mean() - tg['TEAM_PTS'].std()*0.60:.1f} pts")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: WHAT ACTUALLY PREDICTS PLAYER SCORING?
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] WHAT ACTUALLY PREDICTS PLAYER SCORING?")
print("-" * 50)

pg = con.execute("""
    SELECT PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION, GAME_DATE, SEASON,
           POS_GROUP, MIN, PTS, REB, AST, STL, BLK, TOV, FG3M,
           MIN_EWM, PTS_PM_EWM, REB_PM_EWM, AST_PM_EWM, STL_PM_EWM,
           BLK_PM_EWM, TOV_PM_EWM, FG3M_PM_EWM, USG_PCT_EWM,
           FG_PCT_EWM, FG3_PCT_EWM, FT_PCT_EWM,
           IS_HOME, IS_B2B, REST_DAYS,
           OPP_ALLOWED_PTS_POS_AVG10, OPP_ALLOWED_REB_POS_AVG10,
           GAMES_PLAYED
    FROM clean.player_game_stats
    WHERE SEASON >= '2023-24' AND MIN >= 10
    ORDER BY PLAYER_ID, GAME_DATE
""").df()

# Fill
for c in [col for col in pg.columns if '_EWM' in col or '_AVG' in col]:
    pg[c] = pd.to_numeric(pg[c], errors='coerce').fillna(0)
pg['USG_PCT_EWM'] = pg['USG_PCT_EWM'].replace(0, 18.0)

print(f"\n  Player data: {len(pg):,} rows, {pg['PLAYER_ID'].nunique()} unique players")

# 3a. What are the strongest predictors of each stat?
print(f"\n  Feature correlations with actual PTS (simple):")
for col in ['MIN_EWM','PTS_PM_EWM','USG_PCT_EWM','FG_PCT_EWM','FG3_PCT_EWM',
            'IS_HOME','IS_B2B','OPP_ALLOWED_PTS_POS_AVG10','GAMES_PLAYED']:
    if col in pg.columns:
        c = pg['PTS'].corr(pg[col].fillna(0))
        print(f"    {col:<35} {c:>+7.4f}")

# 3b. Is rate×min better or worse than just using PTS_EWM directly?
print(f"\n  Method comparison for PTS prediction:")
if 'PTS_EWM' in pg.columns:
    # Direct EWM of pts (not rate×min)
    mae_direct = mean_absolute_error(pg['PTS'], pg['PTS_EWM'].fillna(pg['PTS'].mean()))
    corr_direct = pg['PTS'].corr(pg['PTS_EWM'])
    print(f"    PTS_EWM direct:             MAE={mae_direct:.3f}  corr={corr_direct:.3f}")
else:
    print("    PTS_EWM column not in player stats — only in team stats")

# Rate×min
pg['pred_rate_x_min'] = pg['PTS_PM_EWM'] * pg['MIN_EWM']
mae_rate = mean_absolute_error(pg['PTS'], pg['pred_rate_x_min'].fillna(pg['PTS'].mean()))
corr_rate = pg['PTS'].corr(pg['pred_rate_x_min'])
print(f"    Rate×Min (PTS_PM_EWM×MIN_EWM): MAE={mae_rate:.3f}  corr={corr_rate:.3f}")

# What if we just use PTS_AVG5?
if 'PTS_AVG5' in pg.columns:
    mae_avg5 = mean_absolute_error(pg['PTS'], pg['PTS_AVG5'].fillna(pg['PTS'].mean()))
    corr_avg5 = pg['PTS'].corr(pg['PTS_AVG5'])
    print(f"    PTS_AVG5 (last 5 games):      MAE={mae_avg5:.3f}  corr={corr_avg5:.3f}")

# Min×rate but use AVG5 instead of EWM
if 'MIN_AVG5' in pg.columns and 'PTS_PM_AVG5' in pg.columns:
    pg['pred_avg5'] = pg['PTS_PM_AVG5'] * pg['MIN_AVG5']
    mae_avg5r = mean_absolute_error(pg['PTS'], pg['pred_avg5'].fillna(pg['PTS'].mean()))
    corr_avg5r = pg['PTS'].corr(pg['pred_avg5'])
    print(f"    Rate×Min (AVG5):               MAE={mae_avg5r:.3f}  corr={corr_avg5r:.3f}")

# 3c. Does the opponent defense adjustment actually help?
print(f"\n  Does opponent defense (OPP_ALLOWED_PTS_POS_AVG10) actually help?")
opp_pts = pg['OPP_ALLOWED_PTS_POS_AVG10'].fillna(0)
pg_with_opp = pg[opp_pts > 0].copy()
if len(pg_with_opp) > 1000:
    # Baseline
    mae_no_opp = mean_absolute_error(pg_with_opp['PTS'], pg_with_opp['pred_rate_x_min'].fillna(pg['PTS'].mean()))
    # With opp adjustment
    lg_pos_pts = pg_with_opp.groupby('POS_GROUP')['PTS'].mean().to_dict()
    pg_with_opp['opp_mult'] = pg_with_opp.apply(
        lambda r: np.clip(r['OPP_ALLOWED_PTS_POS_AVG10'] / max(lg_pos_pts.get(r['POS_GROUP'],13.2), 1), 0.80, 1.20),
        axis=1
    )
    pg_with_opp['pred_with_opp'] = pg_with_opp['pred_rate_x_min'] * pg_with_opp['opp_mult']
    mae_with_opp = mean_absolute_error(pg_with_opp['PTS'], pg_with_opp['pred_with_opp'].fillna(pg['PTS'].mean()))
    print(f"    Without opp defense: MAE={mae_no_opp:.3f}")
    print(f"    With opp defense:    MAE={mae_with_opp:.3f}  delta={mae_no_opp-mae_with_opp:+.3f}")
    # Correlation of OPP feature with actual pts
    c = pg_with_opp['PTS'].corr(pg_with_opp['opp_mult'])
    print(f"    Opp mult correlation with actual PTS: {c:.4f}")

# 3d. Home/Away effect on player scoring
print(f"\n  Home vs Away player scoring:")
home_pts = pg[pg['IS_HOME']==1]['PTS'].mean()
away_pts = pg[pg['IS_HOME']==0]['PTS'].mean()
home_min = pg[pg['IS_HOME']==1]['MIN'].mean()
away_min = pg[pg['IS_HOME']==0]['MIN'].mean()
print(f"    Home: avg PTS={home_pts:.2f}  avg MIN={home_min:.2f}")
print(f"    Away: avg PTS={away_pts:.2f}  avg MIN={away_min:.2f}")
print(f"    Diff: PTS={home_pts-away_pts:+.3f}  MIN={home_min-away_min:+.3f}")
print(f"    Home pts/min: {home_pts/home_min:.4f}  Away: {away_pts/away_min:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: SHOULD WE USE RATE×MIN OR SOMETHING ELSE?
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] ARCHITECTURE QUESTION: RATE×MIN vs ALTERNATIVES")
print("-" * 50)

# 4a. What does rate×min fail at?
pg['error'] = pg['pred_rate_x_min'] - pg['PTS']
pg['abs_error'] = pg['error'].abs()

print(f"\n  Rate×Min error analysis:")
print(f"  Overall: MAE={pg['abs_error'].mean():.3f}  bias={pg['error'].mean():+.3f}")

# Does it fail more for certain types of games?
print(f"\n  Error by actual minutes:")
for lo, hi in [(10,18),(18,26),(26,32),(32,38),(38,48)]:
    g = pg[(pg['MIN']>=lo)&(pg['MIN']<hi)]
    if len(g) < 50: continue
    print(f"    {lo:2d}-{hi:2d} min: MAE={g['abs_error'].mean():.2f}  bias={g['error'].mean():+.2f}  n={len(g):,}")

# 4b. Key question: is the ERROR in pts explained by minutes error?
pg['min_error'] = pg['MIN_EWM'] - pg['MIN']
corr_min_pts_err = pg['min_error'].corr(pg['error'])
print(f"\n  Corr(minutes_error, pts_error): {corr_min_pts_err:.3f}")
print(f"  Interpretation: {corr_min_pts_err:.0%} of pts error is explained by minutes error")
print(f"  → If we had perfect minutes, pts MAE would drop by ~{abs(corr_min_pts_err)*pg['abs_error'].mean():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] DISTRIBUTION REALITY CHECK")
print("-" * 50)

print(f"\n  Actual NBA distribution parameters (2023-26, MIN>=10):")
for stat, col in [("PTS","PTS"),("REB","REB"),("AST","AST"),("STL","STL"),("BLK","BLK"),("FG3M","FG3M")]:
    if col not in pg.columns: continue
    d = pg[col]
    mean_v = d.mean()
    std_v  = d.std()
    var_ratio = (std_v**2) / max(mean_v, 0.01)  # var/mean = overdispersion
    zero_r = (d == 0).mean()
    p90 = d.quantile(0.90)
    print(f"    {stat:<6}: mean={mean_v:.2f}  std={std_v:.2f}  var/mean={var_ratio:.2f}  zero%={zero_r:.0%}  P90={p90:.0f}")

print(f"\n  Our simulator variance ratios (for comparison):")
var_ratios = {"PTS":1.55,"REB":1.45,"AST":1.65,"STL":2.20,"BLK":2.50,"TOV":1.35,"FG3M":1.80}
for stat, vr in var_ratios.items():
    if stat in pg.columns:
        actual_vr = pg[stat].std()**2 / max(pg[stat].mean(), 0.01)
        print(f"    {stat:<6}: our ratio={vr:.2f}  actual ratio={actual_vr:.2f}  {'OK' if abs(vr-actual_vr)<0.5 else 'MISMATCH'}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: WHAT WOULD ACTUALLY IMPROVE THE MODEL?
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] IMPROVEMENT OPPORTUNITIES (DATA-DRIVEN)")
print("-" * 50)

# 6a. How much variance IS predictable from available features?
print(f"\n  Maximum R² achievable for each stat (Ridge, TimeSeriesSplit CV):")
feat_base = ['MIN_EWM','PTS_PM_EWM','USG_PCT_EWM','IS_HOME','IS_B2B']
feat_base = [c for c in feat_base if c in pg.columns]
pg_model = pg.dropna(subset=feat_base).copy()
for stat in ['PTS','REB','AST','FG3M','STL','BLK']:
    if stat not in pg_model.columns: continue
    X = pg_model[feat_base].fillna(0).values
    y = pg_model[stat].values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    tscv = TimeSeriesSplit(n_splits=3)
    r2_cv = cross_val_score(Ridge(alpha=1.0), Xs, y, cv=tscv, scoring='r2')
    mae_cv = cross_val_score(Ridge(alpha=1.0), Xs, y, cv=tscv, scoring='neg_mean_absolute_error')
    null_mae = mean_absolute_error(y, np.full(len(y), y.mean()))
    print(f"    {stat:<6}: R²={r2_cv.mean():.3f}  MAE={-mae_cv.mean():.3f}  "
          f"null_MAE={null_mae:.3f}  improvement={null_mae-(-mae_cv.mean()):+.3f}")

# 6b. Correlation matrix of key features vs targets
print(f"\n  Cross-correlation: features vs targets:")
corr_features = ['MIN_EWM','PTS_PM_EWM','USG_PCT_EWM']
corr_targets  = ['PTS','REB','AST','MIN']
corr_table = pg[corr_features + corr_targets].corr().loc[corr_features, corr_targets]
print(corr_table.round(3).to_string())

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
con.close()
