"""
NBA Model Deep Research
=======================
Comprehensive analysis of every model component to identify exactly
what needs fixing and what the expected improvement would be.

Sections:
  1. Minutes model — current accuracy, what features help, residual patterns
  2. Team total model — pace context, opponent ratings, what's missing
  3. Player stat accuracy by position, minutes bucket, usage
  4. Feature correlation — which EWM features actually predict outcomes
  5. Calibration — do the distributions reflect real uncertainty?
  6. Identify systematic biases by player type
"""
import sys, warnings
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import duckdb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

con = duckdb.connect('data/analytics_database/nba_warehouse.duckdb', read_only=True)

print("=" * 70)
print("NBA MODEL DEEP RESEARCH")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading data...")
pg = con.execute("""
    SELECT * FROM clean.player_game_stats
    WHERE MIN > 5 AND SEASON >= '2022-23'
    ORDER BY PLAYER_ID, GAME_DATE
""").df()

tg = con.execute("""
    SELECT * FROM clean.team_game_stats
    WHERE SEASON >= '2022-23'
    ORDER BY TEAM_ABBREVIATION, GAME_DATE
""").df()

print(f"Player rows: {len(pg):,}  Team rows: {len(tg):,}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: MINUTES MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 1: MINUTES MODEL")
print("=" * 70)

# 1a. Baseline: how does EWM alone perform?
pg_min = pg.dropna(subset=['MIN_EWM','MIN']).copy()
mae_ewm = mean_absolute_error(pg_min['MIN'], pg_min['MIN_EWM'])
corr_ewm = pg_min['MIN'].corr(pg_min['MIN_EWM'])
bias_ewm = (pg_min['MIN_EWM'] - pg_min['MIN']).mean()
print(f"\n1a. EWM baseline:")
print(f"    MAE={mae_ewm:.3f}  corr={corr_ewm:.3f}  bias={bias_ewm:+.3f}")

# 1b. Residual patterns — where does EWM fail most?
pg_min['min_err'] = pg_min['MIN_EWM'] - pg_min['MIN']
pg_min['min_abs_err'] = pg_min['min_err'].abs()

print(f"\n1b. Minutes error by actual minutes bucket:")
for low, high in [(5,15),(15,25),(25,32),(32,40),(40,48)]:
    bucket = pg_min[(pg_min['MIN'] >= low) & (pg_min['MIN'] < high)]
    if len(bucket) < 50: continue
    print(f"    {low:2d}-{high:2d} min: MAE={bucket['min_abs_err'].mean():.2f}  "
          f"bias={bucket['min_err'].mean():+.2f}  n={len(bucket):,}")

print(f"\n1c. Minutes error by USG% bucket:")
if 'USG_PCT_EWM' in pg_min.columns:
    pg_min['usg_bucket'] = pd.cut(pg_min['USG_PCT_EWM'].fillna(15),
                                   bins=[0,12,18,24,30,50], labels=['<12%','12-18%','18-24%','24-30%','>30%'])
    for bucket, grp in pg_min.groupby('usg_bucket', observed=True):
        if len(grp) < 50: continue
        print(f"    USG {str(bucket):<6}: MAE={grp['min_abs_err'].mean():.2f}  "
              f"bias={grp['min_err'].mean():+.2f}  n={len(grp):,}")

# 1d. What features improve minutes prediction?
print(f"\n1d. Feature correlations with actual minutes:")
min_feature_candidates = [
    'MIN_EWM','USG_PCT_EWM','PTS_PM_EWM','REST_DAYS',
    'IS_B2B','PACE_EWM','TEAM_PTS_EWM',
    'OPP_ALLOWED_PTS_POS_AVG10',
]
available = [c for c in min_feature_candidates if c in pg_min.columns]
for col in available:
    c = pg_min[col].corr(pg_min['MIN'])
    print(f"    {col:<35} corr with actual MIN: {c:.3f}")

# 1e. Can a linear model beat EWM?
print(f"\n1e. Linear regression on minutes features (TimeSeriesSplit CV):")
feat_cols = [c for c in available if c != 'MIN_EWM' or True]
feat_cols = [c for c in feat_cols if pg_min[c].notna().sum() > 1000]
train_df = pg_min.dropna(subset=feat_cols + ['MIN']).copy()
X = train_df[feat_cols].fillna(0).values
y = train_df['MIN'].values
if len(train_df) > 500:
    tscv = TimeSeriesSplit(n_splits=5)
    lr_scores = cross_val_score(Ridge(alpha=1.0), X, y,
                                 cv=tscv, scoring='neg_mean_absolute_error')
    print(f"    Ridge regression CV MAE: {-lr_scores.mean():.3f} ± {lr_scores.std():.3f}")
    print(f"    vs EWM baseline MAE:     {mae_ewm:.3f}")
    print(f"    Improvement:             {mae_ewm - (-lr_scores.mean()):+.3f}")

    # Feature importances from GBM
    gbm = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                     learning_rate=0.1, random_state=42)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    gbm.fit(Xs, y)
    importances = sorted(zip(feat_cols, gbm.feature_importances_),
                         key=lambda x: -x[1])
    print(f"\n    GBM feature importances for minutes prediction:")
    for feat, imp in importances:
        print(f"      {feat:<35} {imp:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: TEAM TOTAL MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2: TEAM TOTAL / GAME TOTAL MODEL")
print("=" * 70)

# 2a. What does EWM actually capture?
tg_clean = tg.dropna(subset=['TEAM_PTS','TEAM_PTS_EWM']).copy()
print(f"\n2a. Team EWM baseline:")
print(f"    MAE={mean_absolute_error(tg_clean['TEAM_PTS'], tg_clean['TEAM_PTS_EWM']):.3f}")
print(f"    corr={tg_clean['TEAM_PTS'].corr(tg_clean['TEAM_PTS_EWM']):.3f}")
print(f"    Team PTS std: {tg_clean['TEAM_PTS'].std():.2f}  mean: {tg_clean['TEAM_PTS'].mean():.2f}")

# 2b. What is the variance structure? Is this even predictable game-to-game?
print(f"\n2b. Game-to-game variance analysis:")
tg_clean['prev_pts'] = tg_clean.groupby('TEAM_ABBREVIATION')['TEAM_PTS'].shift(1)
corr_lag1 = tg_clean['TEAM_PTS'].corr(tg_clean['prev_pts'].fillna(tg_clean['TEAM_PTS'].mean()))
print(f"    Corr(today_pts, yesterday_pts): {corr_lag1:.3f}  — autocorrelation")
print(f"    This tells us: {corr_lag1:.0%} of today's score is explained by yesterday's")
print(f"    The rest ({1-corr_lag1:.0%}) is game-specific noise + matchup effects")

# 2c. Does opponent defense help?
print(f"\n2c. Feature correlations with team scoring:")
team_feat_candidates = [
    'TEAM_PTS_EWM','PACE_EWM','TEAM_EFG_EWM',
    'OPP_ALLOWED_PTS_POS_AVG10',
]
# Also check opponent stats
if 'OPP_TEAM_PTS_EWM' in tg.columns:
    team_feat_candidates.append('OPP_TEAM_PTS_EWM')

for col in team_feat_candidates:
    if col in tg_clean.columns:
        c = tg_clean['TEAM_PTS'].corr(tg_clean[col])
        print(f"    {col:<40} corr: {c:.3f}")

# 2d. Pace-adjusted model
print(f"\n2d. Pace-adjusted model test:")
# Simple pace model: pts ≈ pace * off_rtg / 100
if 'PACE_EWM' in tg_clean.columns and 'TEAM_PTS_EWM' in tg_clean.columns:
    # Estimate using pace scaling
    lg_pace = tg_clean['PACE_EWM'].mean()
    lg_pts  = tg_clean['TEAM_PTS'].mean()
    pace_pred = tg_clean['TEAM_PTS_EWM'] * (tg_clean['PACE_EWM'] / lg_pace)
    pace_pred_clipped = pace_pred.clip(85, 145)
    mae_pace = mean_absolute_error(tg_clean['TEAM_PTS'], pace_pred_clipped)
    corr_pace = tg_clean['TEAM_PTS'].corr(pace_pred_clipped)
    print(f"    Pace-scaled EWM: MAE={mae_pace:.3f}  corr={corr_pace:.3f}")
    print(f"    vs plain EWM:    MAE={mean_absolute_error(tg_clean['TEAM_PTS'], tg_clean['TEAM_PTS_EWM']):.3f}  corr={tg_clean['TEAM_PTS'].corr(tg_clean['TEAM_PTS_EWM']):.3f}")

# 2e. What's the theoretical ceiling?
print(f"\n2e. Theoretical prediction ceiling:")
print(f"    If we perfectly predict the mean (113 pts every game):")
naive_pred = np.full(len(tg_clean), tg_clean['TEAM_PTS'].mean())
mae_naive = mean_absolute_error(tg_clean['TEAM_PTS'], naive_pred)
print(f"      MAE = {mae_naive:.3f} (predicting mean always)")
print(f"    NBA team score std ≈ {tg_clean['TEAM_PTS'].std():.1f} pts")
print(f"    That means ±{tg_clean['TEAM_PTS'].std():.1f} pts of unavoidable noise")
print(f"    Best achievable MAE ≈ {tg_clean['TEAM_PTS'].std() * 0.6:.1f} pts (60% of std)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: PLAYER STAT ACCURACY BY SEGMENT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3: PLAYER STAT ACCURACY BY SEGMENT")
print("=" * 70)

# 3a. By position group
print("\n3a. PTS accuracy by position (2025-26):")
curr = pg[pg['SEASON'] == '2025-26'].dropna(subset=['PTS_PM_EWM','MIN_EWM','PTS']).copy()
curr['pred_pts'] = curr['PTS_PM_EWM'] * curr['MIN_EWM']
for pos in ['G','F','C']:
    grp = curr[curr['POS_GROUP'] == pos]
    if len(grp) < 50: continue
    mae = mean_absolute_error(grp['PTS'], grp['pred_pts'])
    corr = grp['PTS'].corr(grp['pred_pts'])
    bias = (grp['pred_pts'] - grp['PTS']).mean()
    print(f"    {pos}: MAE={mae:.3f}  corr={corr:.3f}  bias={bias:+.3f}  n={len(grp):,}")

# 3b. By games played (experience / stability)
print(f"\n3b. PTS accuracy by player experience (games played):")
if 'GAMES_PLAYED' in curr.columns:
    for lo, hi in [(0,15),(15,40),(40,80),(80,200)]:
        grp = curr[(curr['GAMES_PLAYED'] >= lo) & (curr['GAMES_PLAYED'] < hi)]
        if len(grp) < 50: continue
        mae = mean_absolute_error(grp['PTS'], grp['pred_pts'])
        corr = grp['PTS'].corr(grp['pred_pts'])
        print(f"    {lo:3d}-{hi:3d} games: MAE={mae:.3f}  corr={corr:.3f}  n={len(grp):,}")

# 3c. All stats accuracy
print(f"\n3c. All stats — MAE and correlation (2025-26, MIN>10):")
stats_to_check = [
    ('PTS','PTS_PM_EWM'),('REB','REB_PM_EWM'),('AST','AST_PM_EWM'),
    ('FG3M','FG3M_PM_EWM'),('STL','STL_PM_EWM'),('BLK','BLK_PM_EWM'),
    ('TOV','TOV_PM_EWM'),
]
for stat, pm_col in stats_to_check:
    if pm_col not in curr.columns: continue
    grp = curr.dropna(subset=[pm_col, stat])
    pred = grp[pm_col] * grp['MIN_EWM']
    mae = mean_absolute_error(grp[stat], pred)
    corr = grp[stat].corr(pred)
    bias = (pred - grp[stat]).mean()
    avg_act = grp[stat].mean()
    print(f"    {stat:<6}: MAE={mae:.3f}  corr={corr:.3f}  bias={bias:+.3f}  avg_actual={avg_act:.2f}  MAE/avg={mae/max(avg_act,0.01):.1%}")

# 3d. Zero-game accuracy (what % of games does model predict 0 for?)
print(f"\n3d. Zero-inflation reality check (2025-26, MIN>10):")
for stat in ['PTS','REB','AST','FG3M','STL','BLK']:
    if stat not in curr.columns: continue
    actual_zero = (curr[stat] == 0).mean()
    pm_col = f"{stat}_PM_EWM"
    if pm_col in curr.columns:
        pred_zero = (curr[pm_col] * curr['MIN_EWM'] < 0.5).mean()
        print(f"    {stat:<6}: actual_zero_rate={actual_zero:.1%}  model_pred_zero_rate={pred_zero:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: FEATURE IMPORTANCE FOR PTS PREDICTION")
print("=" * 70)

pts_features = [
    'MIN_EWM','PTS_PM_EWM','USG_PCT_EWM','FG_PCT_EWM',
    'FG3_PCT_EWM','FT_PCT_EWM','EFG_PCT_EWM',
    'PACE_EWM','TEAM_PTS_EWM','REST_DAYS','IS_B2B',
]
avail_pts = [c for c in pts_features if c in curr.columns]
pts_df = curr.dropna(subset=avail_pts + ['PTS']).copy()

print(f"\n4a. Simple correlation of each feature with actual PTS:")
for col in avail_pts:
    c = pts_df[col].corr(pts_df['PTS'])
    print(f"    {col:<30} {c:.4f}")

if len(pts_df) > 500:
    X_pts = pts_df[avail_pts].fillna(0).values
    y_pts = pts_df['PTS'].values
    scaler2 = StandardScaler()
    Xs2 = scaler2.fit_transform(X_pts)
    gbm2 = GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                      learning_rate=0.08, random_state=42)
    gbm2.fit(Xs2, y_pts)
    print(f"\n4b. GBM feature importances for PTS prediction:")
    for feat, imp in sorted(zip(avail_pts, gbm2.feature_importances_), key=lambda x:-x[1]):
        print(f"    {feat:<30} {imp:.4f}")

    tscv2 = TimeSeriesSplit(n_splits=5)
    gbm_cv = cross_val_score(GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                         random_state=42),
                              Xs2, y_pts, cv=tscv2, scoring='neg_mean_absolute_error')
    print(f"\n4c. GBM CV MAE for PTS: {-gbm_cv.mean():.3f} vs EWM baseline {mae_ewm:.3f}")
    print(f"    Potential improvement: {mae_ewm - (-gbm_cv.mean()):+.3f} MAE reduction")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: WHAT'S MISSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5: WHAT'S MISSING / NOT USED")
print("=" * 70)

print("\n5a. Available features NOT currently used in projections:")
all_cols = set(pg.columns)
ewm_used = {'MIN_EWM','PTS_PM_EWM','REB_PM_EWM','AST_PM_EWM','STL_PM_EWM',
            'BLK_PM_EWM','TOV_PM_EWM','FG3M_PM_EWM','FG_PCT_EWM','FG3_PCT_EWM',
            'FT_PCT_EWM','PACE_EWM','TEAM_PTS_EWM'}
unused_ewm = [c for c in all_cols if '_EWM' in c and c not in ewm_used]
print(f"    Unused EWM features: {unused_ewm}")

unused_context = [c for c in all_cols if 'OPP_ALLOWED' in c or 'AVG5' in c or 'AVG10' in c]
print(f"\n    Unused opponent-context features ({len(unused_context)}):")
for c in unused_context[:10]:
    notna = pg[c].notna().mean()
    print(f"      {c:<45} non-null: {notna:.0%}")

print("\n5b. Key missing data we don't have at all:")
missing_data = [
    "Starting lineup (not in LeagueGameLog — we use top-5 by pts as proxy)",
    "Injury/DNP status before game (no real-time injury API in free tier)",
    "Blowout/garbage time minutes (systematically distorts low-minute players)",
    "Travel schedule (back-to-back, cross-country games)",
    "Individual matchup data (which defender guards which player)",
    "Home/away split (IS_HOME column — IS present but not used in model)",
]
for m in missing_data:
    print(f"    - {m}")

# 5c. IS_HOME effect
if 'IS_HOME' in pg.columns:
    print(f"\n5c. Home/Away effect on player scoring:")
    home_pts = pg[pg['IS_HOME']==1]['PTS'].mean()
    away_pts = pg[pg['IS_HOME']==0]['PTS'].mean()
    print(f"    Home avg PTS: {home_pts:.2f}  Away avg PTS: {away_pts:.2f}  Diff: {home_pts-away_pts:+.2f}")

print("\n" + "=" * 70)
print("RESEARCH COMPLETE")
print("=" * 70)
con.close()
