# NBA Player Game Projections

Per-game projections for every NBA player (minutes, points, rebounds, assists, 3-pointers and
the PRA / P+R / P+A / R+A combos) and team lines (moneyline, spread, total), with the same
structure as the NHL Game Projections tool.

Stage 1 (this commit): data, model and walk-forward backtest. App tabs come next.

## Data sources (evaluated 2026-10-12)

| Source | Used for |
|---|---|
| ESPN public API | box scores, play-by-play, rosters, injuries, schedule (history from 2018-19) |
| Rotowire NBA lineups | expected / confirmed starting fives, play-likelihood (live, Stage 2) |
| The Odds API | sportsbook odds (Stage 3) |

The NBA's own stats API and CDN, its injury-report PDFs and lineups.com are blocked from this
network; Basketball-Reference is reachable but rate-limited by its terms, so it is a fallback only.

## Run

```
.venv\Scripts\python fetch.py        # download / update ESPN game summaries
.venv\Scripts\python parse.py        # raw -> lake tables
.venv\Scripts\python backtest.py     # walk-forward 2021-26, prints the report
```

## Files

| File | Role |
|---|---|
| `fetch.py`, `parse.py` | ESPN summaries -> player_games, team_games, injuries tables |
| `ew.py` | exponentially weighted running sums (the only way history enters) |
| `features.py` | pregame features: minutes, per-minute rates, pace, opponent defence, absences |
| `model.py` | minutes (if he plays + chance he plays), stats (Poisson GLM), distributions, scores |
| `backtest.py` | walk-forward evaluation vs season-average and last-10 baselines |
