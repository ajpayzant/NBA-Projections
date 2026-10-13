# NBA Player Game Projections

Per-game projections for every NBA player (minutes, points, rebounds, assists, 3-pointers and
the PRA / P+R / P+A / R+A combos) and team lines (moneyline, spread, total), with the same
structure as the NHL Game Projections tool.

Stages so far: (1) data, model and walk-forward backtest; (2) the live slate and app tabs;
(3) sportsbook odds: the Edges tab, the line checker and the Results tab.

## Data sources (evaluated 2026-10-12)

| Source | Used for |
|---|---|
| ESPN public API | box scores, play-by-play, rosters, injuries, schedule (history from 2018-19) |
| Rotowire NBA lineups | expected / confirmed starting fives, play-likelihood (today's and tomorrow's pages) |
| The Odds API | sportsbook odds: DraftKings, FanDuel, BetMGM, Caesars, Fanatics (manual refresh only) |

The NBA's own stats API and CDN, its injury-report PDFs and lineups.com are blocked from this
network; Basketball-Reference is reachable but rate-limited by its terms, so it is a fallback only.

## Run

```
.venv\Scripts\python fetch.py        # download / update ESPN game summaries
.venv\Scripts\python parse.py        # raw -> lake tables
.venv\Scripts\python pbp.py          # play-by-play -> on-court tables
.venv\Scripts\python backtest.py     # walk-forward 2021-26, prints the report
.venv\Scripts\python live.py         # the live slate (refresh.bat runs this and the snapshot)
.venv\Scripts\python odds.py         # sportsbook odds now (the Edges tab's Refresh odds button)
.venv\Scripts\python tracking.py snapshot | grade | backtest START END [MAX_CREDITS]
run_app.bat                           # the app, http://localhost:8680
```

The odds key goes in `.streamlit/secrets.toml` as `odds_api_key` (gitignored), or `ODDS_API_KEY`
in the environment. Credits: a Refresh odds is about 3 plus up to 10 per game starting within 30
hours; grading a day's games is about 200 credits per game (opening and closing lines, 10 prop
markets). Historical prices are cached per day in `data/odds_hist/`, so no day is paid for twice.

## Files

| File | Role |
|---|---|
| `fetch.py`, `parse.py` | ESPN summaries -> player_games, team_games, injuries tables |
| `pbp.py` | play-by-play -> on-court lineups (checked: 96-99.9% of player-games within 1 min of the box) and per-player on-court totals and stints |
| `wowy.py` | with / without: each player's usage and minutes when a specific teammate is off the court or out |
| `ew.py` | exponentially weighted running sums (the only way history enters) |
| `features.py` | pregame features: minutes, per-minute rates, pace, opponent defence, absences |
| `model.py` | minutes (if he plays + chance he plays), stats (Poisson GLM), boosted corrections, distributions, game lines |
| `backtest.py` | walk-forward evaluation vs season-average and last-10 baselines |
| `live.py` | the live slate: schedule, rosters, Rotowire / ESPN lineups, projections, game lines |
| `odds.py` | The Odds API fetch (game lines + 10 player-prop markets) |
| `linecheck.py` | prices one line with the model's distributions (props, double / triple-double, game lines) |
| `edges.py` | every book price against the model: no-vig probabilities, edge, EV |
| `tracking.py` | Results: daily snapshots, grading at the opening and closing lines, historical backtest |
| `app.py` | Streamlit app: Slate, Game, Lineups & minutes, Player, Edges, Results, How it works |
| `research/` | tuning grid, nonlinear headroom check, distribution comparison |
