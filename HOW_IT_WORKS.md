# How the projections work

Every player expected to play tonight gets a projection for **minutes, points, rebounds, assists and
3-pointers** (plus the PRA / P+R / P+A / R+A combos), and every game gets a fair **moneyline,
spread and total**. All of it is built from ESPN box scores and play-by-play since 2018-19, and
checked in a walk-forward backtest: each season projected with models fitted only on earlier seasons.

## 1. Minutes first

Basketball production is mostly minutes, so minutes come first.

- **His minutes if he plays**, from his recent minutes (short and long memory), his minutes in
  tonight's role (starter or bench), how often he starts, back-to-backs and three-games-in-four,
  blowout risk (starters sit when a game is projected lopsided) and the playoffs.
- **The chance he plays at all**, for end-of-bench players who often sit by coach's decision.
- **Absent teammates.** The minutes of regulars who are out tonight are handed to the players who
  usually absorb them. Stars absorb some, rotation players most; a player's own history in earlier
  games those teammates missed counts too.
- A team's minutes add up to 240 plus expected overtime. Minutes added or taken away to get
  there go to the players with room under about 40 minutes: on a deep night coaches trim the
  bench, not the stars.
- Only the 14 players most likely to be in the rotation count as available (teams dress about
  13; the roster also lists two-way players and camp invites). Anyone can be added back on the
  Lineups page.

## 2. Production per minute, in tonight's context

- **His rates** per minute for each stat, measured **against the league at the time** (NBA scoring
  rose about 5% per possession and 3-point volume about 15% since 2018-19), so they are not
  biased by era. New players start from their position group's level.
- **Tonight's context:** game pace (both teams), what the opponent allows of that stat (and to his
  position, for rebounds and assists), home court, rest, and the usage freed by absent teammates.
- **Play-by-play:** who was on the court for every possession is rebuilt from the substitutions
  (matching box-score minutes in 96-99.9% of games). That gives each player's share of team usage,
  rebounds and teammates' baskets while he is on the court, tonight's rebound and basket
  opportunities from both teams' shooting, and how his usage changes when a specific teammate sits.
- A regularised boosted model corrects what these miss (it found curvature in minutes and players
  with short histories).

## 3. Probabilities

Each stat's outcome around the projection has a spread fitted from the data (variance growing with
the projection). The over / under probabilities are **for a game he plays**: books void a player prop
otherwise. Fair odds are those probabilities with no margin.

## 4. Game lines

A team's points come from a fitted blend of two views: the sum of its players' projected points
(which knows tonight's lineup) and team ratings (offence and defence per possession, and pace,
against the league). Margin and total errors are normal, with spreads fitted on earlier seasons.

## 5. How accurate it is (2021-22 to 2025-26, regular season)

| | Model (average miss) | Season average | Last 10 games |
|---|---|---|---|
| Minutes | **4.46** | 5.38 | 5.26 |
| Points | **4.44** | 4.70 | 4.72 |
| Rebounds | **1.84** | 1.94 | 1.96 |
| Assists | **1.29** | 1.35 | 1.36 |
| 3-pointers | **0.86** | 0.88 | 0.89 |

Over / under probabilities are calibrated to within about one point at every common line, for
every stat and every tier of player. Games: 66.2% of winners, margin missed by 10.9 on average.

The backtest uses who actually played and started; late scratches make live use a little worse.

## 6. Using it

- **Slate:** every player projected to play, with fair lines; lineup sources and injuries; game
  lines for every game (the arrows step through alternate spreads and totals).
- **Game:** both teams' projected totals, the game-lines board, starters and bench.
- **Lineups & minutes:** who plays, who starts, anyone's minutes; Save re-projects the team.
- **Player:** tonight against his recent and past numbers, a prop ladder for every stat with fair
  odds and hit rates, his game log, seasons and splits.
- **Edges:** sportsbook prices (DraftKings, FanDuel, BetMGM, Caesars, Fanatics) against the
  model's probability for the same bet, with the edge and expected value. Odds are fetched only
  when you click Refresh odds. "Check a line from a book" prices any line you type in.
- **Results:** the model's final projections graded against the opening and closing lines.

## 7. Edges and what to trust

A book's probability is read from both sides of its line with the margin removed. Double- and
triple-doubles are posted as Yes only, so for those the book's probability keeps its margin.
Books often hear about late scratches and minutes limits first, and closing lines are sharp:
a big edge is more often news the model has not seen than a mispriced line. Check the injury
report first.

Lineups come from, in order: your override, Rotowire's starting five (Confirmed / Expected) and
may-not-play list, ESPN's injury statuses, and the team's last game.
