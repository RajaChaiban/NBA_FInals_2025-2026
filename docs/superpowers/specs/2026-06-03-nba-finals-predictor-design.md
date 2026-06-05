# NBA Finals Predictor (Spurs vs Knicks) — Design

**Date:** 2026-06-03
**Author:** Raja Chaiban (with Claude)

## Goal
Predict the 2025-26 NBA Finals between the San Antonio Spurs and New York Knicks
using ML. Output: single-game win probability + full best-of-7 series outcome.

## Decisions (from brainstorming)
- **Approach:** Hybrid — league-wide per-game win model + Spurs/Knicks-specific
  engineered features.
- **Target:** Both per-game win probability AND series winner (Monte Carlo sim).
- **Models:** XGBoost + Logistic Regression (averaged/ensembled, both calibrated).
- **Features:** full set — form/strength, off/def ratings, four factors,
  situational (rest/B2B/home), matchup H2H, Elo.
- **Data:** 3 seasons (2023-24, 2024-25, 2025-26), Regular Season + Playoffs,
  with playoff games weighted higher via sample_weight.
- **Deliverable:** Jupyter notebook.
- **Data source:** `nba_api` (LeagueGameLog), cached to `data/*.csv`.

## Pipeline
1. **Acquire** all team-games (2 rows/game) per season+type → cache `data/raw_games.csv`.
2. **Construct** one row per game from home perspective; target = `home_win`.
3. **Engineer** pre-game features (leakage-safe shifted rolling stats, Elo, H2H),
   expressed as home−away differentials.
4. **Train** XGBoost + LR with chronological split + time-series CV; playoff weight 2x.
5. **Evaluate** accuracy, log loss, Brier, calibration; ensemble = mean of calibrated probs.
6. **Predict** Spurs vs Knicks single game (home court to better 2025-26 record).
7. **Simulate** best-of-7 (2-2-1-1-1) via Monte Carlo (10k) → P(series win), length dist.

## Leakage guards
- All rolling/season-to-date features shifted to use only prior games.
- Chronological train/validation split (never random).

## Files
- `data/` — cached raw + engineered CSVs
- `fetch_data.py` — one-shot cached data fetcher
- `NBA_Finals_Predictor.ipynb` — main deliverable
