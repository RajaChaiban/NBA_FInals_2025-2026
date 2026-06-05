# 🏀 NBA Finals Predictor — Spurs vs Knicks (2025-26)

An ensemble ML pipeline (**XGBoost + Logistic Regression**) that predicts the
2025-26 NBA Finals between the **San Antonio Spurs** and **New York Knicks**,
using 3 seasons of real game data from `stats.nba.com`.

## 🏆 Final series prediction

**TOSS-UP — essentially 50/50, with the faintest lean to New York.**

| Outcome | Probability |
|---|---|
| **Knicks win the Finals** | **52.4%** |
| **Spurs win the Finals** | **47.6%** |
| Bootstrap 90% CI (Spurs) | 26% – 54% (300 refits; wide → low confidence) |
| Single game, neutral court | Knicks 52.1% / Spurs 47.9% |
| Home court | San Antonio (2-2-1-1-1), worth ~8pp/game |
| Most likely single outcome | Knicks in 6 |

These two teams are genuinely close: the Spurs are stronger over the full regular
season (62-20) and hold home court (~8pp/game), but the model — which weights recent
playoff form — has the Knicks a touch better on a neutral floor, and that edge is
just enough to overcome San Antonio's home court. The honest read is a **coin flip
with wide uncertainty** — the bootstrap CI spans clear-Knicks to slight-Spurs, and
its mean (≈39% Spurs) sits *below* the point estimate, a reminder that the central
number is fragile to resampling.

### Per-game probabilities (by venue)

| Game | Host | P(Spurs win) |
|---|---|---|
| 1, 2, 5, 7 | San Antonio | 56.1% |
| 3, 4, 6 | New York | 39.8% |

> The per-game number varies only by **venue** — each game is simulated as an
> independent contest. The model does not update mid-series or react to momentum.

## Model performance (2025-26 holdout)

| Model | Accuracy | Log Loss | Brier |
|---|---|---|---|
| XGBoost | 67.1% | 0.611 | 0.211 |
| Logistic Regression | 67.8% | 0.605 | 0.208 |
| **Ensemble** | **68.4%** | **0.606** | **0.208** |
| Baseline: Elo+home (no ML) | 66.5% | 0.616 | 0.213 |

The honest benchmark isn't "home always wins" — note that the symmetric set forces
that to 50% by construction, even though the *real* home win rate is ~55%. The bar
that matters is a one-feature **Elo+home** rule (66.5%). The full pipeline beats it
by **+1.9pp** and improves probability quality (log loss 0.616 → 0.606) — but that
edge is **not statistically significant** on this one-season holdout (McNemar
*p* ≈ 0.12), so read it as "**roughly matches a strong Elo baseline**," not a
decisive win. Logistic Regression carries most of the signal; XGBoost is the weakest
of the three, and the ensemble is kept for stability more than a large accuracy gain.

## How it works

1. **Data** (`fetch_data.py`) — pulls all team-game box scores for 2023-24,
   2024-25, 2025-26 (regular season + playoffs) via `nba_api`, cached to
   `data/raw_games.csv` (~3,900 games).
2. **Symmetric dataset** — each game appears twice (one row per team); strength
   features are perspective−opponent differences, while **`is_home`, `is_playoff`
   and their interaction are explicit features** so home-court (incl. a separate
   playoff home-court effect, ~7pp) is learned, not baked into the base rate.
3. **Features** (leakage-safe, pre-game): form/win%, off/def/**net** rating, pace,
   four factors (eFG%, TOV%, OREB%, FT rate), rest/back-to-back, sequential
   **Elo**, head-to-head margin. (Raw point margin dropped — r≈0.999 with net rating.)
4. **Models** — XGBoost + Logistic Regression, sigmoid-calibrated; recency×playoff
   sample weights normalized to mean 1; chronological split; **TimeSeriesSplit
   calibration for evaluation, KFold for the deployed model** (no future to leak
   into for a true forecast). Evaluation includes **walk-forward** (next-season)
   testing and **permutation importance**.
5. **Series simulation** — Monte Carlo best-of-7 (2-2-1-1-1) for the point estimate,
   plus **300 bootstrap refits** to put a 90% credible interval on P(series win) — so
   the number isn't reported with false precision. A **weighting-sensitivity grid**
   (§8b) confirms the TOSS-UP verdict is stable across the recency/playoff knobs
   (P(Spurs) stays in ~[0.47, 0.51] across a 3×3 grid).

## Run it

```bash
pip install nba_api pandas numpy scikit-learn xgboost matplotlib jupyter
python fetch_data.py            # fetch + cache data (skips if cached)
jupyter notebook NBA_Finals_Predictor.ipynb
```

To regenerate the notebook from source: `python build_notebook.py`.

## Files

| File | Purpose |
|---|---|
| `NBA_Finals_Predictor.ipynb` | Main deliverable — full pipeline + plots |
| `fetch_data.py` | Cached data fetcher (nba_api) |
| `build_notebook.py` | Generates the notebook from source |
| `sensitivity_check.py` | Standalone harness: base case, McNemar significance, weighting-sensitivity grid |
| `data/raw_games.csv` | Cached raw box scores |


## Leakage guards

- Rolling/season-to-date features shifted to use only prior games.
- Chronological split (never random); the Finals games are **not** in the data.

## What can swing the result (model blind spots)

The model is built on **team box-score data**, so several real-world factors that
can decide a Finals are invisible to it:

- **Injuries / player availability** — the single biggest one. The model has no
  idea if a star is hurt. Victor Wembanyama missed the back half of 2024-25 with a
  blood clot; if any key player (Wembanyama, Brunson, Towns, Fox) is limited or
  out, the real probabilities shift far more than the model shows.
- **In-series momentum & coaching adjustments** — each game is simulated
  independently; the model doesn't react to a 0-2 hole or a defensive scheme change.
- **Roster changes mid-stream** — trades/buyouts aren't reflected until they show
  up in game data.
- **Variance** — a single hot/cold shooting night or officiating swing. The wide
  bootstrap CI (27–55%) is the model's honest way of saying "low confidence."

For these reasons the output should be read as a **data-driven prior**, not a
certainty — and the headline really is "coin flip."
