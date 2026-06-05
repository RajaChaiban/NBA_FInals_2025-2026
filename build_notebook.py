"""
Builds NBA_Finals_Predictor.ipynb programmatically with nbformat.

Keeping the notebook source here (as code/markdown cell strings) makes it easy to
regenerate a clean, valid notebook and to version-control the logic.

v2 (post-review): symmetric dataset with explicit home-court features, train/serve
consistent serving, normalized weights, time-series calibration, walk-forward
evaluation with honest baselines, and bootstrap uncertainty on the series number.
"""
import sys
import nbformat as nbf

# --colab builds a self-contained Google Colab notebook (installs nba_api and
# fetches data inline instead of reading the local data/raw_games.csv cache).
COLAB = "--colab" in sys.argv
OUT_PATH = "NBA_Finals_Predictor_Colab.ipynb" if COLAB else "NBA_Finals_Predictor.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------- Title
md(r"""
# 🏀 NBA Finals Predictor — Spurs vs Knicks (2025-26)

An ensemble ML pipeline (**XGBoost + Logistic Regression**) that predicts the
2025-26 NBA Finals between the **San Antonio Spurs** and **New York Knicks**.

**Approach:** a league-wide per-game win model (3 seasons, regular season +
playoffs) on a **symmetric** dataset so home-court — including a *separate playoff
home-court effect* — is learned explicitly. Leakage-safe pre-game features, plus
roster-faithful matchup features, then a **Monte Carlo best-of-7 simulation with
bootstrap uncertainty** for the series.

**Pipeline:** data → dataset construction → feature engineering → train/eval →
single-game prediction → series simulation.
""")

if COLAB:
    md(r"""
> **Running in Google Colab.** This notebook is self-contained — the first cell
> installs `nba_api` and the data is fetched live from `stats.nba.com` (cached to
> the Colab session for fast re-runs). Just *Runtime → Run all*.
""")
    code(r"""
# Install dependencies (Colab already has pandas/numpy/scikit-learn/matplotlib)
!pip install -q nba_api xgboost
""")

# ---------------------------------------------------------------- Section 1
md(r"""
## 1. Setup & configuration
""")

code(r"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import xgboost as xgb

warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 60)
np.random.seed(42)

DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "raw_games.csv")

# The two Finals teams (home court is derived from records below, not hardcoded).
TEAM_A = "SAS"   # San Antonio Spurs
TEAM_B = "NYK"   # New York Knicks

# --- Sample-weight scheme -------------------------------------------------
# Each training game's weight = recency_decay * playoff_multiplier, normalized to
# mean 1 (so regularization strength behaves as intended). Most recent PLAYOFF
# games dominate; oldest regular-season games count least.
PLAYOFF_MULT = 2.5            # playoff game weight vs regular-season (1.0)
RECENCY_HALFLIFE_DAYS = 540   # weight halves ~every 1.5 seasons (keeps effective N up)

# Current-roster season for the live matchup (both teams' current cores: 2025-26).
PREDICT_SEASON = "2025-26"

ELO_BASE, ELO_K, ELO_HOME = 1500.0, 20.0, 100.0
N_SIMS = 20000           # Monte Carlo series simulations
N_BOOTSTRAP = 300        # bootstrap refits for series-probability uncertainty
                         # (>=300 so the 5th/95th-pct CI bounds are stable, not noise)
H2H_SHRINK_K = 5         # shrink head-to-head margin toward 0 via n/(n+K) (small-sample guard)

print("Config loaded. Finals:", TEAM_A, "vs", TEAM_B)
""")

# ---------------------------------------------------------------- Section 2
if COLAB:
    md(r"""
## 2. Data acquisition (live + session cache)

Pulls all team-game box scores for 2023-24, 2024-25, 2025-26 (regular season +
playoffs) directly from `stats.nba.com` via **`nba_api`**. The result is cached to
`data/raw_games.csv` in the Colab session, so re-running this cell is instant.
Two rows per game (one per team). First run takes ~30-60s.
""")
    code(r"""
import time
from nba_api.stats.endpoints import leaguegamelog

SEASONS = ["2023-24", "2024-25", "2025-26"]
SEASON_TYPES = ["Regular Season", "Playoffs"]
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_one(season, season_type, retries=3):
    for attempt in range(1, retries + 1):
        try:
            df = leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star=season_type, timeout=60
            ).get_data_frames()[0]
            df["SEASON"] = season
            df["SEASON_TYPE"] = season_type
            return df
        except Exception as e:
            print(f"  retry {attempt} ({season} {season_type}): {type(e).__name__}")
            time.sleep(3 * attempt)
    return pd.DataFrame()


if os.path.exists(RAW_PATH):
    raw = pd.read_csv(RAW_PATH, dtype={"GAME_ID": str})
    print("Loaded cached", RAW_PATH)
else:
    frames = []
    for s in SEASONS:
        for st in SEASON_TYPES:
            print(f"Fetching {s} {st} ...")
            frames.append(fetch_one(s, st))
            time.sleep(0.8)
    raw = pd.concat([f for f in frames if len(f)], ignore_index=True)
    raw.to_csv(RAW_PATH, index=False)
    print("Saved", RAW_PATH)

raw["GAME_ID"] = raw["GAME_ID"].astype(str)
raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
raw = raw.sort_values("GAME_DATE").reset_index(drop=True)
print(f"{len(raw)} team-game rows | {raw['GAME_ID'].nunique()} games")
print(raw.groupby(['SEASON','SEASON_TYPE'])['GAME_ID'].nunique())
""")
else:
    md(r"""
## 2. Data acquisition (cached)

Raw team-game box scores were pulled from `stats.nba.com` via **`nba_api`** and
cached by `fetch_data.py`. If the cache is missing, run `python fetch_data.py`
first. Two rows per game (one per team).
""")
    code(r"""
assert os.path.exists(RAW_PATH), "Missing data/raw_games.csv -- run: python fetch_data.py"
raw = pd.read_csv(RAW_PATH, dtype={"GAME_ID": str})
raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
raw = raw.sort_values("GAME_DATE").reset_index(drop=True)
print(f"{len(raw)} team-game rows | {raw['GAME_ID'].nunique()} games")
print(raw.groupby(['SEASON','SEASON_TYPE'])['GAME_ID'].nunique())
""")

# ---------------------------------------------------------------- Section 3
md(r"""
## 3. Dataset construction

Each game has a home row (`MATCHUP` contains `vs.`) and an away row (`@`). We
join the two so every game also carries its **opponent's** box score — needed to
compute defensive rating, opponent-dependent four factors, etc.
""")

code(r"""
def add_realized_metrics(df):
    '''Compute realized (post-game) advanced metrics from a team's box score
    plus its opponent's box score. These are the per-game ingredients we later
    turn into leakage-safe pre-game rolling features.'''
    df = df.copy()
    # Possessions: symmetric estimate averaging BOTH teams' box scores (textbook).
    # Using one shared possession count means OFF_RTG and DEF_RTG share the same
    # denominator instead of (slightly wrongly) using only the team's own estimate.
    poss_team = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]
    poss_opp = df["OPP_FGA"] - df["OPP_OREB"] + df["OPP_TOV"] + 0.44 * df["OPP_FTA"]
    df["POSS"] = (0.5 * (poss_team + poss_opp)).clip(lower=1)
    df["OFF_RTG"] = 100 * df["PTS"] / df["POSS"]
    df["DEF_RTG"] = 100 * df["OPP_PTS"] / df["POSS"]
    df["NET_RTG"] = df["OFF_RTG"] - df["DEF_RTG"]
    df["PACE"] = df["POSS"]  # possessions per game (both-team avg) ~ pace
    # Four factors
    df["EFG"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"]
    df["TOV_PCT"] = df["TOV"] / df["POSS"]
    df["OREB_PCT"] = df["OREB"] / (df["OREB"] + df["OPP_DREB"]).clip(lower=1)
    df["FT_RATE"] = df["FTA"] / df["FGA"]
    df["WIN"] = (df["WL"] == "W").astype(int)
    df["MARGIN"] = df["PTS"] - df["OPP_PTS"]
    return df


# Self-join on GAME_ID to attach opponent stats to each team-game row.
# (FGA/OREB/TOV/FTA come along so we can compute the opponent's possessions too.)
opp_cols = ["GAME_ID", "TEAM_ABBREVIATION", "PTS", "DREB", "FGA", "OREB", "TOV", "FTA"]
opp = raw[opp_cols].rename(columns={
    "TEAM_ABBREVIATION": "OPP_ABBR", "PTS": "OPP_PTS", "DREB": "OPP_DREB",
    "FGA": "OPP_FGA", "OREB": "OPP_OREB", "TOV": "OPP_TOV", "FTA": "OPP_FTA"})
tg = raw.merge(opp, on="GAME_ID")
tg = tg[tg["TEAM_ABBREVIATION"] != tg["OPP_ABBR"]].copy()
tg["IS_HOME"] = tg["MATCHUP"].str.contains("vs.").astype(int)
# Drop neutral-site games (NBA Cup in Vegas, international) where neither team is
# home -> no valid home/away split. ~10 games out of ~3900, negligible.
home_counts = tg.groupby("GAME_ID")["IS_HOME"].transform("sum")
tg = tg[home_counts == 1].copy()
tg = add_realized_metrics(tg)
print("Team-game rows with opponent stats:", len(tg))
tg[["GAME_DATE","TEAM_ABBREVIATION","OPP_ABBR","IS_HOME","PTS","OPP_PTS","OFF_RTG","DEF_RTG","EFG"]].head()
""")

# ---------------------------------------------------------------- Section 4
md(r"""
## 4. Feature engineering (leakage-safe, pre-game)

For every team-game we compute features using **only games played before it**:

- **Form/strength:** season-to-date & last-10 win%, net/off/def rating, signed streak
- **Efficiency:** pace + four factors (eFG%, TOV%, OREB%, FT rate)
- **Situational:** rest days, back-to-back flag
- **Elo:** sequentially updated rating with home advantage & season regression
- **Matchup (hybrid):** prior head-to-head average margin vs that specific opponent,
  **shrunk toward 0 by `n/(n+K)`** so a 1-2 game cross-conference sample can't dominate

Rolling/expanding stats are **shifted by one game** so the current game never
leaks. Form resets per season; Elo carries over with a 25% regression to the mean.
We dropped raw `MARGIN` because it is ~1.0 correlated with `NET_RTG` (verified in
review) — keeping both just duplicated a feature.
""")

code(r"""
# Metrics we turn into pre-game rolling (last-10) and expanding (season-to-date)
# features. MARGIN intentionally excluded (collinear with NET_RTG, r~0.999).
ROLL_METRICS = ["WIN", "OFF_RTG", "DEF_RTG", "NET_RTG", "PACE",
                "EFG", "TOV_PCT", "OREB_PCT", "FT_RATE"]


def build_form_features(tg):
    '''Per (team, season): shifted last-10 and season-to-date means + streak,
    rest days and back-to-back. Everything is strictly pre-game.'''
    out = []
    for (team, season), g in tg.groupby(["TEAM_ABBREVIATION", "SEASON"], sort=False):
        g = g.sort_values("GAME_DATE").copy()
        sh = g[ROLL_METRICS].shift(1)               # exclude current game
        for m in ROLL_METRICS:
            g[f"l10_{m}"] = sh[m].rolling(10, min_periods=3).mean()
            g[f"std_{m}"] = sh[m].expanding(min_periods=3).mean()
        # signed win/loss streak coming into the game (+n win streak, -n loss streak)
        prev_win = g["WIN"].shift(1)
        streak = []
        cur = 0
        for w in prev_win.values:
            if np.isnan(w):
                cur = 0
            elif w == 1:
                cur = cur + 1 if cur > 0 else 1
            else:
                cur = cur - 1 if cur < 0 else -1
            streak.append(cur)
        g["streak"] = streak
        # rest days & back-to-back
        days = g["GAME_DATE"].diff().dt.days
        g["rest_days"] = days.fillna(3).clip(upper=7)
        g["b2b"] = (days == 1).astype(int)
        out.append(g)
    return pd.concat(out).sort_values("GAME_DATE")


tg = build_form_features(tg)
print("Form features built. Sample columns:",
      [c for c in tg.columns if c.startswith(('l10_','std_'))][:6], "...")
""")

code(r"""
def add_elo(tg):
    '''Sequential Elo over all games in date order. Stores each team's PRE-game
    Elo. Home team gets +ELO_HOME when computing expected score. Ratings regress
    25% toward ELO_BASE at the start of each new season.'''
    tg = tg.sort_values(["GAME_DATE", "GAME_ID"]).copy()
    elo = {}
    season_seen = {}
    pre_elo = {}
    for gid, gg in tg.groupby("GAME_ID", sort=False):
        if len(gg) != 2:
            continue
        home = gg[gg.IS_HOME == 1].iloc[0]
        away = gg[gg.IS_HOME == 0].iloc[0]
        season = home["SEASON"]
        for t in (home["TEAM_ABBREVIATION"], away["TEAM_ABBREVIATION"]):
            if t not in elo:
                elo[t] = ELO_BASE
            if season_seen.get(t) != season:          # new season -> regress
                elo[t] = ELO_BASE + 0.75 * (elo[t] - ELO_BASE)
                season_seen[t] = season
        h, a = home["TEAM_ABBREVIATION"], away["TEAM_ABBREVIATION"]
        rh, ra = elo[h], elo[a]
        pre_elo[(gid, h)] = rh
        pre_elo[(gid, a)] = ra
        exp_h = 1 / (1 + 10 ** (-((rh + ELO_HOME) - ra) / 400))
        res_h = 1.0 if home["WIN"] == 1 else 0.0
        elo[h] = rh + ELO_K * (res_h - exp_h)
        elo[a] = ra + ELO_K * ((1 - res_h) - (1 - exp_h))
    tg["elo_pre"] = [pre_elo.get((r.GAME_ID, r.TEAM_ABBREVIATION), ELO_BASE)
                     for r in tg.itertuples()]
    return tg, elo


tg, final_elo = add_elo(tg)
print("Elo built. Current Elo  SAS=%.0f  NYK=%.0f"
      % (final_elo.get('SAS', 1500), final_elo.get('NYK', 1500)))
""")

code(r"""
def add_h2h(tg):
    '''Prior head-to-head average margin vs the specific opponent (pre-game).
    Sparse for cross-conference pairs -> 0 when unseen.'''
    tg = tg.sort_values(["GAME_DATE", "GAME_ID"]).copy()
    hist = {}
    vals = []
    for r in tg.itertuples():
        key = (r.TEAM_ABBREVIATION, r.OPP_ABBR)
        prior = hist.get(key, [])
        n = len(prior)
        # Shrink toward 0 by n/(n+K): cross-conference pairs meet only ~twice a
        # year, so a 1-2 game average margin must not swing a coin-flip prediction.
        vals.append(np.mean(prior) * n / (n + H2H_SHRINK_K) if n else 0.0)
        hist.setdefault(key, []).append(r.MARGIN)
    tg["h2h_margin"] = vals
    return tg


tg = add_h2h(tg)
print("H2H feature built.")
""")

md(r"""
### 4b. Symmetric modeling table

Each game becomes **two rows** — one from each team's perspective. Strength
features are differences (`perspective − opponent`); **`is_home`, `is_playoff`,
and their interaction `home_x_playoff` are kept as explicit features** so the
model learns home-court directly (and a *separate, larger* playoff home-court
effect) instead of hiding it in the base rate. The target is balanced at 0.5,
which removes the home-perspective bias of the old design.
""")

code(r"""
# Strength/situational features that enter as perspective - opponent differences.
DIFF_BASE = (
    [f"l10_{m}" for m in ROLL_METRICS] +
    [f"std_{m}" for m in ROLL_METRICS] +
    ["streak", "rest_days", "b2b", "elo_pre", "h2h_margin"]
)
# Context features (NOT differenced) -> explicit home-court modeling.
CONTEXT = ["is_home", "is_playoff", "home_x_playoff"]
FEATURES = [f"d_{f}" for f in DIFF_BASE] + CONTEXT

home = tg[tg.IS_HOME == 1].set_index("GAME_ID")
away = tg[tg.IS_HOME == 0].set_index("GAME_ID")
common = home.index.intersection(away.index)
home, away = home.loc[common], away.loc[common]


def build_rows(persp, opp, is_home):
    d = pd.DataFrame(index=persp.index)
    d["GAME_DATE"] = persp["GAME_DATE"]; d["SEASON"] = persp["SEASON"]
    d["SEASON_TYPE"] = persp["SEASON_TYPE"]
    d["persp_team"] = persp["TEAM_ABBREVIATION"]; d["opp_team"] = opp["TEAM_ABBREVIATION"]
    for f in DIFF_BASE:
        d[f"d_{f}"] = persp[f].values - opp[f].values
    d["is_home"] = is_home
    d["is_playoff"] = (persp["SEASON_TYPE"] == "Playoffs").astype(int).values
    d["home_x_playoff"] = d["is_home"] * d["is_playoff"]
    d["won"] = persp["WIN"].values
    return d


games = pd.concat([build_rows(home, away, 1), build_rows(away, home, 0)])
games = games.dropna(subset=[f"d_{f}" for f in DIFF_BASE]).sort_values("GAME_DATE").reset_index(drop=True)
print(f"Symmetric modeling table: {len(games)} rows ({len(games)//2} games) x {len(FEATURES)} features")
print("Target win rate (should be ~0.50): %.3f" % games["won"].mean())
games[["GAME_DATE","persp_team","opp_team","is_home","is_playoff","won","d_elo_pre","d_l10_NET_RTG"]].tail()
""")

# ---------------------------------------------------------------- Section 5
md(r"""
## 5. Train & evaluate — XGBoost + Logistic Regression

**Chronological split** (never random): train 2023-24 + 2024-25, test 2025-26.

**Sample weighting:** `recency_decay × playoff_mult`, **normalized to mean 1** so
the L2 strength of LR isn't distorted (review fix). The exact per-bucket weights
print below.

**Honest baselines:** we compare against *home-always-wins* AND *higher-Elo-wins*
(a one-feature rule) — the latter is the real bar a feature-rich model must beat.
Calibration uses **TimeSeriesSplit** (not random KFold) to stay leakage-consistent.
""")

code(r"""
REF_DATE = games["GAME_DATE"].max()   # "now" = end of conference finals


def raw_weights(df):
    age = (REF_DATE - df["GAME_DATE"]).dt.days.clip(lower=0)
    recency = 0.5 ** (age / RECENCY_HALFLIFE_DAYS)
    playoff = np.where(df["SEASON_TYPE"] == "Playoffs", PLAYOFF_MULT, 1.0)
    return recency * playoff


train = games[games.SEASON.isin(["2023-24", "2024-25"])].copy()
test = games[games.SEASON == "2025-26"].copy()

X_train, y_train = train[FEATURES].values, train["won"].values
X_test, y_test = test[FEATURES].values, test["won"].values
# Normalize weights to mean 1 (so C behaves as intended); report effective N.
w_train = raw_weights(train).values
w_train = w_train / w_train.mean()
ess = (w_train.sum() ** 2) / (w_train ** 2).sum()

print(f"Train rows: {len(train)} | Test rows: {len(test)}")
print(f"Weighted effective sample size: {ess:.0f} ({100*ess/len(w_train):.0f}% of n)")

# Weight table over ALL games (incl. 2025-26) so the recency/playoff ordering is visible.
_wall = raw_weights(games); _wall = _wall / _wall.mean()
wt = games.assign(w=_wall.values).groupby(["SEASON", "SEASON_TYPE"])["w"].mean().round(3)
print("\nMean (normalized) weight by bucket (all games, highest -> lowest):")
print(wt.sort_values(ascending=False).to_string())
""")

code(r"""
XGB_PARAMS = dict(n_estimators=350, max_depth=4, learning_rate=0.03,
                  subsample=0.85, colsample_bytree=0.8, min_child_weight=6,
                  reg_lambda=2.0, eval_metric="logloss", random_state=42)


def make_xgb():
    return xgb.XGBClassifier(**XGB_PARAMS)


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))


def fit_calibrated(estimator_fn, X, y, w, method, cv):
    cal = CalibratedClassifierCV(estimator_fn(), method=method, cv=cv)
    cal.fit(X, y, sample_weight=w)
    return cal


# EVALUATION models: TimeSeriesSplit calibration so folds never see the future.
EVAL_CV = TimeSeriesSplit(4)
xgb_cal = fit_calibrated(make_xgb, X_train, y_train, w_train, "sigmoid", EVAL_CV)
lr_cal = fit_calibrated(make_lr, X_train, y_train, w_train, "sigmoid", EVAL_CV)
xgb_raw = make_xgb().fit(X_train, y_train, sample_weight=w_train)  # for importance
print("Models trained (sigmoid calibration, TimeSeriesSplit).")
""")

code(r"""
def metrics(name, p, y):
    return {"model": name, "accuracy": accuracy_score(y, (p >= 0.5).astype(int)),
            "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p)}


p_xgb = xgb_cal.predict_proba(X_test)[:, 1]
p_lr = lr_cal.predict_proba(X_test)[:, 1]
p_ens = (p_xgb + p_lr) / 2

# Baselines on the SAME test rows
elo_idx = FEATURES.index("d_elo_pre")
home_idx = FEATURES.index("is_home")
p_elo = 1 / (1 + 10 ** (-(X_test[:, elo_idx] + ELO_HOME * X_test[:, home_idx]) / 400))

results = pd.DataFrame([
    metrics("XGBoost", p_xgb, y_test),
    metrics("LogisticRegression", p_lr, y_test),
    metrics("ENSEMBLE (avg)", p_ens, y_test),
    metrics("Baseline: Elo+home (no ML)", p_elo, y_test),
]).set_index("model").round(4)
print("Holdout (2025-26) performance:\n")
print(results)
print(f"\nBaseline: home-always-wins accuracy = {max(y_test.mean(), 1-y_test.mean()):.4f}")

# Is the ensemble's edge over Elo+home REAL or within noise? McNemar's exact test
# on ONE row per game (home perspective) so the paired trials are independent
# (the symmetric table duplicates every game, which would fake a 2x sample).
from scipy.stats import binomtest
hm = test["is_home"].values == 1
yg = y_test[hm].astype(bool)
ens_ok = (p_ens[hm] >= 0.5) == yg
elo_ok = (p_elo[hm] >= 0.5) == yg
b = int(np.sum(ens_ok & ~elo_ok))   # games only the ensemble gets right
c = int(np.sum(~ens_ok & elo_ok))   # games only Elo+home gets right
mcnemar_p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0
print(f"\nEnsemble vs Elo+home (per-game, n={int(hm.sum())}): "
      f"ensemble-only-right={b}, elo-only-right={c}, McNemar p={mcnemar_p:.3f}")
print("  -> " + ("edge IS statistically significant" if mcnemar_p < 0.05
                 else "edge is NOT statistically significant (within noise)"))
""")

md(r"""
> **Reading these numbers honestly.** Logistic Regression is essentially as strong as
> the ensemble on probability quality (log loss), and XGBoost is the **weakest** of
> the three — with ~15 collinear difference features and only a few thousand games
> there isn't enough nonlinear signal for boosting to dominate. The ensemble is kept
> for **stability and a small accuracy bump, not a large edge**. And the gap over the
> Elo+home baseline is modest (~2pp): the **McNemar test above** reports whether it
> clears statistical noise — on this holdout it does **not** reliably (p≈0.12) — so
> the honest claim is "**beats a strong Elo baseline by a small, not-yet-significant
> margin**," not "decisively beats it."
""")

code(r"""
# Walk-forward (next-season) evaluation — the honest test of generalization.
def eval_split(tr_seasons, te_season):
    tr = games[games.SEASON.isin(tr_seasons)]; te = games[games.SEASON == te_season]
    Xtr, ytr = tr[FEATURES].values, tr["won"].values
    Xte, yte = te[FEATURES].values, te["won"].values
    wtr = raw_weights(tr).values; wtr = wtr / wtr.mean()
    xc = fit_calibrated(make_xgb, Xtr, ytr, wtr, "sigmoid", TimeSeriesSplit(4))
    lc = fit_calibrated(make_lr, Xtr, ytr, wtr, "sigmoid", TimeSeriesSplit(4))
    pe = (xc.predict_proba(Xte)[:, 1] + lc.predict_proba(Xte)[:, 1]) / 2
    return accuracy_score(yte, pe >= .5), log_loss(yte, pe)

print("Walk-forward ensemble evaluation:")
for tr_s, te_s in [(["2023-24"], "2024-25"), (["2023-24", "2024-25"], "2025-26")]:
    acc, ll = eval_split(tr_s, te_s)
    print(f"  train {tr_s} -> test {te_s}:  acc={acc:.4f}  logloss={ll:.4f}")
""")

# ---------------------------------------------------------------- Section 6
md(r"""
## 6. Model diagnostics — calibration & permutation importance

We use **permutation importance** (drop in holdout log loss when a feature is
shuffled) rather than XGBoost's gain, which is unreliable under the residual
feature collinearity.
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

for name, p in [("XGBoost", p_xgb), ("LogReg", p_lr), ("Ensemble", p_ens)]:
    frac, mean_pred = calibration_curve(y_test, p, n_bins=8, strategy="quantile")
    ax[0].plot(mean_pred, frac, "o-", label=name)
ax[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
ax[0].set_xlabel("Predicted P(win)"); ax[0].set_ylabel("Observed frequency")
ax[0].set_title("Calibration (2025-26 holdout)"); ax[0].legend()

perm = permutation_importance(xgb_cal, X_test, y_test, scoring="neg_log_loss",
                              n_repeats=10, random_state=42)
imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values().tail(12)
ax[1].barh(imp.index, imp.values, color="#1f77b4")
ax[1].set_title("Permutation importance (top 12)")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- Section 7
md(r"""
## 7. Spurs vs Knicks — single-game prediction

We refit both models on **all 3 seasons** (max signal), then build each team's
Finals feature row with the **same construction used in training** (single-season
2025-26 season-to-date + last-10, current Elo) — eliminating train/serve skew.
Home court is derived from 2025-26 records.
""")

code(r"""
X_all, y_all = games[FEATURES].values, games["won"].values
w_all = raw_weights(games).values; w_all = w_all / w_all.mean()
# FINAL (deployed) models: the Finals is genuinely in the future, so there is no
# future to leak into -> use KFold calibration that USES ALL DATA (incl. the most
# recent games), unlike TimeSeriesSplit which holds the latest fold out of every
# base model. This keeps the live prediction consistent with the bootstrap CI.
DEPLOY_CV = 5
xgb_final = fit_calibrated(make_xgb, X_all, y_all, w_all, "sigmoid", DEPLOY_CV)
lr_final = fit_calibrated(make_lr, X_all, y_all, w_all, "sigmoid", DEPLOY_CV)
print("Final models refit on all", len(games), "rows (KFold-calibrated).")

# Home court -> better 2025-26 regular-season record.
rs = tg[(tg.SEASON == PREDICT_SEASON) & (tg.SEASON_TYPE == "Regular Season")]
recs = {t: (rs[rs.TEAM_ABBREVIATION == t]["WIN"].sum()) for t in (TEAM_A, TEAM_B)}
HOME_TEAM = TEAM_A if recs[TEAM_A] >= recs[TEAM_B] else TEAM_B
AWAY_TEAM = TEAM_B if HOME_TEAM == TEAM_A else TEAM_A
print(f"2025-26 wins: {TEAM_A}={recs[TEAM_A]} {TEAM_B}={recs[TEAM_B]} -> home court: {HOME_TEAM}")
""")

code(r"""
def signed_streak(win_series):
    cur = 0
    for w in win_series:
        cur = (cur + 1 if cur > 0 else 1) if w == 1 else (cur - 1 if cur < 0 else -1)
    return cur


def team_state(team):
    '''Finals feature row, built EXACTLY like training features (single-season
    2025-26 season-to-date + last-10), so there is no train/serve skew.'''
    g = tg[(tg.TEAM_ABBREVIATION == team) & (tg.SEASON == PREDICT_SEASON)].sort_values("GAME_DATE")
    s = {}
    for m in ROLL_METRICS:
        s[f"l10_{m}"] = g[m].tail(10).mean()
        s[f"std_{m}"] = g[m].mean()
    s["streak"] = signed_streak(g["WIN"].values)
    s["rest_days"] = 3.0
    s["b2b"] = 0
    s["elo_pre"] = final_elo.get(team, ELO_BASE)
    other = TEAM_B if team == TEAM_A else TEAM_A
    h2h = tg[(tg.TEAM_ABBREVIATION == team) & (tg.OPP_ABBR == other)
             & (tg.SEASON == PREDICT_SEASON)]
    n_h2h = len(h2h)
    # Same small-sample shrinkage as training (see add_h2h).
    s["h2h_margin"] = h2h["MARGIN"].mean() * n_h2h / (n_h2h + H2H_SHRINK_K) if n_h2h else 0.0
    return s


state_A, state_B = team_state(TEAM_A), team_state(TEAM_B)


def feature_row(persp_state, opp_state, is_home):
    d = {f"d_{f}": persp_state[f] - opp_state[f] for f in DIFF_BASE}
    d["is_home"] = is_home
    d["is_playoff"] = 1
    d["home_x_playoff"] = is_home * 1
    return np.array([[d[f] for f in FEATURES]])


def p_win(persp, opp_state, is_home, models):
    x = feature_row(persp, opp_state, is_home)
    return float(np.mean([m.predict_proba(x)[:, 1][0] for m in models]))


ens = [xgb_final, lr_final]
# P(SAS win) by venue, using ONE convention everywhere: the host is modeled at home
# (is_home=1) and the visitor's win prob is its complement -- identical to how the
# series simulation builds per-game probabilities, so there is no convention skew
# between the single-game readout and the series sim.
p_A_at_home = p_win(state_A, state_B, 1, ens)          # SAS hosts
p_A_at_away = 1 - p_win(state_B, state_A, 1, ens)      # NYK hosts -> P(SAS win)
p_A_neutral = (p_A_at_home + p_A_at_away) / 2

print(f"{TEAM_A} Elo {final_elo.get(TEAM_A):.0f} | {TEAM_B} Elo {final_elo.get(TEAM_B):.0f}")
print(f"P({TEAM_A} win | {TEAM_A} home): {p_A_at_home:.3f}")
print(f"P({TEAM_A} win | {TEAM_A} away): {p_A_at_away:.3f}")
print(f"Implied playoff home-court edge: {100*(p_A_at_home - p_A_neutral):.1f} pp")
print(f"P({TEAM_A} win, neutral): {p_A_neutral:.3f}")
""")

# ---------------------------------------------------------------- Section 8
md(r"""
## 8. Series simulation — Monte Carlo + bootstrap uncertainty

The Finals use **2-2-1-1-1**; the home-court team hosts games 1, 2, 5, 7. The
point estimate uses the calibrated ensemble's per-venue probabilities. To avoid
**false precision**, we also **bootstrap-refit** the model `N_BOOTSTRAP` times on
resampled training data and re-run the series each time — yielding a credible
interval on P(series win), not just a single number.
""")

code(r"""
def home_schedule(home_team):
    # games 1,2,5,7 at home_team; 3,4,6 at the other team
    pattern = [home_team, home_team, None, None, home_team, None, home_team]
    other = TEAM_B if home_team == TEAM_A else TEAM_A
    return [h if h else other for h in pattern]


GAME_HOMES = home_schedule(HOME_TEAM)


def p_A_per_game(models):
    '''P(TEAM_A wins) for each of the 7 games given the venue.'''
    out = []
    for host in GAME_HOMES:
        if host == TEAM_A:
            out.append(p_win(state_A, state_B, 1, models))
        else:                      # TEAM_A is the visitor
            out.append(1 - p_win(state_B, state_A, 1, models))
    return out


def simulate_series(p_by_game, n_sims, seed=7):
    rng = np.random.default_rng(seed)
    a_series, lengths, winner_len = 0, {4: 0, 5: 0, 6: 0, 7: 0}, []
    for _ in range(n_sims):
        a = b = 0
        for gi in range(7):
            if rng.random() < p_by_game[gi]:
                a += 1
            else:
                b += 1
            if a == 4 or b == 4:
                L = gi + 1
                break
        a_series += (a == 4); lengths[L] += 1; winner_len.append((a == 4, L))
    return a_series / n_sims, lengths, winner_len


p_by_game = p_A_per_game(ens)
p_series_A, length_counts, winner_len = simulate_series(p_by_game, N_SIMS)
p_series_B = 1 - p_series_A

from collections import Counter
exact = Counter((TEAM_A if w else TEAM_B, L) for w, L in winner_len)
top_exact = exact.most_common(1)[0]

print(f"Point estimate  P({TEAM_A} series win): {p_series_A:.3f}")
print(f"                P({TEAM_B} series win): {p_series_B:.3f}")
print(f"Most likely outcome: {top_exact[0][0]} in {top_exact[0][1]} "
      f"({100*top_exact[1]/N_SIMS:.1f}% of sims)")
""")

code(r"""
# Bootstrap: refit (fast, uncalibrated) ensemble on resampled rows -> CI on P(series).
rng = np.random.default_rng(123)
boot_series = []
n = len(X_all)
for _ in range(N_BOOTSTRAP):
    idx = rng.integers(0, n, n)
    Xb, yb, wb = X_all[idx], y_all[idx], w_all[idx]
    if len(np.unique(yb)) < 2:
        continue
    # Calibrate inside the bootstrap (cv=3) so the CI is on the SAME scale as the
    # calibrated point estimate -- otherwise the band wouldn't contain it.
    mb_x = CalibratedClassifierCV(make_xgb(), method="sigmoid", cv=3).fit(Xb, yb, sample_weight=wb)
    mb_l = CalibratedClassifierCV(make_lr(), method="sigmoid", cv=3).fit(Xb, yb, sample_weight=wb)
    pbg = p_A_per_game([mb_x, mb_l])
    ps, _, _ = simulate_series(pbg, 2500)
    boot_series.append(ps)

boot_series = np.array(boot_series)
lo, hi = np.percentile(boot_series, [5, 95])
print(f"Bootstrap P({TEAM_A} series win): {boot_series.mean():.3f}  "
      f"90% CI [{lo:.3f}, {hi:.3f}]  (n={len(boot_series)} refits)")
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
lengths = [4, 5, 6, 7]
a_by = [sum(1 for w, L in winner_len if w and L == k) / N_SIMS for k in lengths]
b_by = [sum(1 for w, L in winner_len if (not w) and L == k) / N_SIMS for k in lengths]
xp = np.arange(4)
ax[0].bar(xp - 0.2, a_by, 0.4, label=TEAM_A, color="#c8102e")
ax[0].bar(xp + 0.2, b_by, 0.4, label=TEAM_B, color="#1d428a")
ax[0].set_xticks(xp); ax[0].set_xticklabels([f"in {k}" for k in lengths])
ax[0].set_ylabel("Probability"); ax[0].set_title("Series outcome distribution"); ax[0].legend()

ax[1].hist(boot_series, bins=15, color="#c8102e", alpha=0.8)
ax[1].axvline(0.5, color="k", ls="--", alpha=0.6)
ax[1].axvline(boot_series.mean(), color="navy", lw=2, label=f"mean {boot_series.mean():.2f}")
ax[1].set_title(f"Bootstrap P({TEAM_A} wins series)"); ax[1].set_xlabel("probability"); ax[1].legend()
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- Section 8b
md(r"""
### 8b. Sensitivity to the weighting knobs

`PLAYOFF_MULT` and `RECENCY_HALFLIFE_DAYS` are judgment calls, and the verdict is a
coin flip — so we check that the series number is **stable** across a grid of both.
If P(SAS series win) stays inside ~[0.45, 0.55] everywhere, the TOSS-UP call does
**not** hinge on how the knobs were tuned.
""")

code(r"""
def weights_for(df, halflife, pmult):
    age = (REF_DATE - df["GAME_DATE"]).dt.days.clip(lower=0)
    w = (0.5 ** (age / halflife)) * np.where(df["SEASON_TYPE"] == "Playoffs", pmult, 1.0)
    return w / w.mean()


HL_GRID = [365, 540, 730]      # recency half-life (days)
PM_GRID = [1.0, 1.5, 2.5]      # playoff multiplier (1.0 = playoffs weighted like RS)
print("P(SAS series win) across weighting knobs:")
print("            " + "  ".join(f"pm={pm:<4}" for pm in PM_GRID))
grid_vals = []
for hl in HL_GRID:
    row_out = []
    for pm in PM_GRID:
        wv = weights_for(games, hl, pm).values
        xf = fit_calibrated(make_xgb, X_all, y_all, wv, "sigmoid", DEPLOY_CV)
        lf = fit_calibrated(make_lr, X_all, y_all, wv, "sigmoid", DEPLOY_CV)
        ps, _, _ = simulate_series(p_A_per_game([xf, lf]), 8000)
        row_out.append(ps); grid_vals.append(ps)
    print(f"  hl={hl:4d}d:  " + "  ".join(f"{v:6.3f}" for v in row_out))

stable = max(grid_vals) < 0.55 and min(grid_vals) > 0.45
print(f"\nRange across all {len(grid_vals)} settings: "
      f"[{min(grid_vals):.3f}, {max(grid_vals):.3f}]  -> "
      f"{'TOSS-UP is stable to the knobs' if stable else 'verdict moves with the knobs'}")
""")

# ---------------------------------------------------------------- Section 9
md(r"""
## 9. Final readout
""")

code(r"""
favorite = TEAM_A if p_series_A >= 0.5 else TEAM_B
fav_prob = max(p_series_A, p_series_B)
edge = abs(p_series_A - 0.5)
names = {"SAS": "San Antonio Spurs", "NYK": "New York Knicks"}
ens_acc = results.loc["ENSEMBLE (avg)", "accuracy"]
elo_acc = results.loc["Baseline: Elo+home (no ML)", "accuracy"]

# Honest framing: with a CI this wide, anything inside ~[0.45, 0.55] is a coin flip.
if edge < 0.05:
    verdict = f"TOSS-UP - essentially 50/50 (faint lean {names[favorite]})"
elif edge < 0.10:
    verdict = f"LEAN {names[favorite]}"
elif edge < 0.20:
    verdict = f"MODERATE edge: {names[favorite]}"
else:
    verdict = f"STRONG edge: {names[favorite]}"

print("=" * 62)
print("        NBA FINALS 2025-26 PREDICTION")
print(f"   {names[TEAM_A]}  vs  {names[TEAM_B]}")
print("=" * 62)
print(f"\n  VERDICT: {verdict}")
print(f"\n  P({TEAM_A} wins series): {p_series_A:.1%}   P({TEAM_B} wins series): {p_series_B:.1%}")
print(f"  Bootstrap 90% CI on P({TEAM_A} wins): [{lo:.1%}, {hi:.1%}]  (wide -> low confidence)")
print(f"  Single game (neutral court): {TEAM_A} {p_A_neutral:.1%} / {TEAM_B} {1-p_A_neutral:.1%}")
print(f"  Home court: {names[HOME_TEAM]} (2-2-1-1-1), worth ~{100*(p_A_at_home-p_A_neutral):.0f}pp/game")
print(f"  Most likely single outcome: {top_exact[0][0]} in {top_exact[0][1]}")
print(f"\n  Holdout accuracy: ensemble {ens_acc:.1%}  vs  Elo-only baseline {elo_acc:.1%}")
print(f"  (the ML adds {100*(ens_acc-elo_acc):.1f}pp of accuracy over a one-feature Elo rule)")
print(f"  Models: XGBoost + Logistic Regression (sigmoid-calibrated ensemble)")
print("=" * 62)

# Persist the headline numbers so downstream artifacts (make_chart.py) read from a
# single source of truth instead of hardcoding values that silently go stale.
import json
with open("finals_prediction.json", "w") as f:
    json.dump({
        "p_spurs_series": round(100 * p_series_A, 1),
        "p_knicks_series": round(100 * p_series_B, 1),
        "ensemble_accuracy": round(100 * ens_acc, 1),
        "home_team": HOME_TEAM,
        "most_likely": f"{top_exact[0][0]} in {top_exact[0][1]}",
    }, f, indent=2)
print("Wrote finals_prediction.json")
""")

# ----------------------------------------------------------------
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
with open(OUT_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote {OUT_PATH} with {len(cells)} cells (COLAB={COLAB})")
