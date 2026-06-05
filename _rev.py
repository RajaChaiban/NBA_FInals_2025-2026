#!/usr/bin/env python
# coding: utf-8

# # 🏀 NBA Finals Predictor — Spurs vs Knicks (2025-26)
# 
# An ensemble ML pipeline (**XGBoost + Logistic Regression**) that predicts the
# 2025-26 NBA Finals between the **San Antonio Spurs** and **New York Knicks**.
# 
# **Approach:** a league-wide per-game win model (3 seasons, regular season +
# playoffs) with leakage-safe pre-game features, plus Spurs/Knicks-specific
# matchup features, then a **Monte Carlo best-of-7 simulation** for the series.
# 
# **Pipeline:** data → dataset construction → feature engineering → train/eval →
# single-game prediction → series simulation.

# ## 1. Setup & configuration

# In[1]:


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
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import xgboost as xgb

warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 60)
np.random.seed(42)

DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "raw_games.csv")

# The two Finals teams + home court (better 2025-26 record gets it).
TEAM_A = "SAS"   # San Antonio Spurs (62-20, West #1) -> HOME COURT
TEAM_B = "NYK"   # New York Knicks   (53-29, East)

# --- Sample-weight scheme -------------------------------------------------
# Each training game's weight = recency_decay * playoff_multiplier, so the most
# recent PLAYOFF games (2025-26) dominate and the oldest regular-season games
# (2023-24) count least. Tune these two knobs to taste.
PLAYOFF_MULT = 2.5            # playoff game weight vs regular-season (1.0)
RECENCY_HALFLIFE_DAYS = 365   # a game's weight halves every ~1 season into the past

# --- Roster-era guard ----------------------------------------------------
# Only the seasons where each Finals team's CURRENT core was intact (see research):
#   Spurs: Fox arrived Feb 2025 but Wembanyama was lost to a blood clot 2 weeks
#          later -> the Fox+Wemby core is only healthy together in 2025-26.
#   Knicks: KAT + Bridges joined for 2024-25 -> current core = 2024-25 onward.
ROSTER_ERA = {"SAS": ["2025-26"], "NYK": ["2024-25", "2025-26"]}

ELO_BASE, ELO_K, ELO_HOME = 1500.0, 20.0, 100.0
N_SIMS = 20000           # Monte Carlo series simulations

print("Config loaded. Finals:", TEAM_A, "vs", TEAM_B)


# ## 2. Data acquisition (cached)
# 
# Raw team-game box scores were pulled from `stats.nba.com` via **`nba_api`** and
# cached by `fetch_data.py`. If the cache is missing, run `python fetch_data.py`
# first. Two rows per game (one per team).

# In[2]:


assert os.path.exists(RAW_PATH), "Missing data/raw_games.csv -- run: python fetch_data.py"
raw = pd.read_csv(RAW_PATH, dtype={"GAME_ID": str})
raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
raw = raw.sort_values("GAME_DATE").reset_index(drop=True)
print(f"{len(raw)} team-game rows | {raw['GAME_ID'].nunique()} games")
print(raw.groupby(['SEASON','SEASON_TYPE'])['GAME_ID'].nunique())


# ## 3. Dataset construction
# 
# Each game has a home row (`MATCHUP` contains `vs.`) and an away row (`@`). We
# join the two so every game also carries its **opponent's** box score — needed to
# compute defensive rating, opponent-dependent four factors, etc.

# In[3]:


def add_realized_metrics(df):
    '''Compute realized (post-game) advanced metrics from a team's box score
    plus its opponent's box score. These are the per-game ingredients we later
    turn into leakage-safe pre-game rolling features.'''
    df = df.copy()
    # Possessions (standard estimate) and tempo
    poss = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]
    df["POSS"] = poss.clip(lower=1)
    df["OFF_RTG"] = 100 * df["PTS"] / df["POSS"]
    df["DEF_RTG"] = 100 * df["OPP_PTS"] / df["POSS"]
    df["NET_RTG"] = df["OFF_RTG"] - df["DEF_RTG"]
    df["PACE"] = df["POSS"]  # per-game possessions ~ pace proxy
    # Four factors
    df["EFG"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"]
    df["TOV_PCT"] = df["TOV"] / df["POSS"]
    df["OREB_PCT"] = df["OREB"] / (df["OREB"] + df["OPP_DREB"]).clip(lower=1)
    df["FT_RATE"] = df["FTA"] / df["FGA"]
    df["WIN"] = (df["WL"] == "W").astype(int)
    df["MARGIN"] = df["PTS"] - df["OPP_PTS"]
    return df


# Self-join on GAME_ID to attach opponent stats to each team-game row.
opp_cols = ["GAME_ID", "TEAM_ABBREVIATION", "PTS", "DREB"]
opp = raw[opp_cols].rename(columns={
    "TEAM_ABBREVIATION": "OPP_ABBR", "PTS": "OPP_PTS", "DREB": "OPP_DREB"})
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


# ## 4. Feature engineering (leakage-safe, pre-game)
# 
# For every team-game we compute features using **only games played before it**:
# 
# - **Form/strength:** season-to-date & last-10 win%, point differential, signed streak
# - **Efficiency:** off/def/net rating, pace, four factors (eFG%, TOV%, OREB%, FT rate)
# - **Situational:** rest days, back-to-back flag
# - **Elo:** sequentially updated rating with home advantage & season regression
# - **Matchup (hybrid):** prior head-to-head average margin vs that specific opponent
# 
# Rolling/expanding stats are **shifted by one game** so the current game never
# leaks. Form resets per season; Elo carries over with a 25% regression to the mean.

# In[4]:


# Metrics we turn into pre-game rolling (last-10) and expanding (season-to-date) features.
ROLL_METRICS = ["WIN", "MARGIN", "OFF_RTG", "DEF_RTG", "NET_RTG", "PACE",
                "EFG", "TOV_PCT", "OREB_PCT", "FT_RATE"]


def build_form_features(tg):
    '''Per (team, season): shifted last-10 and season-to-date means + streak,
    rest days and back-to-back. Everything is strictly pre-game.'''
    out = []
    for (team, season), g in tg.groupby(["TEAM_ABBREVIATION", "SEASON"], sort=False):
        g = g.sort_values("GAME_DATE").copy()
        sh = g[ROLL_METRICS].shift(1)               # exclude current game
        for m in ROLL_METRICS:
            g[f"l10_{m}"] = sh[m].rolling(10, min_periods=1).mean()
            g[f"std_{m}"] = sh[m].expanding(min_periods=1).mean()
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


# In[5]:


def add_elo(tg):
    '''Sequential Elo over all games in date order. Stores each team's PRE-game
    Elo. Home team gets +ELO_HOME when computing expected score. Ratings regress
    25% toward ELO_BASE at the start of each new season.'''
    tg = tg.sort_values(["GAME_DATE", "GAME_ID"]).copy()
    elo = {}
    season_seen = {}
    pre_elo = {}
    # iterate game by game (each game = 2 rows; process once per GAME_ID)
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


# In[6]:


def add_h2h(tg):
    '''Prior head-to-head average margin vs the specific opponent (pre-game,
    across the whole window). Sparse for cross-conference pairs -> 0 when unseen.'''
    tg = tg.sort_values(["GAME_DATE", "GAME_ID"]).copy()
    hist = {}
    vals = []
    for r in tg.itertuples():
        key = (r.TEAM_ABBREVIATION, r.OPP_ABBR)
        prior = hist.get(key, [])
        vals.append(np.mean(prior) if prior else 0.0)
        hist.setdefault(key, []).append(r.MARGIN)
    tg["h2h_margin"] = vals
    return tg


tg = add_h2h(tg)
print("H2H feature built.")


# In[7]:


# Assemble the game-level modeling table: one row per game (home perspective),
# features expressed as home - away differentials.
FEATURE_BASE = (
    [f"l10_{m}" for m in ROLL_METRICS] +
    [f"std_{m}" for m in ROLL_METRICS] +
    ["streak", "rest_days", "b2b", "elo_pre", "h2h_margin"]
)

home = tg[tg.IS_HOME == 1].set_index("GAME_ID")
away = tg[tg.IS_HOME == 0].set_index("GAME_ID")

games = pd.DataFrame(index=home.index)
games["GAME_DATE"] = home["GAME_DATE"]
games["SEASON"] = home["SEASON"]
games["SEASON_TYPE"] = home["SEASON_TYPE"]
games["home_team"] = home["TEAM_ABBREVIATION"]
games["away_team"] = away["TEAM_ABBREVIATION"]
games["home_win"] = home["WIN"]

DIFF_FEATURES = []
for f in FEATURE_BASE:
    games[f"d_{f}"] = home[f] - away[f]
    DIFF_FEATURES.append(f"d_{f}")

games = games.dropna(subset=DIFF_FEATURES).sort_values("GAME_DATE").reset_index(drop=True)
games["is_playoff"] = (games["SEASON_TYPE"] == "Playoffs").astype(int)
print(f"Modeling table: {len(games)} games x {len(DIFF_FEATURES)} features")
print("Home win base rate: %.3f" % games["home_win"].mean())
games[["GAME_DATE","home_team","away_team","home_win","d_elo_pre","d_l10_NET_RTG"]].tail()


# ## 5. Train & evaluate — XGBoost + Logistic Regression
# 
# **Chronological split** (never random — avoids time leakage):
# - **Train:** 2023-24 + 2024-25 (both season types)
# - **Test:** 2025-26 (regular season + playoffs)
# 
# **Sample weighting:** each game's training weight = `recency_decay × playoff_mult`,
# so **2025-26 playoff games carry the most weight** and 2023-24 regular-season games
# the least (recency halves every ~365 days; playoffs ×2.5). The exact per-bucket
# weights are printed below. We also run `TimeSeriesSplit` CV on the train set. Both
# models are **probability-calibrated**; we report accuracy, log loss and Brier
# score, and the ensemble averages the two calibrated probabilities.

# In[8]:


# Reference "now" = the latest game in the data (end of the conference finals).
REF_DATE = games["GAME_DATE"].max()


def sample_weights(df):
    '''Per-game training weight = recency decay * playoff multiplier.
    Recency halves every RECENCY_HALFLIFE_DAYS; playoffs get PLAYOFF_MULT.
    Result: 2025-26 playoffs weigh most, 2023-24 regular season least.'''
    age_days = (REF_DATE - df["GAME_DATE"]).dt.days.clip(lower=0)
    recency = 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)
    playoff = np.where(df["SEASON_TYPE"] == "Playoffs", PLAYOFF_MULT, 1.0)
    return recency * playoff


games["weight"] = sample_weights(games)

train = games[games.SEASON.isin(["2023-24", "2024-25"])].copy()
test = games[games.SEASON == "2025-26"].copy()

X_train, y_train = train[DIFF_FEATURES].values, train["home_win"].values
X_test, y_test = test[DIFF_FEATURES].values, test["home_win"].values
w_train = train["weight"].values

print(f"Train: {len(train)} games  | Test: {len(test)} games")
print(f"Test home-win base rate: {y_test.mean():.3f}  (naive 'home always wins' accuracy)")

# Show the average sample weight per season/type (what the model actually sees)
wt_table = (games.groupby(["SEASON", "SEASON_TYPE"])["weight"]
            .mean().round(3).reset_index()
            .sort_values("weight", ascending=False))
print("\nAverage training weight by bucket (highest -> lowest):")
print(wt_table.to_string(index=False))


# In[9]:


def fit_xgb(X, y, w):
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        reg_lambda=1.5, eval_metric="logloss", random_state=42)
    m.fit(X, y, sample_weight=w)
    return m


def fit_lr(X, y, w):
    base = make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.5, max_iter=2000))
    base.fit(X, y, logisticregression__sample_weight=w)
    return base


# Calibrate both via cross-val on the training data (prefit-free, sklearn 1.8 API)
xgb_cal = CalibratedClassifierCV(
    xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                      subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
                      reg_lambda=1.5, eval_metric="logloss", random_state=42),
    method="isotonic", cv=3)
xgb_cal.fit(X_train, y_train, sample_weight=w_train)

lr_cal = CalibratedClassifierCV(
    make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
    method="sigmoid", cv=3)
lr_cal.fit(X_train, y_train, sample_weight=w_train)

# Raw (uncalibrated) XGB too, for feature importances
xgb_raw = fit_xgb(X_train, y_train, w_train)
print("Models trained.")


# In[10]:


def report(name, model, X, y):
    p = model.predict_proba(X)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y, (p >= 0.5).astype(int)),
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
    }


p_xgb = xgb_cal.predict_proba(X_test)[:, 1]
p_lr = lr_cal.predict_proba(X_test)[:, 1]
p_ens = (p_xgb + p_lr) / 2

rows = [report("XGBoost", xgb_cal, X_test, y_test),
        report("LogisticRegression", lr_cal, X_test, y_test)]
rows.append({"model": "ENSEMBLE (avg)",
             "accuracy": accuracy_score(y_test, (p_ens >= 0.5).astype(int)),
             "log_loss": log_loss(y_test, p_ens),
             "brier": brier_score_loss(y_test, p_ens)})
results = pd.DataFrame(rows).set_index("model").round(4)
print("Holdout (2025-26) performance:\n")
print(results)
print(f"\nBaseline (home always wins): {max(y_test.mean(), 1-y_test.mean()):.4f}")


# In[11]:


# Time-series CV log loss on the training set (sanity check for stability)
tscv = TimeSeriesSplit(n_splits=5)
cv_ll = []
for tr, va in tscv.split(X_train):
    m = fit_xgb(X_train[tr], y_train[tr], w_train[tr])
    cv_ll.append(log_loss(y_train[va], m.predict_proba(X_train[va])[:, 1]))
print("XGB TimeSeriesSplit log loss: %.4f +/- %.4f" % (np.mean(cv_ll), np.std(cv_ll)))


# ## 6. Model diagnostics — calibration & feature importance

# In[12]:


fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# Calibration curves
for name, p in [("XGBoost", p_xgb), ("LogReg", p_lr), ("Ensemble", p_ens)]:
    frac, mean_pred = calibration_curve(y_test, p, n_bins=8, strategy="quantile")
    ax[0].plot(mean_pred, frac, "o-", label=name)
ax[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
ax[0].set_xlabel("Predicted P(home win)"); ax[0].set_ylabel("Observed frequency")
ax[0].set_title("Calibration (2025-26 holdout)"); ax[0].legend()

# Feature importance (XGB gain)
imp = (pd.Series(xgb_raw.feature_importances_, index=DIFF_FEATURES)
       .sort_values().tail(12))
ax[1].barh(imp.index, imp.values, color="#1f77b4")
ax[1].set_title("XGBoost feature importance (top 12)")
plt.tight_layout(); plt.show()


# ## 7. Spurs vs Knicks — single-game prediction
# 
# For the actual Finals we refit both models on **all 3 seasons** (max signal),
# then build the matchup feature row from each team's **current state** — their
# season-to-date stats and latest Elo, through the conference finals.

# In[13]:


# Refit on ALL data for the live prediction (same recency*playoff weighting)
X_all, y_all = games[DIFF_FEATURES].values, games["home_win"].values
w_all = games["weight"].values

xgb_final = CalibratedClassifierCV(
    xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                      subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
                      reg_lambda=1.5, eval_metric="logloss", random_state=42),
    method="isotonic", cv=3).fit(X_all, y_all, sample_weight=w_all)
lr_final = CalibratedClassifierCV(
    make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
    method="sigmoid", cv=3).fit(X_all, y_all, sample_weight=w_all)
print("Final models refit on all", len(games), "games.")


# In[14]:


def current_state(team):
    '''Each team's pre-Finals state, ROSTER-FAITHFUL:
      - stable strength (std_*) averaged over the team's current-roster seasons
        (ROSTER_ERA) so a different-roster past season can't pollute it;
      - recent form (l10_*) from the last 10 games actually played (the 2025-26
        playoffs), which is current-roster by construction;
      - H2H only counts meetings where BOTH teams' current cores were intact.'''
    era = ROSTER_ERA[team]
    g_era = tg[(tg.TEAM_ABBREVIATION == team) & (tg.SEASON.isin(era))].sort_values("GAME_DATE")
    g_recent = g_era.tail(10)               # last 10 games (current roster)
    last = g_era.iloc[-1]
    state = {}
    for m in ROLL_METRICS:
        state[f"l10_{m}"] = g_recent[m].mean()   # recent form
        state[f"std_{m}"] = g_era[m].mean()      # roster-era strength
    state["streak"] = last["streak"]
    state["rest_days"] = 3.0                       # Finals: well-rested
    state["b2b"] = 0                               # no back-to-backs in Finals
    state["elo_pre"] = final_elo.get(team, ELO_BASE)
    # H2H only in seasons where BOTH teams had their current core (intersection).
    other = TEAM_B if team == TEAM_A else TEAM_A
    shared_era = [s for s in ROSTER_ERA[team] if s in ROSTER_ERA[other]]
    h2h = tg[(tg.TEAM_ABBREVIATION == team) & (tg.OPP_ABBR == other)
             & (tg.SEASON.isin(shared_era))]
    state["h2h_margin"] = h2h["MARGIN"].mean() if len(h2h) else 0.0
    return state


state_A, state_B = current_state(TEAM_A), current_state(TEAM_B)


def matchup_row(home_state, away_state):
    return np.array([[home_state[f] - away_state[f] for f in FEATURE_BASE]])


def game_prob_home(home_team):
    '''P(home_team wins) for a single game where home_team is at home.'''
    if home_team == TEAM_A:
        x = matchup_row(state_A, state_B)
    else:
        x = matchup_row(state_B, state_A)
    p = (xgb_final.predict_proba(x)[:, 1] + lr_final.predict_proba(x)[:, 1]) / 2
    return float(p[0])


# Per-team single-game win prob when each is at home
p_A_home = game_prob_home(TEAM_A)          # SAS at home -> P(SAS win)
p_B_home = game_prob_home(TEAM_B)          # NYK at home -> P(NYK win)
# Neutral-ish single number: P(SAS beats NYK) averaged over venue
p_A_neutral = (p_A_home + (1 - p_B_home)) / 2

print(f"{TEAM_A} Elo {final_elo.get(TEAM_A):.0f} | {TEAM_B} Elo {final_elo.get(TEAM_B):.0f}")
print(f"P({TEAM_A} win | {TEAM_A} home): {p_A_home:.3f}")
print(f"P({TEAM_B} win | {TEAM_B} home): {p_B_home:.3f}")
print(f"P({TEAM_A} win, venue-averaged): {p_A_neutral:.3f}")


# ## 8. Series simulation — Monte Carlo best-of-7
# 
# The Finals use the **2-2-1-1-1** format. San Antonio has home court (better
# record), so games **1, 2, 5, 7** are in SAS and **3, 4, 6** in NYK. We simulate
# the series `N_SIMS` times using each game's venue-specific win probability and
# report P(series win) plus the distribution of series lengths.

# In[15]:


# Home team for each of the 7 games (2-2-1-1-1), TEAM_A has home court.
GAME_HOMES = [TEAM_A, TEAM_A, TEAM_B, TEAM_B, TEAM_A, TEAM_B, TEAM_A]
# P(TEAM_A wins) for each game given who's home
p_A_by_game = [game_prob_home(TEAM_A) if h == TEAM_A else 1 - game_prob_home(TEAM_B)
               for h in GAME_HOMES]


def simulate_series(p_A_by_game, n_sims):
    a_wins_series = 0
    length_counts = {4: 0, 5: 0, 6: 0, 7: 0}
    winner_len = []  # (a_won, length)
    rng = np.random.default_rng(7)
    for _ in range(n_sims):
        a, b = 0, 0
        for gi in range(7):
            if rng.random() < p_A_by_game[gi]:
                a += 1
            else:
                b += 1
            if a == 4 or b == 4:
                length = gi + 1
                break
        a_won = a == 4
        a_wins_series += a_won
        length_counts[length] += 1
        winner_len.append((a_won, length))
    return a_wins_series / n_sims, length_counts, winner_len


p_series_A, length_counts, winner_len = simulate_series(p_A_by_game, N_SIMS)
p_series_B = 1 - p_series_A

# Most likely exact outcome (e.g., SAS in 6)
from collections import Counter
exact = Counter()
for a_won, length in winner_len:
    team = TEAM_A if a_won else TEAM_B
    exact[(team, length)] += 1
top_exact = exact.most_common(1)[0]

print(f"P({TEAM_A} win series): {p_series_A:.3f}")
print(f"P({TEAM_B} win series): {p_series_B:.3f}")
print(f"Most likely outcome: {top_exact[0][0]} in {top_exact[0][1]} "
      f"({100*top_exact[1]/N_SIMS:.1f}% of sims)")


# In[16]:


# Visualize series-length distribution split by winner
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

lengths = [4, 5, 6, 7]
a_by_len = [sum(1 for w, l in winner_len if w and l == L) / N_SIMS for L in lengths]
b_by_len = [sum(1 for w, l in winner_len if (not w) and l == L) / N_SIMS for L in lengths]
xpos = np.arange(len(lengths))
ax[0].bar(xpos - 0.2, a_by_len, 0.4, label=TEAM_A, color="#c8102e")
ax[0].bar(xpos + 0.2, b_by_len, 0.4, label=TEAM_B, color="#1d428a")
ax[0].set_xticks(xpos); ax[0].set_xticklabels([f"in {L}" for L in lengths])
ax[0].set_ylabel("Probability"); ax[0].set_title("Series outcome distribution")
ax[0].legend()

ax[1].bar([TEAM_A, TEAM_B], [p_series_A, p_series_B], color=["#c8102e", "#1d428a"])
ax[1].set_ylim(0, 1); ax[1].set_title("P(win the Finals)")
for i, v in enumerate([p_series_A, p_series_B]):
    ax[1].text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
plt.tight_layout(); plt.show()


# ## 9. Final readout

# In[17]:


favorite = TEAM_A if p_series_A >= 0.5 else TEAM_B
fav_prob = max(p_series_A, p_series_B)
conf = ("TOSS-UP" if fav_prob < 0.55 else "LEAN" if fav_prob < 0.65
        else "MODERATE" if fav_prob < 0.75 else "STRONG")
names = {"SAS": "San Antonio Spurs", "NYK": "New York Knicks"}

print("=" * 56)
print("        NBA FINALS 2025-26 PREDICTION")
print(f"   {names[TEAM_A]}  vs  {names[TEAM_B]}")
print("=" * 56)
print(f"\n  PREDICTED CHAMPION: {names[favorite]}")
print(f"  Series win probability: {fav_prob:.1%}   [{conf}]")
print(f"\n  Single game (venue-averaged): {TEAM_A} {p_A_neutral:.1%} / {TEAM_B} {1-p_A_neutral:.1%}")
print(f"  Home court: {names[TEAM_A]} (2-2-1-1-1)")
print(f"  Most likely outcome: {top_exact[0][0]} in {top_exact[0][1]}")
print(f"\n  Model holdout accuracy (2025-26): {results.loc['ENSEMBLE (avg)','accuracy']:.1%}")
print(f"  Models: XGBoost + Logistic Regression (calibrated ensemble)")
print("=" * 56)

