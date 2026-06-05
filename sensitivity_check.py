"""
Standalone verification harness for the NBA Finals predictor (post-review v2).

Reproduces the *fixed* pipeline end-to-end and reports:
  1. base-case P(SAS series win) and per-venue single-game probabilities;
  2. the McNemar significance test of ensemble vs Elo+home on the holdout;
  3. a sensitivity grid of P(SAS series win) over the two weighting knobs
     (PLAYOFF_MULT x RECENCY_HALFLIFE_DAYS), to confirm the TOSS-UP verdict is
     not an artifact of how those knobs were set.

This mirrors build_notebook.py so the numbers match the notebook. Run:
    python sensitivity_check.py
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from scipy.stats import binomtest
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "raw_games.csv")
TEAM_A, TEAM_B = "SAS", "NYK"
PREDICT_SEASON = "2025-26"
PLAYOFF_MULT = 2.5
RECENCY_HALFLIFE_DAYS = 540
H2H_SHRINK_K = 5
ELO_BASE, ELO_K, ELO_HOME = 1500.0, 20.0, 100.0
N_SIMS = 20000
DEPLOY_CV = 5

ROLL_METRICS = ["WIN", "OFF_RTG", "DEF_RTG", "NET_RTG", "PACE",
                "EFG", "TOV_PCT", "OREB_PCT", "FT_RATE"]
DIFF_BASE = ([f"l10_{m}" for m in ROLL_METRICS] + [f"std_{m}" for m in ROLL_METRICS] +
             ["streak", "rest_days", "b2b", "elo_pre", "h2h_margin"])
CONTEXT = ["is_home", "is_playoff", "home_x_playoff"]
FEATURES = [f"d_{f}" for f in DIFF_BASE] + CONTEXT


# ----------------------------------------------------------------- data + features
def add_realized_metrics(df):
    df = df.copy()
    poss_team = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]
    poss_opp = df["OPP_FGA"] - df["OPP_OREB"] + df["OPP_TOV"] + 0.44 * df["OPP_FTA"]
    df["POSS"] = (0.5 * (poss_team + poss_opp)).clip(lower=1)
    df["OFF_RTG"] = 100 * df["PTS"] / df["POSS"]
    df["DEF_RTG"] = 100 * df["OPP_PTS"] / df["POSS"]
    df["NET_RTG"] = df["OFF_RTG"] - df["DEF_RTG"]
    df["PACE"] = df["POSS"]
    df["EFG"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"]
    df["TOV_PCT"] = df["TOV"] / df["POSS"]
    df["OREB_PCT"] = df["OREB"] / (df["OREB"] + df["OPP_DREB"]).clip(lower=1)
    df["FT_RATE"] = df["FTA"] / df["FGA"]
    df["WIN"] = (df["WL"] == "W").astype(int)
    df["MARGIN"] = df["PTS"] - df["OPP_PTS"]
    return df


def build_form_features(tg):
    out = []
    for (_, _), g in tg.groupby(["TEAM_ABBREVIATION", "SEASON"], sort=False):
        g = g.sort_values("GAME_DATE").copy()
        sh = g[ROLL_METRICS].shift(1)
        for m in ROLL_METRICS:
            g[f"l10_{m}"] = sh[m].rolling(10, min_periods=3).mean()
            g[f"std_{m}"] = sh[m].expanding(min_periods=3).mean()
        prev_win = g["WIN"].shift(1)
        streak, cur = [], 0
        for w in prev_win.values:
            if np.isnan(w):
                cur = 0
            elif w == 1:
                cur = cur + 1 if cur > 0 else 1
            else:
                cur = cur - 1 if cur < 0 else -1
            streak.append(cur)
        g["streak"] = streak
        days = g["GAME_DATE"].diff().dt.days
        g["rest_days"] = days.fillna(3).clip(upper=7)
        g["b2b"] = (days == 1).astype(int)
        out.append(g)
    return pd.concat(out).sort_values("GAME_DATE")


def add_elo(tg):
    tg = tg.sort_values(["GAME_DATE", "GAME_ID"]).copy()
    elo, season_seen, pre_elo = {}, {}, {}
    for gid, gg in tg.groupby("GAME_ID", sort=False):
        if len(gg) != 2:
            continue
        home = gg[gg.IS_HOME == 1].iloc[0]
        away = gg[gg.IS_HOME == 0].iloc[0]
        season = home["SEASON"]
        for t in (home["TEAM_ABBREVIATION"], away["TEAM_ABBREVIATION"]):
            if t not in elo:
                elo[t] = ELO_BASE
            if season_seen.get(t) != season:
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


def add_h2h(tg):
    tg = tg.sort_values(["GAME_DATE", "GAME_ID"]).copy()
    hist, vals = {}, []
    for r in tg.itertuples():
        key = (r.TEAM_ABBREVIATION, r.OPP_ABBR)
        prior = hist.get(key, [])
        n = len(prior)
        vals.append(np.mean(prior) * n / (n + H2H_SHRINK_K) if n else 0.0)
        hist.setdefault(key, []).append(r.MARGIN)
    tg["h2h_margin"] = vals
    return tg


def build_rows(persp, opp, is_home):
    d = pd.DataFrame(index=persp.index)
    d["GAME_DATE"] = persp["GAME_DATE"]; d["SEASON"] = persp["SEASON"]
    d["SEASON_TYPE"] = persp["SEASON_TYPE"]
    for f in DIFF_BASE:
        d[f"d_{f}"] = persp[f].values - opp[f].values
    d["is_home"] = is_home
    d["is_playoff"] = (persp["SEASON_TYPE"] == "Playoffs").astype(int).values
    d["home_x_playoff"] = d["is_home"] * d["is_playoff"]
    d["won"] = persp["WIN"].values
    return d


# ----------------------------------------------------------------- models
def make_xgb():
    return xgb.XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.03,
                             subsample=0.85, colsample_bytree=0.8, min_child_weight=6,
                             reg_lambda=2.0, eval_metric="logloss", random_state=42)


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))


def fit_calibrated(fn, X, y, w, cv):
    cal = CalibratedClassifierCV(fn(), method="sigmoid", cv=cv)
    cal.fit(X, y, sample_weight=w)
    return cal


def signed_streak(win_series):
    cur = 0
    for w in win_series:
        cur = (cur + 1 if cur > 0 else 1) if w == 1 else (cur - 1 if cur < 0 else -1)
    return cur


def simulate_series(p_by_game, n_sims, seed=7):
    rng = np.random.default_rng(seed)
    a_series = 0
    for _ in range(n_sims):
        a = b = 0
        for gi in range(7):
            if rng.random() < p_by_game[gi]:
                a += 1
            else:
                b += 1
            if a == 4 or b == 4:
                break
        a_series += (a == 4)
    return a_series / n_sims


# ----------------------------------------------------------------- build everything
def main():
    raw = pd.read_csv(RAW_PATH, dtype={"GAME_ID": str})
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
    raw = raw.sort_values("GAME_DATE").reset_index(drop=True)

    opp_cols = ["GAME_ID", "TEAM_ABBREVIATION", "PTS", "DREB", "FGA", "OREB", "TOV", "FTA"]
    opp = raw[opp_cols].rename(columns={
        "TEAM_ABBREVIATION": "OPP_ABBR", "PTS": "OPP_PTS", "DREB": "OPP_DREB",
        "FGA": "OPP_FGA", "OREB": "OPP_OREB", "TOV": "OPP_TOV", "FTA": "OPP_FTA"})
    tg = raw.merge(opp, on="GAME_ID")
    tg = tg[tg["TEAM_ABBREVIATION"] != tg["OPP_ABBR"]].copy()
    tg["IS_HOME"] = tg["MATCHUP"].str.contains("vs.").astype(int)
    home_counts = tg.groupby("GAME_ID")["IS_HOME"].transform("sum")
    tg = tg[home_counts == 1].copy()
    tg = add_realized_metrics(tg)
    tg = build_form_features(tg)
    tg, final_elo = add_elo(tg)
    tg = add_h2h(tg)

    home = tg[tg.IS_HOME == 1].set_index("GAME_ID")
    away = tg[tg.IS_HOME == 0].set_index("GAME_ID")
    common = home.index.intersection(away.index)
    home, away = home.loc[common], away.loc[common]
    games = pd.concat([build_rows(home, away, 1), build_rows(away, home, 0)])
    games = games.dropna(subset=[f"d_{f}" for f in DIFF_BASE]).sort_values("GAME_DATE").reset_index(drop=True)

    REF_DATE = games["GAME_DATE"].max()
    X_all, y_all = games[FEATURES].values, games["won"].values

    def weights_for(df, halflife, pmult):
        age = (REF_DATE - df["GAME_DATE"]).dt.days.clip(lower=0)
        w = (0.5 ** (age / halflife)) * np.where(df["SEASON_TYPE"] == "Playoffs", pmult, 1.0)
        return w / w.mean()

    # --- team states (independent of the weighting knobs) ---
    def team_state(team):
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
        h2h = tg[(tg.TEAM_ABBREVIATION == team) & (tg.OPP_ABBR == other) & (tg.SEASON == PREDICT_SEASON)]
        n = len(h2h)
        s["h2h_margin"] = h2h["MARGIN"].mean() * n / (n + H2H_SHRINK_K) if n else 0.0
        return s

    state_A, state_B = team_state(TEAM_A), team_state(TEAM_B)

    def feature_row(persp, opp_state, is_home):
        d = {f"d_{f}": persp[f] - opp_state[f] for f in DIFF_BASE}
        d["is_home"] = is_home; d["is_playoff"] = 1; d["home_x_playoff"] = is_home
        return np.array([[d[f] for f in FEATURES]])

    def p_win(persp, opp_state, is_home, models):
        x = feature_row(persp, opp_state, is_home)
        return float(np.mean([m.predict_proba(x)[:, 1][0] for m in models]))

    # home court -> better 2025-26 RS record
    rs = tg[(tg.SEASON == PREDICT_SEASON) & (tg.SEASON_TYPE == "Regular Season")]
    recs = {t: rs[rs.TEAM_ABBREVIATION == t]["WIN"].sum() for t in (TEAM_A, TEAM_B)}
    HOME_TEAM = TEAM_A if recs[TEAM_A] >= recs[TEAM_B] else TEAM_B
    other = TEAM_B if HOME_TEAM == TEAM_A else TEAM_A
    pattern = [HOME_TEAM, HOME_TEAM, None, None, HOME_TEAM, None, HOME_TEAM]
    GAME_HOMES = [h if h else other for h in pattern]

    def p_A_per_game(models):
        out = []
        for host in GAME_HOMES:
            if host == TEAM_A:
                out.append(p_win(state_A, state_B, 1, models))
            else:
                out.append(1 - p_win(state_B, state_A, 1, models))
        return out

    # ---------------- base case ----------------
    w_base = weights_for(games, RECENCY_HALFLIFE_DAYS, PLAYOFF_MULT).values
    xgb_f = fit_calibrated(make_xgb, X_all, y_all, w_base, DEPLOY_CV)
    lr_f = fit_calibrated(make_lr, X_all, y_all, w_base, DEPLOY_CV)
    ens = [xgb_f, lr_f]
    p_home = p_win(state_A, state_B, 1, ens)
    p_away = 1 - p_win(state_B, state_A, 1, ens)
    p_neutral = (p_home + p_away) / 2
    p_series_base = simulate_series(p_A_per_game(ens), N_SIMS)
    print("=" * 66)
    print("BASE CASE (pm=2.5, halflife=540d)")
    print(f"  P(SAS win | SAS home) = {p_home:.3f}   P(SAS win | NYK home) = {p_away:.3f}")
    print(f"  P(SAS win, neutral)   = {p_neutral:.3f}")
    print(f"  P(SAS SERIES win)     = {p_series_base:.3f}   "
          f"(home court: {HOME_TEAM}, worth ~{100*(p_home-p_neutral):.0f}pp/game)")

    # ---------------- McNemar: ensemble vs Elo+home on holdout ----------------
    train = games[games.SEASON.isin(["2023-24", "2024-25"])]
    test = games[games.SEASON == "2025-26"]
    Xtr, ytr = train[FEATURES].values, train["won"].values
    Xte, yte = test[FEATURES].values, test["won"].values
    wtr = weights_for(train, RECENCY_HALFLIFE_DAYS, PLAYOFF_MULT).values
    xc = fit_calibrated(make_xgb, Xtr, ytr, wtr, TimeSeriesSplit(4))
    lc = fit_calibrated(make_lr, Xtr, ytr, wtr, TimeSeriesSplit(4))
    p_ens = (xc.predict_proba(Xte)[:, 1] + lc.predict_proba(Xte)[:, 1]) / 2
    eidx, hidx = FEATURES.index("d_elo_pre"), FEATURES.index("is_home")
    p_elo = 1 / (1 + 10 ** (-(Xte[:, eidx] + ELO_HOME * Xte[:, hidx]) / 400))
    acc_ens = accuracy_score(yte, p_ens >= 0.5)
    acc_elo = accuracy_score(yte, p_elo >= 0.5)
    hm = test["is_home"].values == 1
    yg = yte[hm].astype(bool)
    eok = (p_ens[hm] >= 0.5) == yg
    bok = (p_elo[hm] >= 0.5) == yg
    b = int(np.sum(eok & ~bok)); c = int(np.sum(~eok & bok))
    pval = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0
    print("\nHOLDOUT (2025-26): ensemble vs Elo+home")
    print(f"  accuracy  ensemble={acc_ens:.3f}  elo+home={acc_elo:.3f}  "
          f"(delta {100*(acc_ens-acc_elo):+.1f}pp, logloss {log_loss(yte,p_ens):.3f})")
    print(f"  McNemar per-game (n={int(hm.sum())}): only-ens-right={b}, only-elo-right={c}, "
          f"p={pval:.3f} -> {'SIGNIFICANT' if pval<0.05 else 'NOT significant (within noise)'}")

    # ---------------- sensitivity grid ----------------
    print("\nSENSITIVITY: P(SAS series win) across weighting knobs")
    HL_GRID, PM_GRID = [365, 540, 730], [1.0, 1.5, 2.5]
    print("            " + "  ".join(f"pm={pm:<4}" for pm in PM_GRID))
    grid = []
    for hl in HL_GRID:
        row = []
        for pm in PM_GRID:
            wv = weights_for(games, hl, pm).values
            xf = fit_calibrated(make_xgb, X_all, y_all, wv, DEPLOY_CV)
            lf = fit_calibrated(make_lr, X_all, y_all, wv, DEPLOY_CV)
            ps = simulate_series(p_A_per_game([xf, lf]), 8000)
            row.append(ps); grid.append(ps)
        print(f"  hl={hl:4d}d:  " + "  ".join(f"{v:6.3f}" for v in row))
    stable = max(grid) < 0.55 and min(grid) > 0.45
    print(f"\n  Range across all {len(grid)} settings: [{min(grid):.3f}, {max(grid):.3f}]"
          f"  -> {'TOSS-UP STABLE to the knobs' if stable else 'verdict MOVES with the knobs'}")
    print("=" * 66)


if __name__ == "__main__":
    main()
