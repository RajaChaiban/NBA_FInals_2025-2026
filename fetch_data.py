"""
Fetch and cache all NBA team-game box scores for the 3-season window.

Pulls Regular Season + Playoffs for 2023-24, 2024-25, 2025-26 from stats.nba.com
via nba_api, concatenates, and caches to data/raw_games.csv.

stats.nba.com is rate-limited and flaky, so we sleep between calls and retry.
Re-running loads from cache unless --refresh is passed.
"""
import os
import sys
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_PATH = os.path.join(DATA_DIR, "raw_games.csv")

SEASONS = ["2023-24", "2024-25", "2025-26"]
SEASON_TYPES = ["Regular Season", "Playoffs"]


def fetch_one(season, season_type, retries=3):
    """Fetch a single season+type, with retries on transient failures."""
    for attempt in range(1, retries + 1):
        try:
            df = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                timeout=60,
            ).get_data_frames()[0]
            df["SEASON"] = season
            df["SEASON_TYPE"] = season_type
            return df
        except Exception as e:  # noqa: BLE001 - network can throw many things
            print(f"  attempt {attempt} failed: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(3 * attempt)
    print(f"  GIVING UP on {season} {season_type}")
    return pd.DataFrame()


def fetch_all():
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = []
    for season in SEASONS:
        for stype in SEASON_TYPES:
            print(f"Fetching {season} {stype} ...")
            df = fetch_one(season, stype)
            print(f"  -> {len(df)} team-game rows")
            if len(df):
                frames.append(df)
            time.sleep(0.8)  # be polite to stats.nba.com
    if not frames:
        print("ERROR: no data fetched.")
        sys.exit(1)
    allg = pd.concat(frames, ignore_index=True)
    allg.to_csv(RAW_PATH, index=False)
    print(f"\nSaved {len(allg)} rows ({allg['GAME_ID'].nunique()} games) -> {RAW_PATH}")
    return allg


def load_or_fetch(refresh=False):
    if not refresh and os.path.exists(RAW_PATH):
        print(f"Loading cached {RAW_PATH}")
        return pd.read_csv(RAW_PATH, dtype={"GAME_ID": str})
    return fetch_all()


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    df = load_or_fetch(refresh=refresh)
    print("\nBy season/type:")
    print(df.groupby(["SEASON", "SEASON_TYPE"])["GAME_ID"].nunique())
