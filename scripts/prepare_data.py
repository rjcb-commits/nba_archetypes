"""Join per-100-possessions and advanced stats, filter to qualifying players, build the feature matrix.

Uses nba_api column naming convention (different from basketball-reference).

Usage:
    python scripts/prepare_data.py --season 2025
    python scripts/prepare_data.py --season 2025 --min-games 30 --min-mpg 15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
OUT = REPO / "data" / "prepared"

# Features used for clustering. Source columns (nba_api naming):
#   per100 table   -> already per-100-possessions when fetched with that per_mode
#   advanced table -> already in rate / pct form
FEATURES = [
    "3PA_per100",   # FG3A from per100
    "FTA_per100",   # FTA from per100
    "TS_PCT",       # advanced
    "EFG_PCT",      # advanced
    "AST_per100",   # AST from per100
    "TOV_per100",   # TOV from per100
    "OREB_PCT",     # advanced
    "DREB_PCT",     # advanced
    "STL_per100",   # STL from per100
    "BLK_per100",   # BLK from per100
    "USG_PCT",      # advanced
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--min-games", type=int, default=30,
                   help="Minimum games played to qualify (default 30)")
    p.add_argument("--min-mpg", type=float, default=15.0,
                   help="Minimum minutes per game to qualify (default 15)")
    return p.parse_args()


def main():
    args = parse_args()

    per100_path = RAW / f"{args.season}_per100.csv"
    adv_path = RAW / f"{args.season}_advanced.csv"

    if not per100_path.exists() or not adv_path.exists():
        raise FileNotFoundError(
            f"Run fetch_data.py --season {args.season} first."
        )

    per100 = pd.read_csv(per100_path)
    advanced = pd.read_csv(adv_path)
    print(f"Loaded per100 ({len(per100):,} rows) and advanced ({len(advanced):,} rows)")

    # Rename per100 counting stats so they don't collide with advanced
    per100_renamed = per100.rename(columns={
        "FG3A": "3PA_per100",
        "FTA": "FTA_per100",
        "AST": "AST_per100",
        "TOV": "TOV_per100",
        "STL": "STL_per100",
        "BLK": "BLK_per100",
    })

    # Pull GP and MIN from advanced (per-game minutes), join on PLAYER_ID
    merged = per100_renamed.merge(
        advanced[["PLAYER_ID", "TS_PCT", "EFG_PCT", "USG_PCT",
                  "OREB_PCT", "DREB_PCT", "AST_PCT"]],
        on="PLAYER_ID",
        how="inner",
    )
    print(f"Merged rows: {len(merged):,}")

    # Compute MPG from advanced's MIN (which is per-game in nba_api advanced output)
    # Note: per100's MIN is per-100-possessions so we don't use it for the filter.
    # Pull per-game MIN from a fresh advanced lookup since we already have GP from per100.
    merged = merged.merge(
        advanced[["PLAYER_ID", "MIN"]].rename(columns={"MIN": "MIN_advanced"}),
        on="PLAYER_ID",
        how="left",
    )
    merged["MPG"] = merged["MIN_advanced"]  # advanced "MIN" is per-game

    # Apply qualifying filter
    qualified = merged[
        (merged["GP"] >= args.min_games) & (merged["MPG"] >= args.min_mpg)
    ].copy()
    print(f"After filter (GP >= {args.min_games}, MPG >= {args.min_mpg}): "
          f"{len(qualified):,} qualifying players")

    keep = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE",
            "GP", "MPG"] + FEATURES
    missing = [c for c in keep if c not in qualified.columns]
    if missing:
        raise KeyError(
            f"Missing expected columns: {missing}\n"
            f"Available: {sorted(qualified.columns)}"
        )

    out_df = qualified[keep].dropna(subset=FEATURES).reset_index(drop=True)
    out_df = out_df.rename(columns={
        "PLAYER_NAME": "Player",
        "TEAM_ABBREVIATION": "Team",
    })
    print(f"After dropping rows with missing features: {len(out_df):,}")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{args.season}_features.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(f"Features: {FEATURES}")


if __name__ == "__main__":
    main()
