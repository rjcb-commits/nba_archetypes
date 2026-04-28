"""Join per-100-possessions and advanced stats, filter to qualifying players, build the feature matrix.

Outputs a parquet file at data/prepared/{season}_features.parquet that the
clustering step consumes.

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

# Features used for clustering. All are per-100-possessions or advanced rates,
# i.e., volume-normalized. Raw counting stats would bias toward starters with
# heavy minutes regardless of their playing style.
FEATURES = [
    # Shot mix (what kind of shots they take)
    "3PA_per100",
    "FTA_per100",

    # Scoring efficiency
    "TS%",
    "eFG%",

    # Playmaking
    "AST_per100",
    "TOV_per100",

    # Rebounding split (offensive vs defensive tells you a lot about role)
    "ORB%",
    "DRB%",

    # Defense
    "STL_per100",
    "BLK_per100",

    # Usage and load
    "USG%",
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

    per_poss_path = RAW / f"{args.season}_per_poss.csv"
    advanced_path = RAW / f"{args.season}_advanced.csv"

    if not per_poss_path.exists() or not advanced_path.exists():
        raise FileNotFoundError(
            f"Run fetch_data.py --season {args.season} first."
        )

    per_poss = pd.read_csv(per_poss_path)
    advanced = pd.read_csv(advanced_path)

    # basketball-reference uses team abbrev "TOT" for traded players (combined row).
    # Keep only the TOT row when a player was traded mid-season; otherwise their
    # single-team row.
    per_poss = _dedupe_traded(per_poss)
    advanced = _dedupe_traded(advanced)

    # Rename per-100-possessions counting stats so they don't collide with advanced
    rename_per100 = {c: f"{c}_per100" for c in
                     ["FG", "FGA", "3P", "3PA", "FT", "FTA",
                      "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PTS"]}
    per_poss = per_poss.rename(columns=rename_per100)

    # Merge on player + position + age (a player has one row in each table)
    merged = per_poss.merge(
        advanced[["Player", "Pos", "Age", "G", "MP", "TS%", "eFG%",
                  "ORB%", "DRB%", "USG%"]],
        on=["Player", "Pos", "Age"],
        suffixes=("_pp", ""),
        how="inner",
    )
    print(f"Merged rows: {len(merged):,}")

    # Apply qualifying filter
    mpg = merged["MP"] / merged["G"]
    qualified = merged[(merged["G"] >= args.min_games) & (mpg >= args.min_mpg)].copy()
    print(f"After filter (games >= {args.min_games}, MPG >= {args.min_mpg}): "
          f"{len(qualified):,} qualifying players")

    # Keep only the columns we need for clustering + identifying
    keep = ["Player", "Pos", "Age", "G", "MP"] + FEATURES
    missing = [c for c in keep if c not in qualified.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}\n"
                       f"Available: {sorted(qualified.columns)}")

    out_df = qualified[keep].dropna(subset=FEATURES).reset_index(drop=True)
    print(f"After dropping rows with missing features: {len(out_df):,}")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{args.season}_features.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(f"Features: {FEATURES}")


def _dedupe_traded(df: pd.DataFrame) -> pd.DataFrame:
    """For traded players, basketball-reference creates one TOT row plus rows per team.
    Keep only TOT when present; otherwise the single team row.
    """
    counts = df.groupby("Player").size()
    multi = counts[counts > 1].index
    single = df[~df["Player"].isin(multi)]
    tot_rows = df[df["Player"].isin(multi) & (df["Tm"] == "TOT")]
    return pd.concat([single, tot_rows], ignore_index=True)


if __name__ == "__main__":
    main()
