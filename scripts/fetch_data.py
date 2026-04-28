"""Pull NBA per-100-possessions and advanced stats from stats.nba.com via nba_api.

Uses the official NBA Stats API (wrapped by the nba_api package) which is more
reliable than scraping basketball-reference (which blocks datacenter IPs).

Usage:
    python scripts/fetch_data.py --season 2025
    # 2025 means the 2024-25 season (season ending year)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import LeagueDashPlayerStats

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"

THROTTLE_SECONDS = 2  # nba_api is more permissive than bb-ref


def season_str(season_ending: int) -> str:
    """Convert season ending year (e.g., 2025) to NBA's format (e.g., '2024-25')."""
    return f"{season_ending - 1}-{str(season_ending)[-2:]}"


def fetch_measure(season: str, measure: str) -> pd.DataFrame:
    """Fetch a stats table for the season and measure type.

    measure: 'Advanced', 'Per100Possessions', 'Base', 'Misc', etc.
    """
    print(f"Fetching {measure} for {season}...")
    endpoint = LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense=measure,
        per_mode_detailed="Per100Possessions" if measure == "Base" else "Totals",
        timeout=30,
    )
    df = endpoint.get_data_frames()[0]
    print(f"  got {len(df):,} rows, {len(df.columns)} columns")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025,
                        help="Season ending year (e.g., 2025 for the 2024-25 season)")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    season = season_str(args.season)

    # Per-100 base stats (counting stats normalized per 100 possessions)
    per100 = fetch_measure(season, "Base")
    per100.to_csv(RAW / f"{args.season}_per100.csv", index=False)
    print(f"  wrote {RAW / f'{args.season}_per100.csv'}")
    time.sleep(THROTTLE_SECONDS)

    # Advanced stats (TS%, USG%, ratings, etc.)
    advanced = fetch_measure(season, "Advanced")
    advanced.to_csv(RAW / f"{args.season}_advanced.csv", index=False)
    print(f"  wrote {RAW / f'{args.season}_advanced.csv'}")

    print(f"\nDone. Raw stats for {args.season} ({season}) in {RAW}")


if __name__ == "__main__":
    main()
