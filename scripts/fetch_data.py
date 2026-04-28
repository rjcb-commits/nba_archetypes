"""Pull NBA per-100-possessions and advanced stats from basketball-reference for a season.

basketball-reference rate-limits to ~1 request per 6 seconds, so we sleep between calls.
Outputs raw CSVs to data/raw/ that the prepare step consumes.

Usage:
    python scripts/fetch_data.py --season 2025
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"

BASE = "https://www.basketball-reference.com/leagues/NBA_{season}_{stat}.html"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; nba-archetypes/0.1; portfolio project)",
}
THROTTLE_SECONDS = 6


def fetch_table(season: int, stat: str) -> pd.DataFrame:
    """Fetch a single basketball-reference league stats table."""
    url = BASE.format(season=season, stat=stat)
    print(f"GET {url}")
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    # basketball-reference wraps the table in HTML comments inside the page;
    # pandas.read_html still finds it via the table elements
    tables = pd.read_html(response.text)
    if not tables:
        raise RuntimeError(f"No tables found at {url}")

    df = tables[0]

    # The header row repeats every ~25 rows; drop those duplicate header rows
    df = df[df["Player"] != "Player"].reset_index(drop=True)

    # Cast numeric columns where possible
    for col in df.columns:
        if col in ("Player", "Pos", "Tm", "Team", "Awards"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025,
                        help="Season ending year (e.g., 2025 for the 2024-25 season)")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    for stat in ("per_poss", "advanced"):
        df = fetch_table(args.season, stat)
        out = RAW / f"{args.season}_{stat}.csv"
        df.to_csv(out, index=False)
        print(f"  wrote {out} ({len(df):,} rows)")
        time.sleep(THROTTLE_SECONDS)

    print(f"\nDone. Raw stats for {args.season} in {RAW}")


if __name__ == "__main__":
    main()
