# NBA Player Archetypes

Unsupervised clustering of NBA players using per-100-possession and advanced stats. Groups players into statistical profiles that read differently from the traditional five-position framework (PG / SG / SF / PF / C). The groupings depend on the season, features, filters, scaling, and choice of K — they are one representation, not a fixed truth about the game.

**Live demo:** [nbaarchetypes.streamlit.app](https://nbaarchetypes.streamlit.app/)

## The question

If we cluster players on what they do on the floor (shot mix, scoring efficiency, playmaking, rebounding split, defense, usage), how many groups emerge, and how well do they line up with the position labels?

## What's in here

- `scripts/fetch_data.py` pulls per-100-possession and advanced stats from stats.nba.com via the `nba_api` package for a given season.
- `scripts/prepare_data.py` filters to qualifying players, joins the two stat tables, builds the feature matrix.
- `scripts/cluster.py` runs K-means across a range of K, picks the best with silhouette + elbow, saves the model and cluster assignments.
- `app/streamlit_app.py` interactive viewer: pick a player, see their cluster, see similar players, view the PCA scatter.
- `notebooks/` exploratory notebooks (free to iterate before formalizing in scripts).

## Stack

- Python 3.11+
- pandas, numpy, scikit-learn for data and modeling
- matplotlib, seaborn, plotly for static and interactive viz
- Streamlit for the deployed app
- stats.nba.com as the data source (via the `nba_api` package)

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate    # macOS / Linux
pip install -r requirements.txt
```

## Usage

```bash
# 1. Pull the raw stats for a season (default: most recent)
python scripts/fetch_data.py --season 2025

# 2. Prepare the feature matrix (filters to qualifying players)
python scripts/prepare_data.py --season 2025

# 3. Cluster and pick K
python scripts/cluster.py --season 2025 --k-range 4 12

# 4. Run the Streamlit app
streamlit run app/streamlit_app.py
```

## Project layout

```
data/
  raw/          NBA Stats API CSVs (gitignored)
  prepared/     joined and feature-engineered parquet (gitignored)
scripts/
  fetch_data.py
  prepare_data.py
  cluster.py
app/
  streamlit_app.py
outputs/
  figures/      PCA scatter, cluster radar charts (gitignored)
  models/       fitted KMeans + scaler (gitignored)
notebooks/
  01_explore.ipynb
```

## Method notes

- **Per-100-possessions, not per-game.** Per-100-possession statistics make player production more comparable across playing time and pace. They measure production rates, not efficiency by themselves.
- **Qualifying filter.** Default minimum: 30 games played and 15 minutes per game. Removes random call-ups and rookies who barely played, which would otherwise pollute clusters with noise.
- **Standardization.** All features standardized before clustering. K-means is distance-based and unscaled features (e.g., usage rate at 30% vs blocks per 100 at 0.5) would dominate by accident.
- **K selection.** Silhouette score across K = 4 to 12, sanity-checked against the elbow on inertia. Final K is whichever balances statistical fit with interpretability (you want clusters you can name).
- **Cluster validation.** A high silhouette score on uninterpretable clusters is worse than a moderate score on archetypes you can describe in two sentences. The script prints the centroid profile for each cluster so you can read the result, not just measure it.

## Source data

Player statistics are retrieved from stats.nba.com using the `nba_api` Python package and its `LeagueDashPlayerStats` endpoint. Per-100-possession and advanced measures are pulled per season and saved to `data/raw/`.

## License

MIT for the code. Underlying player statistics are the property of the NBA and provided via stats.nba.com; consult the NBA's terms of use for any redistribution.
