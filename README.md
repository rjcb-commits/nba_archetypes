# NBA Player Archetypes

Unsupervised clustering of NBA players using per-100-possession and advanced stats. Surfaces the player archetypes that actually exist in the modern game, which look different from the traditional five-position framework (PG / SG / SF / PF / C).

**Live demo:** _coming soon (Streamlit Community Cloud)_

## The question

If we cluster players purely on what they do on the floor (shot mix, scoring efficiency, playmaking, rebounding split, defense, usage), how many archetypes actually emerge? Do they line up with the position labels we use, or has the modern game outgrown that taxonomy?

## What's in here

- `scripts/fetch_data.py` pulls per-100-possession and advanced stats from basketball-reference for a given season.
- `scripts/prepare_data.py` filters to qualifying players, joins the two stat tables, builds the feature matrix.
- `scripts/cluster.py` runs K-means across a range of K, picks the best with silhouette + elbow, saves the model and cluster assignments.
- `app/streamlit_app.py` interactive viewer: pick a player, see their cluster, see similar players, view the PCA scatter.
- `notebooks/` exploratory notebooks (free to iterate before formalizing in scripts).

## Stack

- Python 3.11+
- pandas, numpy, scikit-learn for data and modeling
- matplotlib, seaborn, plotly for static and interactive viz
- Streamlit for the deployed app
- basketball-reference as the data source (HTML tables via `pandas.read_html`)

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
  raw/          basketball-reference HTML tables, parsed to CSV (gitignored)
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

- **Per-100-possessions, not per-game.** Per-game stats are biased by minutes played. A bench scorer who comes in for 12 minutes and chucks 3s looks more efficient on per-game metrics than a starter doing the same volume of work.
- **Qualifying filter.** Default minimum: 30 games played and 15 minutes per game. Removes random call-ups and rookies who barely played, which would otherwise pollute clusters with noise.
- **Standardization.** All features standardized before clustering. K-means is distance-based and unscaled features (e.g., usage rate at 30% vs blocks per 100 at 0.5) would dominate by accident.
- **K selection.** Silhouette score across K = 4 to 12, sanity-checked against the elbow on inertia. Final K is whichever balances statistical fit with interpretability (you want clusters you can name).
- **Cluster validation.** A high silhouette score on uninterpretable clusters is worse than a moderate score on archetypes you can describe in two sentences. The script prints the centroid profile for each cluster so you can read the result, not just measure it.

## Source data

Per-100-possessions stats: `https://www.basketball-reference.com/leagues/NBA_{season}_per_poss.html`
Advanced stats: `https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html`

basketball-reference asks for a 6-second delay between requests. The fetch script respects this.

## License

MIT for code. Stats are public from basketball-reference.
