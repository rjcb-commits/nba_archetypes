"""Interactive viewer for the NBA player archetypes.

Lets a user pick a player, see which cluster they belong to, see similar
players inside that cluster, and view the PCA scatter of all players colored
by cluster.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "outputs" / "models"

DEFAULT_SEASON = 2025

st.set_page_config(page_title="NBA Player Archetypes", layout="wide")


@st.cache_data
def load_artifacts(season: int):
    assignments = pd.read_parquet(MODELS / f"{season}_assignments.parquet")
    centroids = pd.read_csv(MODELS / f"{season}_centroids.csv")
    with open(MODELS / f"{season}_kmeans.pkl", "rb") as f:
        bundle = pickle.load(f)
    return assignments, centroids, bundle


def main():
    st.title("NBA Player Archetypes")
    st.write(
        "K-means clustering on per-100-possession and advanced stats. "
        "Pick a player to see their archetype, similar players, and the "
        "shape of all archetypes in PCA space."
    )

    # Season selector (only one season for now, but easy to extend)
    season = st.sidebar.number_input(
        "Season", min_value=2014, max_value=2025, value=DEFAULT_SEASON, step=1,
    )

    try:
        assignments, centroids, bundle = load_artifacts(season)
    except FileNotFoundError:
        st.error(
            f"No model found for season {season}. "
            f"Run `python scripts/cluster.py --season {season}` first."
        )
        st.stop()

    k = bundle["k"]
    features = bundle["features"]

    # Player picker
    players = sorted(assignments["Player"].unique())
    selected = st.sidebar.selectbox("Player", players, index=0)

    selected_row = assignments[assignments["Player"] == selected].iloc[0]
    selected_cluster = int(selected_row["cluster"])

    st.subheader(f"{selected} — Cluster {selected_cluster}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("**Player stats (per 100 / advanced rates):**")
        player_stats = selected_row[features].to_frame(name="value").round(2)
        st.dataframe(player_stats, use_container_width=True)

    with col2:
        st.write("**Cluster centroid (the average player in this archetype):**")
        cluster_centroid = (
            centroids[centroids["cluster"] == selected_cluster]
            .iloc[0][features]
            .to_frame(name="centroid")
            .round(2)
        )
        st.dataframe(cluster_centroid, use_container_width=True)

    # Similar players in cluster (5 closest by feature distance)
    st.write(f"**Five closest players to {selected} within cluster {selected_cluster}:**")
    cluster_members = assignments[assignments["cluster"] == selected_cluster].copy()
    selected_features = selected_row[features].to_numpy()
    cluster_members["distance"] = np.linalg.norm(
        cluster_members[features].to_numpy() - selected_features, axis=1
    )
    closest = (
        cluster_members[cluster_members["Player"] != selected]
        .nsmallest(5, "distance")[["Player", "Pos", "distance"] + features]
        .round(2)
    )
    st.dataframe(closest, use_container_width=True)

    # PCA scatter of everyone colored by cluster, with selected player highlighted
    st.subheader("All players in PCA space, colored by cluster")
    fig = px.scatter(
        assignments,
        x="pc1",
        y="pc2",
        color=assignments["cluster"].astype(str),
        hover_data=["Player", "Pos"],
        labels={"color": "Cluster"},
    )
    selected_xy = assignments[assignments["Player"] == selected]
    fig.add_scatter(
        x=selected_xy["pc1"],
        y=selected_xy["pc2"],
        mode="markers",
        marker=dict(size=18, color="black", line=dict(color="white", width=2)),
        name=selected,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cluster size summary
    st.subheader("Cluster sizes")
    sizes = (
        assignments["cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="players")
    )
    st.dataframe(sizes, use_container_width=True)


if __name__ == "__main__":
    main()
