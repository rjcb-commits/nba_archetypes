"""Interactive viewer for the NBA player archetypes.

Pick a player, see which archetype they belong to, see similar players in the
same archetype, and view all players in PCA space colored by archetype.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "outputs" / "models"

DEFAULT_SEASON = 2025

# Map cluster id -> archetype name. Determined by reading the centroid profile
# for each cluster after running cluster.py with --pick 8 and seed=42.
# Add a new sub-dict per season once 2014 (or others) are clustered.
CLUSTER_NAMES = {
    2025: {
        0: "Secondary Creator",
        1: "Heavy-Usage PG",
        2: "Rim-Protecting Big",
        3: "3&D Wing",
        4: "Stretch Big",
        5: "Primary Star",
        6: "Low-Efficiency Wing",
        7: "Defensive Forward",
    },
}

# One-line archetype descriptions used in cluster cards and hover text.
CLUSTER_DESCRIPTIONS = {
    "Secondary Creator": "High 3PA with real playmaking, mid-tier usage",
    "Heavy-Usage PG": "High passing volume, low efficiency, high turnovers",
    "Rim-Protecting Big": "Almost zero 3PA, elite rim efficiency, top blocks",
    "3&D Wing": "High 3PA, low usage, low turnover, the modern role player",
    "Stretch Big": "Mix of perimeter and interior, hybrid 4/5",
    "Primary Star": "Highest usage and FT rate, big assist + turnover load",
    "Low-Efficiency Wing": "Shoots threes at low percentages",
    "Defensive Forward": "Highest steal rate, high blocks, modest offense",
}

# 8 distinct, semantically-coded colors for the archetypes.
ARCHETYPE_COLORS = {
    "Primary Star":         "#d62728",  # red, the featured / star color
    "Secondary Creator":    "#e89914",  # amber
    "Heavy-Usage PG":       "#ff7f0e",  # orange
    "3&D Wing":             "#0a2540",  # navy, matches portfolio
    "Low-Efficiency Wing":  "#7f7f7f",  # gray
    "Defensive Forward":    "#2ca02c",  # green
    "Stretch Big":          "#9467bd",  # purple
    "Rim-Protecting Big":   "#8c564b",  # brown
}

st.set_page_config(
    page_title="NBA Player Archetypes",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_artifacts(season: int):
    assignments = pd.read_parquet(MODELS / f"{season}_assignments.parquet")
    centroids = pd.read_csv(MODELS / f"{season}_centroids.csv")
    with open(MODELS / f"{season}_kmeans.pkl", "rb") as f:
        bundle = pickle.load(f)

    # Apply human-readable archetype names
    name_map = CLUSTER_NAMES.get(season, {})
    assignments["archetype"] = assignments["cluster"].map(name_map)
    centroids["archetype"] = centroids["cluster"].map(name_map)
    return assignments, centroids, bundle


def build_pca_figure(assignments: pd.DataFrame, selected_player: str | None) -> go.Figure:
    """Hero PCA scatter: every player as a dot colored by archetype, centroids
    marked with X and labeled, selected player highlighted in black."""
    fig = go.Figure()

    # Compute per-archetype centroid in PCA space for label placement
    pca_centroids = (
        assignments.groupby("archetype")[["pc1", "pc2"]]
        .mean()
        .reset_index()
    )

    # One trace per archetype so the legend has named entries
    for archetype in CLUSTER_NAMES[2025].values():
        sub = assignments[assignments["archetype"] == archetype]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["pc1"],
            y=sub["pc2"],
            mode="markers",
            name=archetype,
            marker=dict(
                size=10,
                color=ARCHETYPE_COLORS.get(archetype, "#777"),
                opacity=0.75,
                line=dict(color="white", width=0.5),
            ),
            customdata=np.stack([sub["Player"], sub["Team"], sub["archetype"]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]} · %{customdata[2]}<br>"
                "<extra></extra>"
            ),
        ))

    # Cluster centroid markers (X)
    fig.add_trace(go.Scatter(
        x=pca_centroids["pc1"],
        y=pca_centroids["pc2"],
        mode="markers+text",
        name="Centroids",
        showlegend=False,
        marker=dict(
            symbol="x",
            size=18,
            color="black",
            line=dict(color="white", width=2),
        ),
        text=pca_centroids["archetype"],
        textposition="top center",
        textfont=dict(size=12, color="#0a2540", family="Inter, system-ui, sans-serif"),
        hoverinfo="skip",
    ))

    # Selected player highlight
    if selected_player:
        sel = assignments[assignments["Player"] == selected_player]
        if not sel.empty:
            fig.add_trace(go.Scatter(
                x=sel["pc1"],
                y=sel["pc2"],
                mode="markers",
                name=selected_player,
                marker=dict(
                    symbol="circle",
                    size=22,
                    color="black",
                    line=dict(color="white", width=3),
                ),
                hovertemplate=f"<b>{selected_player}</b><extra></extra>",
            ))

    fig.update_layout(
        height=620,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="#fafaf7",
        paper_bgcolor="#fafaf7",
        font=dict(family="Inter, system-ui, sans-serif", color="#0a2540"),
        xaxis=dict(
            title="Principal component 1",
            zeroline=False,
            gridcolor="#e0dfd9",
            showline=True,
            linecolor="#0a2540",
        ),
        yaxis=dict(
            title="Principal component 2",
            zeroline=False,
            gridcolor="#e0dfd9",
            showline=True,
            linecolor="#0a2540",
        ),
        legend=dict(
            orientation="v",
            y=1,
            x=1.01,
            bgcolor="rgba(250,250,247,0.8)",
            bordercolor="#0a2540",
            borderwidth=1,
        ),
        hovermode="closest",
    )
    return fig


def render_cluster_cards(assignments: pd.DataFrame, centroids: pd.DataFrame):
    """Eight small cards across two rows showing each archetype: name, count, top reps."""
    archetypes = list(CLUSTER_NAMES[2025].values())
    cols = st.columns(4)
    for i, archetype in enumerate(archetypes):
        members = assignments[assignments["archetype"] == archetype]
        n = len(members)
        # Top 3 representative players: closest to cluster centroid in PCA space
        cluster_id = centroids[centroids["archetype"] == archetype]["cluster"].iloc[0]
        cluster_members = assignments[assignments["cluster"] == cluster_id].copy()
        cluster_pca_center = cluster_members[["pc1", "pc2"]].mean()
        cluster_members["d"] = np.linalg.norm(
            cluster_members[["pc1", "pc2"]].to_numpy(dtype=float)
            - cluster_pca_center.to_numpy(dtype=float),
            axis=1,
        )
        reps = cluster_members.nsmallest(3, "d")["Player"].tolist()

        with cols[i % 4]:
            color = ARCHETYPE_COLORS.get(archetype, "#777")
            st.markdown(
                f"""<div style="
                    border-left: 4px solid {color};
                    padding: 8px 12px;
                    margin-bottom: 12px;
                    background: #f0efe9;
                    border-radius: 4px;
                ">
                <div style="font-weight: 600; color: #0a2540; font-size: 0.95rem;">
                    {archetype}
                </div>
                <div style="color: #555; font-size: 0.8rem; margin: 4px 0;">
                    {CLUSTER_DESCRIPTIONS.get(archetype, '')}
                </div>
                <div style="color: #777; font-size: 0.75rem;">
                    n={n} · {', '.join(reps)}
                </div>
                </div>""",
                unsafe_allow_html=True,
            )


def main():
    st.title("NBA Player Archetypes")
    st.caption(
        "K-means clustering on per-100-possession and advanced stats finds eight "
        "player archetypes in the 2024-25 NBA season. None of them line up with "
        "the traditional five-position framework."
    )

    # Sidebar: season + player
    season = st.sidebar.selectbox(
        "Season",
        options=[2025],
        index=0,
        format_func=lambda y: f"{y - 1}-{str(y)[-2:]}",
        help="Only 2024-25 is available right now. Earlier seasons coming.",
    )

    try:
        assignments, centroids, bundle = load_artifacts(season)
    except FileNotFoundError:
        st.error(
            f"No model found for season {season}. "
            f"Run `python scripts/cluster.py --season {season}` first."
        )
        st.stop()

    features = bundle["features"]

    players = sorted(assignments["Player"].unique())
    default_index = players.index("Nikola Jokić") if "Nikola Jokić" in players else 0
    selected = st.sidebar.selectbox("Player", players, index=default_index)

    selected_row = assignments[assignments["Player"] == selected].iloc[0]
    selected_archetype = selected_row["archetype"]

    # Hero PCA scatter
    st.markdown("### The eight archetypes")
    st.plotly_chart(
        build_pca_figure(assignments, selected),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Archetype quick-reference cards
    st.markdown("### Archetype reference")
    render_cluster_cards(assignments, centroids)

    # Selected player section
    st.markdown("---")
    archetype_color = ARCHETYPE_COLORS.get(selected_archetype, "#0a2540")
    st.markdown(
        f"### {selected} <span style='color:{archetype_color}'>·</span> "
        f"<span style='color:{archetype_color}; font-weight:600'>{selected_archetype}</span>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Player stats**")
        st.dataframe(
            selected_row[features].to_frame(name="value").round(2),
            use_container_width=True,
        )
    with col2:
        st.markdown(f"**{selected_archetype} centroid**")
        cluster_id = int(selected_row["cluster"])
        st.dataframe(
            centroids[centroids["cluster"] == cluster_id]
                .iloc[0][features].to_frame(name="centroid").round(2),
            use_container_width=True,
        )

    st.markdown(f"**Five most similar players within {selected_archetype}**")
    cluster_members = assignments[assignments["cluster"] == int(selected_row["cluster"])].copy()
    selected_features = selected_row[features].to_numpy(dtype=float)
    member_features = cluster_members[features].to_numpy(dtype=float)
    cluster_members["distance"] = np.linalg.norm(
        member_features - selected_features, axis=1
    )
    closest = (
        cluster_members[cluster_members["Player"] != selected]
        .nsmallest(5, "distance")[["Player", "Team", "distance"] + features]
        .round(2)
    )
    st.dataframe(closest, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
