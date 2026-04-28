"""Interactive viewer for the NBA player archetypes.

Pick a player, see which archetype they belong to (radar chart vs the
archetype centroid), see their headshot and 5 most similar teammates in
the cluster, and view all 322 players in PCA space colored by archetype.

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

# Cluster id -> archetype name. Determined from cluster.py centroid printout
# at K=8 with seed=42. Add a new sub-dict per season once others are clustered.
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

CLUSTER_DESCRIPTIONS = {
    "Secondary Creator": "High 3PA with real playmaking, mid-tier usage",
    "Heavy-Usage PG": "High passing volume, low efficiency, high turnovers",
    "Rim-Protecting Big": "Almost zero 3PA, elite rim efficiency, top blocks",
    "3&D Wing": "High 3PA, low usage, low turnover, modern role player",
    "Stretch Big": "Mix of perimeter and interior, hybrid 4/5",
    "Primary Star": "Highest usage and FT rate, big assist + turnover load",
    "Low-Efficiency Wing": "Shoots threes at low percentages",
    "Defensive Forward": "Highest steal rate, high blocks, modest offense",
}

ARCHETYPE_COLORS = {
    "Primary Star":         "#d62728",
    "Secondary Creator":    "#e89914",
    "Heavy-Usage PG":       "#ff7f0e",
    "3&D Wing":             "#0a2540",
    "Low-Efficiency Wing":  "#7f7f7f",
    "Defensive Forward":    "#2ca02c",
    "Stretch Big":          "#9467bd",
    "Rim-Protecting Big":   "#8c564b",
}

# Friendly labels for the radar chart axes
FEATURE_LABELS = {
    "3PA_per100":  "3PA",
    "FTA_per100":  "FTA",
    "TS_PCT":      "TS%",
    "EFG_PCT":     "eFG%",
    "AST_per100":  "AST",
    "TOV_per100":  "TOV",
    "OREB_PCT":    "OREB%",
    "DREB_PCT":    "DREB%",
    "STL_per100":  "STL",
    "BLK_per100":  "BLK",
    "USG_PCT":     "USG%",
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

    name_map = CLUSTER_NAMES.get(season, {})
    assignments["archetype"] = assignments["cluster"].map(name_map)
    centroids["archetype"] = centroids["cluster"].map(name_map)
    return assignments, centroids, bundle


def player_photo_url(player_id: int) -> str:
    """NBA.com hosts headshots at this predictable pattern."""
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(player_id)}.png"


def normalize(values: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """Min-max scale to 0..1, with safety against zero ranges."""
    rng = maxs - mins
    rng = np.where(rng == 0, 1, rng)
    return (values - mins) / rng


def build_radar_figure(
    player_row: pd.Series,
    centroid_row: pd.Series,
    features: list[str],
    mins: np.ndarray,
    maxs: np.ndarray,
    player_name: str,
    archetype_name: str,
    archetype_color: str,
) -> go.Figure:
    """Player profile vs archetype centroid as overlaid spider charts."""
    feature_labels = [FEATURE_LABELS.get(f, f) for f in features]

    player_vals = player_row[features].to_numpy(dtype=float)
    centroid_vals = centroid_row[features].to_numpy(dtype=float)

    player_norm = normalize(player_vals, mins, maxs)
    centroid_norm = normalize(centroid_vals, mins, maxs)

    # Close the polygon by repeating the first value at the end
    def close(arr):
        return np.concatenate([arr, arr[:1]])

    theta = feature_labels + [feature_labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=close(centroid_norm),
        theta=theta,
        fill="toself",
        name=f"{archetype_name} (avg)",
        line=dict(color=archetype_color, width=2),
        fillcolor=f"rgba{(*tuple(int(archetype_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)), 0.20)}",
    ))
    fig.add_trace(go.Scatterpolar(
        r=close(player_norm),
        theta=theta,
        fill="toself",
        name=player_name,
        line=dict(color="#0a2540", width=3),
        fillcolor="rgba(10, 37, 64, 0.15)",
    ))

    fig.update_layout(
        height=480,
        margin=dict(l=40, r=40, t=40, b=40),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="#d0cfc8"),
            angularaxis=dict(gridcolor="#d0cfc8", linecolor="#0a2540"),
            bgcolor="#fafaf7",
        ),
        paper_bgcolor="#fafaf7",
        plot_bgcolor="#fafaf7",
        font=dict(family="Inter, system-ui, sans-serif", color="#0a2540", size=12),
        showlegend=True,
        legend=dict(
            orientation="h",
            y=-0.05,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def build_cluster_size_figure(assignments: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of player count per archetype, sorted descending."""
    counts = (
        assignments.groupby("archetype")
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=True)
    )
    colors = [ARCHETYPE_COLORS.get(a, "#777") for a in counts["archetype"]]

    fig = go.Figure(go.Bar(
        x=counts["n"],
        y=counts["archetype"],
        orientation="h",
        marker=dict(color=colors),
        text=counts["n"],
        textposition="outside",
        textfont=dict(color="#0a2540", size=12),
        hovertemplate="<b>%{y}</b><br>%{x} players<extra></extra>",
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=40, t=20, b=20),
        plot_bgcolor="#fafaf7",
        paper_bgcolor="#fafaf7",
        font=dict(family="Inter, system-ui, sans-serif", color="#0a2540"),
        xaxis=dict(showgrid=True, gridcolor="#e0dfd9", zeroline=False),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    return fig


def build_pca_figure(
    assignments: pd.DataFrame,
    selected_player: str | None,
    filter_archetype: str | None = None,
) -> go.Figure:
    """Hero PCA scatter. If filter_archetype is set, dim everything else."""
    fig = go.Figure()

    pca_centroids = (
        assignments.groupby("archetype")[["pc1", "pc2"]]
        .mean()
        .reset_index()
    )

    for archetype in CLUSTER_NAMES[2025].values():
        sub = assignments[assignments["archetype"] == archetype]
        if sub.empty:
            continue
        is_focus = filter_archetype is None or filter_archetype == archetype
        opacity = 0.85 if is_focus else 0.10
        fig.add_trace(go.Scatter(
            x=sub["pc1"],
            y=sub["pc2"],
            mode="markers",
            name=archetype,
            marker=dict(
                size=10,
                color=ARCHETYPE_COLORS.get(archetype, "#777"),
                opacity=opacity,
                line=dict(color="white", width=0.5),
            ),
            customdata=np.stack([sub["Player"], sub["Team"], sub["archetype"]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]} · %{customdata[2]}<br>"
                "<extra></extra>"
            ),
        ))

    # Floating cluster name labels (no marker, just styled text annotations)
    for _, row in pca_centroids.iterrows():
        archetype = row["archetype"]
        if filter_archetype is not None and filter_archetype != archetype:
            continue
        color = ARCHETYPE_COLORS.get(archetype, "#777")
        fig.add_annotation(
            x=row["pc1"],
            y=row["pc2"],
            text=f"<b>{archetype}</b>",
            showarrow=False,
            font=dict(size=12, color=color, family="Inter, system-ui, sans-serif"),
            bgcolor="rgba(250,250,247,0.92)",
            bordercolor=color,
            borderwidth=1.5,
            borderpad=4,
            xanchor="center",
            yanchor="middle",
        )

    if selected_player:
        sel = assignments[assignments["Player"] == selected_player]
        if not sel.empty:
            fig.add_trace(go.Scatter(
                x=sel["pc1"],
                y=sel["pc2"],
                mode="markers",
                name=selected_player,
                marker=dict(symbol="circle", size=22, color="black",
                            line=dict(color="white", width=3)),
                hovertemplate=f"<b>{selected_player}</b><extra></extra>",
            ))

    fig.update_layout(
        height=620,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="#fafaf7",
        paper_bgcolor="#fafaf7",
        font=dict(family="Inter, system-ui, sans-serif", color="#0a2540"),
        xaxis=dict(title="Principal component 1", zeroline=False,
                   gridcolor="#e0dfd9", showline=True, linecolor="#0a2540"),
        yaxis=dict(title="Principal component 2", zeroline=False,
                   gridcolor="#e0dfd9", showline=True, linecolor="#0a2540"),
        legend=dict(orientation="v", y=1, x=1.01,
                    bgcolor="rgba(250,250,247,0.8)",
                    bordercolor="#0a2540", borderwidth=1),
        hovermode="closest",
    )
    return fig


def render_cluster_cards(assignments: pd.DataFrame, centroids: pd.DataFrame):
    """Eight clickable cards. Clicking sets the scatter filter via session state."""
    archetypes = list(CLUSTER_NAMES[2025].values())
    cols = st.columns(4)
    current_filter = st.session_state.get("filter_archetype")

    for i, archetype in enumerate(archetypes):
        members = assignments[assignments["archetype"] == archetype]
        n = len(members)
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
            is_active = current_filter == archetype
            border_style = "3px solid" if is_active else "4px solid"
            bg_color = "#e8e6dd" if is_active else "#f0efe9"
            st.markdown(
                f"""<div style="
                    border-left: {border_style} {color};
                    padding: 8px 12px;
                    margin-bottom: 6px;
                    background: {bg_color};
                    border-radius: 4px;
                ">
                <div style="font-weight: 600; color: #0a2540; font-size: 0.95rem;">
                    {archetype}
                </div>
                <div style="color: #555; font-size: 0.78rem; margin: 4px 0;">
                    {CLUSTER_DESCRIPTIONS.get(archetype, '')}
                </div>
                <div style="color: #777; font-size: 0.72rem;">
                    n={n} · {', '.join(reps)}
                </div>
                </div>""",
                unsafe_allow_html=True,
            )
            label = "Clear filter" if is_active else "Show on scatter"
            if st.button(label, key=f"btn_{archetype}", use_container_width=True):
                st.session_state["filter_archetype"] = None if is_active else archetype
                st.rerun()


def render_player_section(
    selected_row: pd.Series,
    assignments: pd.DataFrame,
    centroids: pd.DataFrame,
    features: list[str],
):
    """Selected player: photo, header, radar vs centroid, similar players with photos."""
    selected = selected_row["Player"]
    selected_archetype = selected_row["archetype"]
    archetype_color = ARCHETYPE_COLORS.get(selected_archetype, "#0a2540")

    # Pre-compute min/max per feature for radar normalization
    feature_matrix = assignments[features].to_numpy(dtype=float)
    mins = feature_matrix.min(axis=0)
    maxs = feature_matrix.max(axis=0)

    # Header row: photo on left, name + archetype on right
    head_cols = st.columns([1, 5])
    with head_cols[0]:
        st.image(player_photo_url(selected_row["PLAYER_ID"]), width=140)
    with head_cols[1]:
        st.markdown(
            f"### {selected}<br>"
            f"<span style='color:{archetype_color}; font-size: 1.1rem; font-weight:600'>"
            f"{selected_archetype}</span> "
            f"<span style='color:#777; font-size:0.9rem'>· "
            f"{selected_row['Team']} · age {int(selected_row['AGE'])}</span>",
            unsafe_allow_html=True,
        )
        st.caption(CLUSTER_DESCRIPTIONS.get(selected_archetype, ""))

    # Radar
    cluster_id = int(selected_row["cluster"])
    centroid_row = centroids[centroids["cluster"] == cluster_id].iloc[0]
    fig = build_radar_figure(
        selected_row, centroid_row, features, mins, maxs,
        selected, selected_archetype, archetype_color,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Similar players with photos
    st.markdown(f"**Five most similar players within {selected_archetype}**")
    cluster_members = assignments[assignments["cluster"] == cluster_id].copy()
    selected_features = selected_row[features].to_numpy(dtype=float)
    member_features = cluster_members[features].to_numpy(dtype=float)
    cluster_members["distance"] = np.linalg.norm(
        member_features - selected_features, axis=1
    )
    closest = cluster_members[cluster_members["Player"] != selected].nsmallest(5, "distance")

    photo_cols = st.columns(5)
    for col, (_, p) in zip(photo_cols, closest.iterrows()):
        with col:
            st.image(player_photo_url(p["PLAYER_ID"]), use_container_width=True)
            st.markdown(
                f"<div style='text-align:center; font-weight:600; color:#0a2540; font-size:0.9rem'>"
                f"{p['Player']}</div>"
                f"<div style='text-align:center; color:#777; font-size:0.75rem'>"
                f"{p['Team']}</div>",
                unsafe_allow_html=True,
            )


def main():
    st.title("NBA Player Archetypes")
    st.caption(
        "K-means clustering on per-100-possession and advanced stats finds eight "
        "player archetypes in the 2024-25 NBA season. None of them line up with "
        "the traditional five-position framework."
    )

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

    # Hero PCA scatter (filterable via cluster cards)
    st.markdown("### The eight archetypes in PCA space")
    filter_archetype = st.session_state.get("filter_archetype")
    if filter_archetype:
        st.caption(f"Filtered to **{filter_archetype}**. Click the card again to clear.")
    st.plotly_chart(
        build_pca_figure(assignments, selected, filter_archetype),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Cluster cards (clickable)
    st.markdown("### Archetype reference")
    render_cluster_cards(assignments, centroids)

    # Cluster sizes
    st.markdown("### Players per archetype")
    st.caption(
        "3&D Wings dominate the modern game. Rim-Protecting Bigs are the smallest "
        "group, evidence of how the center position has thinned out."
    )
    st.plotly_chart(
        build_cluster_size_figure(assignments),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Selected player section
    st.markdown("---")
    render_player_section(selected_row, assignments, centroids, features)


if __name__ == "__main__":
    main()
