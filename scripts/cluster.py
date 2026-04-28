"""Cluster NBA players using K-means across a range of K, pick the best, save outputs.

For each K in the search range:
- Standardize features
- Fit K-means
- Compute silhouette score and inertia

Print the silhouette / elbow comparison so you can sanity-check the choice.
Save the fitted model, scaler, and cluster assignments for the Streamlit app.

Usage:
    python scripts/cluster.py --season 2025
    python scripts/cluster.py --season 2025 --k-range 4 12 --pick 7
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make Windows console handle non-ASCII player names (Jokic, Doncic, etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
PREPARED = REPO / "data" / "prepared"
MODELS = REPO / "outputs" / "models"
FIGURES = REPO / "outputs" / "figures"

FEATURES = [
    "3PA_per100", "FTA_per100", "TS_PCT", "EFG_PCT",
    "AST_per100", "TOV_per100", "OREB_PCT", "DREB_PCT",
    "STL_per100", "BLK_per100", "USG_PCT",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--k-range", type=int, nargs=2, default=[4, 12],
                   help="Search range for K, inclusive on both ends")
    p.add_argument("--pick", type=int, default=None,
                   help="If set, force this K instead of using best silhouette")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(PREPARED / f"{args.season}_features.parquet")
    print(f"Loaded {len(df):,} qualifying players for season {args.season}")

    X = df[FEATURES].to_numpy()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Search across K
    k_low, k_high = args.k_range
    results = []
    for k in range(k_low, k_high + 1):
        km = KMeans(n_clusters=k, random_state=args.seed, n_init=20)
        labels = km.fit_predict(X_std)
        sil = silhouette_score(X_std, labels)
        results.append({"k": k, "silhouette": sil, "inertia": km.inertia_})

    results_df = pd.DataFrame(results)
    print("\nK selection diagnostics:")
    print(results_df.to_string(index=False))

    chosen_k = args.pick if args.pick is not None else int(
        results_df.loc[results_df["silhouette"].idxmax(), "k"]
    )
    print(f"\nUsing K = {chosen_k}")

    # Refit at chosen K
    final_model = KMeans(n_clusters=chosen_k, random_state=args.seed, n_init=20)
    df["cluster"] = final_model.fit_predict(X_std)

    # PCA for 2D viz
    pca = PCA(n_components=2, random_state=args.seed)
    df[["pc1", "pc2"]] = pca.fit_transform(X_std)

    # Cluster centroid profiles in original feature space
    centroids = pd.DataFrame(
        scaler.inverse_transform(final_model.cluster_centers_),
        columns=FEATURES,
    )
    centroids["cluster"] = range(chosen_k)
    centroids["n_players"] = df["cluster"].value_counts().sort_index().to_numpy()

    print("\nCluster centroid profiles (original units):")
    print(centroids.round(2).to_string(index=False))

    # Show 5 representative players from each cluster
    print("\nRepresentative players (closest to centroid) per cluster:")
    for c in range(chosen_k):
        cluster_pts = X_std[df["cluster"] == c]
        center = final_model.cluster_centers_[c]
        dists = np.linalg.norm(cluster_pts - center, axis=1)
        names = df[df["cluster"] == c]["Player"].to_numpy()
        ordered = names[np.argsort(dists)][:5]
        print(f"  Cluster {c}: {', '.join(ordered)}")

    # Save artifacts
    MODELS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    with open(MODELS / f"{args.season}_kmeans.pkl", "wb") as f:
        pickle.dump({
            "model": final_model,
            "scaler": scaler,
            "pca": pca,
            "features": FEATURES,
            "k": chosen_k,
            "season": args.season,
        }, f)

    df.to_parquet(MODELS / f"{args.season}_assignments.parquet", index=False)
    centroids.to_csv(MODELS / f"{args.season}_centroids.csv", index=False)

    with open(MODELS / f"{args.season}_diagnostics.json", "w") as f:
        json.dump({
            "k_selection": results_df.to_dict(orient="records"),
            "chosen_k": chosen_k,
        }, f, indent=2)

    print(f"\nSaved model, assignments, centroids, and diagnostics to {MODELS}")


if __name__ == "__main__":
    main()
