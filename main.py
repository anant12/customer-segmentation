"""
Customer Segmentation using RFM features, K-Means, and DBSCAN.

Dataset expected: marketing_campaign.csv (tab-separated despite its .csv extension).
Run:
    python main.py --data marketing_campaign.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PURCHASE_COLUMNS = [
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
]


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the tab-separated marketing campaign dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Place marketing_campaign.csv beside main.py or pass --data PATH."
        )

    # The supplied dataset is tab-separated although its extension is .csv.
    raw = pd.read_csv(csv_path, sep="\\t")

    required = {"ID", "Recency", *PURCHASE_COLUMNS}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    monetary_columns = [column for column in raw.columns if column.startswith("Mnt")]
    if not monetary_columns:
        raise ValueError("No spending columns beginning with 'Mnt' were found.")

    return raw


def build_rfm(raw: pd.DataFrame) -> pd.DataFrame:
    """Create one Recency/Frequency/Monetary record per customer."""
    monetary_columns = [column for column in raw.columns if column.startswith("Mnt")]
    rfm = pd.DataFrame({
        "CustomerID": raw["ID"],
        "Recency": raw["Recency"],
        "Frequency": raw[PURCHASE_COLUMNS].sum(axis=1),
        "Monetary": raw[monetary_columns].sum(axis=1),
    }).dropna()

    if rfm.empty:
        raise ValueError("No customer records remain after removing missing RFM values.")
    return rfm


def prepare_features(rfm: pd.DataFrame):
    """Log-transform Frequency/Monetary and standardize the three RFM features."""
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    features[["Frequency", "Monetary"]] = np.log1p(
        features[["Frequency", "Monetary"]]
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return features, scaled


def cluster_customers(rfm: pd.DataFrame, scaled_features: np.ndarray):
    """Fit K-Means and DBSCAN and return the fitted labels and diagnostics."""
    if len(rfm) < 5:
        raise ValueError("At least 5 customer records are required for this DBSCAN setup.")

    rfm = rfm.copy()
    rfm["KMeans"] = KMeans(
        n_clusters=4,
        n_init=10,
        random_state=42,
    ).fit_predict(scaled_features)

    distances, _ = NearestNeighbors(n_neighbors=5).fit(
        scaled_features
    ).kneighbors(scaled_features)
    eps = float(np.quantile(distances[:, -1], 0.85))

    rfm["DBSCAN"] = DBSCAN(
        eps=eps,
        min_samples=5,
    ).fit_predict(scaled_features)

    labels = rfm["KMeans"]
    silhouette = (
        float(silhouette_score(scaled_features, labels))
        if len(np.unique(labels)) > 1 else float("nan")
    )
    return rfm, eps, silhouette


def create_personas(rfm: pd.DataFrame) -> pd.DataFrame:
    """Profile K-Means groups and apply the notebook's rule-based personas."""
    profiles = rfm.groupby("KMeans")[
        ["Recency", "Frequency", "Monetary"]
    ].median()
    overall = rfm[["Recency", "Frequency", "Monetary"]].median()

    profiles["Persona"] = np.select(
        [
            (profiles.Recency <= overall.Recency)
            & (profiles.Frequency >= overall.Frequency)
            & (profiles.Monetary >= overall.Monetary),

            (profiles.Recency > overall.Recency)
            & (profiles.Monetary >= overall.Monetary),

            (profiles.Recency <= overall.Recency)
            & (profiles.Frequency >= overall.Frequency),

            (profiles.Recency > overall.Recency)
            & (profiles.Frequency < overall.Frequency),
        ],
        [
            "Champions / VIP",
            "High-value at risk",
            "Frequent buyers",
            "Dormant customers",
        ],
        default="Occasional buyers",
    )

    profiles["Marketing Action"] = profiles["Persona"].map({
        "Champions / VIP": "Offer loyalty rewards or referral perks",
        "High-value at risk": "Send a personalized win-back offer",
        "Frequent buyers": "Offer bundles or replenishment reminders",
        "Dormant customers": "Try a reactivation message",
        "Occasional buyers": "Send product recommendations",
    })
    return profiles


def save_visualizations(rfm: pd.DataFrame, scaled_features: np.ndarray, output_dir: Path):
    """Save PCA and t-SNE cluster plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pca_points = PCA(n_components=2).fit_transform(scaled_features)
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_points[:, 0], pca_points[:, 1],
                c=rfm["KMeans"], cmap="tab10", s=18)
    plt.title("K-Means clusters shown with PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(output_dir / "kmeans_pca.png", dpi=150)
    plt.close()

    if len(rfm) > 30:
        perplexity = 30
    else:
        perplexity = max(2, min(5, len(rfm) - 1))

    tsne_points = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(scaled_features)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(tsne_points[:, 0], tsne_points[:, 1],
                    c=rfm["KMeans"], cmap="tab10", s=18)
    axes[0].set_title("t-SNE colored by K-Means")
    axes[1].scatter(tsne_points[:, 0], tsne_points[:, 1],
                    c=rfm["DBSCAN"], cmap="tab10", s=18)
    axes[1].set_title("t-SNE colored by DBSCAN")
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_kmeans_dbscan.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Segment customers with RFM features, K-Means, and DBSCAN."
    )
    parser.add_argument(
        "--data", type=Path, default=Path("marketing_campaign.csv"),
        help="Path to the tab-separated marketing_campaign.csv dataset."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs"),
        help="Directory where CSV results and plots will be saved."
    )
    args = parser.parse_args()

    raw = load_data(args.data)
    print(f"Dataset shape: {raw.shape}")

    rfm = build_rfm(raw)
    _, scaled_features = prepare_features(rfm)
    rfm, eps, silhouette = cluster_customers(rfm, scaled_features)
    profiles = create_personas(rfm)

    args.output.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(args.output / "customer_segments.csv", index=False)
    profiles.to_csv(args.output / "customer_personas.csv")

    print(f"DBSCAN eps: {eps:.3f}")
    print(f"K-Means silhouette score: {silhouette:.3f}")
    print("\nCluster label combinations:")
    print(rfm[["KMeans", "DBSCAN"]].value_counts())
    print("\nCustomer personas:")
    print(profiles)

    save_visualizations(rfm, scaled_features, args.output)
    print(f"\nResults and plots saved in: {args.output.resolve()}")


if __name__ == "__main__":
    main()
