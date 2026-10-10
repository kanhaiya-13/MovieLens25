"""
visualize.py
============
Publication-quality plots for the Item-KNN project.

All plots are saved as PNGs to cfg.results/.
No plt.show() is called (headless / script mode).

Functions
---------
plot_rating_distribution(ratings_df, cfg)
plot_user_item_histograms(ratings_df, cfg)
plot_similarity_heatmap(model, movies_df, cfg, top_n=30)
plot_k_vs_metrics(grid_df, cfg)
plot_similarity_vs_metrics(grid_df, cfg)
plot_sim_value_histogram(model, cfg)
plot_all(model, ratings_df, movies_df, grid_df, cfg)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

from config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

sns.set_theme(style="darkgrid", palette="muted")
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path.name)


# ---------------------------------------------------------------------------
# 1. Rating distribution
# ---------------------------------------------------------------------------

def plot_rating_distribution(ratings_df: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> None:
    """Bar chart of rating frequencies."""
    counts = ratings_df["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index.astype(str), counts.values, color=COLORS[0], edgecolor="white", width=0.6)
    ax.set_xlabel("Rating", fontsize=12)
    ax.set_ylabel("Number of ratings", fontsize=12)
    ax.set_title("Rating Distribution -- MovieLens 25M", fontsize=14, fontweight="bold")
    for x, y in zip(range(len(counts)), counts.values):
        ax.text(x, y * 1.01, f"{y:,}", ha="center", va="bottom", fontsize=8)
    _savefig(fig, cfg.results / "rating_distribution.png")


# ---------------------------------------------------------------------------
# 2. Ratings-per-user / ratings-per-item histograms (log scale)
# ---------------------------------------------------------------------------

def plot_user_item_histograms(ratings_df: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> None:
    """Log-scale histograms of activity per user and per item."""
    user_counts = ratings_df.groupby("user_idx").size()
    item_counts = ratings_df.groupby("item_idx").size()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, counts, label, color in [
        (axes[0], user_counts, "Ratings per User",  COLORS[0]),
        (axes[1], item_counts, "Ratings per Item",  COLORS[1]),
    ]:
        ax.hist(counts, bins=60, color=color, edgecolor="white", log=True)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Count (log scale)", fontsize=11)
        ax.set_title(label + " Distribution", fontsize=13, fontweight="bold")
        ax.axvline(counts.median(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Median={counts.median():.0f}")
        ax.legend()

    fig.suptitle("Activity Distributions -- MovieLens 25M", fontsize=15, fontweight="bold")
    _savefig(fig, cfg.results / "activity_histograms.png")


# ---------------------------------------------------------------------------
# 3. Similarity heatmap (top-30 popular movies)
# ---------------------------------------------------------------------------

def plot_similarity_heatmap(
    model,
    movies_df: pd.DataFrame,
    cfg: Config = DEFAULT_CONFIG,
    top_n: int = 30,
) -> None:
    """Heatmap of item-item similarity for the N most popular items."""
    if not hasattr(model, "S") or model.S is None:
        logger.warning("Model not fitted; skipping similarity heatmap.")
        return
    if "item_idx" not in movies_df.columns:
        logger.warning("movies_df missing item_idx; skipping heatmap.")
        return

    # Most popular items (most ratings in train)
    item_pop = np.diff(model.user_item_matrix.tocsc().indptr)  # per-column nnz
    top_items = np.argsort(-item_pop)[:top_n]

    S_sub = model.S[np.ix_(top_items, top_items)].toarray()

    idx_to_title = movies_df.set_index("item_idx")["title"].to_dict()
    labels = [idx_to_title.get(int(i), str(i))[:25] for i in top_items]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        S_sub, ax=ax,
        xticklabels=labels, yticklabels=labels,
        cmap="viridis", vmin=0, vmax=1,
        linewidths=0.3, linecolor="gray",
        cbar_kws={"label": "Similarity"},
    )
    ax.set_title(f"Item-Item Similarity -- Top {top_n} Popular Movies", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    _savefig(fig, cfg.results / "similarity_heatmap.png", dpi=120)


# ---------------------------------------------------------------------------
# 4. K vs RMSE and K vs NDCG@10 per similarity type
# ---------------------------------------------------------------------------

def plot_k_vs_metrics(grid_df: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> None:
    """Line plots: K vs RMSE and K vs NDCG@10, one line per similarity type."""
    if grid_df.empty:
        return

    # Aggregate over shrinkage -- use the best shrinkage per K per similarity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric, ax, title, best_fn in [
        ("rmse",    axes[0], "K vs RMSE",    "min"),
        ("ndcg@10", axes[1], "K vs NDCG@10", "max"),
    ]:
        agg = (
            grid_df.groupby(["similarity", "k_neighbors"])[metric]
            .agg(best_fn)
            .reset_index()
        )
        for i, sim in enumerate(cfg.grid_similarities):
            sub = agg[agg["similarity"] == sim].sort_values("k_neighbors")
            if sub.empty:
                continue
            ax.plot(sub["k_neighbors"], sub[metric], marker="o", label=sim, color=COLORS[i], linewidth=2)

        ax.set_xlabel("K (neighbours)", fontsize=11)
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend()
        ax.xaxis.set_major_locator(plt.MultipleLocator(10))

    fig.suptitle("Hyperparameter Sensitivity -- Validation Set", fontsize=15, fontweight="bold")
    _savefig(fig, cfg.results / "k_vs_metrics.png")


# ---------------------------------------------------------------------------
# 5. Bar chart: similarity type vs key metrics
# ---------------------------------------------------------------------------

def plot_similarity_vs_metrics(grid_df: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> None:
    """Grouped bar chart of best-per-similarity metrics."""
    if grid_df.empty:
        return

    metrics = ["rmse", "ndcg@10", "recall@10", "map@10"]
    agg_rows = []
    for sim in cfg.grid_similarities:
        sub = grid_df[grid_df["similarity"] == sim]
        # For RMSE: best = lowest; for others: best = highest
        best_rmse = sub.loc[sub["rmse"].idxmin()] if not sub.empty else {}
        best_rank = sub.loc[sub["ndcg@10"].idxmax()] if not sub.empty else {}
        agg_rows.append({
            "similarity": sim,
            "rmse":      best_rmse.get("rmse",   0),
            "ndcg@10":   best_rank.get("ndcg@10", 0),
            "recall@10": best_rank.get("recall@10", 0),
            "map@10":    best_rank.get("map@10", 0),
        })
    agg_df = pd.DataFrame(agg_rows)

    x = np.arange(len(cfg.grid_similarities))
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5), sharey=False)
    for ax, metric in zip(axes, metrics):
        vals = agg_df[metric].values
        bars = ax.bar(x, vals, color=COLORS[:len(x)], edgecolor="white", width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(agg_df["similarity"], rotation=15, fontsize=9)
        ax.set_title(metric.upper(), fontsize=12, fontweight="bold")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Best Metrics per Similarity Type -- Validation Set", fontsize=14, fontweight="bold")
    _savefig(fig, cfg.results / "similarity_vs_metrics.png")


# ---------------------------------------------------------------------------
# 6. Histogram of nonzero similarity values
# ---------------------------------------------------------------------------

def plot_sim_value_histogram(model, cfg: Config = DEFAULT_CONFIG) -> None:
    """Distribution of nonzero similarity values in the top-K matrix."""
    if not hasattr(model, "S") or model.S is None:
        return
    vals = model.S.data
    if len(vals) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals, bins=80, color=COLORS[2], edgecolor="white", log=True)
    ax.axvline(float(np.median(vals)), color="red", linestyle="--",
               label=f"Median={np.median(vals):.4f}")
    ax.set_xlabel("Similarity Value", fontsize=11)
    ax.set_ylabel("Count (log scale)", fontsize=11)
    ax.set_title("Distribution of Nonzero Similarity Values", fontsize=13, fontweight="bold")
    ax.legend()
    _savefig(fig, cfg.results / "sim_value_histogram.png")


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def plot_all(
    model,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    grid_df: Optional[pd.DataFrame],
    cfg: Config = DEFAULT_CONFIG,
) -> None:
    """Generate all plots."""
    logger.info("Generating all plots ...")
    plot_rating_distribution(ratings_df, cfg)
    plot_user_item_histograms(ratings_df, cfg)
    plot_similarity_heatmap(model, movies_df, cfg, top_n=30)
    plot_sim_value_histogram(model, cfg)
    if grid_df is not None and not grid_df.empty:
        plot_k_vs_metrics(grid_df, cfg)
        plot_similarity_vs_metrics(grid_df, cfg)
    logger.info("All plots saved to %s", cfg.results)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_data, load_movies
    from split import split_data
    from item_knn import ItemKNN

    cfg = DEFAULT_CONFIG
    if "--sample" in sys.argv:
        cfg = cfg.copy_with(sample_n=1_000_000)

    ratings_df, user_map, item_map = load_data(cfg)
    movies_df = load_movies(cfg)
    train_df, _, _ = split_data(ratings_df, cfg)

    model = ItemKNN(cfg)
    model.fit(train_df, n_users=len(user_map), n_items=len(item_map))

    grid_csv = cfg.results / "grid_results.csv"
    grid_df = pd.read_csv(grid_csv) if grid_csv.exists() else None

    plot_all(model, ratings_df, movies_df, grid_df, cfg)
    print("Plots saved to", cfg.results)
