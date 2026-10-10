"""
data_loader.py
==============
Loads the MovieLens 25M dataset, applies configurable quality filters,
remaps userId / movieId to contiguous 0-based int32 indices, and caches
everything as Parquet for fast subsequent loads.

Public API
----------
load_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    Returns (ratings_df, user_map_df, item_map_df).
    All DataFrames use int32 / float32 dtypes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _try_kaggle_download(raw_dir: Path) -> bool:
    """Attempt to download the dataset via kagglehub. Returns True on success."""
    try:
        import kagglehub  # type: ignore
        logger.info("Attempting kagglehub download ...")
        path = kagglehub.dataset_download("garymk/movielens-25m-dataset")
        src = Path(path)
        # kagglehub returns the local cache dir; link or copy
        if not (raw_dir / "ratings.csv").exists():
            import shutil
            shutil.copytree(str(src), str(raw_dir), dirs_exist_ok=True)
        logger.info("kagglehub download complete -> %s", raw_dir)
        return True
    except Exception as exc:
        logger.warning("kagglehub download failed: %s", exc)
        return False


def _print_manual_instructions(raw_dir: Path) -> None:
    msg = (
        "\n" + "=" * 70 + "\n"
        "  Could not auto-download MovieLens 25M.\n"
        "  Please download it manually:\n"
        "  1. Go to https://www.kaggle.com/datasets/garymk/movielens-25m-dataset\n"
        "  2. Click 'Download' (you need a Kaggle account).\n"
        "  3. Extract the ZIP and place its contents in:\n"
        f"     {raw_dir}\n"
        "  4. Ensure these files exist:\n"
        f"     {raw_dir / 'ratings.csv'}\n"
        f"     {raw_dir / 'movies.csv'}\n"
        "=" * 70 + "\n"
    )
    print(msg)


def _ensure_data(raw_dir: Path) -> None:
    """Make sure ratings.csv and movies.csv exist, or exit with instructions."""
    if (raw_dir / "ratings.csv").exists() and (raw_dir / "movies.csv").exists():
        return
    logger.info("Data not found at %s, attempting download ...", raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if _try_kaggle_download(raw_dir):
        return
    _print_manual_instructions(raw_dir)
    raise FileNotFoundError(
        f"ratings.csv / movies.csv not found in {raw_dir}. "
        "See instructions above."
    )


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------

RATINGS_DTYPES = {
    "userId":    "int32",
    "movieId":   "int32",
    "rating":    "float32",
    "timestamp": "int32",
}

MOVIES_DTYPES = {
    "movieId": "int32",
    "title":   "str",
    "genres":  "str",
}


def _read_ratings(raw_dir: Path, sample_n: int = 0) -> pd.DataFrame:
    """Read ratings.csv with pyarrow engine and explicit dtypes."""
    csv_path = raw_dir / "ratings.csv"
    logger.info("Reading ratings from %s ...", csv_path)
    t0 = time.perf_counter()

    # Read with int64 first (CSV has int64), then downcast
    df = pd.read_csv(
        csv_path,
        engine="pyarrow",
        dtype={"userId": "int64", "movieId": "int64", "rating": "float32", "timestamp": "int64"},
    )
    if sample_n > 0:
        logger.info("Sampling first %d ratings (--sample mode)", sample_n)
        df = df.iloc[:sample_n].copy()

    # Downcast to int32 / float32 to halve memory footprint
    df["userId"]    = df["userId"].astype("int32")
    df["movieId"]   = df["movieId"].astype("int32")
    df["rating"]    = df["rating"].astype("float32")
    df["timestamp"] = df["timestamp"].astype("int32")

    elapsed = time.perf_counter() - t0
    logger.info(
        "Loaded %s ratings in %.1f s  (%.1f MB RAM)",
        f"{len(df):,}", elapsed, df.memory_usage(deep=True).sum() / 1e6,
    )
    return df


def _read_movies(raw_dir: Path) -> pd.DataFrame:
    """Read movies.csv."""
    csv_path = raw_dir / "movies.csv"
    logger.info("Reading movies from %s ...", csv_path)
    df = pd.read_csv(csv_path, engine="pyarrow", dtype={"movieId": "int64", "title": "str", "genres": "str"})
    df["movieId"] = df["movieId"].astype("int32")
    return df


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _print_stats(df: pd.DataFrame, label: str) -> None:
    n_users  = df["userId"].nunique()
    n_items  = df["movieId"].nunique()
    n_ratings = len(df)
    sparsity = 1.0 - n_ratings / (n_users * n_items)
    logger.info(
        "[%s] users=%s  items=%s  ratings=%s  sparsity=%.4f%%",
        label, f"{n_users:,}", f"{n_items:,}", f"{n_ratings:,}", sparsity * 100,
    )


def _filter_ratings(
    df: pd.DataFrame,
    min_per_user: int,
    min_per_item: int,
) -> pd.DataFrame:
    """
    Iteratively remove users with < min_per_user ratings and items with
    < min_per_item ratings until the dataset is stable.
    """
    _print_stats(df, "BEFORE filtering")
    iteration = 0
    while True:
        iteration += 1
        n_before = len(df)

        # Filter items
        item_counts = df["movieId"].value_counts()
        valid_items = item_counts[item_counts >= min_per_item].index
        df = df[df["movieId"].isin(valid_items)]

        # Filter users
        user_counts = df["userId"].value_counts()
        valid_users = user_counts[user_counts >= min_per_user].index
        df = df[df["userId"].isin(valid_users)]

        n_after = len(df)
        logger.debug("  Filter iter %d: %s -> %s ratings", iteration, f"{n_before:,}", f"{n_after:,}")
        if n_after == n_before:
            break   # convergence

    _print_stats(df, "AFTER filtering")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# ID remapping
# ---------------------------------------------------------------------------

def _remap_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Remap original userId / movieId to contiguous 0-based int32 indices.

    Returns
    -------
    ratings_df : DataFrame with columns [user_idx, item_idx, rating, timestamp]
    user_map   : DataFrame [userId (original), user_idx (0-based)]
    item_map   : DataFrame [movieId (original), item_idx (0-based)]
    """
    # Sort for deterministic mapping
    unique_users = np.sort(df["userId"].unique())
    unique_items = np.sort(df["movieId"].unique())

    user_map = pd.DataFrame({
        "userId":   unique_users.astype("int32"),
        "user_idx": np.arange(len(unique_users), dtype="int32"),
    })
    item_map = pd.DataFrame({
        "movieId":  unique_items.astype("int32"),
        "item_idx": np.arange(len(unique_items), dtype="int32"),
    })

    # Merge to apply mappings
    df = df.merge(user_map, on="userId", how="left")
    df = df.merge(item_map, on="movieId", how="left")
    df = df[["user_idx", "item_idx", "rating", "timestamp"]].copy()
    df["user_idx"] = df["user_idx"].astype("int32")
    df["item_idx"] = df["item_idx"].astype("int32")

    logger.info(
        "Remapped IDs: %d users -> [0..%d], %d items -> [0..%d]",
        len(unique_users), len(unique_users) - 1,
        len(unique_items), len(unique_items) - 1,
    )
    return df, user_map, item_map


# ---------------------------------------------------------------------------
# Parquet cache
# ---------------------------------------------------------------------------

def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved %s  (%s rows)", path.name, f"{len(df):,}")


def _load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Loaded cached %s  (%s rows)", path.name, f"{len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(
    cfg: Config = DEFAULT_CONFIG,
    force_reload: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load, filter, remap, and cache the MovieLens 25M dataset.

    Parameters
    ----------
    cfg : Config
        Configuration object.
    force_reload : bool
        If True, ignore the parquet cache and re-process from CSV.

    Returns
    -------
    ratings_df : pd.DataFrame
        Columns: [user_idx (int32), item_idx (int32), rating (float32), timestamp (int32)]
    user_map : pd.DataFrame
        Columns: [userId (int32), user_idx (int32)]
    item_map : pd.DataFrame
        Columns: [movieId (int32), item_idx (int32)]
    """
    cfg.ensure_dirs()

    # Parquet cache paths
    suffix = f"_sample{cfg.sample_n}" if cfg.sample_n > 0 else ""
    ratings_pq = cfg.data_proc / f"ratings_filtered{suffix}.parquet"
    user_map_pq = cfg.data_proc / f"user_map{suffix}.parquet"
    item_map_pq = cfg.data_proc / f"item_map{suffix}.parquet"

    if not force_reload and ratings_pq.exists() and user_map_pq.exists() and item_map_pq.exists():
        logger.info("Loading data from parquet cache ...")
        ratings_df = _load_parquet(ratings_pq)
        user_map   = _load_parquet(user_map_pq)
        item_map   = _load_parquet(item_map_pq)
        _print_stats(ratings_df.rename(columns={"user_idx": "userId", "item_idx": "movieId"}), "CACHED")
        return ratings_df, user_map, item_map

    # --- Full pipeline ---
    _ensure_data(cfg.data_raw)
    raw_df = _read_ratings(cfg.data_raw, sample_n=cfg.sample_n)
    filtered = _filter_ratings(raw_df, cfg.min_ratings_per_user, cfg.min_ratings_per_item)
    ratings_df, user_map, item_map = _remap_ids(filtered)

    _save_parquet(ratings_df, ratings_pq)
    _save_parquet(user_map,   user_map_pq)
    _save_parquet(item_map,   item_map_pq)

    return ratings_df, user_map, item_map


def load_movies(cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """
    Load movies.csv and join with item_map so we have item_idx alongside movieId.

    Returns
    -------
    pd.DataFrame with columns: [movieId, title, genres, item_idx]
    """
    movies = _read_movies(cfg.data_raw)
    suffix = f"_sample{cfg.sample_n}" if cfg.sample_n > 0 else ""
    item_map_pq = cfg.data_proc / f"item_map{suffix}.parquet"
    if item_map_pq.exists():
        item_map = _load_parquet(item_map_pq)
        movies = movies.merge(item_map, on="movieId", how="inner")
    return movies


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = DEFAULT_CONFIG
    if "--sample" in sys.argv:
        cfg = cfg.copy_with(sample_n=1_000_000)
    ratings, umap, imap = load_data(cfg, force_reload=True)
    print(ratings.dtypes)
    print(ratings.head())
    print(f"Users: {umap.shape[0]:,}  Items: {imap.shape[0]:,}")
