"""
split.py
========
Per-user temporal train / validation / test split.

Design
------
For every user, ratings are sorted by timestamp (ascending).
The oldest 80 % go to TRAIN, next 10 % to VALIDATION, newest 10 % to TEST.
Users with fewer than `min_ratings_to_split` ratings go entirely to TRAIN.

All statistics (means, similarities) must be computed from TRAIN ONLY.
The parquet files written here are reused by all other models in the study.

Public API
----------
split_data(ratings_df, cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    Returns (train, val, test).
load_splits(cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    Loads from parquet cache (calls split_data if missing).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def split_data(
    ratings_df: pd.DataFrame,
    cfg: Config = DEFAULT_CONFIG,
    force_reload: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a per-user temporal 80/10/10 split.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Columns: [user_idx, item_idx, rating, timestamp]
    cfg : Config
    force_reload : bool
        Recompute even if parquet cache exists.

    Returns
    -------
    train, val, test : pd.DataFrame
        Same schema as ratings_df, split by temporal position.
    """
    cfg.ensure_dirs()
    suffix = f"_sample{cfg.sample_n}" if cfg.sample_n > 0 else ""
    train_pq = cfg.data_proc / f"train{suffix}.parquet"
    val_pq   = cfg.data_proc / f"val{suffix}.parquet"
    test_pq  = cfg.data_proc / f"test{suffix}.parquet"

    if not force_reload and train_pq.exists() and val_pq.exists() and test_pq.exists():
        logger.info("Loading splits from parquet cache ...")
        return (
            pd.read_parquet(train_pq, engine="pyarrow"),
            pd.read_parquet(val_pq,   engine="pyarrow"),
            pd.read_parquet(test_pq,  engine="pyarrow"),
        )

    logger.info("Computing per-user temporal split ...")
    t0 = time.perf_counter()

    # Sort entire dataset by user then timestamp for efficient groupby
    df = ratings_df.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)

    # Vectorised split using cumulative counts within each user group
    # ----------------------------------------------------------------
    # For each user: positions 0 .. n-1
    # train   : positions < ceil(0.80 * n)
    # val     : positions in [ceil(0.80*n), ceil(0.90*n))
    # test    : positions >= ceil(0.90*n)

    user_sizes = df.groupby("user_idx", sort=False).size()

    # Build a flat array of per-row split labels (0=train, 1=val, 2=test)
    split_labels = np.empty(len(df), dtype=np.int8)
    idx = 0
    for uid, n in user_sizes.items():
        n = int(n)
        if n < cfg.min_ratings_to_split:
            # All go to train
            split_labels[idx : idx + n] = 0
        else:
            train_end = int(np.ceil(cfg.train_frac * n))
            val_end   = int(np.ceil((cfg.train_frac + cfg.val_frac) * n))
            # Ensure at least 1 item in val and test when n >= min_ratings_to_split
            val_end   = max(val_end, train_end + 1)
            val_end   = min(val_end, n - 1)
            split_labels[idx : idx + train_end]       = 0
            split_labels[idx + train_end : idx + val_end] = 1
            split_labels[idx + val_end : idx + n]     = 2
        idx += n

    df = df.copy()
    df["_split"] = split_labels

    train = df[df["_split"] == 0].drop(columns="_split").reset_index(drop=True)
    val   = df[df["_split"] == 1].drop(columns="_split").reset_index(drop=True)
    test  = df[df["_split"] == 2].drop(columns="_split").reset_index(drop=True)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Split complete in %.1f s -- train=%s  val=%s  test=%s",
        elapsed, f"{len(train):,}", f"{len(val):,}", f"{len(test):,}",
    )
    _log_split_stats(train, val, test)

    # Save to parquet
    for split_df, path in [(train, train_pq), (val, val_pq), (test, test_pq)]:
        split_df.to_parquet(path, index=False, engine="pyarrow")
        logger.info("Saved %s", path.name)

    return train, val, test


def load_splits(
    cfg: Config = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the splits from parquet. Raises FileNotFoundError if not yet computed.

    Returns
    -------
    train, val, test : pd.DataFrame
    """
    suffix = f"_sample{cfg.sample_n}" if cfg.sample_n > 0 else ""
    train_pq = cfg.data_proc / f"train{suffix}.parquet"
    val_pq   = cfg.data_proc / f"val{suffix}.parquet"
    test_pq  = cfg.data_proc / f"test{suffix}.parquet"
    for p in [train_pq, val_pq, test_pq]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run split_data() first.")
    return (
        pd.read_parquet(train_pq, engine="pyarrow"),
        pd.read_parquet(val_pq,   engine="pyarrow"),
        pd.read_parquet(test_pq,  engine="pyarrow"),
    )


def _log_split_stats(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    for name, df in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        n_users = df["user_idx"].nunique()
        n_items = df["item_idx"].nunique()
        logger.info(
            "  [%s]  ratings=%s  users=%s  items=%s  avg_rating=%.3f",
            name, f"{len(df):,}", f"{n_users:,}", f"{n_items:,}", df["rating"].mean(),
        )


# ---------------------------------------------------------------------------
# Leakage validation helper (used in unit tests)
# ---------------------------------------------------------------------------

def assert_no_temporal_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    sample_users: int = 1000,
    seed: int = 42,
) -> None:
    """
    For a random sample of users, assert that every train timestamp <=
    every val timestamp <= every test timestamp.

    Raises AssertionError if leakage is detected.
    """
    rng = np.random.default_rng(seed)
    all_users = np.intersect1d(
        train["user_idx"].unique(),
        test["user_idx"].unique(),
    )
    sampled = rng.choice(all_users, size=min(sample_users, len(all_users)), replace=False)

    for uid in sampled:
        t_train = train.loc[train["user_idx"] == uid, "timestamp"].values
        t_test  = test.loc[test["user_idx"]  == uid, "timestamp"].values
        if len(t_train) > 0 and len(t_test) > 0:
            assert t_train.max() <= t_test.max(), (
                f"Temporal leakage for user {uid}: "
                f"train max ts {t_train.max()} > test max ts {t_test.max()}"
            )
        if uid in val["user_idx"].values:
            t_val = val.loc[val["user_idx"] == uid, "timestamp"].values
            if len(t_train) > 0 and len(t_val) > 0:
                assert t_train.max() <= t_val.max(), (
                    f"Temporal leakage (train/val) for user {uid}"
                )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from data_loader import load_data

    cfg = DEFAULT_CONFIG
    if "--sample" in sys.argv:
        cfg = cfg.copy_with(sample_n=1_000_000)
    ratings, _, _ = load_data(cfg)
    train, val, test = split_data(ratings, cfg, force_reload=True)
    print(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    print("Checking for temporal leakage ...")
    assert_no_temporal_leakage(train, val, test)
    print("No leakage detected  ")
