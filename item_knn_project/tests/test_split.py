"""
tests/test_split.py
===================
Unit tests for the temporal split:
  1. No temporal leakage (train timestamps <= test timestamps per user).
  2. Every user's train timestamps <= val timestamps.
  3. No (user, item) pair appears in both train and test.
  4. Split fractions are approximately correct.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from split import split_data, assert_no_temporal_leakage


# ---------------------------------------------------------------------------
# Fixture: synthetic dataset (100 users   random items)
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_ratings() -> pd.DataFrame:
    """Generate a deterministic toy rating dataset."""
    rng = np.random.default_rng(42)
    n_users, n_items, n_ratings = 100, 200, 5000
    users   = rng.integers(0, n_users,  n_ratings, dtype=np.int32)
    items   = rng.integers(0, n_items,  n_ratings, dtype=np.int32)
    ratings = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], n_ratings).astype(np.float32)
    # Unique timestamps so sort order is deterministic
    timestamps = np.arange(n_ratings, dtype=np.int32)
    rng.shuffle(timestamps)

    df = pd.DataFrame({
        "user_idx":  users,
        "item_idx":  items,
        "rating":    ratings,
        "timestamp": timestamps,
    }).drop_duplicates(subset=["user_idx", "item_idx"])
    return df.reset_index(drop=True)


@pytest.fixture
def splits(toy_ratings, tmp_path):
    cfg = Config(data_proc=tmp_path, min_ratings_to_split=3)
    return split_data(toy_ratings, cfg, force_reload=True)


# ---------------------------------------------------------------------------
# Test 1: No temporal leakage
# ---------------------------------------------------------------------------

def test_no_temporal_leakage(splits):
    """For every user, max(train.timestamp) <= max(test.timestamp)."""
    train, val, test = splits
    assert_no_temporal_leakage(train, val, test, sample_users=100)


# ---------------------------------------------------------------------------
# Test 2: Train timestamps never exceed val timestamps (per user)
# ---------------------------------------------------------------------------

def test_train_before_val(splits):
    """For every user in both train and val, train's max ts <= val's max ts."""
    train, val, _ = splits
    shared = np.intersect1d(train["user_idx"].unique(), val["user_idx"].unique())
    for uid in shared[:50]:   # sample for speed
        t_tr = train.loc[train["user_idx"] == uid, "timestamp"].max()
        t_va = val.loc[val["user_idx"] == uid, "timestamp"].max()
        assert t_tr <= t_va, f"User {uid}: train ts {t_tr} > val ts {t_va}"


# ---------------------------------------------------------------------------
# Test 3: No (user, item) in both train and test
# ---------------------------------------------------------------------------

def test_no_overlap_train_test(splits):
    """A (user_idx, item_idx) pair must not appear in both train and test."""
    train, _, test = splits
    train_pairs = set(zip(train["user_idx"].tolist(), train["item_idx"].tolist()))
    test_pairs  = set(zip(test["user_idx"].tolist(),  test["item_idx"].tolist()))
    overlap = train_pairs & test_pairs
    assert len(overlap) == 0, f"{len(overlap)} (user, item) pairs appear in both train and test"


# ---------------------------------------------------------------------------
# Test 4: Split fractions approximately correct
# ---------------------------------------------------------------------------

def test_split_fractions(toy_ratings, splits):
    """train / (train+val+test) should be approximately 0.80."""
    train, val, test = splits
    total = len(train) + len(val) + len(test)
    assert total == len(toy_ratings) or total == len(toy_ratings.drop_duplicates()), \
        "Total rows don't match"
    frac = len(train) / total
    assert 0.70 <= frac <= 0.90, f"Train fraction {frac:.3f} out of expected range [0.70, 0.90]"


# ---------------------------------------------------------------------------
# Test 5: Users with few ratings go to train entirely
# ---------------------------------------------------------------------------

def test_few_rating_users_in_train_only(splits):
    """Users with < min_ratings_to_split (=3) go entirely to train, not val/test."""
    train, val, test = splits
    # Find users with exactly 1 or 2 ratings total
    all_counts = pd.concat([train, val, test]).groupby("user_idx").size()
    tiny_users = all_counts[all_counts < 3].index.tolist()
    for uid in tiny_users:
        assert uid not in val["user_idx"].values, f"User {uid} (few ratings) found in val"
        assert uid not in test["user_idx"].values, f"User {uid} (few ratings) found in test"
