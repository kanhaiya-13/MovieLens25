"""
tests/test_model.py
===================
Unit tests for ItemKNN:
  1. Predictions are within [0.5, 5.0].
  2. Recommendations never include seen items.
  3. Cold-start users/items return valid fallbacks (no crashes).
  4. save/load round-trip preserves predictions.
  5. predict_batch matches predict() for individual calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from item_knn import ItemKNN


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_model(tmp_path):
    """Return a fitted ItemKNN on a small toy dataset."""
    rng = np.random.default_rng(7)
    n_users, n_items, n_ratings = 30, 40, 800
    users   = rng.integers(0, n_users, n_ratings, dtype=np.int32)
    items   = rng.integers(0, n_items, n_ratings, dtype=np.int32)
    ratings = rng.choice([1., 2., 3., 4., 5.], n_ratings).astype(np.float32)
    timestamps = np.arange(n_ratings, dtype=np.int32)

    train_df = pd.DataFrame({
        "user_idx":  users,
        "item_idx":  items,
        "rating":    ratings,
        "timestamp": timestamps,
    }).drop_duplicates(subset=["user_idx", "item_idx"]).reset_index(drop=True)

    cfg = Config(
        data_proc=tmp_path,
        k_neighbors=10,
        block_size=20,
        min_common_users=1,
        shrinkage_lambda=0.0,
        similarity="cosine",
        sample_n=0,
        rating_min=0.5,
        rating_max=5.0,
    )
    model = ItemKNN(cfg)
    model.fit(train_df, n_users=n_users, n_items=n_items)
    return model, train_df, n_users, n_items


# ---------------------------------------------------------------------------
# Test 1: Predictions clipped to [0.5, 5.0]
# ---------------------------------------------------------------------------

def test_predictions_in_range(fitted_model):
    """All predicted ratings must be in [0.5, 5.0]."""
    model, train_df, n_users, n_items = fitted_model
    rng = np.random.default_rng(42)
    u_arr = rng.integers(0, n_users, 200, dtype=np.int32)
    i_arr = rng.integers(0, n_items, 200, dtype=np.int32)
    preds = model.predict_batch(u_arr, i_arr)
    assert preds.min() >= 0.5 - 1e-6, f"Min prediction {preds.min()} < 0.5"
    assert preds.max() <= 5.0 + 1e-6, f"Max prediction {preds.max()} > 5.0"


# ---------------------------------------------------------------------------
# Test 2: Recommendations never include seen items
# ---------------------------------------------------------------------------

def test_no_seen_items_in_recs(fitted_model):
    """Recommended items must not appear in the user's training history."""
    model, train_df, n_users, n_items = fitted_model
    for uid in range(min(n_users, 10)):
        seen = set(train_df.loc[train_df["user_idx"] == uid, "item_idx"].tolist())
        recs = model.recommend(uid, N=10, exclude_seen=True)
        rec_items = {item for item, _ in recs}
        overlap = seen & rec_items
        assert len(overlap) == 0, f"User {uid}: seen items in recs: {overlap}"


# ---------------------------------------------------------------------------
# Test 3: Cold-start -- no crashes
# ---------------------------------------------------------------------------

def test_cold_start_no_crash(fitted_model):
    """Unseen users and items must return a fallback, not crash."""
    model, _, n_users, n_items = fitted_model
    cold_u = np.array([n_users + 1, n_users + 2], dtype=np.int32)
    cold_i = np.array([n_items + 1, n_items + 2], dtype=np.int32)

    # predict_batch with cold user/item
    preds = model.predict_batch(cold_u, cold_i)
    assert len(preds) == 2
    assert np.all(np.isfinite(preds))

    # recommend with cold user
    recs = model.recommend(n_users + 99, N=5)
    assert isinstance(recs, list)   # empty list is fine


# ---------------------------------------------------------------------------
# Test 4: Save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_consistency(fitted_model, tmp_path):
    """Loaded model must produce identical predictions to the original."""
    model, train_df, n_users, n_items = fitted_model
    save_dir = tmp_path / "saved_model"
    model.save(save_dir)

    model2 = ItemKNN(model.cfg)
    model2.load(save_dir)

    rng = np.random.default_rng(0)
    u_arr = rng.integers(0, n_users, 50, dtype=np.int32)
    i_arr = rng.integers(0, n_items, 50, dtype=np.int32)
    p1 = model.predict_batch(u_arr, i_arr)
    p2 = model2.predict_batch(u_arr, i_arr)
    np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=1e-6,
                                err_msg="Loaded model predictions differ from original")


# ---------------------------------------------------------------------------
# Test 5: predict_batch matches individual predict()
# ---------------------------------------------------------------------------

def test_batch_matches_single(fitted_model):
    """predict_batch(u, i) == predict(u, i) for individual pairs."""
    model, train_df, n_users, n_items = fitted_model
    rng = np.random.default_rng(1)
    u_arr = rng.integers(0, n_users, 30, dtype=np.int32)
    i_arr = rng.integers(0, n_items, 30, dtype=np.int32)
    batch = model.predict_batch(u_arr, i_arr)
    singles = np.array([model.predict(int(u), int(i)) for u, i in zip(u_arr, i_arr)])
    np.testing.assert_allclose(batch, singles, rtol=1e-4, atol=1e-4,
                                err_msg="predict_batch and predict() disagree")
