"""
tests/test_similarity.py
========================
Unit tests for the similarity module:
  1. Similarity matrix is symmetric.
  2. Diagonal is zero (an item is not its own neighbour).
  3. All values are in [0, 1] (positive only, post-shrinkage).
  4. Each row has at most K nonzero entries.
  5. min_common_users filter works (pairs below threshold are 0).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from similarity import compute_similarity, build_item_user_matrix


# ---------------------------------------------------------------------------
# Fixture: tiny deterministic rating dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_train() -> pd.DataFrame:
    """5 users, 10 items, ~50 ratings (dense enough for testing)."""
    rng = np.random.default_rng(0)
    data = []
    for u in range(15):
        for i in rng.choice(20, size=rng.integers(5, 15), replace=False):
            data.append({
                "user_idx":  np.int32(u),
                "item_idx":  np.int32(i),
                "rating":    np.float32(rng.choice([1.0, 2.0, 3.0, 4.0, 5.0])),
                "timestamp": np.int32(rng.integers(1, 1000)),
            })
    df = pd.DataFrame(data).drop_duplicates(subset=["user_idx", "item_idx"])
    return df.reset_index(drop=True)


@pytest.fixture(params=["cosine", "adjusted_cosine", "pearson"])
def sim_matrix(tiny_train, request, tmp_path):
    cfg = Config(
        data_proc    = tmp_path,
        k_neighbors  = 5,
        block_size   = 5,
        min_common_users = 1,
        shrinkage_lambda = 0.0,
        similarity   = request.param,
        sample_n     = 0,
    )
    S = compute_similarity(tiny_train, cfg, n_users=15, n_items=20, force_reload=True)
    return S, cfg


# ---------------------------------------------------------------------------
# Test 1: Similarity score consistency (symmetric when both directions stored)
# ---------------------------------------------------------------------------

def test_similarity_consistent(sim_matrix):
    """
    The stored similarity values are consistent: if both S[i,j] and S[j,i]
    are stored (i.e., both appear in each other's top-K), they must be equal.

    Note: The top-K matrix is intentionally NOT globally symmetric -- storing
    top-K per row means S[i,j] > 0 does NOT imply S[j,i] > 0 (i may be in
    j's top-K but j might not be in i's top-K).  That is correct behaviour.
    We verify only mutual entries.
    """
    S, _ = sim_matrix
    coo = S.tocoo()
    for i, j, v in zip(coo.row, coo.col, coo.data):
        s_ji = float(S[j, i])
        if s_ji > 0:
            # Both are stored -- they must agree (same dot product, same norms)
            assert abs(float(v) - s_ji) < 1e-4, (
                f"S[{i},{j}]={float(v):.6f}  !=  S[{j},{i}]={s_ji:.6f}"
            )


# ---------------------------------------------------------------------------
# Test 2: Diagonal excluded
# ---------------------------------------------------------------------------

def test_diagonal_is_zero(sim_matrix):
    """Items must not be their own neighbours."""
    S, _ = sim_matrix
    diag = S.diagonal()
    assert np.all(diag == 0), f"Nonzero diagonal entries found: {diag[diag != 0]}"


# ---------------------------------------------------------------------------
# Test 3: Values in [0, 1]
# ---------------------------------------------------------------------------

def test_values_in_range(sim_matrix):
    """All stored similarity values must be in (0, 1]."""
    S, _ = sim_matrix
    if S.nnz == 0:
        return
    assert S.data.min() > 0,  f"Negative similarity: {S.data.min()}"
    assert S.data.max() <= 1.0 + 1e-5, f"Similarity > 1: {S.data.max()}"


# ---------------------------------------------------------------------------
# Test 4: At most K neighbours per row
# ---------------------------------------------------------------------------

def test_at_most_k_neighbours(sim_matrix):
    """Each row of S has at most K nonzero entries."""
    S, cfg = sim_matrix
    row_nnz = np.diff(S.tocsr().indptr)
    assert row_nnz.max() <= cfg.k_neighbors, (
        f"Row has {row_nnz.max()} neighbours > K={cfg.k_neighbors}"
    )


# ---------------------------------------------------------------------------
# Test 5: min_common_users filter
# ---------------------------------------------------------------------------

def test_min_common_users_filter(tiny_train, tmp_path):
    """
    With min_common_users set very high, most/all similarities should be 0.
    With min_common_users=1, similarities exist.
    """
    n_users, n_items = 15, 20

    cfg_low = Config(
        data_proc=tmp_path / "low",
        k_neighbors=5, block_size=5,
        min_common_users=1, shrinkage_lambda=0.0, similarity="cosine",
    )
    S_low = compute_similarity(tiny_train, cfg_low, n_users=n_users, n_items=n_items, force_reload=True)

    cfg_high = Config(
        data_proc=tmp_path / "high",
        k_neighbors=5, block_size=5,
        min_common_users=50, shrinkage_lambda=0.0, similarity="cosine",   # impossible threshold
    )
    S_high = compute_similarity(tiny_train, cfg_high, n_users=n_users, n_items=n_items, force_reload=True)

    assert S_low.nnz >= S_high.nnz, "Higher min_common_users should produce <= nonzeros"
    assert S_high.nnz == 0, "With min_common=50 on a 15-user dataset, S should be all zeros"
