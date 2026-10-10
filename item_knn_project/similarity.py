"""
similarity.py
=============
Blockwise top-K item-item similarity computation for Item-KNN CF.

Three similarity variants
--------------------------
1. cosine
   sim(i, j) = (r_i   r_j) / (||r_i|| ||r_j||)
   where r_i is the raw rating vector of item i across all users.

2. adjusted_cosine
   Subtract each USER's mean rating before computing cosine.
   sim(i, j) = ((r_i -  _u)   (r_j -  _u)) / ...
   Captures the fact that different users use the rating scale differently.

3. pearson (sparse approximation)
   Subtract each ITEM's mean rating, then cosine on the centred matrix.
   This is an efficient approximation of Pearson because we include ALL users
   in the dot product, not just co-rated users.  The exact Pearson would use:
       sim_exact(i,j) =  _{u   U_ij} (r_ui    _i)(r_uj    _j) / ...
   where U_ij is the set of users who rated BOTH i and j.
   A validation function computes the exact Pearson on ~100 random item pairs
   and compares it to the approximation; the error is logged.

Shrinkage
---------
   sim_shrunk(i, j) = sim_raw(i, j)   n_common(i, j) / (n_common(i, j) +  )
   This dampens the similarity between items that share very few co-raters.
     = cfg.shrinkage_lambda (default 100).

Output
------
A scipy.sparse CSR matrix of shape (n_items, n_items) where each row i
contains at most K nonzero entries: the top-K most similar items (positive
sim only), excluding item i itself.

Public API
----------
compute_similarity(train_df, cfg) -> scipy.sparse.csr_matrix
load_similarity(cfg)              -> scipy.sparse.csr_matrix
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import numpy as np
import psutil
import scipy.sparse as sp
from tqdm import tqdm

from config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build the item-user matrix from TRAIN
# ---------------------------------------------------------------------------

def build_item_user_matrix(
    train_df: "pd.DataFrame",
    n_users: int,
    n_items: int,
) -> sp.csr_matrix:
    """
    Build a sparse item   user rating matrix (CSR, float32).

    Parameters
    ----------
    train_df : pd.DataFrame  [user_idx, item_idx, rating, timestamp]
    n_users, n_items : int

    Returns
    -------
    R : scipy.sparse.csr_matrix, shape (n_items, n_users), float32
    """
    users  = train_df["user_idx"].to_numpy(dtype=np.int32)
    items  = train_df["item_idx"].to_numpy(dtype=np.int32)
    ratings = train_df["rating"].to_numpy(dtype=np.float32)

    R = sp.csr_matrix(
        (ratings, (items, users)),
        shape=(n_items, n_users),
        dtype=np.float32,
    )
    logger.info(
        "Item User matrix: shape=%s  nnz=%s  density=%.4f%%",
        R.shape, f"{R.nnz:,}", R.nnz / (R.shape[0] * R.shape[1]) * 100,
    )
    return R


# ---------------------------------------------------------------------------
# Per-variant centring helpers
# ---------------------------------------------------------------------------

def _subtract_user_means(R: sp.csr_matrix) -> sp.csr_matrix:
    """
    Adjusted cosine: subtract each user's mean rating.
    R is (n_items   n_users); we work column-wise.
    Returns a new CSR matrix (values centred per user).
    """
    # user mean from column sums; CSC for column-wise access
    Rcsc = R.tocsc()
    user_mean = np.zeros(R.shape[1], dtype=np.float32)
    for j in range(R.shape[1]):
        col = Rcsc.getcol(j)
        if col.nnz > 0:
            user_mean[j] = col.data.mean()

    # Subtract: for every nonzero (i, j), subtract user_mean[j]
    R2 = R.copy().astype(np.float32)
    R2 = R2.tocoo()
    R2.data -= user_mean[R2.col]
    return R2.tocsr()


def _subtract_item_means(R: sp.csr_matrix) -> sp.csr_matrix:
    """
    Pearson approx: subtract each item's mean rating.
    R is (n_items   n_users); row-wise centring.
    Returns a new CSR matrix.
    """
    R2 = R.copy().astype(np.float32).tocsr()
    # Item mean from row mean of nonzero entries
    row_means = np.zeros(R.shape[0], dtype=np.float32)
    for i in range(R.shape[0]):
        row = R2.getrow(i)
        if row.nnz > 0:
            row_means[i] = row.data.mean()

    R2 = R2.tocoo()
    R2.data -= row_means[R2.row]
    return R2.tocsr()


def _centre_matrix(R: sp.csr_matrix, similarity: str) -> sp.csr_matrix:
    """Apply the centring step appropriate for the chosen similarity variant."""
    if similarity == "cosine":
        return R
    elif similarity == "adjusted_cosine":
        logger.info("Computing user means for adjusted cosine ...")
        return _subtract_user_means(R)
    elif similarity == "pearson":
        logger.info("Computing item means for Pearson approximation ...")
        return _subtract_item_means(R)
    else:
        raise ValueError(f"Unknown similarity: {similarity!r}. Choose cosine / adjusted_cosine / pearson.")


def _row_norms(R: sp.csr_matrix) -> np.ndarray:
    """L2 norm of each row of a sparse matrix. Returns float32 array."""
    # Efficient: sum of squares per row
    norms = np.sqrt(np.array(R.power(2).sum(axis=1)).ravel()).astype(np.float32)
    return norms


# ---------------------------------------------------------------------------
# Co-rating count matrix  (needed for support filter + shrinkage)
# ---------------------------------------------------------------------------

def _build_corating_matrix_block(
    R_bin_block: sp.csr_matrix,   # (block_size   n_users), binarised
    R_bin_full:  sp.csr_matrix,   # (n_items   n_users), binarised
) -> np.ndarray:
    """
    Return co-rating count matrix for the current block:
      corating[b, j] = number of users who rated both item b (in block) and item j.
    Shape: (block_size, n_items).
    """
    # (block_size   n_items) -- dense int16 sufficient up to 32K users; use int32 to be safe
    counts = (R_bin_block @ R_bin_full.T).toarray().astype(np.int32)
    return counts


# ---------------------------------------------------------------------------
# Top-K selection helper
# ---------------------------------------------------------------------------

def _top_k_per_row(
    sim_block: np.ndarray,   # (block_size, n_items) dense float32
    corating:  np.ndarray,   # (block_size, n_items) int32
    block_start: int,
    k: int,
    min_common: int,
    shrinkage_lambda: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Apply filters and shrinkage, then select the top-K neighbours for each
    item in the block.

    Returns
    -------
    rows, cols, vals : lists of int32/float32 arrays, one entry per item in block.
    """
    rows_out, cols_out, vals_out = [], [], []

    block_size = sim_block.shape[0]
    item_indices = np.arange(block_size, dtype=np.int32) + block_start

    for b in range(block_size):
        sim_row = sim_block[b].copy()           # (n_items,)
        cor_row = corating[b]                    # (n_items,)
        item_i  = item_indices[b]

        # 1. Zero out self-similarity
        sim_row[item_i] = 0.0

        # 2. Support filter
        sim_row[cor_row < min_common] = 0.0

        # 3. Positive only
        sim_row[sim_row <= 0.0] = 0.0

        # 4. Shrinkage
        if shrinkage_lambda > 0:
            nz = sim_row > 0
            if nz.any():
                n_com = cor_row[nz].astype(np.float32)
                sim_row[nz] *= n_com / (n_com + shrinkage_lambda)

        # 5. Top-K via argpartition (O(n), faster than argsort)
        nz_count = int((sim_row > 0).sum())
        if nz_count == 0:
            continue
        actual_k = min(k, nz_count)
        # argpartition gives k smallest; we want k largest -> negate trick
        top_k_idx = np.argpartition(-sim_row, actual_k - 1)[:actual_k]
        top_k_idx = top_k_idx[sim_row[top_k_idx] > 0]
        if len(top_k_idx) == 0:
            continue

        rows_out.append(np.full(len(top_k_idx), item_i, dtype=np.int32))
        cols_out.append(top_k_idx.astype(np.int32))
        vals_out.append(sim_row[top_k_idx].astype(np.float32))

    return rows_out, cols_out, vals_out


# ---------------------------------------------------------------------------
# Main blockwise similarity computation
# ---------------------------------------------------------------------------

def compute_similarity(
    train_df: "pd.DataFrame",
    cfg: Config = DEFAULT_CONFIG,
    n_users: int | None = None,
    n_items: int | None = None,
    force_reload: bool = False,
) -> sp.csr_matrix:
    """
    Compute the top-K item-item similarity matrix from the training set.

    Parameters
    ----------
    train_df : pd.DataFrame  [user_idx, item_idx, rating, timestamp]
    cfg : Config
    n_users, n_items : int -- inferred from data if None
    force_reload : bool -- recompute even if cache exists

    Returns
    -------
    S : scipy.sparse.csr_matrix, shape (n_items, n_items), float32
        S[i, j] = shrunk similarity between items i and j.
        Each row has at most K nonzero entries.
    """
    cfg.ensure_dirs()
    suffix = (
        f"_sample{cfg.sample_n}" if cfg.sample_n > 0 else ""
    )
    cache_name = (
        f"similarity_{cfg.similarity}_k{cfg.k_neighbors}"
        f"_sh{int(cfg.shrinkage_lambda)}_mc{cfg.min_common_users}{suffix}.npz"
    )
    cache_path = cfg.data_proc / cache_name

    if not force_reload and cache_path.exists():
        logger.info("Loading similarity from cache: %s", cache_path.name)
        return sp.load_npz(str(cache_path))

    if n_users is None:
        n_users = int(train_df["user_idx"].max()) + 1
    if n_items is None:
        n_items = int(train_df["item_idx"].max()) + 1

    logger.info(
        "Computing %s similarity  (K=%d, shrink=%.0f, min_common=%d, block=%d) ...",
        cfg.similarity, cfg.k_neighbors, cfg.shrinkage_lambda,
        cfg.min_common_users, cfg.block_size,
    )

    t0 = time.perf_counter()
    process = psutil.Process()

    # Build item user matrix and centre it
    R = build_item_user_matrix(train_df, n_users, n_items)
    R_centred = _centre_matrix(R, cfg.similarity)

    # L2-normalise each item's (centred) vector for cosine computation
    norms = _row_norms(R_centred)
    # Avoid division by zero for items with zero norm after centring
    norms = np.where(norms == 0, 1.0, norms).astype(np.float32)

    # Normalised matrix (CSR, float32) -- each row is a unit vector
    # We scale in-place to save memory
    R_norm = R_centred.copy()
    # Row-scale: multiply row i by 1/norms[i]
    diag_inv = sp.diags(1.0 / norms, dtype=np.float32)
    R_norm = diag_inv @ R_norm     # (n_items   n_users)

    # Binarised matrix for co-rating counts
    R_bin = (R > 0).astype(np.int8)
    R_bin_csr = R_bin.tocsr()

    # Accumulate COO data for the final sparse matrix
    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_vals: list[np.ndarray] = []

    n_blocks = int(np.ceil(n_items / cfg.block_size))
    pbar = tqdm(total=n_blocks, desc=f"Similarity [{cfg.similarity}]", unit="block")

    for blk in range(n_blocks):
        blk_start = blk * cfg.block_size
        blk_end   = min(blk_start + cfg.block_size, n_items)

        t_blk = time.perf_counter()

        R_block = R_norm[blk_start:blk_end]          # (block_size   n_users)
        R_bin_block = R_bin_csr[blk_start:blk_end]   # (block_size   n_users)

        # sim_dense[b, j] = normalised dot product -> cosine similarity
        # Shape: (block_size, n_items)
        sim_dense = (R_block @ R_norm.T).toarray().astype(np.float32)

        # Co-rating counts (needed for support filter + shrinkage)
        corating = _build_corating_matrix_block(R_bin_block, R_bin_csr)

        rows_b, cols_b, vals_b = _top_k_per_row(
            sim_dense,
            corating,
            block_start=blk_start,
            k=cfg.k_neighbors,
            min_common=cfg.min_common_users,
            shrinkage_lambda=cfg.shrinkage_lambda,
        )
        all_rows.extend(rows_b)
        all_cols.extend(cols_b)
        all_vals.extend(vals_b)

        # Free block memory explicitly
        del sim_dense, corating, R_block, R_bin_block
        gc.collect()

        elapsed_blk = time.perf_counter() - t_blk
        ram_mb = process.memory_info().rss / 1e6
        pbar.set_postfix({"t/blk": f"{elapsed_blk:.1f}s", "RAM": f"{ram_mb:.0f}MB"})
        pbar.update(1)

    pbar.close()

    # Build final CSR matrix
    if all_rows:
        row_arr = np.concatenate(all_rows, dtype=np.int32)
        col_arr = np.concatenate(all_cols, dtype=np.int32)
        val_arr = np.concatenate(all_vals, dtype=np.float32)
    else:
        row_arr = np.array([], dtype=np.int32)
        col_arr = np.array([], dtype=np.int32)
        val_arr = np.array([], dtype=np.float32)

    S = sp.csr_matrix(
        (val_arr, (row_arr, col_arr)),
        shape=(n_items, n_items),
        dtype=np.float32,
    )

    elapsed = time.perf_counter() - t0
    ram_mb = process.memory_info().rss / 1e6
    logger.info(
        "Similarity done in %.1f s  nnz=%s  shape=%s  RAM=%.0f MB",
        elapsed, f"{S.nnz:,}", S.shape, ram_mb,
    )

    sp.save_npz(str(cache_path), S)
    logger.info("Saved similarity matrix -> %s", cache_path.name)

    return S


def load_similarity(cfg: Config = DEFAULT_CONFIG) -> sp.csr_matrix:
    """Load the pre-computed similarity matrix from cache."""
    suffix = f"_sample{cfg.sample_n}" if cfg.sample_n > 0 else ""
    cache_name = (
        f"similarity_{cfg.similarity}_k{cfg.k_neighbors}"
        f"_sh{int(cfg.shrinkage_lambda)}_mc{cfg.min_common_users}{suffix}.npz"
    )
    cache_path = cfg.data_proc / cache_name
    if not cache_path.exists():
        raise FileNotFoundError(f"Similarity cache not found: {cache_path}")
    logger.info("Loading similarity from %s", cache_path.name)
    return sp.load_npz(str(cache_path))


# ---------------------------------------------------------------------------
# Exact Pearson validation
# ---------------------------------------------------------------------------

def _exact_pearson(R: sp.csr_matrix, i: int, j: int) -> float:
    """
    Compute the exact Pearson correlation between items i and j,
    using ONLY co-rated users.

    sim_exact(i,j) =  _{u   U_ij} (r_ui    _i')(r_uj    _j') /
                      (  (r_ui    _i')      (r_uj    _j') )

    where  _i' and  _j' are item means computed over co-rated users only.
    """
    ri = np.array(R.getrow(i).todense()).ravel()
    rj = np.array(R.getrow(j).todense()).ravel()
    mask = (ri != 0) & (rj != 0)
    if mask.sum() < 2:
        return np.nan
    ri_co = ri[mask] - ri[mask].mean()
    rj_co = rj[mask] - rj[mask].mean()
    denom = np.linalg.norm(ri_co) * np.linalg.norm(rj_co)
    if denom == 0:
        return np.nan
    return float(np.dot(ri_co, rj_co) / denom)


def validate_pearson_approximation(
    train_df: "pd.DataFrame",
    S_approx: sp.csr_matrix,
    cfg: Config = DEFAULT_CONFIG,
    n_pairs: int = 100,
    seed: int = 42,
) -> dict:
    """
    Compare the sparse Pearson approximation against exact Pearson on
    `n_pairs` random item pairs that share at least min_common_users raters.

    Returns
    -------
    dict with 'mean_abs_error', 'max_abs_error', 'n_valid_pairs'.
    """
    logger.info("Validating Pearson approximation on %d random item pairs ...", n_pairs)
    n_users = int(train_df["user_idx"].max()) + 1
    n_items = int(train_df["item_idx"].max()) + 1
    R = build_item_user_matrix(train_df, n_users, n_items)

    rng = np.random.default_rng(seed)
    candidates = rng.choice(n_items, size=n_pairs * 10, replace=False)

    errors = []
    checked = 0
    for k in range(0, len(candidates) - 1, 2):
        i, j = int(candidates[k]), int(candidates[k + 1])
        exact = _exact_pearson(R, i, j)
        if np.isnan(exact):
            continue
        # Approximation: look up S_approx
        approx = S_approx[i, j]  # 0 if not in top-K
        errors.append(abs(exact - approx))
        checked += 1
        if checked >= n_pairs:
            break

    if errors:
        mae = float(np.mean(errors))
        max_e = float(np.max(errors))
        logger.info(
            "Pearson validation (%d pairs): mean |error|=%.4f  max |error|=%.4f",
            checked, mae, max_e,
        )
        return {"mean_abs_error": mae, "max_abs_error": max_e, "n_valid_pairs": checked}
    else:
        logger.warning("No valid pairs found for Pearson validation.")
        return {"mean_abs_error": None, "max_abs_error": None, "n_valid_pairs": 0}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import pandas as pd
    from data_loader import load_data
    from split import split_data

    cfg = DEFAULT_CONFIG
    if "--sample" in sys.argv:
        cfg = cfg.copy_with(sample_n=1_000_000)

    ratings, umap, imap = load_data(cfg)
    train, val, test = split_data(ratings, cfg)
    n_users = int(umap.shape[0])
    n_items = int(imap.shape[0])

    S = compute_similarity(train, cfg, n_users=n_users, n_items=n_items, force_reload=True)
    print("Similarity matrix:", S.shape, "nnz:", S.nnz)
