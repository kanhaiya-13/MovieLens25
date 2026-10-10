"""
item_knn.py
===========
Item-based K-Nearest-Neighbours Collaborative Filtering model.

Common interface (so other models can be swapped in via evaluate.py):
    fit(train_df)
    predict(user_idx, item_idx) -> float
    predict_batch(user_idx_array, item_idx_array) -> np.ndarray
    recommend(user_idx, N, exclude_seen=True) -> list[tuple[int, float]]
    save(path) / load(path)

Rating prediction formula (mean-centred weighted average)
---------------------------------------------------------
For user u and target item i:

    r (u,i) =  _i +  _j [ sim(i,j)   (r_uj    _j) ] /  _j |sim(i,j)|

where the sum is over the top-K neighbors j of i that user u has rated.
 _i is the mean rating of item i (train-only).

Fallback chain when no usable neighbors exist:
    item mean -> user mean -> global mean

Clipping: r  is clipped to [rating_min, rating_max].

Top-N recommendation
--------------------
Scores for all unseen items are:
  "weighted_sum" mode : score(u,i) = u_ratings_vec @ S[i, :]    (sparse matmul)
  "mean_centered" mode: subtract item means, same sparse matmul, then add  _i

Both modes use np.argpartition for O(n) top-N selection.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from config import Config, DEFAULT_CONFIG
from similarity import compute_similarity, load_similarity

logger = logging.getLogger(__name__)


class ItemKNN:
    """
    Item-based K-Nearest-Neighbours Collaborative Filtering.

    Parameters
    ----------
    cfg : Config
        Hyperparameters and paths.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, cfg: Config = DEFAULT_CONFIG) -> None:
        self.cfg = cfg
        self._is_fitted = False

        # Set after fit()
        self.n_users: int = 0
        self.n_items: int = 0
        self.S: Optional[sp.csr_matrix] = None          # item item similarity
        self.user_item_matrix: Optional[sp.csr_matrix] = None  # user item (train)
        self.item_means: Optional[np.ndarray] = None    #  _i per item
        self.user_means: Optional[np.ndarray] = None    #  _u per user
        self.global_mean: float = 0.0
        self.train_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        n_users: Optional[int] = None,
        n_items: Optional[int] = None,
        force_similarity: bool = False,
    ) -> "ItemKNN":
        """
        Fit the model on the training data.

        1. Compute (or load) the item-item similarity matrix.
        2. Build a sparse user item matrix for fast lookups.
        3. Compute per-item means, per-user means, and the global mean.

        Parameters
        ----------
        train_df : pd.DataFrame  [user_idx, item_idx, rating, timestamp]
        n_users, n_items : int  (inferred from data if None)
        force_similarity : bool  recompute similarity even if cached.

        Returns
        -------
        self
        """
        t0 = time.perf_counter()
        logger.info("ItemKNN.fit() starting ...")

        self.train_df = train_df.copy()
        self.n_users = int(n_users or train_df["user_idx"].max() + 1)
        self.n_items = int(n_items or train_df["item_idx"].max() + 1)

        #    Item-item similarity                                       
        self.S = compute_similarity(
            train_df,
            self.cfg,
            n_users=self.n_users,
            n_items=self.n_items,
            force_reload=force_similarity,
        )

        #    User Item sparse matrix (CSR)                              
        users   = train_df["user_idx"].to_numpy(dtype=np.int32)
        items   = train_df["item_idx"].to_numpy(dtype=np.int32)
        ratings = train_df["rating"].to_numpy(dtype=np.float32)

        self.user_item_matrix = sp.csr_matrix(
            (ratings, (users, items)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )

        #    Statistics (train only)                                    
        # Item means: average over nonzero entries in each item column
        col_sums  = np.array(self.user_item_matrix.sum(axis=0)).ravel()  # (n_items,)
        col_counts = np.diff(self.user_item_matrix.tocsc().indptr)        # (n_items,)
        self.item_means = np.where(
            col_counts > 0, col_sums / np.maximum(col_counts, 1), 0.0
        ).astype(np.float32)

        # User means
        row_sums   = np.array(self.user_item_matrix.sum(axis=1)).ravel()  # (n_users,)
        row_counts = np.diff(self.user_item_matrix.indptr)                 # (n_users,)
        self.user_means = np.where(
            row_counts > 0, row_sums / np.maximum(row_counts, 1), 0.0
        ).astype(np.float32)

        # Global mean
        self.global_mean = float(
            train_df["rating"].sum() / max(len(train_df), 1)
        )

        self._is_fitted = True
        elapsed = time.perf_counter() - t0
        logger.info("ItemKNN.fit() done in %.1f s", elapsed)
        logger.info(
            "  n_users=%d  n_items=%d  global_mean=%.4f",
            self.n_users, self.n_items, self.global_mean,
        )
        return self

    # ------------------------------------------------------------------
    # Rating prediction
    # ------------------------------------------------------------------

    def predict(self, user_idx: int, item_idx: int) -> float:
        """
        Predict the rating for a single (user, item) pair.

        Uses mean-centred weighted average over the user's rated neighbours
        of the target item.
        """
        return float(self.predict_batch(
            np.array([user_idx], dtype=np.int32),
            np.array([item_idx], dtype=np.int32),
        )[0])

    def predict_batch(
        self,
        user_idx_array: np.ndarray,
        item_idx_array: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorised rating prediction for many (user, item) pairs.

        For each pair (u, i):
          1. Retrieve the top-K neighbours of i from S[i, :].
          2. For each neighbour j that u has rated, compute
             sim(i,j)   (r_uj    _j).
          3. r  =  _i +   numerator /   |sim(i,j)|
          4. Fallback chain: item mean -> user mean -> global mean.
          5. Clip to [rating_min, rating_max].

        Parameters
        ----------
        user_idx_array : np.ndarray [int32], shape (N,)
        item_idx_array : np.ndarray [int32], shape (N,)

        Returns
        -------
        preds : np.ndarray [float32], shape (N,)
        """
        self._check_fitted()

        user_idx_array = np.asarray(user_idx_array, dtype=np.int32)
        item_idx_array = np.asarray(item_idx_array, dtype=np.int32)
        N = len(user_idx_array)
        preds = np.empty(N, dtype=np.float32)

        # Bounds check: mark out-of-range as "cold"
        valid_u = (user_idx_array >= 0) & (user_idx_array < self.n_users)
        valid_i = (item_idx_array >= 0) & (item_idx_array < self.n_items)

        # --- Cold-start fallback for invalid indices ---
        cold_mask = ~(valid_u & valid_i)
        preds[cold_mask] = self.global_mean

        # Group pairs by unique items to amortise neighbour look-up
        # (For a target item i, S[i, :] gives neighbours for ALL users at once)
        pairs_mask = ~cold_mask
        if not pairs_mask.any():
            return np.clip(preds, self.cfg.rating_min, self.cfg.rating_max).astype(np.float32)

        valid_pairs = np.where(pairs_mask)[0]
        u_arr = user_idx_array[valid_pairs]
        i_arr = item_idx_array[valid_pairs]

        # Iterate over unique target items (typically much fewer than pairs)
        unique_items = np.unique(i_arr)
        # Map from item_idx -> positions in valid_pairs array
        item_to_pos = {item: [] for item in unique_items}
        for pos, item in zip(range(len(u_arr)), i_arr):
            item_to_pos[int(item)].append(pos)

        S_csr = self.S.tocsr()
        UIM   = self.user_item_matrix   # user item CSR

        for item_i, positions in item_to_pos.items():
            positions = np.array(positions, dtype=np.int32)
            users_i   = u_arr[positions]

            # Neighbours of item i (sparse row)
            nbr_row = S_csr.getrow(item_i)
            if nbr_row.nnz == 0:
                # No neighbours -> item mean fallback
                preds[valid_pairs[positions]] = self.item_means[item_i]
                continue

            nbr_idx = nbr_row.indices          # neighbour item indices
            nbr_sim = nbr_row.data             # similarity values

            mu_i       = self.item_means[item_i]
            mu_nbrs    = self.item_means[nbr_idx]          # shape (K,)

            # For each user in users_i, look up their ratings for nbr_idx
            # UIM[user, :][:, nbr_idx] -- iterate per unique user in users_i
            unique_users_i, user_inv = np.unique(users_i, return_inverse=True)
            for ui_pos, u in enumerate(unique_users_i):
                user_row = UIM.getrow(int(u))
                # Ratings for neighbour items
                r_nbrs = np.asarray(user_row[:, nbr_idx].todense()).ravel()   # shape (K,)
                rated_mask = r_nbrs > 0

                idx_in_positions = positions[user_inv == ui_pos]

                if not rated_mask.any():
                    # No rated neighbours -> item mean -> user mean fallback
                    fallback = mu_i if mu_i > 0 else self.user_means[int(u)]
                    if fallback == 0:
                        fallback = self.global_mean
                    preds[valid_pairs[idx_in_positions]] = fallback
                    continue

                sim_rated  = nbr_sim[rated_mask]          # subset of sims
                r_uj       = r_nbrs[rated_mask]            # user's ratings for those
                mu_j       = mu_nbrs[rated_mask]           # item means of neighbours

                numerator   = float((sim_rated * (r_uj - mu_j)).sum())
                denominator = float(np.abs(sim_rated).sum())

                pred = mu_i + (numerator / max(denominator, 1e-8))
                preds[valid_pairs[idx_in_positions]] = pred

        return np.clip(preds, self.cfg.rating_min, self.cfg.rating_max).astype(np.float32)

    # ------------------------------------------------------------------
    # Top-N recommendation
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_idx: int,
        N: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """
        Return the top-N recommended items for a user.

        Scoring:
          "weighted_sum"  : score(i) = u_ratings_vec   S[:, i]
          "mean_centered" : score(i) = (u_ratings_vec - item_means[user_rated])   S[:, i] +  _i

        Parameters
        ----------
        user_idx : int
        N : int
        exclude_seen : bool  -- exclude items already rated in train.

        Returns
        -------
        list of (item_idx, score), sorted by score descending.
        """
        self._check_fitted()

        if user_idx < 0 or user_idx >= self.n_users:
            return []   # cold-start user

        user_vec = self.user_item_matrix.getrow(user_idx)  # (1, n_items) CSR

        if self.cfg.topn_mode == "mean_centered":
            # Subtract item means from user ratings (sparse)
            u_coo = user_vec.tocoo()
            adjusted_data = u_coo.data - self.item_means[u_coo.col]
            user_vec_adj = sp.csr_matrix(
                (adjusted_data, (u_coo.row, u_coo.col)),
                shape=user_vec.shape, dtype=np.float32,
            )
            mat = user_vec_adj @ self.S  # (1, n_items)
            arr = mat.toarray() if sp.issparse(mat) else np.asarray(mat)
            scores = arr.ravel() + self.item_means
        else:
            # weighted_sum: scores = user_ratings @ S  (1   n_items)
            mat = user_vec @ self.S
            arr = mat.toarray() if sp.issparse(mat) else np.asarray(mat)
            scores = arr.ravel()  # (n_items,)

        scores = scores.astype(np.float32)

        if exclude_seen:
            seen_items = user_vec.indices
            scores[seen_items] = -np.inf

        # Top-N via argpartition (O(n), avoids full sort)
        if N >= self.n_items:
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, N)[:N]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [(int(idx), float(scores[idx])) for idx in top_idx if not np.isinf(scores[idx])]

    def recommend_batch(
        self,
        user_indices: list[int] | np.ndarray,
        N: int = 10,
        exclude_seen: bool = True,
    ) -> dict[int, list[tuple[int, float]]]:
        """
        Recommend for multiple users efficiently.

        Returns
        -------
        dict mapping user_idx -> list of (item_idx, score)
        """
        self._check_fitted()
        results = {}

        user_indices = np.asarray(user_indices, dtype=np.int32)
        valid = user_indices[(user_indices >= 0) & (user_indices < self.n_users)]

        # Batch the user item submatrix
        U_sub = self.user_item_matrix[valid, :]   # (batch, n_items)

        if self.cfg.topn_mode == "mean_centered":
            # Subtract item means from ratings (sparse, vectorised)
            U_sub_coo = U_sub.tocoo()
            U_sub_coo.data -= self.item_means[U_sub_coo.col]
            U_sub = U_sub_coo.tocsr()
            mat = U_sub @ self.S
            arr = mat.toarray() if sp.issparse(mat) else np.asarray(mat)
            all_scores = arr.astype(np.float32)
            all_scores += self.item_means[np.newaxis, :]
        else:
            mat = U_sub @ self.S
            arr = mat.toarray() if sp.issparse(mat) else np.asarray(mat)
            all_scores = arr.astype(np.float32)  # (batch, n_items)

        for bi, uid in enumerate(valid):
            uid = int(uid)
            scores = all_scores[bi]
            if exclude_seen:
                seen = self.user_item_matrix.getrow(uid).indices
                scores = scores.copy()
                scores[seen] = -np.inf
            if N >= self.n_items:
                top_idx = np.argsort(-scores)
            else:
                top_idx = np.argpartition(-scores, N)[:N]
                top_idx = top_idx[np.argsort(-scores[top_idx])]
            results[uid] = [
                (int(idx), float(scores[idx]))
                for idx in top_idx
                if not np.isinf(scores[idx])
            ]
        return results

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """
        Serialise the fitted model to disk.
        The similarity matrix is saved as .npz; the rest as pickle.
        """
        self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        sp.save_npz(str(path / "similarity.npz"), self.S)
        sp.save_npz(str(path / "user_item.npz"),  self.user_item_matrix)
        state = {
            "n_users":     self.n_users,
            "n_items":     self.n_items,
            "item_means":  self.item_means,
            "user_means":  self.user_means,
            "global_mean": self.global_mean,
            "cfg":         self.cfg,
        }
        with open(path / "state.pkl", "wb") as f:
            pickle.dump(state, f)
        logger.info("Model saved to %s", path)

    def load(self, path: Path | str) -> "ItemKNN":
        """Load a previously saved model from disk."""
        path = Path(path)
        self.S                = sp.load_npz(str(path / "similarity.npz"))
        self.user_item_matrix = sp.load_npz(str(path / "user_item.npz"))
        with open(path / "state.pkl", "rb") as f:
            state = pickle.load(f)
        self.n_users     = state["n_users"]
        self.n_items     = state["n_items"]
        self.item_means  = state["item_means"]
        self.user_means  = state["user_means"]
        self.global_mean = state["global_mean"]
        self.cfg         = state["cfg"]
        self._is_fitted  = True
        logger.info(
            "Model loaded from %s  (n_users=%d, n_items=%d)",
            path, self.n_users, self.n_items,
        )
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

    def model_size_bytes(self) -> int:
        """Return approximate disk size of the saved model."""
        total = 0
        if self.S is not None:
            total += self.S.data.nbytes + self.S.indices.nbytes + self.S.indptr.nbytes
        if self.user_item_matrix is not None:
            m = self.user_item_matrix
            total += m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
        if self.item_means is not None:
            total += self.item_means.nbytes
        if self.user_means is not None:
            total += self.user_means.nbytes
        return total
