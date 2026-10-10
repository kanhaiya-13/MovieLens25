"""
evaluate.py
===========
Model-agnostic evaluation for recommender systems.

Any model that exposes:
    predict_batch(user_idx_array, item_idx_array) -> np.ndarray[float32]
    recommend(user_idx, N, exclude_seen=True)     -> list[(item_idx, score)]
    recommend_batch(user_indices, N)              -> dict[int, list[(item_idx, score)]]

can be evaluated without modification to this file.

Metrics
-------
Rating prediction
    RMSE, MAE  (on all test (u,i) pairs)

Top-N ranking  (at K = 5, 10, 20 by default)
    Precision@K, Recall@K, F1@K, HitRate@K, NDCG@K, MAP@K
    Relevance threshold: rating >= cfg.relevance_threshold (default 4.0)
    Evaluated on users with >= 1 relevant test item.
    Candidates = ALL items not seen in train (most rigorous; not sampled negatives).

Beyond-accuracy
    Catalog coverage @K  -- fraction of items ever recommended
    Diversity @K         -- mean intra-list genre Jaccard dissimilarity
    Novelty @K           -- mean -log2(popularity) of recommended items

Efficiency
    Similarity build time, fit time, peak memory, inference time per user,
    model size on disk (if saved).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rating prediction metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    diff = y_true.astype(np.float64) - y_pred.astype(np.float64)
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true.astype(np.float64) - y_pred.astype(np.float64))))


def evaluate_ratings(
    model,
    test_df: pd.DataFrame,
    cfg: Config = DEFAULT_CONFIG,
) -> dict[str, float]:
    """
    Evaluate rating prediction on all (u, i) pairs in test_df.

    Parameters
    ----------
    model : any object with predict_batch()
    test_df : pd.DataFrame  [user_idx, item_idx, rating]

    Returns
    -------
    dict with 'rmse', 'mae', 'n_pairs'.
    """
    u_arr = test_df["user_idx"].to_numpy(dtype=np.int32)
    i_arr = test_df["item_idx"].to_numpy(dtype=np.int32)
    y_true = test_df["rating"].to_numpy(dtype=np.float32)

    logger.info("Predicting %s test pairs ...", f"{len(u_arr):,}")
    t0 = time.perf_counter()
    y_pred = model.predict_batch(u_arr, i_arr)
    elapsed = time.perf_counter() - t0

    result = {
        "rmse":           rmse(y_true, y_pred),
        "mae":            mae(y_true, y_pred),
        "n_pairs":        int(len(y_true)),
        "pred_time_s":    elapsed,
    }
    logger.info("  RMSE=%.4f  MAE=%.4f  ({:,} pairs in {:.1f}s)".format(
        len(y_true), elapsed).format(**{}),
    )
    logger.info("  RMSE=%.4f  MAE=%.4f", result["rmse"], result["mae"])
    return result


# ---------------------------------------------------------------------------
# Ranking metrics helpers
# ---------------------------------------------------------------------------

def _dcg(relevances: np.ndarray) -> float:
    """Discounted Cumulative Gain for a ranked list of binary relevances."""
    if len(relevances) == 0:
        return 0.0
    positions = np.arange(1, len(relevances) + 1, dtype=np.float64)
    return float(np.sum(relevances / np.log2(positions + 1)))


def _ndcg_at_k(recommended: list[int], relevant_set: set[int], k: int) -> float:
    """NDCG@K for a single user."""
    hits = np.array([1.0 if r in relevant_set else 0.0 for r in recommended[:k]])
    dcg  = _dcg(hits)
    # Ideal: relevant items ranked first
    ideal = np.zeros(k)
    n_rel = min(len(relevant_set), k)
    ideal[:n_rel] = 1.0
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def _ap_at_k(recommended: list[int], relevant_set: set[int], k: int) -> float:
    """Average Precision@K for a single user."""
    if not relevant_set:
        return 0.0
    hits  = 0
    score = 0.0
    for pos, item in enumerate(recommended[:k], start=1):
        if item in relevant_set:
            hits += 1
            score += hits / pos
    return score / min(len(relevant_set), k)


def _precision_at_k(recommended: list[int], relevant_set: set[int], k: int) -> float:
    hits = sum(1 for r in recommended[:k] if r in relevant_set)
    return hits / k


def _recall_at_k(recommended: list[int], relevant_set: set[int], k: int) -> float:
    if not relevant_set:
        return 0.0
    hits = sum(1 for r in recommended[:k] if r in relevant_set)
    return hits / len(relevant_set)


def _hit_rate_at_k(recommended: list[int], relevant_set: set[int], k: int) -> float:
    return float(any(r in relevant_set for r in recommended[:k]))


# ---------------------------------------------------------------------------
# Full ranking evaluation
# ---------------------------------------------------------------------------

def evaluate_ranking(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config = DEFAULT_CONFIG,
    movies_df: Optional[pd.DataFrame] = None,
) -> dict[str, float]:
    """
    Evaluate top-N ranking on the test set.

    Candidates = ALL items not seen in train (most rigorous).
    Relevance  = rating >= cfg.relevance_threshold.
    Users with no relevant test items are excluded.

    If cfg.eval_sample_users > 0, evaluates on a random sample (seed=cfg.eval_seed).

    Parameters
    ----------
    model        : any object with recommend_batch()
    train_df     : pd.DataFrame  [user_idx, item_idx, rating, timestamp]
    test_df      : pd.DataFrame  [user_idx, item_idx, rating, timestamp]
    cfg          : Config
    movies_df    : pd.DataFrame  [item_idx, title, genres]  (for diversity/novelty)

    Returns
    -------
    dict of metric_name -> value
    """
    logger.info("Building test relevance sets ...")
    # Users with at least 1 relevant test item
    rel_mask = test_df["rating"] >= cfg.relevance_threshold
    rel_test = test_df[rel_mask]

    users_with_rel = rel_test["user_idx"].unique()
    logger.info("  Users with >= 1 relevant test item: %s", f"{len(users_with_rel):,}")

    # Sample users if needed
    if cfg.eval_sample_users > 0 and len(users_with_rel) > cfg.eval_sample_users:
        rng = np.random.default_rng(cfg.eval_seed)
        eval_users = rng.choice(users_with_rel, size=cfg.eval_sample_users, replace=False)
        logger.info("  Sampling %d users for ranking eval (seed=%d)", cfg.eval_sample_users, cfg.eval_seed)
    else:
        eval_users = users_with_rel
        logger.info("  Evaluating all %d qualifying users", len(eval_users))

    # Pre-compute item popularity for novelty
    item_pop = train_df["item_idx"].value_counts().to_dict()
    n_train  = len(train_df)

    # Genre lookup for diversity
    genre_sets: Optional[dict[int, set[str]]] = None
    if movies_df is not None and "item_idx" in movies_df.columns and "genres" in movies_df.columns:
        genre_sets = {}
        for _, row in movies_df.iterrows():
            genres = set(row["genres"].split("|")) - {"(no genres listed)"}
            genre_sets[int(row["item_idx"])] = genres

    # Build per-user data structures
    train_items_per_user: dict[int, set[int]] = (
        train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    )
    test_rel_per_user: dict[int, set[int]] = (
        rel_test[rel_test["user_idx"].isin(eval_users)]
        .groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )

    max_k = max(cfg.topn_ks)

    # --- Batch recommend ---
    logger.info("Running recommend_batch for %d users ...", len(eval_users))
    t0 = time.perf_counter()
    recs_dict = model.recommend_batch(list(eval_users), N=max_k, exclude_seen=True)
    rec_time = time.perf_counter() - t0
    logger.info("  recommend_batch done in %.1f s  (%.2f ms/user)",
                rec_time, 1000 * rec_time / max(len(eval_users), 1))

    # --- Accumulate per-user metrics ---
    ks = sorted(cfg.topn_ks)
    metric_sums: dict[str, float] = {f"{m}@{k}": 0.0 for k in ks for m in
                                      ["precision", "recall", "f1", "hit", "ndcg", "map"]}
    n_eval = 0
    all_recommended_items: dict[int, set[int]] = {k: set() for k in ks}

    diversity_vals: dict[int, list[float]] = {k: [] for k in ks}
    novelty_vals:   dict[int, list[float]] = {k: [] for k in ks}

    for uid in eval_users:
        uid = int(uid)
        relevant = test_rel_per_user.get(uid, set())
        if not relevant:
            continue

        raw_recs = recs_dict.get(uid, [])
        rec_items = [item for item, _ in raw_recs]

        n_eval += 1
        for k in ks:
            rec_k = rec_items[:k]
            p  = _precision_at_k(rec_k, relevant, k)
            r  = _recall_at_k(rec_k, relevant, k)
            h  = _hit_rate_at_k(rec_k, relevant, k)
            nd = _ndcg_at_k(rec_k, relevant, k)
            ap = _ap_at_k(rec_k, relevant, k)
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

            metric_sums[f"precision@{k}"] += p
            metric_sums[f"recall@{k}"]    += r
            metric_sums[f"f1@{k}"]        += f1
            metric_sums[f"hit@{k}"]       += h
            metric_sums[f"ndcg@{k}"]      += nd
            metric_sums[f"map@{k}"]       += ap

            all_recommended_items[k].update(rec_k)

            # Novelty
            if item_pop:
                novs = [-np.log2(item_pop.get(it, 1) / n_train + 1e-10) for it in rec_k]
                if novs:
                    novelty_vals[k].append(float(np.mean(novs)))

            # Diversity (intra-list genre dissimilarity)
            if genre_sets and len(rec_k) > 1:
                dissims = []
                for a in range(len(rec_k)):
                    for b in range(a + 1, len(rec_k)):
                        ga = genre_sets.get(rec_k[a], set())
                        gb = genre_sets.get(rec_k[b], set())
                        union = len(ga | gb)
                        if union > 0:
                            jacc = len(ga & gb) / union
                            dissims.append(1.0 - jacc)
                if dissims:
                    diversity_vals[k].append(float(np.mean(dissims)))

    # --- Aggregate ---
    results: dict[str, float] = {}
    n = max(n_eval, 1)
    for metric, total in metric_sums.items():
        results[metric] = total / n

    n_items = int(train_df["item_idx"].max()) + 1
    for k in ks:
        results[f"coverage@{k}"] = len(all_recommended_items[k]) / n_items
        if novelty_vals[k]:
            results[f"novelty@{k}"] = float(np.mean(novelty_vals[k]))
        if diversity_vals[k]:
            results[f"diversity@{k}"] = float(np.mean(diversity_vals[k]))

    results["n_eval_users"]     = float(n_eval)
    results["rec_time_s"]       = rec_time
    results["rec_time_ms_user"] = 1000 * rec_time / max(n_eval, 1)

    # Log summary
    k_report = 10 if 10 in ks else ks[0]
    logger.info(
        "Ranking eval (%d users): Precision@%d=%.4f  Recall@%d=%.4f  "
        "NDCG@%d=%.4f  MAP@%d=%.4f",
        n_eval, k_report, results.get(f"precision@{k_report}", 0),
        k_report, results.get(f"recall@{k_report}", 0),
        k_report, results.get(f"ndcg@{k_report}", 0),
        k_report, results.get(f"map@{k_report}", 0),
    )
    return results


# ---------------------------------------------------------------------------
# Full evaluation  (ratings + ranking)
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config = DEFAULT_CONFIG,
    movies_df: Optional[pd.DataFrame] = None,
    fit_time_s: float = 0.0,
    sim_time_s: float = 0.0,
    peak_ram_mb: float = 0.0,
    model_name: str = "item_knn",
) -> dict[str, float]:
    """
    Run the full evaluation suite (rating + ranking) and return a flat dict
    of metrics.

    This dict can be directly written to results/final_metrics.csv so that
    all other models can append rows for the comparison table.
    """
    logger.info("=== Evaluating model: %s ===", model_name)

    rating_metrics  = evaluate_ratings(model, test_df, cfg)
    ranking_metrics = evaluate_ranking(model, train_df, test_df, cfg, movies_df)

    model_size = 0
    if hasattr(model, "model_size_bytes"):
        model_size = model.model_size_bytes()

    results = {
        "model":            model_name,
        "similarity":       getattr(getattr(model, "cfg", cfg), "similarity", ""),
        "k_neighbors":      getattr(getattr(model, "cfg", cfg), "k_neighbors", 0),
        "shrinkage_lambda": getattr(getattr(model, "cfg", cfg), "shrinkage_lambda", 0),
        **rating_metrics,
        **ranking_metrics,
        "fit_time_s":       fit_time_s,
        "sim_time_s":       sim_time_s,
        "peak_ram_mb":      peak_ram_mb,
        "model_size_bytes": model_size,
    }
    return results


# ---------------------------------------------------------------------------
# Toy-example unit-test helpers (used in tests/test_metrics.py)
# ---------------------------------------------------------------------------

def _toy_ndcg() -> None:
    """
    Hand-computed NDCG@3 example.
    Recommended: [A, B, C]  Relevant: {A, C}
    Gains:  1, 0, 1
    DCG  = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
    IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.631 = 1.631
    NDCG = 1.5 / 1.631   0.920
    """
    rec = [0, 1, 2]    # A=0, B=1, C=2
    rel = {0, 2}        # A and C are relevant
    expected = 1.5 / (1 / np.log2(2) + 1 / np.log2(3))
    actual   = _ndcg_at_k(rec, rel, k=3)
    assert abs(actual - expected) < 1e-6, f"NDCG mismatch: {actual} vs {expected}"


def _toy_map() -> None:
    """
    Hand-computed MAP@4.
    Recommended: [A, B, C, D]  Relevant: {A, C}
    Hits at positions 1, 3.
    Precisions: 1/1=1.0, 2/3 0.667
    AP = (1.0 + 0.667) / min(2,4) = 1.667/2 = 0.8335
    """
    rec = [0, 1, 2, 3]
    rel = {0, 2}
    expected = (1.0 + 2 / 3) / 2
    actual   = _ap_at_k(rec, rel, k=4)
    assert abs(actual - expected) < 1e-6, f"MAP mismatch: {actual} vs {expected}"


if __name__ == "__main__":
    # Quick self-test
    _toy_ndcg()
    _toy_map()
    print("Metric self-tests passed  ")
