# Implementation Plan: Item-KNN Collaborative Filtering on MovieLens 25M

## 1. Project Layout

```
d:\MovieLens25\item_knn_project\
  config.py            ← All hyperparams & paths (dataclass + argparse bridge)
  data_loader.py       ← Read CSV → filter → remap IDs → parquet cache
  split.py             ← Per-user temporal 80/10/10 split → parquet cache
  similarity.py        ← Blockwise top-K item-item similarity (sparse CSR)
  item_knn.py          ← ItemKNN class: fit / predict / recommend / save / load
  evaluate.py          ← Model-agnostic metrics (RMSE, MAE, NDCG, MAP, …)
  run_experiments.py   ← CLI: grid search + final run + tables
  visualize.py         ← Matplotlib plots → results/
  tests/
    test_split.py
    test_similarity.py
    test_model.py
    test_metrics.py
  requirements.txt
  README.md
  results/             ← CSVs, JSON, PNGs, saved model, examples.txt
data/
  raw/ml-25m/          ← symlinked or copied from d:\MovieLens25\ml-25m\
  processed/           ← parquet caches (ratings, splits, mappings)
```

## 2. Data Flow

```
ratings.csv  ──→  data_loader.py
                  │ 1. Read with pandas pyarrow engine
                  │ 2. Filter: min_ratings_per_user=20, min_ratings_per_item=50
                  │    (iterative until stable)
                  │ 3. Remap userId, movieId → 0-based int32 indices
                  │ 4. Cache → data/processed/ratings_filtered.parquet
                  │ 5. Save mappings → data/processed/user_map.parquet,
                  │                    data/processed/item_map.parquet
                  ↓
             split.py
                  │ 6. Per-user temporal sort
                  │ 7. 80% train / 10% val / 10% test
                  │ 8. Save → data/processed/train.parquet,
                  │           data/processed/val.parquet,
                  │           data/processed/test.parquet
                  ↓
            similarity.py  (uses TRAIN ONLY)
                  │ 9. Build sparse item×user CSR matrix R (float32)
                  │ 10. Blockwise similarity (cosine/adj_cosine/pearson)
                  │ 11. Apply min_common_users filter + shrinkage
                  │ 12. Keep top-K per item, save sparse CSR
                  ↓
             item_knn.py
                  │ 13. fit(train_df) → calls similarity
                  │ 14. predict_batch(users, items)
                  │ 15. recommend(user, N)
                  ↓
             evaluate.py
                  │ 16. RMSE, MAE on test pairs
                  │ 17. Precision/Recall/F1/HitRate/NDCG/MAP @K
                  │ 18. Coverage, diversity, novelty
                  │ 19. Efficiency stats
                  ↓
            run_experiments.py → results/grid_results.csv
                               → results/final_metrics.json
                               → results/final_metrics.csv
```

## 3. Key Design Decisions

### Memory / Scalability
- **Never build dense U×I matrix.** Everything is scipy.sparse CSR/CSC (float32, int32).
- Similarity computed in **blocks of 1000 items**: for block B, compute `(R_B.T @ R).toarray()`, keep only top-K neighbors, free the block.
- Peak RAM during similarity: O(block_size × n_items × 4 bytes) ≈ 1000 × 62000 × 4 = 248 MB per block — well within 64 GB.
- Predictions vectorized: for a user's rating vector (sparse), multiply by similarity matrix (sparse CSR) → get weighted scores for all items at once.

### Similarity Variants
- **Cosine**: normalize each item's rating vector, then dot product.
- **Adjusted Cosine**: subtract *user* mean from each rating, then cosine. This captures that users rate on different scales.
- **Pearson (approx)**: subtract *item* mean, then cosine. Efficient sparse approximation — documented with exact Pearson validation on 100 random pairs.

### Shrinkage
- `sim_shrunk = sim_raw × n_common / (n_common + lambda)` where lambda=100.
- Prevents items with only 1-2 co-raters from having artificially high similarity.

### Temporal Split (no leakage)
- For each user: sort ratings by timestamp ASC, slice [0:80%] → train, [80%:90%] → val, [90%:] → test.
- Users with < 3 ratings go entirely to train (can't split meaningfully).
- All statistics (means, similarities) computed from train only.

### Evaluation
- Rating prediction: RMSE, MAE on all (user, item) pairs in test.
- Top-N: rank ALL unseen items (not just sampled negatives) for proper NDCG/Recall.
- Sample 10,000 users (seed=42) for ranking eval if full set is too slow.
- Relevance threshold: rating ≥ 4.0.

### Common Interface (for other models)
```python
class BaseRecommender:
    def fit(self, train_df: pd.DataFrame) -> None: ...
    def predict_batch(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray: ...
    def recommend(self, user_idx: int, N: int, exclude_seen: bool = True) -> list[tuple[int, float]]: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```
`evaluate.py` only calls `predict_batch` and `recommend`, so any model implementing those works.

## 4. Build Order
1. `requirements.txt` + `config.py`
2. `data_loader.py` — test on 1M sample
3. `split.py` — verify no leakage with unit test
4. `similarity.py` — blockwise, all 3 variants
5. `item_knn.py` — fit, predict, recommend, save/load
6. `evaluate.py` — all metrics
7. `run_experiments.py` — grid + final
8. `visualize.py` — plots
9. `tests/` — pytest suite
10. `README.md`
11. Run end-to-end, fix errors, report real results

## 5. Expected Runtime Estimates (full 25M, i9-13900K)
- Data loading + filtering: ~60s
- Split: ~30s
- Similarity (cosine, K=50, block=1000): ~10–20 min (CPU, scipy)
- Grid search (9 configs × similarity): ~2–3 hrs total
- Evaluation (10K users, all items): ~5–10 min
- Total end-to-end for one config: ~30–40 min

## 6. Assumptions & Limitations
- Exact Pearson (co-rated only) is ≈ 3–5× slower than the sparse approx; we validate the approximation on 100 pairs and document the error.
- Cold-start: users/items not in train receive fallback predictions (global mean).
- GPU similarity path (--device cuda) uses PyTorch sparse matmul if available.
- The "all-items" ranking eval is the most rigorous but slowest; the sample of 10K users makes it feasible.
