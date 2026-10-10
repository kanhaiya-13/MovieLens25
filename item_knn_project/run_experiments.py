"""
run_experiments.py
==================
CLI entry-point for the Item-KNN comparative study.

Usage
-----
  # Grid search on validation set (finds best hyperparams)
  python run_experiments.py --mode grid [--sample] [--device cuda]

  # Retrain best config and evaluate on test set
  python run_experiments.py --mode final [--sample] [--device cuda]

  # Both in one go
  python run_experiments.py --mode all [--sample]

Outputs (written to item_knn_project/results/)
-------
  grid_results.csv        -- one row per (sim, K, shrinkage) on validation
  final_metrics.json      -- best config evaluated on test (full detail)
  final_metrics.csv       -- flat one-row-per-model format for comparison table
  model/                  -- saved model artefacts
  examples.txt            -- demo recommendations
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

# Ensure project root is in path when run from anywhere
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, DEFAULT_CONFIG
from data_loader import load_data, load_movies
from split import split_data, load_splits
from item_knn import ItemKNN
from evaluate import evaluate_model, evaluate_ratings, evaluate_ranking
from similarity import validate_pearson_approximation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _peak_ram_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def _print_table(rows: list[dict], title: str = "") -> None:
    if not rows:
        return
    if title:
        print(f"\n{'='*70}\n{title}\n{'='*70}")
    keys = list(rows[0].keys())
    col_widths = {k: max(len(str(k)), max(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    header = "  ".join(str(k).ljust(col_widths[k]) for k in keys)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row.get(k, "")).ljust(col_widths[k]) for k in keys))


def _save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %s (%d rows)", path.name, len(rows))


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid_search(cfg: Config, train_df: pd.DataFrame, val_df: pd.DataFrame,
                    n_users: int, n_items: int, movies_df: pd.DataFrame | None) -> list[dict]:
    """
    Exhaustive grid search on the VALIDATION set.
    Returns list of result dicts (one per config).
    """
    results = []
    combinations = [
        (sim, k, sh)
        for sim in cfg.grid_similarities
        for k   in cfg.grid_k_neighbors
        for sh  in cfg.grid_shrinkages
    ]
    total = len(combinations)
    logger.info("Grid search: %d combinations on validation set", total)

    for idx, (sim, k, sh) in enumerate(combinations, start=1):
        logger.info("[%d/%d] sim=%s  K=%d  shrink=%.0f", idx, total, sim, k, sh)
        exp_cfg = cfg.copy_with(similarity=sim, k_neighbors=k, shrinkage_lambda=sh)

        model = ItemKNN(exp_cfg)
        t0 = time.perf_counter()
        model.fit(train_df, n_users=n_users, n_items=n_items)
        fit_time = time.perf_counter() - t0

        ram = _peak_ram_mb()

        # Rating metrics on val (up to 100k pairs for grid search efficiency)
        val_eval_df = val_df if len(val_df) <= 100_000 else val_df.sample(100_000, random_state=cfg.random_seed)
        rating_m = evaluate_ratings(model, val_eval_df, exp_cfg)

        # Ranking metrics on val (reduced eval for speed in grid)
        rank_cfg = exp_cfg.copy_with(eval_sample_users=min(cfg.eval_sample_users, 3000))
        ranking_m = evaluate_ranking(model, train_df, val_df, rank_cfg, movies_df)

        row = {
            "similarity":       sim,
            "k_neighbors":      k,
            "shrinkage_lambda": sh,
            "rmse":             round(rating_m["rmse"], 4),
            "mae":              round(rating_m["mae"],  4),
            "ndcg@10":          round(ranking_m.get("ndcg@10", 0), 4),
            "recall@10":        round(ranking_m.get("recall@10", 0), 4),
            "precision@10":     round(ranking_m.get("precision@10", 0), 4),
            "map@10":           round(ranking_m.get("map@10", 0), 4),
            "fit_time_s":       round(fit_time, 1),
            "peak_ram_mb":      round(ram, 0),
        }
        results.append(row)
        logger.info(
            "  RMSE=%.4f  MAE=%.4f  NDCG@10=%.4f  Recall@10=%.4f",
            row["rmse"], row["mae"], row["ndcg@10"], row["recall@10"],
        )

    return results


def _pick_best(results: list[dict], metric: str) -> dict:
    """Return the config row with the highest value for `metric`."""
    # Lower is better for RMSE/MAE
    if metric in ("rmse", "mae"):
        return min(results, key=lambda r: r[metric])
    return max(results, key=lambda r: r[metric])


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

def run_final(
    cfg: Config,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    movies_df: pd.DataFrame | None,
    best_config: dict | None = None,
) -> dict:
    """
    Retrain on the BEST config and evaluate on TEST set.
    """
    if best_config is not None:
        cfg = cfg.copy_with(
            similarity       = best_config["similarity"],
            k_neighbors      = int(best_config["k_neighbors"]),
            shrinkage_lambda = float(best_config["shrinkage_lambda"]),
        )
        logger.info(
            "Final eval with best config: sim=%s  K=%d  shrink=%.0f",
            cfg.similarity, cfg.k_neighbors, cfg.shrinkage_lambda,
        )
    else:
        logger.info("Final eval with default config: %s", cfg)

    model = ItemKNN(cfg)
    t0 = time.perf_counter()
    model.fit(train_df, n_users=n_users, n_items=n_items, force_similarity=True)
    fit_time = time.perf_counter() - t0
    peak_ram = _peak_ram_mb()

    # Save model
    model_path = cfg.results / "model"
    model.save(model_path)
    model_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())

    metrics = evaluate_model(
        model, train_df, test_df, cfg, movies_df,
        fit_time_s=fit_time, peak_ram_mb=peak_ram, model_name="item_knn",
    )
    metrics["model_size_bytes"] = model_size

    # Pearson validation (if similarity is pearson)
    if cfg.similarity == "pearson":
        from similarity import compute_similarity
        S = model.S
        pv = validate_pearson_approximation(train_df, S, cfg, n_pairs=cfg.pearson_validation_pairs)
        metrics.update({f"pearson_val_{k}": v for k, v in pv.items()})

    return metrics, model


# ---------------------------------------------------------------------------
# Demo output
# ---------------------------------------------------------------------------

def show_recommendations(
    model: ItemKNN,
    user_idx: int,
    item_map: pd.DataFrame,
    movies_df: pd.DataFrame,
    train_df: pd.DataFrame,
    N: int = 10,
) -> str:
    """Return a formatted string of top-rated train items and recommendations."""
    # Build item_idx -> title/genres lookup
    if "item_idx" in movies_df.columns:
        idx_to_movie = movies_df.set_index("item_idx")[["title", "genres"]].to_dict("index")
    else:
        idx_to_movie = {}

    # User's top-5 train ratings
    user_train = train_df[train_df["user_idx"] == user_idx].nlargest(5, "rating")
    lines = [f"\n{'='*60}", f"  User {user_idx} -- Top-5 rated movies (train)", f"{'='*60}"]
    for _, row in user_train.iterrows():
        m = idx_to_movie.get(int(row["item_idx"]), {"title": "Unknown", "genres": ""})
        lines.append(f"  * {row['rating']:.1f}  {m['title']}  [{m['genres']}]")

    # Recommendations
    recs = model.recommend(user_idx, N=N, exclude_seen=True)
    lines.append(f"\n  Top-{N} recommendations:")
    lines.append(f"  {'Rank':<5} {'Score':>7}  Title  [Genres]")
    lines.append("  " + "-" * 55)
    for rank, (item_idx, score) in enumerate(recs, start=1):
        m = idx_to_movie.get(item_idx, {"title": "Unknown", "genres": ""})
        lines.append(f"  {rank:<5} {score:>7.3f}  {m['title']}  [{m['genres']}]")

    return "\n".join(lines)


def similar_movies(
    model: ItemKNN,
    title: str,
    movies_df: pd.DataFrame,
    N: int = 10,
) -> str:
    """Return a formatted string of N most similar movies to `title`."""
    if "item_idx" not in movies_df.columns:
        return "item_idx column not found in movies_df"

    # Find item_idx for title
    match = movies_df[movies_df["title"].str.contains(title, case=False, na=False)]
    if match.empty:
        return f"Movie '{title}' not found in dataset."

    row = match.iloc[0]
    seed_idx = int(row["item_idx"])
    idx_to_movie = movies_df.set_index("item_idx")[["title", "genres"]].to_dict("index")

    import scipy.sparse as sp
    S = model.S.tocsr()
    sim_row = S.getrow(seed_idx)
    if sim_row.nnz == 0:
        return f"No neighbours found for '{row['title']}'."

    nbr_idx = sim_row.indices
    nbr_sim = sim_row.data
    order   = np.argsort(-nbr_sim)[:N]

    lines = [f"\n{'='*60}", f"  Movies similar to: {row['title']}  [{row['genres']}]", f"{'='*60}"]
    for rank, pos in enumerate(order, start=1):
        m = idx_to_movie.get(int(nbr_idx[pos]), {"title": "Unknown", "genres": ""})
        lines.append(f"  {rank:<5} sim={nbr_sim[pos]:.4f}  {m['title']}  [{m['genres']}]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Item-KNN CF on MovieLens 25M -- grid search + final evaluation"
    )
    p.add_argument("--mode",   choices=["grid", "final", "all"], default="all",
                   help="grid=grid search only; final=best+test eval; all=both")
    p.add_argument("--sample", type=int, default=0, metavar="N",
                   help="Use first N ratings (0=full dataset)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                   help="Compute device for similarity (cuda requires torch)")
    p.add_argument("--config", type=str, default=None,
                   help="Optional path to a config JSON file (overrides defaults)")
    p.add_argument("--best-metric", type=str, default=None,
                   help="Metric to select best grid config (default: cfg.grid_best_metric)")
    p.add_argument("--force-reload", action="store_true",
                   help="Ignore all parquet caches and reprocess from scratch")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    #    Setup                                                          
    cfg = DEFAULT_CONFIG.copy_with(
        sample_n = args.sample,
        device   = args.device,
    )
    if args.best_metric:
        cfg = cfg.copy_with(grid_best_metric=args.best_metric)
    if args.config:
        # Optional: load JSON overrides
        with open(args.config) as f:
            overrides = json.load(f)
        cfg = cfg.copy_with(**overrides)

    cfg.ensure_dirs()

    log_path = cfg.results / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_path), mode="a"),
        ],
    )
    logger.info("Starting run -- mode=%s  sample=%d  device=%s", args.mode, args.sample, args.device)

    #    Load data                                                      
    ratings_df, user_map, item_map = load_data(cfg, force_reload=args.force_reload)
    n_users = int(user_map.shape[0])
    n_items = int(item_map.shape[0])
    logger.info("Dataset: %d users, %d items, %d ratings", n_users, n_items, len(ratings_df))

    train_df, val_df, test_df = split_data(ratings_df, cfg, force_reload=args.force_reload)
    logger.info("Splits: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    movies_df = load_movies(cfg)

    best_config: dict | None = None

    #    Grid search                                                    
    if args.mode in ("grid", "all"):
        grid_results = run_grid_search(cfg, train_df, val_df, n_users, n_items, movies_df)

        grid_csv = cfg.results / "grid_results.csv"
        _save_csv(grid_results, grid_csv)
        print(f"\nGrid results saved to {grid_csv}")
        _print_table(grid_results, title="Grid Search Results (Validation Set)")

        # Best by NDCG@10 (primary)
        best_ndcg = _pick_best(grid_results, cfg.grid_best_metric)
        best_rmse = _pick_best(grid_results, "rmse")

        print(f"\n{'='*60}")
        print(f"  Best by {cfg.grid_best_metric}: sim={best_ndcg['similarity']}  "
              f"K={best_ndcg['k_neighbors']}  shrink={best_ndcg['shrinkage_lambda']}  "
              f"-> {cfg.grid_best_metric}={best_ndcg[cfg.grid_best_metric]}")
        print(f"  Best by RMSE:     sim={best_rmse['similarity']}  "
              f"K={best_rmse['k_neighbors']}  shrink={best_rmse['shrinkage_lambda']}  "
              f"-> RMSE={best_rmse['rmse']}")
        print(f"{'='*60}")
        note = ""
        if best_ndcg != best_rmse:
            note = "NOTE: Best NDCG and best RMSE configs differ -- using best NDCG for final run."
            print(f"\n  [!]  {note}\n")

        best_config = best_ndcg
        # Save best config info
        with open(cfg.results / "best_config.json", "w") as f:
            json.dump({"best_metric": cfg.grid_best_metric,
                       "best_config": best_config,
                       "best_rmse_config": best_rmse,
                       "note": note}, f, indent=2)

    #    Final evaluation                                               
    if args.mode in ("final", "all"):
        if best_config is None and (cfg.results / "best_config.json").exists():
            with open(cfg.results / "best_config.json") as f:
                best_config = json.load(f)["best_config"]
            logger.info("Loaded best config from best_config.json")

        final_metrics, model = run_final(
            cfg, train_df, test_df, n_users, n_items, movies_df, best_config
        )

        # Save JSON
        json_path = cfg.results / "final_metrics.json"
        with open(json_path, "w") as f:
            # Convert numpy types for JSON serialisation
            json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in final_metrics.items()}, f, indent=2)
        logger.info("Final metrics saved to %s", json_path)

        # Save CSV (one-row-per-model flat format)
        csv_path = cfg.results / "final_metrics.csv"
        _save_csv([{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in final_metrics.items()}], csv_path)

        # Print summary table
        summary_keys = [
            "model", "similarity", "k_neighbors", "shrinkage_lambda",
            "rmse", "mae",
            "ndcg@10", "recall@10", "precision@10", "map@10",
            "coverage@10", "diversity@10", "novelty@10",
            "fit_time_s", "peak_ram_mb",
        ]
        summary = [{k: round(v, 4) if isinstance(v, float) else v
                    for k, v in final_metrics.items() if k in summary_keys}]
        _print_table(summary, title="Final Test-Set Results")

        #    Demo output                                                
        examples_path = cfg.results / "examples.txt"
        examples_lines = []

        # Pick a user who has   20 train ratings
        sample_users = train_df.groupby("user_idx").size()
        good_users = sample_users[sample_users >= 20].index
        demo_user = int(good_users[0]) if len(good_users) > 0 else 0

        rec_str = show_recommendations(model, demo_user, item_map, movies_df, train_df, N=10)
        examples_lines.append(rec_str)
        print(rec_str)

        sim_str = similar_movies(model, "Toy Story", movies_df, N=10)
        examples_lines.append(sim_str)
        print(sim_str)

        with open(examples_path, "w", encoding="utf-8") as f:
            f.write("\n".join(examples_lines))
        logger.info("Examples saved to %s", examples_path)

    logger.info("All done.")


if __name__ == "__main__":
    main()
