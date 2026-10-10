"""
config.py
=========
Central configuration for the Item-KNN project.
All hyperparameters and paths live here; other modules import from this file.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths  (all relative to the repo root so the project is portable)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent          # d:\MovieLens25
DATA_RAW  = REPO_ROOT / "ml-25m"                  # data lives directly here
DATA_PROC = REPO_ROOT / "data" / "processed"
RESULTS   = Path(__file__).parent / "results"


@dataclass
class Config:
    #    Paths                                                               
    data_raw: Path  = DATA_RAW
    data_proc: Path = DATA_PROC
    results:  Path  = RESULTS

    #    Data filtering                                                      
    min_ratings_per_user: int   = 20
    min_ratings_per_item: int   = 50

    #    Split                                                               
    train_frac: float  = 0.80
    val_frac:   float  = 0.10
    # test_frac  = 1 - train_frac - val_frac
    min_ratings_to_split: int = 3   # users with fewer go entirely to train
    random_seed: int = 42

    #    Similarity                                                          
    similarity: str  = "adjusted_cosine"   # cosine | adjusted_cosine | pearson
    k_neighbors: int = 50                  # top-K neighbors per item
    block_size:  int = 1000                # items per block during computation
    min_common_users: int = 5              # co-rating support threshold
    shrinkage_lambda: float = 100.0        # shrinkage regularizer

    #    Model                                                               
    rating_min: float = 0.5
    rating_max: float = 5.0
    topn_mode: str    = "weighted_sum"     # weighted_sum | mean_centered

    #    Evaluation                                                          
    relevance_threshold: float  = 4.0
    topn_ks: list[int]          = field(default_factory=lambda: [5, 10, 20])
    eval_sample_users: int      = 10_000  # 0 = all users
    eval_seed: int              = 42

    #    Grid search                                                         
    grid_similarities: list[str]   = field(default_factory=lambda: ["cosine", "adjusted_cosine", "pearson"])
    grid_k_neighbors:  list[int]   = field(default_factory=lambda: [20, 50, 100])
    grid_shrinkages:   list[float] = field(default_factory=lambda: [0.0, 50.0, 100.0])
    grid_best_metric: str          = "ndcg@10"   # metric used to pick best config

    #    Runtime                                                             
    device: str   = "cpu"    # cpu | cuda
    n_jobs: int   = -1       # -1 = all cores (used where joblib is applicable)
    sample_n: int = 0        # 0 = full dataset; >0 = take first sample_n ratings

    #    Misc                                                                
    log_level: str = "INFO"

    #    Pearson validation                                                  
    pearson_validation_pairs: int = 100   # random pairs to validate approx vs exact

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.data_proc.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)

    def copy_with(self, **kwargs) -> "Config":
        """Return a shallow copy with overridden fields."""
        import copy
        c = copy.copy(self)
        for k, v in kwargs.items():
            setattr(c, k, v)
        return c


# Default singleton used when no custom config is supplied
DEFAULT_CONFIG = Config()
