"""
tests/test_metrics.py
=====================
Unit tests for evaluation metric functions against hand-computed toy examples.

Toy example for NDCG@3:
  recommended: [A, B, C]   relevant: {A, C}
  gains:  1, 0, 1
  DCG  = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
  IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
  NDCG = 1.5 / 1.6309   0.9197

Toy example for MAP@4:
  recommended: [A, B, C, D]   relevant: {A, C}
  Hits at positions 1 and 3 -> Precisions: 1/1, 2/3
  AP = (1.0 + 2/3) / 2 = 0.8333

Toy example for Precision/Recall/F1:
  recommended: [A, B, C, D, E]  K=5  relevant: {A, C, E}
  Precision@5 = 3/5 = 0.6
  Recall@5    = 3/3 = 1.0
  F1@5        = 2*0.6*1.0/(0.6+1.0) = 0.75

Toy example for RMSE / MAE:
  y_true = [1, 2, 3]  y_pred = [1.5, 2.5, 2.5]
  Errors = [0.5, 0.5, 0.5]
  MAE  = 0.5
  RMSE = 0.5
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate import (
    _ndcg_at_k,
    _ap_at_k,
    _precision_at_k,
    _recall_at_k,
    _hit_rate_at_k,
    rmse,
    mae,
)


# ---------------------------------------------------------------------------
# NDCG@K
# ---------------------------------------------------------------------------

def test_ndcg_toy():
    rec = [0, 1, 2]   # A=0 relevant, B=1 not, C=2 relevant
    rel = {0, 2}
    expected = 1.5 / (1 / np.log2(2) + 1 / np.log2(3))
    assert abs(_ndcg_at_k(rec, rel, k=3) - expected) < 1e-6


def test_ndcg_perfect():
    """All relevant items recommended first -> NDCG=1.0"""
    rec = [0, 1, 2, 3]
    rel = {0, 1}
    assert abs(_ndcg_at_k(rec, rel, k=2) - 1.0) < 1e-6


def test_ndcg_zero():
    """No relevant items in recommendations -> NDCG=0."""
    rec = [0, 1]
    rel = {2, 3}
    assert _ndcg_at_k(rec, rel, k=2) == 0.0


def test_ndcg_empty_recs():
    assert _ndcg_at_k([], {0, 1}, k=5) == 0.0


# ---------------------------------------------------------------------------
# MAP@K
# ---------------------------------------------------------------------------

def test_map_toy():
    rec = [0, 1, 2, 3]   # A=0, C=2 relevant
    rel = {0, 2}
    expected = (1.0 + 2 / 3) / 2
    assert abs(_ap_at_k(rec, rel, k=4) - expected) < 1e-6


def test_map_perfect():
    rec = [0, 1, 2]
    rel = {0, 1, 2}
    assert abs(_ap_at_k(rec, rel, k=3) - 1.0) < 1e-6


def test_map_no_hits():
    rec = [0, 1]
    rel = {2, 3}
    assert _ap_at_k(rec, rel, k=2) == 0.0


# ---------------------------------------------------------------------------
# Precision / Recall / F1 / HitRate
# ---------------------------------------------------------------------------

def test_precision_toy():
    rec = [0, 1, 2, 3, 4]   # A=0, C=2, E=4 relevant
    rel = {0, 2, 4}
    assert abs(_precision_at_k(rec, rel, k=5) - 0.6) < 1e-9


def test_recall_toy():
    rec = [0, 1, 2, 3, 4]
    rel = {0, 2, 4}
    assert abs(_recall_at_k(rec, rel, k=5) - 1.0) < 1e-9


def test_recall_empty_relevant():
    assert _recall_at_k([0, 1], set(), k=2) == 0.0


def test_hit_rate_hit():
    assert _hit_rate_at_k([0, 1, 2], {2}, k=3) == 1.0


def test_hit_rate_miss():
    assert _hit_rate_at_k([0, 1], {3}, k=2) == 0.0


# ---------------------------------------------------------------------------
# RMSE / MAE
# ---------------------------------------------------------------------------

def test_rmse_toy():
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_pred = np.array([1.5, 2.5, 2.5], dtype=np.float32)
    assert abs(rmse(y_true, y_pred) - 0.5) < 1e-6


def test_mae_toy():
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_pred = np.array([1.5, 2.5, 2.5], dtype=np.float32)
    assert abs(mae(y_true, y_pred) - 0.5) < 1e-6


def test_rmse_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_mae_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0
