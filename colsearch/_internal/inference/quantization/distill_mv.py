"""
Multi-view distill rerank head.

Numpy-only forward pass for the small MLP that reorders the rroq158 top-K
shortlist into the final top-k. Trained offline on a held-out set of
queries (see ``research/low_bit_roq/bench_beir_e2e.py:_train_mv_distill_on_pairs``)
and shipped as a frozen ``.npz`` (~10 KB) per index.

Status (PROGRESS.md [2026-04-19]):
    The distillation head recovers ~50% of the ternary→roq4 NN50* gap on
    offline distortion fixtures, but on real BEIR qrels (nfcorpus) it has
    so far regressed Recall@10 (0.3456 → 0.1160) without any improvement
    to candidate recall (R@100 unchanged at 0.3688). The reranker is
    actively re-ordering the right candidates the wrong way, likely
    because qrel-positive grades are bimodal and the BCE training signal
    doesn't separate "weakly relevant" from "irrelevant" within the top-K.

This module is therefore shipped behind ``SearchConfig.distill_rerank =
False`` by default. It exists so we can iterate on the training-time
features (more queries / pairwise hinge loss / non-binary qrel grades)
without re-shipping kernel code.

Features (must match training-time order):

    [0] score_rroq         base rroq158 MaxSim score
    [1] score_ternary      no-centroid ternary MaxSim score (rotated frame)
    [2] qc                 sum_q max_d <q, c_d>            (centroid-only)
    [3] r_norm_avg         mean ||r_d|| over the doc tokens
    [4] disagree           |score_rroq - score_ternary|
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MultiViewDistillHead:
    """Frozen 5→32→32→1 MLP forward pass."""

    W1: np.ndarray   # (5, 32)
    b1: np.ndarray   # (32,)
    W2: np.ndarray   # (32, 32)
    b2: np.ndarray   # (32,)
    W3: np.ndarray   # (32, 1)
    b3: np.ndarray   # (1,)
    feat_mean: np.ndarray   # (5,)
    feat_std: np.ndarray    # (5,)

    def forward(self, features: np.ndarray) -> np.ndarray:
        """``features`` is (N, 5). Returns ``(N,)`` rerank scores (higher = more relevant)."""
        if features.ndim != 2 or features.shape[1] != 5:
            raise ValueError(f"expected (N, 5) features, got {features.shape}")
        x = (features.astype(np.float32) - self.feat_mean) / self.feat_std
        x = np.maximum(0.0, x @ self.W1 + self.b1)
        x = np.maximum(0.0, x @ self.W2 + self.b2)
        return (x @ self.W3 + self.b3).reshape(-1)

    @classmethod
    def from_npz(cls, path: Path | str) -> "MultiViewDistillHead":
        data = np.load(str(path))
        return cls(
            W1=data["W1"].astype(np.float32),
            b1=data["b1"].astype(np.float32),
            W2=data["W2"].astype(np.float32),
            b2=data["b2"].astype(np.float32),
            W3=data["W3"].astype(np.float32),
            b3=data["b3"].astype(np.float32),
            feat_mean=data["feat_mean"].astype(np.float32),
            feat_std=data["feat_std"].astype(np.float32),
        )

    def to_npz(self, path: Path | str) -> None:
        np.savez(
            str(path),
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
            feat_mean=self.feat_mean, feat_std=self.feat_std,
        )


def from_torch_state_dict(state_dict: dict, feat_mean: np.ndarray,
                          feat_std: np.ndarray) -> MultiViewDistillHead:
    """Convert a torch ``Sequential(Linear, ReLU, Linear, ReLU, Linear)``
    state_dict (as produced by ``bench_beir_e2e.py:_train_mv_distill_on_pairs``)
    into a numpy MLP for serve-time.
    """
    return MultiViewDistillHead(
        W1=state_dict["0.weight"].T.astype(np.float32),
        b1=state_dict["0.bias"].astype(np.float32),
        W2=state_dict["2.weight"].T.astype(np.float32),
        b2=state_dict["2.bias"].astype(np.float32),
        W3=state_dict["4.weight"].T.astype(np.float32),
        b3=state_dict["4.bias"].astype(np.float32),
        feat_mean=feat_mean.astype(np.float32),
        feat_std=feat_std.astype(np.float32),
    )


def build_features_for_shortlist(
    score_rroq: np.ndarray,    # (N,)
    score_ternary: np.ndarray, # (N,)
    qc: np.ndarray,            # (N,)
    r_norm_avg: np.ndarray,    # (N,)
) -> np.ndarray:
    """Build the (N, 5) feature matrix the head expects."""
    disagree = np.abs(score_rroq - score_ternary)
    return np.stack([score_rroq, score_ternary, qc, r_norm_avg, disagree], axis=1).astype(np.float32)


__all__ = [
    "MultiViewDistillHead",
    "build_features_for_shortlist",
    "from_torch_state_dict",
]
