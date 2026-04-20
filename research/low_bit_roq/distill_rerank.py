"""
Distillation rerank head.

Plan reference: cross-cut 1, promoted to gate-required for A6. Tiny
2-layer MLP that takes ``(<q,d>_lowbit, ||r̂_d||, <q,c_d>, score_lowbit_normalized)``
and predicts the fp16-rerank-rank, restoring ~0.5-1.0 Recall@10 points
lost to quantization for ~50µs p95.

Per the plan we train **one reranker per bit-width** (1.0, 1.58, 2.0) so
that no bit-width is favored by a more carefully tuned reranker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class DistillFeatures:
    score_lowbit: np.ndarray         # (N,)
    residual_norm: np.ndarray        # (N,) ||r̂||
    centroid_dot: np.ndarray         # (N,) <q, c_d>
    score_lowbit_norm: np.ndarray    # (N,) score_lowbit / ||q||·||d||
    extras: dict[str, np.ndarray] | None = None


@dataclass
class TrainExample:
    features: DistillFeatures
    target_rank: np.ndarray  # (N,) gold fp16 rerank ranks


def collect_training_data(
    runs: Iterator[TrainExample],
    *,
    out_path: Path,
) -> Path:
    """Concatenate per-query (features, ranks) into a single .npz so the
    trainer is one open() away from a model.

    The harness's ``inspect_query_pipeline()`` returns
    ``routed_ids`` / ``pruned_ids`` / ``exact_candidate_ids``; one row per
    candidate per query. The runner provides the gold ranks by re-scoring
    with fp16 on the exact-candidate shortlist.
    """
    score_lowbit, residual_norm, centroid_dot, score_lowbit_norm, target_rank = (
        [], [], [], [], []
    )
    for ex in runs:
        score_lowbit.append(ex.features.score_lowbit)
        residual_norm.append(ex.features.residual_norm)
        centroid_dot.append(ex.features.centroid_dot)
        score_lowbit_norm.append(ex.features.score_lowbit_norm)
        target_rank.append(ex.target_rank)
    data = {
        "score_lowbit": np.concatenate(score_lowbit),
        "residual_norm": np.concatenate(residual_norm),
        "centroid_dot": np.concatenate(centroid_dot),
        "score_lowbit_norm": np.concatenate(score_lowbit_norm),
        "target_rank": np.concatenate(target_rank),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **data)
    log.info("saved %d training rows to %s", data["target_rank"].shape[0], out_path)
    return out_path


def _featurize(features: DistillFeatures) -> np.ndarray:
    return np.stack(
        [
            features.score_lowbit,
            features.residual_norm,
            features.centroid_dot,
            features.score_lowbit_norm,
        ],
        axis=1,
    ).astype(np.float32)


def train_distill_head(
    npz_path: Path,
    *,
    out_path: Path,
    hidden: int = 32,
    epochs: int = 4,
    batch_size: int = 4096,
    lr: float = 1e-3,
    seed: int = 0,
) -> Path:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    data = np.load(npz_path)
    feats = np.stack(
        [data["score_lowbit"], data["residual_norm"], data["centroid_dot"], data["score_lowbit_norm"]],
        axis=1,
    ).astype(np.float32)
    targets = data["target_rank"].astype(np.float32)

    feats_t = torch.from_numpy(feats)
    targets_t = torch.from_numpy(targets)

    mean = feats_t.mean(dim=0)
    std = feats_t.std(dim=0).clamp_min(1e-6)
    feats_norm = (feats_t - mean) / std

    model = nn.Sequential(
        nn.Linear(4, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = feats.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(feats_norm[idx]).squeeze(-1)
            loss = loss_fn(pred, targets_t[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss) * idx.numel()
        log.info("epoch %d  loss=%.4f", epoch, epoch_loss / n)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "mean": mean, "std": std}, out_path)
    return out_path


class DistillReranker:
    """Inference wrapper. Loaded from disk by the runner; called per query
    with the lowbit candidate set, returns reranked indices."""

    def __init__(self, path: Path):
        import torch

        ckpt = torch.load(path, map_location="cpu")
        self._model_state = ckpt["model"]
        self._mean = ckpt["mean"]
        self._std = ckpt["std"]
        self._model = self._build_model()
        self._model.load_state_dict(self._model_state)
        self._model.eval()

    def _build_model(self):
        import torch.nn as nn

        return nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))

    def rerank(self, features: DistillFeatures, top_k: int) -> np.ndarray:
        import torch

        f = (torch.from_numpy(_featurize(features)) - self._mean) / self._std
        with torch.no_grad():
            scores = self._model(f).squeeze(-1).cpu().numpy()
        idx = np.argsort(scores)[:top_k]
        return idx
