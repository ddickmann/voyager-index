"""Shard layout and assignment policies for offline builds."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from ..config import BuildConfig, StorageLayout

def _index_dir(cfg: BuildConfig) -> Path:
    cache_base = Path.home() / ".cache" / "shard-bench"
    suffix = "_uniform" if cfg.uniform_shard_tokens else ""
    pool_suffix = f"_pool{cfg.pooling.pool_factor}" if cfg.pooling.enabled else ""
    router_suffix = f"_{cfg.router_type.value}"
    return cache_base / f"index_{cfg.corpus_size}_{cfg.compression.value}_{cfg.layout.value}{router_suffix}{pool_suffix}{suffix}"

def assign_storage_shards(
    pooled_offsets: List[Tuple[int, int]],
    n_shards: int,
    seed: int,
    layout: StorageLayout,
    proxy_weights: torch.Tensor | None = None,
) -> np.ndarray:
    lengths = np.array([e - s for s, e in pooled_offsets], dtype=np.int64)
    n_docs = len(pooled_offsets)
    rng = np.random.RandomState(seed)

    if layout == StorageLayout.RANDOM:
        return rng.randint(0, n_shards, size=n_docs).astype(np.int32)

    if proxy_weights is not None and proxy_weights.numel() > 0 and layout in (
        StorageLayout.PROXY_GROUPED, StorageLayout.CENTROID_GROUPED,
    ):
        W = proxy_weights.detach().cpu().numpy().astype(np.float32)
        order = _proxy_order(W, seed)
    else:
        order = np.argsort(-lengths)

    assignments = np.zeros(n_docs, dtype=np.int32)

    if layout == StorageLayout.TOKEN_BALANCED:
        shard_loads = np.zeros(n_shards, dtype=np.int64)
        for doc_idx in order:
            sid = int(np.argmin(shard_loads))
            assignments[doc_idx] = sid
            shard_loads[sid] += lengths[doc_idx]
        return assignments

    total_tokens = max(1, int(lengths.sum()))
    target = max(1, int(np.ceil(total_tokens / n_shards)))
    sid = 0
    current = 0
    for doc_idx in order:
        if sid < n_shards - 1 and current >= target:
            sid += 1
            current = 0
        assignments[doc_idx] = sid
        current += int(lengths[doc_idx])
    return assignments

def _proxy_order(weights: np.ndarray, seed: int) -> np.ndarray:
    try:
        from sklearn.cluster import MiniBatchKMeans
        n_clusters = min(256, max(16, int(np.sqrt(weights.shape[0]))))
        km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, random_state=seed)
        labels = km.fit_predict(weights)
        centroid_norm = np.linalg.norm(km.cluster_centers_, axis=1)
        return np.lexsort((centroid_norm[labels], labels)).astype(np.int64)
    except Exception:
        proj = weights[:, 0] if weights.shape[1] > 0 else np.zeros(weights.shape[0], dtype=np.float32)
        return np.argsort(proj, kind="mergesort")

