"""
GPU Centroid Router: k-means training, GPU centroid scoring, shard selection.

At build time:
  1. Train centroids via k-means on a sample of token embeddings
  2. Assign each document to its dominant centroid
  3. Map centroids to shards

At query time:
  1. Score query tokens against all centroids: Q @ C^T  (GPU matmul)
  2. Aggregate per-shard scores from the centroid scores
  3. Select top-k shards
  4. Expand to a bounded working set (cap total docs)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CentroidRouter:
    """
    GPU-resident centroid router for shard selection.

    The centroid table and shard metadata live on GPU.
    A single query-vs-centroids matmul selects which shards to fetch.
    """

    def __init__(
        self,
        centroid_table: torch.Tensor,
        centroid_to_shard: Dict[int, int],
        shard_centroid_hist: Dict[int, List[int]],
        shard_doc_counts: Dict[int, int],
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.centroid_table = centroid_table.to(self.device, dtype=torch.float16)
        self.n_centroids = self.centroid_table.shape[0]
        self.dim = self.centroid_table.shape[1]

        self.centroid_to_shard = centroid_to_shard
        self.shard_centroid_hist = shard_centroid_hist
        self.shard_doc_counts = shard_doc_counts

        # Only track shards that actually contain documents
        self._shard_ids = sorted(sid for sid, cnt in shard_doc_counts.items() if cnt > 0)
        self._n_shards = len(self._shard_ids)

        # Pre-build shard-centroid membership matrix on GPU: (n_shards, n_centroids)
        membership = torch.zeros(self._n_shards, self.n_centroids, dtype=torch.float16)
        shard_idx_map = {sid: idx for idx, sid in enumerate(self._shard_ids)}
        for sid, cids in shard_centroid_hist.items():
            if sid in shard_idx_map:
                for cid in cids:
                    if 0 <= cid < self.n_centroids:
                        membership[shard_idx_map[sid], cid] = 1.0
        self._shard_membership = membership.to(self.device)
        self._shard_idx_map = shard_idx_map

    @classmethod
    def train(
        cls,
        all_vectors: np.ndarray,
        doc_offsets: List[Tuple[int, int]],
        n_centroids: int = 1024,
        n_shards: int = 256,
        sample_fraction: float = 0.1,
        max_iter: int = 50,
        seed: int = 42,
        device: str = "cuda",
    ) -> Tuple["CentroidRouter", np.ndarray, Dict[int, int]]:
        """
        Train centroids, assign docs to shards, return (router, shard_assignments, c2s).

        Returns:
            router: trained CentroidRouter
            shard_assignments: array of shape (n_docs,) with shard IDs
            centroid_to_shard: mapping from centroid_id to shard_id
        """
        n_vectors = all_vectors.shape[0]
        dim = all_vectors.shape[1]
        n_docs = len(doc_offsets)

        # Sample tokens for k-means
        rng = np.random.RandomState(seed)
        n_sample = max(n_centroids * 40, int(n_vectors * sample_fraction))
        n_sample = min(n_sample, n_vectors)
        sample_idx = rng.choice(n_vectors, size=n_sample, replace=False)
        sample = all_vectors[sample_idx].astype(np.float32)

        logger.info("Training %d centroids on %d sampled vectors (dim=%d)...", n_centroids, n_sample, dim)

        try:
            import faiss
            kmeans = faiss.Kmeans(dim, n_centroids, niter=max_iter, seed=seed, verbose=False, gpu=False)
            kmeans.train(sample)
            centroid_table = kmeans.centroids.copy()
        except ImportError:
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(n_clusters=n_centroids, max_iter=max_iter, random_state=seed, batch_size=4096)
            km.fit(sample)
            centroid_table = km.cluster_centers_.astype(np.float32)

        logger.info("Centroids trained. Assigning docs to dominant centroids...")

        # Assign each doc to its dominant centroid (the centroid with highest
        # aggregate similarity across the doc's tokens)
        doc_dominant_centroid = np.zeros(n_docs, dtype=np.int32)
        ct = torch.from_numpy(centroid_table).float()

        for i, (s, e) in enumerate(doc_offsets):
            doc_vecs = torch.from_numpy(all_vectors[s:e].astype(np.float32))
            sims = doc_vecs @ ct.T  # (n_tokens, n_centroids)
            per_token_best = sims.argmax(dim=-1)  # (n_tokens,)
            # Dominant = most frequently assigned centroid
            counts = torch.bincount(per_token_best, minlength=n_centroids)
            doc_dominant_centroid[i] = int(counts.argmax().item())

        # Map centroids to shards: distribute centroids round-robin or greedily
        centroids_per_shard = max(1, n_centroids // n_shards)
        centroid_order = np.argsort(doc_dominant_centroid)  # group by centroid
        centroid_to_shard: Dict[int, int] = {}

        # Sort centroids by their ID, assign consecutive groups to shards
        for cid in range(n_centroids):
            centroid_to_shard[cid] = cid % n_shards

        # Assign each doc to the shard of its dominant centroid
        shard_assignments = np.array(
            [centroid_to_shard[doc_dominant_centroid[i]] for i in range(n_docs)],
            dtype=np.int32,
        )

        # Build shard metadata
        shard_centroid_hist: Dict[int, List[int]] = {}
        for cid, sid in centroid_to_shard.items():
            shard_centroid_hist.setdefault(sid, []).append(cid)

        shard_doc_counts: Dict[int, int] = {}
        for sid in range(n_shards):
            shard_doc_counts[sid] = int(np.sum(shard_assignments == sid))

        router = cls(
            centroid_table=torch.from_numpy(centroid_table),
            centroid_to_shard=centroid_to_shard,
            shard_centroid_hist=shard_centroid_hist,
            shard_doc_counts=shard_doc_counts,
            device=device,
        )

        logger.info(
            "Router trained: %d centroids -> %d shards, doc distribution p50=%d p95=%d",
            n_centroids, n_shards,
            int(np.median(list(shard_doc_counts.values()))),
            int(np.percentile(list(shard_doc_counts.values()), 95)),
        )

        return router, shard_assignments, centroid_to_shard

    # ------------------------------------------------------------------
    # Query-time routing
    # ------------------------------------------------------------------

    def route(
        self,
        query_vectors: torch.Tensor,
        top_shards: int = 8,
        max_docs: int = 10_000,
    ) -> List[int]:
        """
        Route a single query to the best shards.

        Args:
            query_vectors: (n_query_tokens, dim) on any device
            top_shards: max number of shards to return
            max_docs: cap on total documents across returned shards

        Returns:
            List of shard IDs, ordered by routing score (best first).
        """
        q = query_vectors.to(self.device, dtype=torch.float16)
        if q.dim() == 3:
            q = q.squeeze(0)

        # Score query tokens vs centroids: (n_tokens, n_centroids)
        token_centroid_scores = q @ self.centroid_table.T

        # Per-centroid score: max across query tokens (MaxSim-style aggregation)
        centroid_scores = token_centroid_scores.max(dim=0).values  # (n_centroids,)

        # Per-shard score: sum of centroid scores for centroids in that shard
        # Using the pre-built membership matrix: (n_shards, n_centroids)
        shard_scores = self._shard_membership @ centroid_scores  # (n_shards,)

        # Select top shards
        k = min(top_shards, self._n_shards)
        _, top_indices = shard_scores.topk(k)

        # Expand with doc budget
        selected = []
        total_docs = 0
        for idx in top_indices.cpu().tolist():
            sid = self._shard_ids[idx]
            n_docs = self.shard_doc_counts.get(sid, 0)
            if total_docs + n_docs > max_docs and selected:
                break
            selected.append(sid)
            total_docs += n_docs

        return selected

    def route_batch(
        self,
        query_batch: torch.Tensor,
        top_shards: int = 8,
        max_docs: int = 10_000,
    ) -> List[List[int]]:
        """Route a batch of queries. query_batch: (B, n_tokens, dim)."""
        results = []
        for i in range(query_batch.shape[0]):
            results.append(self.route(query_batch[i], top_shards, max_docs))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.centroid_table.cpu(), path / "centroid_table.pt")
        state = {
            "centroid_to_shard": {str(k): v for k, v in self.centroid_to_shard.items()},
            "shard_centroid_hist": {str(k): v for k, v in self.shard_centroid_hist.items()},
            "shard_doc_counts": {str(k): v for k, v in self.shard_doc_counts.items()},
        }
        with open(path / "router_state.json", "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path: Path, device: str = "cuda") -> "CentroidRouter":
        path = Path(path)
        ct = torch.load(path / "centroid_table.pt", weights_only=True)
        with open(path / "router_state.json") as f:
            state = json.load(f)
        return cls(
            centroid_table=ct,
            centroid_to_shard={int(k): v for k, v in state["centroid_to_shard"].items()},
            shard_centroid_hist={int(k): v for k, v in state["shard_centroid_hist"].items()},
            shard_doc_counts={int(k): v for k, v in state["shard_doc_counts"].items()},
            device=device,
        )
