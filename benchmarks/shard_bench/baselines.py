"""
Baseline implementations for comparison.

Baseline A: GPU-only working-set MaxSim
  Everything for a subset loaded fully on GPU.
  This is the best-case kernel throughput ceiling.

Baseline C: Dense single-vector retrieval
  Mean-pooled vectors with FAISS flat/HNSW.
  This is the throughput and memory floor.
"""
from __future__ import annotations

import logging
import time
from typing import List, Tuple

import numpy as np
import torch

from colsearch._internal.inference.shard_engine.scorer import brute_force_maxsim, score_and_topk
from colsearch._internal.inference.shard_engine.profiler import Timer, QueryProfile

logger = logging.getLogger(__name__)


# ======================================================================
# Baseline A: GPU-only MaxSim (kernel ceiling)
# ======================================================================

class BaselineGpuMaxSim:
    """
    Load a document subset fully into GPU VRAM, score with Triton MaxSim.
    Measures pure kernel throughput without any fetch or transfer overhead.
    """

    def __init__(
        self,
        doc_vecs: list,
        doc_ids: List[int],
        dim: int,
        device: str = "cuda",
        max_docs: int = 0,
    ):
        self.doc_ids = doc_ids
        self.dim = dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if max_docs and len(doc_vecs) > max_docs:
            doc_vecs = doc_vecs[:max_docs]
            self.doc_ids = doc_ids[:max_docs]

        logger.info("Baseline A: loading %d docs to GPU...", len(self.doc_ids))
        max_tok = max(v.shape[0] for v in doc_vecs)
        n = len(doc_vecs)

        self.doc_emb = torch.zeros(n, max_tok, dim, dtype=torch.float16, device=self.device)
        self.doc_mask = torch.zeros(n, max_tok, dtype=torch.float32, device=self.device)

        for i, v in enumerate(doc_vecs):
            t = v.shape[0]
            self.doc_emb[i, :t] = torch.from_numpy(v.astype(np.float16))
            self.doc_mask[i, :t] = 1.0

        mem_gb = self.doc_emb.nelement() * 2 / 1e9
        logger.info("Baseline A ready: %d docs, %d max_tokens, %.2f GB GPU", n, max_tok, mem_gb)

    def search(self, query: torch.Tensor, k: int = 10) -> QueryProfile:
        q = query.to(self.device)
        if q.dim() == 2:
            q = q.unsqueeze(0)

        prof = QueryProfile()

        with Timer(sync_cuda=True) as t:
            ids, scores = score_and_topk(
                q, self.doc_emb, self.doc_mask, self.doc_ids, k=k,
            )

        prof.maxsim_ms = t.elapsed_ms
        prof.total_ms = t.elapsed_ms
        prof.num_docs_scored = len(self.doc_ids)
        prof.retrieved_ids = ids
        prof.retrieved_scores = scores
        return prof


# ======================================================================
# Baseline C: Dense single-vector (floor)
# ======================================================================

class BaselineDenseSingleVector:
    """
    Mean-pool all documents to single vectors, search with FAISS.
    No late interaction — measures the quality and throughput floor.
    """

    def __init__(
        self,
        doc_vecs: list,
        doc_ids: List[int],
        dim: int,
        index_type: str = "flat",
    ):
        self.doc_ids = doc_ids
        self.dim = dim

        logger.info("Baseline C: mean-pooling %d docs (dim=%d)...", len(doc_ids), dim)
        pooled = np.zeros((len(doc_ids), dim), dtype=np.float32)
        for i, v in enumerate(doc_vecs):
            pooled[i] = np.mean(v.astype(np.float32), axis=0)

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        pooled = pooled / norms

        try:
            import faiss
            if index_type == "hnsw":
                self.index = faiss.IndexHNSWFlat(dim, 32)
                self.index.hnsw.efConstruction = 200
            else:
                self.index = faiss.IndexFlatIP(dim, )
            self.index.add(pooled)
            self._use_faiss = True
            logger.info("Baseline C: FAISS %s index built (%d vectors)", index_type, self.index.ntotal)
        except ImportError:
            self._pooled = torch.from_numpy(pooled)
            self._use_faiss = False
            logger.warning("FAISS not available, using PyTorch brute-force for dense baseline")

        self._query_pool_cache: dict = {}

    def search(self, query: torch.Tensor, k: int = 10) -> QueryProfile:
        """query: (n_tokens, dim) or (1, n_tokens, dim)"""
        prof = QueryProfile()

        if isinstance(query, torch.Tensor):
            q_np = query.cpu().numpy().astype(np.float32)
        else:
            q_np = query.astype(np.float32)
        if q_np.ndim == 3:
            q_np = q_np[0]

        q_pooled = np.mean(q_np, axis=0, keepdims=True)
        norm = np.linalg.norm(q_pooled, axis=1, keepdims=True)
        if norm > 0:
            q_pooled = q_pooled / norm

        with Timer() as t:
            if self._use_faiss:
                scores, indices = self.index.search(q_pooled, k)
                ids = [self.doc_ids[i] for i in indices[0] if 0 <= i < len(self.doc_ids)]
                sc = scores[0].tolist()[:len(ids)]
            else:
                sims = (self._pooled @ torch.from_numpy(q_pooled.T)).squeeze(-1)
                topk = sims.topk(min(k, len(self.doc_ids)))
                ids = [self.doc_ids[i] for i in topk.indices.tolist()]
                sc = topk.values.tolist()

        prof.total_ms = t.elapsed_ms
        prof.routing_ms = t.elapsed_ms
        prof.num_docs_scored = len(self.doc_ids)
        prof.retrieved_ids = ids
        prof.retrieved_scores = sc
        return prof
