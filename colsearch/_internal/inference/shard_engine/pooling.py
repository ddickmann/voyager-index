from __future__ import annotations

import heapq
from typing import List, Sequence, Tuple

import numpy as np
import torch


class TokenPooler:
    """Index-time token pooling for document multi-vectors.

    The implementation is intentionally self-contained and dependency-light.
    It performs greedy hierarchical merging of adjacent tokens until the token
    budget is reduced by ``pool_factor``. Queries are never pooled.
    """

    def __init__(
        self,
        method: str = "hierarchical",
        pool_factor: int = 2,
        protected_tokens: int = 0,
    ) -> None:
        if pool_factor < 1:
            raise ValueError("pool_factor must be >= 1")
        self.method = method
        self.pool_factor = int(pool_factor)
        self.protected_tokens = int(max(0, protected_tokens))

    def pool_docs(
        self,
        flat_doc_vectors: torch.Tensor | np.ndarray,
        doc_offsets: Sequence[Tuple[int, int]],
    ) -> tuple[torch.Tensor, List[Tuple[int, int]], torch.Tensor]:
        """Pool a flattened document matrix.

        Returns a new flattened tensor, new offsets, and per-doc pooled counts.
        """
        if isinstance(flat_doc_vectors, np.ndarray):
            flat = torch.from_numpy(flat_doc_vectors)
        else:
            flat = flat_doc_vectors
        flat = flat.detach().cpu().to(torch.float32)

        pooled_chunks: List[torch.Tensor] = []
        pooled_offsets: List[Tuple[int, int]] = []
        pooled_counts: List[int] = []
        pos = 0
        for start, end in doc_offsets:
            doc = flat[int(start):int(end)]
            pooled = self.pool_document(doc)
            pooled_chunks.append(pooled)
            pooled_offsets.append((pos, pos + pooled.shape[0]))
            pooled_counts.append(int(pooled.shape[0]))
            pos += int(pooled.shape[0])

        dim = flat.shape[1] if flat.numel() else 0
        if pooled_chunks:
            pooled_flat = torch.cat(pooled_chunks, dim=0).to(torch.float16)
        else:
            pooled_flat = torch.empty((0, dim), dtype=torch.float16)
        pooled_counts_t = torch.tensor(pooled_counts, dtype=torch.int32)
        return pooled_flat, pooled_offsets, pooled_counts_t

    def pool_document(self, doc_vectors: torch.Tensor) -> torch.Tensor:
        if doc_vectors.ndim != 2:
            raise ValueError("doc_vectors must be 2D")
        n_tokens = int(doc_vectors.shape[0])
        if self.pool_factor <= 1 or n_tokens <= max(2, self.protected_tokens):
            return doc_vectors.detach().cpu().to(torch.float32)
        if self.method != "hierarchical":
            raise ValueError(f"unsupported pooling method: {self.method}")

        target = max(self.protected_tokens, int(np.ceil(n_tokens / self.pool_factor)))
        if target >= n_tokens:
            return doc_vectors.detach().cpu().to(torch.float32)
        return self._hierarchical_pool(doc_vectors.detach().cpu().to(torch.float32), target)

    def _hierarchical_pool(self, doc_vectors: torch.Tensor, target_tokens: int) -> torch.Tensor:
        n, dim = doc_vectors.shape
        # Cluster state.
        vecs = doc_vectors.clone()
        weights = torch.ones(n, dtype=torch.float32)
        prev = [-1] + list(range(0, n - 1))
        nxt = list(range(1, n)) + [-1]
        active = [True] * n
        versions = [0] * n

        def cosine(i: int, j: int) -> float:
            a = vecs[i]
            b = vecs[j]
            denom = (torch.linalg.norm(a) * torch.linalg.norm(b)).item()
            if denom == 0.0:
                return -1.0
            return float(torch.dot(a, b).item() / denom)

        protected = set(range(min(self.protected_tokens, n)))

        heap: List[tuple[float, int, int, int, int]] = []
        for i in range(n - 1):
            if i in protected and (i + 1) in protected:
                continue
            heapq.heappush(heap, (-cosine(i, i + 1), i, i + 1, versions[i], versions[i + 1]))

        active_count = n
        while active_count > target_tokens and heap:
            neg_sim, i, j, vi, vj = heapq.heappop(heap)
            if not (0 <= i < n and 0 <= j < n):
                continue
            if not (active[i] and active[j]):
                continue
            if nxt[i] != j or versions[i] != vi or versions[j] != vj:
                continue
            if i in protected and j in protected:
                continue

            wi = float(weights[i].item())
            wj = float(weights[j].item())
            vecs[i] = (wi * vecs[i] + wj * vecs[j]) / max(wi + wj, 1e-6)
            weights[i] = wi + wj
            versions[i] += 1

            # Remove j.
            active[j] = False
            active_count -= 1
            right = nxt[j]
            nxt[i] = right
            if right != -1:
                prev[right] = i
            versions[j] += 1

            left = prev[i]
            if left != -1 and active[left]:
                heapq.heappush(heap, (-cosine(left, i), left, i, versions[left], versions[i]))
            if right != -1 and active[right]:
                heapq.heappush(heap, (-cosine(i, right), i, right, versions[i], versions[right]))

        out: List[torch.Tensor] = []
        idx = 0
        while idx != -1 and idx < n:
            if active[idx]:
                out.append(vecs[idx].unsqueeze(0))
            idx = nxt[idx]

        if not out:
            return doc_vectors[:target_tokens]
        return torch.cat(out, dim=0)
