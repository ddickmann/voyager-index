"""Backend adapters and durable fallback models for LEMUR routing."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

class _TorchMipsIndex:
    """GPU-accelerated MIPS index (brute-force matmul + topk)."""

    def __init__(self, device: str = "cpu") -> None:
        self._device = torch.device(device)
        self.weights = torch.empty((0, 0), dtype=torch.float32)
        self.ids: List[int] = []

    def build(self, weights: torch.Tensor, ids: Sequence[int]) -> None:
        self.weights = weights.detach().to(self._device, dtype=torch.float32).contiguous()
        self.ids = [int(x) for x in ids]

    def add(self, weights: torch.Tensor, ids: Sequence[int]) -> None:
        weights = weights.detach().to(self._device, dtype=torch.float32).contiguous()
        if self.weights.numel() == 0:
            self.weights = weights
        else:
            self.weights = torch.cat([self.weights, weights], dim=0)
        self.ids.extend(int(x) for x in ids)

    def search(self, queries: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.weights.numel() == 0:
            return np.empty((queries.shape[0], 0), dtype=np.float32), np.empty((queries.shape[0], 0), dtype=np.int64)
        q = queries.detach().to(self._device, dtype=torch.float32)
        scores = q @ self.weights.T
        topk = min(k, scores.shape[1])
        vals, idx = torch.topk(scores, topk, dim=1)
        return vals.cpu().numpy(), idx.cpu().numpy().astype(np.int64)

class _ProjectionFallbackModel:
    """Very small self-contained proxy if official LEMUR is unavailable.

    This is not a research-faithful reimplementation of LEMUR. It is a durable
    fallback that keeps the pipeline operational with a predictable interface.
    """

    def __init__(self, index_dir: Path, projection_dim: int = 128, seed: int = 42) -> None:
        self.index_dir = Path(index_dir)
        self.projection_dim = int(projection_dim)
        self.seed = int(seed)
        self.proj: Optional[torch.Tensor] = None

    def fit(self, train: torch.Tensor, train_counts: torch.Tensor, epochs: int = 0, verbose: bool = False) -> None:
        dim = int(train.shape[1])
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        proj = torch.randn(dim, self.projection_dim, generator=g, dtype=torch.float32)
        proj = torch.nn.functional.normalize(proj, dim=0)
        self.proj = proj.contiguous()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.proj, self.index_dir / "fallback_proj.pt")

    def compute_features(self, query_pair: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if self.proj is None:
            self.proj = torch.load(self.index_dir / "fallback_proj.pt", weights_only=True)
        q_tokens, q_counts = query_pair
        return _aggregate_doc_matrix(q_tokens, q_counts) @ self.proj

    def compute_weights(self, docs: torch.Tensor, doc_counts: torch.Tensor) -> torch.Tensor:
        if self.proj is None:
            self.proj = torch.load(self.index_dir / "fallback_proj.pt", weights_only=True)
        return _aggregate_doc_matrix(docs, doc_counts) @ self.proj

def _aggregate_doc_matrix(flat_vectors: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Fallback aggregation: mean + max pooling mixed together."""
    flat_vectors = flat_vectors.detach().cpu().to(torch.float32)
    counts = counts.detach().cpu().to(torch.long)
    out = []
    pos = 0
    for count in counts.tolist():
        count = int(count)
        block = flat_vectors[pos:pos + count]
        pos += count
        if count == 0:
            out.append(torch.zeros(flat_vectors.shape[1], dtype=torch.float32))
            continue
        mean = block.mean(dim=0)
        mx = block.max(dim=0).values
        out.append(torch.nn.functional.normalize(0.5 * (mean + mx), dim=0))
    return torch.stack(out, dim=0) if out else torch.empty((0, flat_vectors.shape[1]), dtype=torch.float32)

