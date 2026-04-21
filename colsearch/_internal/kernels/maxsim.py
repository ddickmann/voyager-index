"""
Lazy MaxSim export with a PyTorch fallback when Triton is unavailable.

Production guidance for the OSS surface:
- default path: exact Triton MaxSim in FP16
- optional fast path: Triton INT8 where supported
- experimental path: FP8
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)
_TRITON_FALLBACK_CACHE: set[tuple] = set()

try:
    from .triton_maxsim import fast_colbert_scores as _triton_fast_colbert_scores
    TRITON_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on local Triton install
    _triton_fast_colbert_scores = None
    TRITON_AVAILABLE = False
    logger.debug("Triton MaxSim unavailable, using PyTorch fallback: %s", exc)


def _convert_to_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, list):
        return torch.stack([_convert_to_tensor(item) for item in value])
    return torch.tensor(value)


def _pad_embeddings(items: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not items:
        raise ValueError("Expected at least one embedding tensor")

    max_tokens = max(item.shape[0] for item in items)
    dim = items[0].shape[-1]
    padded = torch.zeros((len(items), max_tokens, dim), dtype=items[0].dtype, device=items[0].device)
    mask = torch.zeros((len(items), max_tokens), dtype=torch.float32, device=items[0].device)

    for idx, item in enumerate(items):
        length = item.shape[0]
        padded[idx, :length] = item
        mask[idx, :length] = 1.0

    return padded, mask


def _cpu_colbert_scores(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    q = queries_embeddings.to(torch.float32)
    d = documents_embeddings.to(torch.float32)

    target_elements = 8_000_000
    per_doc_cost = max(q.shape[1] * d.shape[1] * q.shape[2], 1)
    doc_chunk_size = max(1, min(d.shape[0], target_elements // per_doc_cost))
    outputs = []

    for start in range(0, d.shape[0], doc_chunk_size):
        stop = min(start + doc_chunk_size, d.shape[0])
        d_chunk = d[start:stop]
        sim = torch.einsum("ash,bth->abst", q, d_chunk)

        if documents_mask is not None:
            dm = documents_mask[start:stop].to(dtype=torch.bool, device=sim.device)
            sim = sim.masked_fill(~dm[None, :, None, :], float("-inf"))

        max_sim = sim.max(dim=-1).values

        if queries_mask is not None:
            qm = queries_mask.to(dtype=max_sim.dtype, device=max_sim.device)
            max_sim = max_sim * qm[:, None, :]

        max_sim = torch.nan_to_num(max_sim, neginf=0.0)
        outputs.append(max_sim.sum(dim=-1))

    return torch.cat(outputs, dim=1) if outputs else torch.empty((q.shape[0], 0), device=q.device, dtype=q.dtype)


def compute_maxsim_token_coverage_matrix(
    query_vectors: np.ndarray,
    candidate_vectors: Sequence[np.ndarray],
    *,
    device: torch.device,
) -> np.ndarray:
    """
    Per-query-token coverage: for each query token row and each candidate, the
    max cosine similarity to that candidate's token rows, mapped from [-1, 1]
    dot product to [0, 1].

    This is the same late-interaction geometry summarized by :func:`fast_colbert_scores`
    (sum over query tokens of max over document tokens); here we expose the
    per-query-token matrix needed by the fulfilment knapsack precompute.

    Implementation uses the same masked einsum path as the PyTorch fallback in
    this module (GPU when ``device`` is CUDA).
    """
    if not candidate_vectors:
        return np.zeros((int(query_vectors.shape[0]), 0), dtype=np.float32)
    query = torch.as_tensor(query_vectors, device=device, dtype=torch.float32)
    query_token_count = int(query.shape[0])
    max_tokens = max(vector.shape[0] for vector in candidate_vectors)
    target_elements = 24_000_000 if device.type == "cuda" else 8_000_000
    doc_chunk_size = max(
        1,
        min(len(candidate_vectors), target_elements // max(query_token_count * max_tokens, 1)),
    )
    coverage_chunks: list[np.ndarray] = []

    for start in range(0, len(candidate_vectors), doc_chunk_size):
        stop = min(start + doc_chunk_size, len(candidate_vectors))
        chunk_vectors = candidate_vectors[start:stop]
        chunk_max_tokens = max(vector.shape[0] for vector in chunk_vectors)
        dim = int(chunk_vectors[0].shape[1])
        padded = np.zeros((len(chunk_vectors), chunk_max_tokens, dim), dtype=np.float32)
        mask = np.zeros((len(chunk_vectors), chunk_max_tokens), dtype=bool)
        for index, vector in enumerate(chunk_vectors):
            padded[index, : vector.shape[0], :] = vector
            mask[index, : vector.shape[0]] = True
        docs = torch.as_tensor(padded, device=device, dtype=torch.float32)
        mask_t = torch.as_tensor(mask, device=device, dtype=torch.bool)
        sims = torch.einsum("ntd,qd->ntq", docs, query)
        sims = sims.masked_fill(~mask_t.unsqueeze(-1), float("-inf"))
        max_per_query = sims.max(dim=1).values.transpose(0, 1).contiguous()
        max_per_query = torch.nan_to_num(max_per_query, nan=0.0, neginf=0.0, posinf=1.0)
        coverage = ((max_per_query + 1.0) * 0.5).clamp(0.0, 1.0)
        coverage_chunks.append(coverage.detach().cpu().numpy().astype(np.float32, copy=False))

    return (
        np.concatenate(coverage_chunks, axis=1)
        if coverage_chunks
        else np.zeros((query_token_count, 0), dtype=np.float32)
    )


def _triton_cache_key(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
    use_quantization: bool,
    quantization_mode: str,
) -> tuple:
    return (
        str(queries_embeddings.device),
        tuple(queries_embeddings.shape),
        tuple(documents_embeddings.shape),
        bool(use_quantization),
        quantization_mode,
    )


def _triton_supported(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
) -> bool:
    if _triton_fast_colbert_scores is None:
        return False
    if queries_embeddings.device.type != "cuda" or documents_embeddings.device.type != "cuda":
        return False
    if queries_embeddings.shape[-1] < 16 or documents_embeddings.shape[-1] < 16:
        return False
    if min(queries_embeddings.shape[0], queries_embeddings.shape[1], documents_embeddings.shape[0], documents_embeddings.shape[1]) < 1:
        return False
    return True


def fast_colbert_scores(
    queries_embeddings: torch.Tensor | List[torch.Tensor],
    documents_embeddings: torch.Tensor | List[torch.Tensor],
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    use_quantization: bool = False,
    quantization_mode: str = "int8",
) -> torch.Tensor:
    """
    Score late-interaction embeddings with Triton when available.

    The canonical OSS default is exact FP16 Triton MaxSim. Quantized modes are
    opt-in: `int8` is the main speed-oriented profile, while `fp8` should be
    treated as experimental.
    """
    if isinstance(queries_embeddings, list):
        query_tensors = [_convert_to_tensor(item) for item in queries_embeddings]
        queries_embeddings, inferred_query_mask = _pad_embeddings(query_tensors)
        if queries_mask is None:
            queries_mask = inferred_query_mask
    else:
        queries_embeddings = _convert_to_tensor(queries_embeddings)

    if isinstance(documents_embeddings, list):
        document_tensors = [_convert_to_tensor(item) for item in documents_embeddings]
        documents_embeddings, inferred_document_mask = _pad_embeddings(document_tensors)
        if documents_mask is None:
            documents_mask = inferred_document_mask
    else:
        documents_embeddings = _convert_to_tensor(documents_embeddings)

    if queries_embeddings.dim() != 3 or documents_embeddings.dim() != 3:
        raise ValueError("queries_embeddings and documents_embeddings must be 3D")

    if queries_embeddings.shape[-1] != documents_embeddings.shape[-1]:
        raise ValueError("Embedding dimensions must match")

    cache_key = _triton_cache_key(
        queries_embeddings,
        documents_embeddings,
        use_quantization=use_quantization,
        quantization_mode=quantization_mode,
    )
    if _triton_supported(queries_embeddings, documents_embeddings) and cache_key not in _TRITON_FALLBACK_CACHE:
        try:
            return _triton_fast_colbert_scores(
                queries_embeddings=queries_embeddings,
                documents_embeddings=documents_embeddings,
                queries_mask=queries_mask,
                documents_mask=documents_mask,
                use_quantization=use_quantization,
                quantization_mode=quantization_mode,
            )
        except Exception as exc:  # pragma: no cover - depends on local Triton runtime
            logger.warning(
                "Falling back to PyTorch MaxSim because Triton execution failed: %s",
                exc,
            )
            _TRITON_FALLBACK_CACHE.add(cache_key)

    if use_quantization:
        logger.warning(
            "Quantized Triton MaxSim is unavailable in the PyTorch fallback; "
            "running full-precision scoring instead."
        )

    return _cpu_colbert_scores(
        queries_embeddings,
        documents_embeddings,
        queries_mask=queries_mask,
        documents_mask=documents_mask,
    )


__all__ = [
    "TRITON_AVAILABLE",
    "compute_maxsim_token_coverage_matrix",
    "fast_colbert_scores",
]
