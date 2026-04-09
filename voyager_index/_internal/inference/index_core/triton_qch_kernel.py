"""
Triton kernel for batched max-gather qCH proxy scoring on GPU.

For each document, computes:
    score(doc) = 1.0 - (1/n_query) * sum_qi max_{c in codes[off:off+len]} scores[qi * n_fine + c]

where scores = query_centroid_scores (n_query x n_fine), already IDF-weighted.

Supports document lengths up to 2048 tokens without truncation.
Uses @triton.autotune for automatic block-size selection.
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

MAX_SUPPORTED_DOC_LEN = 2048


def _next_power_of_2(x: int) -> int:
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()


if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 32}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=8, num_stages=3),
        ],
        key=["n_query", "n_fine", "MAX_DOC_LEN"],
    )
    @triton.jit
    def _qch_max_gather_kernel(
        scores_ptr,     # [n_query * n_fine] float32
        codes_ptr,      # [total_codes] int32 (u16 widened)
        offsets_ptr,    # [n_docs] int32
        lengths_ptr,    # [n_docs] int32
        out_ptr,        # [n_docs] float32
        n_docs,
        n_query: tl.constexpr,
        n_fine: tl.constexpr,
        MAX_DOC_LEN: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        doc_ids = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = doc_ids < n_docs

        off = tl.load(offsets_ptr + doc_ids, mask=valid, other=0)
        length = tl.load(lengths_ptr + doc_ids, mask=valid, other=0)

        total = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for qi in range(n_query):
            base = qi * n_fine
            max_val = tl.full([BLOCK_SIZE], value=-1e30, dtype=tl.float32)
            for ci in range(MAX_DOC_LEN):
                mask = valid & (ci < length)
                code_idx = off + ci
                code = tl.load(codes_ptr + code_idx, mask=mask, other=0)
                score = tl.load(scores_ptr + base + code, mask=mask, other=-1e30)
                max_val = tl.where(mask & (score > max_val), score, max_val)
            total += tl.where(valid, max_val, 0.0)

        inv_nq = 1.0 / n_query
        result = 1.0 - total * inv_nq
        tl.store(out_ptr + doc_ids, result, mask=valid)


def qch_max_gather_gpu(
    scores: torch.Tensor,
    codes: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    n_query: int,
    n_fine: int,
) -> torch.Tensor:
    """
    GPU batched max-gather qCH proxy scoring via Triton.

    Args:
        scores: (n_query * n_fine,) float32 on GPU
        codes:  (total_codes,) int32 on GPU
        offsets: (n_docs,) int32 on GPU
        lengths: (n_docs,) int32 on GPU
        n_query: number of query vectors
        n_fine: codebook size

    Returns:
        (n_docs,) float32 on GPU -- proxy qCH scores (lower = closer)

    Falls back to PyTorch path if max doc length > 2048.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed; GPU qCH scoring unavailable")

    n_docs = offsets.shape[0]
    if n_docs == 0:
        return torch.empty(0, dtype=torch.float32, device=scores.device)

    max_len = int(lengths.max().item())

    if max_len > MAX_SUPPORTED_DOC_LEN:
        logger.warning(
            "max doc length %d exceeds Triton limit %d; falling back to PyTorch",
            max_len, MAX_SUPPORTED_DOC_LEN,
        )
        return qch_max_gather_torch(scores, codes, offsets, lengths, n_query, n_fine)

    if max_len == 0:
        return torch.ones(n_docs, dtype=torch.float32, device=scores.device)

    MAX_DOC_LEN = _next_power_of_2(max_len)

    out = torch.empty(n_docs, dtype=torch.float32, device=scores.device)

    def _grid(meta):
        return (math.ceil(n_docs / meta["BLOCK_SIZE"]),)

    _qch_max_gather_kernel[_grid](
        scores, codes, offsets, lengths, out,
        n_docs,
        n_query=n_query, n_fine=n_fine, MAX_DOC_LEN=MAX_DOC_LEN,
    )
    return out


def qch_max_gather_torch(
    scores: torch.Tensor,
    codes: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    n_query: int,
    n_fine: int,
) -> torch.Tensor:
    """
    Pure PyTorch fallback for max-gather qCH scoring (no Triton needed).

    Memory-efficient: iterates over query vectors instead of materializing
    a (n_query, n_docs, n_fine) intermediate tensor.
    """
    n_docs = offsets.shape[0]
    max_len = int(lengths.max().item()) if n_docs > 0 else 0

    if max_len == 0:
        return torch.ones(n_docs, dtype=torch.float32, device=scores.device)

    # Build padded code indices: (n_docs, max_len)
    idx = torch.arange(max_len, device=codes.device).unsqueeze(0).expand(n_docs, -1)
    mask = idx < lengths.unsqueeze(1)
    flat_idx = (offsets.unsqueeze(1) + idx).clamp(max=codes.shape[0] - 1)
    padded_codes = codes[flat_idx.long()]  # (n_docs, max_len)

    scores_2d = scores.view(n_query, n_fine)  # (n_query, n_fine)

    # Accumulate max-gather per query vector to avoid huge intermediate
    total = torch.zeros(n_docs, dtype=torch.float32, device=scores.device)
    for qi in range(n_query):
        row_scores = scores_2d[qi][padded_codes.long()]  # (n_docs, max_len)
        row_scores = row_scores.masked_fill(~mask, -1e30)
        total += row_scores.max(dim=1).values

    return 1.0 - total / n_query
