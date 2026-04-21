"""
Triton kernels for GPU-accelerated qCH proxy scoring.

Two kernels:
1. qch_max_gather_kernel: Query-time scoring (query vs all docs)
2. qch_pairwise_kernel: Build-time batch pairwise scoring (doc vs doc pairs)

The pairwise kernel is used during NN-Descent graph construction to offload
the O(n*k) distance computations per iteration to GPU. It uses the precomputed
centroid_dists[n_fine, n_fine] L2 lookup table and computes:

    qCH(a, b) = 1.0 - (1/|a|) * sum_{ca in a} max_{cb in b} (1.0 - d(ca,cb)^2/2)

where d(ca,cb) = centroid_dists[ca * n_fine + cb].
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


if TRITON_AVAILABLE:

    @triton.jit
    def _qch_pairwise_kernel(
        centroid_dists_ptr,  # [n_fine * n_fine] float32 — L2 distances
        codes_ptr,           # [total_codes] int32
        offsets_ptr,         # [n_docs] int32
        lengths_ptr,         # [n_docs] int32
        pairs_a_ptr,         # [n_pairs] int32 — doc_a IDs
        pairs_b_ptr,         # [n_pairs] int32 — doc_b IDs
        out_ptr,             # [n_pairs] float32
        n_pairs,
        n_fine: tl.constexpr,
        MAX_LEN_A: tl.constexpr,
        MAX_LEN_B: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= n_pairs:
            return

        doc_a = tl.load(pairs_a_ptr + pid)
        doc_b = tl.load(pairs_b_ptr + pid)

        off_a = tl.load(offsets_ptr + doc_a)
        len_a = tl.load(lengths_ptr + doc_a)
        off_b = tl.load(offsets_ptr + doc_b)
        len_b = tl.load(lengths_ptr + doc_b)

        total = 0.0

        for ca_i in range(MAX_LEN_A):
            mask_a = ca_i < len_a
            ca = tl.load(codes_ptr + off_a + ca_i, mask=mask_a, other=0)
            row_base = ca * n_fine

            max_ip = tl.full([], value=-1e30, dtype=tl.float32)
            for cb_i in range(MAX_LEN_B):
                mask_b = mask_a & (cb_i < len_b)
                cb = tl.load(codes_ptr + off_b + cb_i, mask=mask_b, other=0)
                l2 = tl.load(centroid_dists_ptr + row_base + cb, mask=mask_b, other=0.0)
                ip = 1.0 - l2 * l2 * 0.5
                max_ip = tl.where(mask_b & (ip > max_ip), ip, max_ip)

            total += tl.where(mask_a, max_ip, 0.0)

        len_a_f = tl.maximum(len_a.to(tl.float32), 1.0)
        result = 1.0 - total / len_a_f
        tl.store(out_ptr + pid, result)


def qch_pairwise_batch_gpu(
    centroid_dists: torch.Tensor,
    codes: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    pairs_a: torch.Tensor,
    pairs_b: torch.Tensor,
    n_fine: int,
) -> torch.Tensor:
    """
    GPU batch pairwise qCH distance computation for NN-Descent graph construction.

    For each pair (doc_a, doc_b), computes the asymmetric Chamfer proxy:
        qCH(a,b) = 1 - mean_{ca in a} max_{cb in b} IP(ca, cb)
    where IP is derived from L2 distances in centroid_dists.

    Args:
        centroid_dists: (n_fine * n_fine,) float32 on GPU — pairwise L2 between centroids
        codes:   (total_codes,) int32 on GPU — all doc centroid codes concatenated
        offsets: (n_docs,) int32 on GPU — start offset per doc
        lengths: (n_docs,) int32 on GPU — number of codes per doc
        pairs_a: (n_pairs,) int32 on GPU — first doc in each pair
        pairs_b: (n_pairs,) int32 on GPU — second doc in each pair
        n_fine:  codebook size

    Returns:
        (n_pairs,) float32 on GPU — qCH distances (lower = more similar)
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed; GPU pairwise qCH unavailable")

    n_pairs = pairs_a.shape[0]
    if n_pairs == 0:
        return torch.empty(0, dtype=torch.float32, device=centroid_dists.device)

    max_len_a_ids = pairs_a.unique()
    max_len_b_ids = pairs_b.unique()
    max_len_a = int(lengths[max_len_a_ids.long()].max().item()) if max_len_a_ids.numel() > 0 else 0
    max_len_b = int(lengths[max_len_b_ids.long()].max().item()) if max_len_b_ids.numel() > 0 else 0

    if max_len_a == 0:
        return torch.ones(n_pairs, dtype=torch.float32, device=centroid_dists.device)

    MAX_LEN_A = _next_power_of_2(min(max_len_a, MAX_SUPPORTED_DOC_LEN))
    MAX_LEN_B = _next_power_of_2(min(max_len_b, MAX_SUPPORTED_DOC_LEN))

    out = torch.empty(n_pairs, dtype=torch.float32, device=centroid_dists.device)

    grid = (n_pairs,)
    _qch_pairwise_kernel[grid](
        centroid_dists, codes, offsets, lengths,
        pairs_a, pairs_b, out,
        n_pairs,
        n_fine=n_fine,
        MAX_LEN_A=MAX_LEN_A,
        MAX_LEN_B=MAX_LEN_B,
    )
    return out


def qch_pairwise_batch_torch(
    centroid_dists: torch.Tensor,
    codes: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    pairs_a: torch.Tensor,
    pairs_b: torch.Tensor,
    n_fine: int,
) -> torch.Tensor:
    """Pure PyTorch fallback for pairwise qCH — used for correctness testing."""
    n_pairs = pairs_a.shape[0]
    if n_pairs == 0:
        return torch.empty(0, dtype=torch.float32, device=centroid_dists.device)

    dists_2d = centroid_dists.view(n_fine, n_fine)
    results = torch.empty(n_pairs, dtype=torch.float32, device=centroid_dists.device)

    for i in range(n_pairs):
        a_id = int(pairs_a[i].item())
        b_id = int(pairs_b[i].item())
        off_a = int(offsets[a_id].item())
        len_a = int(lengths[a_id].item())
        off_b = int(offsets[b_id].item())
        len_b = int(lengths[b_id].item())

        if len_a == 0:
            results[i] = 1.0
            continue

        codes_a = codes[off_a:off_a + len_a].long()
        codes_b = codes[off_b:off_b + len_b].long()

        total = 0.0
        for ca in codes_a:
            row = dists_2d[ca]
            l2_vals = row[codes_b]
            ip_vals = 1.0 - l2_vals * l2_vals * 0.5
            total += ip_vals.max().item()

        results[i] = 1.0 - total / len_a

    return results


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
