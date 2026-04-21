"""GPU-vectorized single_maxsim for LEMUR training.

Replaces the upstream pure-Python loop implementation with
scatter_reduce-based segmented max, eliminating O(n_docs)
Python-loop overhead during training and weight computation.

API-compatible drop-in replacement for the original single_maxsim.py.
"""
from __future__ import annotations

from typing import Sequence

import torch


def _single_maxsim_vectorized(
    corpus: torch.Tensor,
    seg_ids: torch.Tensor,
    n_segments: int,
    queries: torch.Tensor,
    block_queries: int = 2048,
) -> torch.Tensor:
    """Vectorized segmented-max via scatter_reduce_ (no Python loops over docs)."""
    device = corpus.device
    n_queries = queries.shape[0]
    out_parts = []

    for qstart in range(0, n_queries, block_queries):
        qend = min(qstart + block_queries, n_queries)
        q_block = queries[qstart:qend]
        bsz = q_block.shape[0]

        scores = q_block @ corpus.T  # (bsz, total_tokens)
        ms = torch.full(
            (bsz, n_segments), -float("inf"),
            device=device, dtype=scores.dtype,
        )
        ms.scatter_reduce_(
            1,
            seg_ids.unsqueeze(0).expand(bsz, -1),
            scores,
            reduce="amax",
            include_self=False,
        )
        out_parts.append(ms)

    return torch.cat(out_parts, dim=0)


def _single_maxsim_blocked_legacy(
    corpus: torch.Tensor,
    counts: list[int],
    num_segments: int,
    queries: torch.Tensor,
    block_bytes: int,
) -> torch.Tensor:
    """Fallback for CPU tensors or extremely tight memory — original blocked impl."""
    num_queries = queries.shape[0]
    bytes_per_score = queries.element_size()
    max_block = max(1, block_bytes // (num_queries * bytes_per_score))

    max_scores = torch.full(
        (num_queries, num_segments),
        -float("inf"),
        device=queries.device,
        dtype=queries.dtype,
    )

    total_rows = corpus.shape[0]
    segment_idx = 0
    seg_start = 0
    seg_end = counts[0] if num_segments else 0
    block_start = 0

    while block_start < total_rows and segment_idx < num_segments:
        block_end = min(block_start + max_block, total_rows)
        block_scores = queries @ corpus[block_start:block_end].T

        while segment_idx < num_segments and seg_end <= block_start:
            segment_idx += 1
            if segment_idx >= num_segments:
                break
            seg_start = seg_end
            seg_end = seg_start + counts[segment_idx]
        if segment_idx >= num_segments:
            break

        while segment_idx < num_segments and seg_start < block_end:
            local_start = max(seg_start, block_start)
            local_end = min(seg_end, block_end)
            col_start = local_start - block_start
            col_end = local_end - block_start
            segment_scores = block_scores[:, col_start:col_end]
            block_max = segment_scores.max(dim=1).values
            max_scores[:, segment_idx] = torch.maximum(
                max_scores[:, segment_idx], block_max
            )

            if seg_end <= block_end:
                segment_idx += 1
                if segment_idx >= num_segments:
                    break
                seg_start = seg_end
                seg_end = seg_start + counts[segment_idx]
            else:
                break

        block_start = block_end

    return max_scores


def single_maxsim(
    corpus: torch.Tensor,
    corpus_counts: torch.Tensor | Sequence[int],
    queries: torch.Tensor,
    block_bytes: int | None = None,
) -> torch.Tensor:
    if not isinstance(corpus, torch.Tensor):
        raise TypeError("corpus must be a torch.Tensor")
    if not isinstance(queries, torch.Tensor):
        raise TypeError("queries must be a torch.Tensor")

    if corpus.dtype != torch.float32:
        raise TypeError("corpus must be float32")
    if queries.dtype != torch.float32:
        raise TypeError("queries must be float32")
    if corpus.ndim != 2:
        raise ValueError("corpus must be 2D")
    if queries.ndim != 2:
        raise ValueError("queries must be 2D")

    if corpus.shape[1] != queries.shape[1]:
        raise ValueError("corpus and queries must have the same number of columns")
    if corpus.device != queries.device:
        raise ValueError("corpus and queries must be on the same device")

    if isinstance(corpus_counts, torch.Tensor):
        if corpus_counts.dtype not in (torch.int32, torch.int64):
            raise TypeError("corpus_counts must be int32 or int64")
        if corpus_counts.ndim != 1:
            raise ValueError("corpus_counts must be 1D")
        counts = corpus_counts.tolist()
    else:
        try:
            counts = list(corpus_counts)
        except TypeError as exc:
            raise TypeError(
                "corpus_counts must be a torch.Tensor or a sequence of ints"
            ) from exc

    if sum(counts) != corpus.shape[0]:
        raise ValueError("corpus_counts must sum to the number of rows in corpus")
    if any(count <= 0 for count in counts):
        raise ValueError("corpus_counts entries must be positive")

    num_segments = len(counts)

    # ---- GPU fast path: vectorized scatter_reduce ----
    if corpus.is_cuda:
        counts_t = torch.tensor(counts, device=corpus.device, dtype=torch.int64)
        seg_ids = torch.repeat_interleave(
            torch.arange(num_segments, device=corpus.device), counts_t,
        )
        # Choose query block size to stay within ~2 GB intermediate
        tokens_per_block = max(1, int(2e9 / (corpus.shape[0] * corpus.element_size())))
        return _single_maxsim_vectorized(
            corpus, seg_ids, num_segments, queries,
            block_queries=tokens_per_block,
        )

    # ---- CPU path: blocked legacy for memory-constrained builds ----
    if block_bytes is not None and block_bytes > 0:
        return _single_maxsim_blocked_legacy(
            corpus, counts, num_segments, queries, block_bytes,
        )

    # ---- CPU unblocked: single matmul + Python loop (small data) ----
    scores = queries @ corpus.T
    max_scores = torch.empty(
        (queries.shape[0], num_segments),
        device=queries.device,
        dtype=queries.dtype,
    )
    offset = 0
    for idx, count in enumerate(counts):
        max_scores[:, idx] = scores[:, offset : offset + count].max(dim=1).values
        offset += count

    return max_scores
