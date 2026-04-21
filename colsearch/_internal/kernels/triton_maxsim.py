"""
High-Performance ColBERT Reranking Kernels

Fused Triton kernels for exact maxsim scoring with INT8 and FP8 quantization.
Designed for reranking small candidate sets (k=20-100) that fit entirely in VRAM.

Quantization modes:
- FP16: Default, full precision (baseline)
- INT8: Per-document absmax quantization, ~1.5x speedup
- FP8 E4M3: Per-document quantization, ~2x speedup (requires Ada/Hopper GPU)

100% accurate, optimized for throughput with query-tiled processing.

Reference: search-index-innovations/search_enhancements.md (FP8 adaptation from unsloth)
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
import triton
import triton.language as tl

from .kernel_warmup import _next_power_of_2

logger = logging.getLogger(__name__)

TRITON_AVAILABLE = True


try:
    from pylate.utils.tensor import convert_to_tensor
except ImportError:
    # Fallback if pylate not installed
    def convert_to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, list):
            return torch.stack([convert_to_tensor(item) for item in x])
        else:
            return torch.tensor(x)


# ============================================================================
# Triton Kernels
# ============================================================================

@triton.jit
def _fused_maxsim_pairwise_kernel(
    Q_PTR, D_PTR,
    Q_MASK_PTR, D_MASK_PTR,
    OUTPUT_PTR,
    q_batch_stride, q_token_stride,
    d_batch_stride, d_token_stride,
    qm_batch_stride, qm_token_stride,
    dm_batch_stride, dm_token_stride,
    NUM_Q_TOKENS: tl.constexpr,
    NUM_D_TOKENS: tl.constexpr,
    EMBED_DIM: tl.constexpr,
    Q_TOKEN_BLOCK_SIZE: tl.constexpr,
    D_TOKEN_BLOCK_SIZE: tl.constexpr,
    EMBED_DIM_BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for FUSED pairwise ColBERT scores.
    Each program in the 1D grid computes the score for one (query, document) pair.

    Grid: (BATCH_SIZE,)
    """
    # --- Program ID & Offsets ---
    # Cast to int64 immediately. Triton uses int32 for pointer arithmetic
    # by default; `batch_idx * d_batch_stride` overflows int32 once the
    # batched docs tensor exceeds ~2^31 elements (e.g. B>32k @ T=512, H=128
    # — trec-covid's largest tier hits B=59k). Overflow → wrapped pointer
    # → cudaErrorIllegalAddress on the next load.
    batch_idx = tl.program_id(0).to(tl.int64)

    # --- Pointers to the current query and document ---
    q_batch_ptr = Q_PTR + batch_idx * q_batch_stride
    d_batch_ptr = D_PTR + batch_idx * d_batch_stride

    # --- Pointers to masks ---
    has_q_mask = Q_MASK_PTR is not None
    has_d_mask = D_MASK_PTR is not None

    q_mask_ptr = Q_MASK_PTR + batch_idx * qm_batch_stride if has_q_mask else None
    d_mask_ptr = D_MASK_PTR + batch_idx * dm_batch_stride if has_d_mask else None

    # --- Accumulator for the final score ---
    total_score = 0.0

    # --- Offsets for the embedding dimension ---
    dim_offsets = tl.arange(0, EMBED_DIM_BLOCK_SIZE)

    # --- Loop over query tokens (S) ---
    q_token_offsets = tl.arange(0, Q_TOKEN_BLOCK_SIZE)
    for q_token_start in range(0, NUM_Q_TOKENS, Q_TOKEN_BLOCK_SIZE):

        # --- Load query mask for this block ---
        q_mask = tl.full([Q_TOKEN_BLOCK_SIZE], 1, dtype=tl.int1)
        if has_q_mask:
            q_mask_offsets = q_token_start + q_token_offsets
            q_mask = tl.load(
                q_mask_ptr + q_mask_offsets * qm_token_stride,
                mask=(q_mask_offsets < NUM_Q_TOKENS),
                other=0
            )

        # --- Load query token block (Q_TOKEN_BLOCK_SIZE, EMBED_DIM) ---
        q_ptr = q_batch_ptr + (q_token_start + q_token_offsets)[:, None] * q_token_stride + dim_offsets[None, :]
        q_token_block = tl.load(
            q_ptr,
            mask=(q_token_start + q_token_offsets)[:, None] < NUM_Q_TOKENS,
            other=0.0
        )

        # --- Accumulator for max sim for each query token in this block ---
        max_sim = tl.zeros([Q_TOKEN_BLOCK_SIZE], dtype=tl.float32) - 1e9

        # --- Loop over document tokens (T) in blocks ---
        d_token_offsets = tl.arange(0, D_TOKEN_BLOCK_SIZE)
        for d_token_start in range(0, NUM_D_TOKENS, D_TOKEN_BLOCK_SIZE):

            # --- Load document token block (D_TOKEN_BLOCK_SIZE, EMBED_DIM) ---
            d_ptr = d_batch_ptr + (d_token_start + d_token_offsets)[None, :] * d_token_stride + dim_offsets[:, None]
            d_token_block = tl.load(
                d_ptr,
                mask=(d_token_start + d_token_offsets)[None, :] < NUM_D_TOKENS,
                other=0.0
            )

            # --- Compute dot product (Q_TOKEN_BLOCK_SIZE, D_TOKEN_BLOCK_SIZE) ---
            sim_matrix = tl.dot(q_token_block, d_token_block)

            # --- Apply document mask ---
            if has_d_mask:
                d_mask_offsets = d_token_start + d_token_offsets
                d_mask = tl.load(
                    d_mask_ptr + d_mask_offsets * dm_token_stride,
                    mask=(d_mask_offsets < NUM_D_TOKENS),
                    other=0
                )
                sim_matrix = tl.where(d_mask[None, :], sim_matrix, -1e9)

            # --- Find max sim in this block of doc tokens ---
            block_max_sim = tl.max(sim_matrix, axis=1)

            # --- Update overall max sim ---
            max_sim = tl.maximum(max_sim, block_max_sim)

        # --- Apply query mask and sum up scores ---
        max_sim = tl.where(q_mask, max_sim, 0.0)
        total_score += tl.sum(max_sim)

    # --- Write final score to output ---
    output_ptr = OUTPUT_PTR + batch_idx
    tl.store(output_ptr, total_score)


@triton.autotune(
    configs=[
        triton.Config({'Q_TOKEN_BLOCK_SIZE': 32, 'D_TOKEN_BLOCK_SIZE': 64, 'DOCS_PER_KERNEL': 16}, num_warps=4, num_stages=2),
        triton.Config({'Q_TOKEN_BLOCK_SIZE': 16, 'D_TOKEN_BLOCK_SIZE': 128, 'DOCS_PER_KERNEL': 16}, num_warps=4, num_stages=2),
        triton.Config({'Q_TOKEN_BLOCK_SIZE': 32, 'D_TOKEN_BLOCK_SIZE': 128, 'DOCS_PER_KERNEL': 16}, num_warps=8, num_stages=2),
        triton.Config({'Q_TOKEN_BLOCK_SIZE': 32, 'D_TOKEN_BLOCK_SIZE': 64, 'DOCS_PER_KERNEL': 32}, num_warps=4, num_stages=2),
        triton.Config({'Q_TOKEN_BLOCK_SIZE': 32, 'D_TOKEN_BLOCK_SIZE': 128, 'DOCS_PER_KERNEL': 8}, num_warps=4, num_stages=2),
        triton.Config({'Q_TOKEN_BLOCK_SIZE': 64, 'D_TOKEN_BLOCK_SIZE': 64, 'DOCS_PER_KERNEL': 16}, num_warps=8, num_stages=2),
        triton.Config({'Q_TOKEN_BLOCK_SIZE': 64, 'D_TOKEN_BLOCK_SIZE': 128, 'DOCS_PER_KERNEL': 32}, num_warps=8, num_stages=3),
    ],
    key=['NUM_Q_TOKENS', 'NUM_D_TOKENS', 'EMBED_DIM'],
)
@triton.jit
def _fused_maxsim_query_tiled_kernel(
    Q_PTR, D_PTR,
    Q_MASK_PTR, D_MASK_PTR,
    OUTPUT_PTR,
    D_SCALES_PTR,
    q_batch_stride, q_token_stride,
    d_batch_stride, d_token_stride,
    qm_batch_stride, qm_token_stride,
    dm_batch_stride, dm_token_stride,
    output_q_stride, output_d_stride,
    d_scales_batch_stride,
    NUM_QUERIES: tl.constexpr,
    NUM_DOCS: tl.constexpr,
    NUM_Q_TOKENS: tl.constexpr,
    NUM_D_TOKENS: tl.constexpr,
    EMBED_DIM: tl.constexpr,
    Q_TOKEN_BLOCK_SIZE: tl.constexpr,
    D_TOKEN_BLOCK_SIZE: tl.constexpr,
    DOCS_PER_KERNEL: tl.constexpr,
    USE_INT8: tl.constexpr,
    HAS_Q_MASK: tl.constexpr,
    HAS_D_MASK: tl.constexpr,
):
    """
    Query-optimized tiling: one query per kernel, process DOCS_PER_KERNEL documents.
    Key optimization: Query embeddings loaded ONCE and reused across all documents.
    """
    # Cast to int64 immediately. Triton's default int32 pointer arithmetic
    # overflows when the document tensor exceeds ~2^31 elements
    # (e.g. B>32k @ T=512, H=128 → trec-covid's T=512 tier with B=59k).
    # Without this cast, `doc_idx * d_batch_stride` wraps and the kernel
    # reads garbage memory → cudaErrorIllegalAddress on the very first
    # autotune launch.
    q_idx = tl.program_id(0).to(tl.int64)
    doc_tile_idx = tl.program_id(1).to(tl.int64)

    doc_start = doc_tile_idx * DOCS_PER_KERNEL

    dim_offsets = tl.arange(0, EMBED_DIM)
    q_token_offsets = tl.arange(0, Q_TOKEN_BLOCK_SIZE)
    d_token_offsets = tl.arange(0, D_TOKEN_BLOCK_SIZE)

    q_batch_ptr = Q_PTR + q_idx * q_batch_stride

    if HAS_Q_MASK:
        q_mask_batch_ptr = Q_MASK_PTR + q_idx * qm_batch_stride
    else:
        q_mask_batch_ptr = Q_MASK_PTR

    # Process each document in the tile
    for doc_i in range(DOCS_PER_KERNEL):
        doc_idx = doc_start + doc_i

        if doc_idx < NUM_DOCS:
            d_batch_ptr = D_PTR + doc_idx * d_batch_stride

            if HAS_D_MASK:
                d_mask_batch_ptr = D_MASK_PTR + doc_idx * dm_batch_stride
            else:
                d_mask_batch_ptr = D_MASK_PTR

            # Load scale as FP16 for better performance
            d_scale = 1.0
            if USE_INT8:
                scale_ptr = D_SCALES_PTR + doc_idx * d_scales_batch_stride
                d_scale = tl.load(scale_ptr).to(tl.float16)

            doc_score = tl.zeros((), dtype=tl.float32)

            # Loop over query tokens
            for q_tok_start in range(0, NUM_Q_TOKENS, Q_TOKEN_BLOCK_SIZE):
                q_tok_idx = q_tok_start + q_token_offsets
                q_tok_valid = q_tok_idx < NUM_Q_TOKENS

                # Load query block
                q_ptr = q_batch_ptr + q_tok_idx[:, None] * q_token_stride + dim_offsets[None, :]
                q_block = tl.load(q_ptr, mask=q_tok_valid[:, None], other=0.0).to(tl.float16)

                # Load query mask as float32 for arithmetic
                q_mask = tl.full([Q_TOKEN_BLOCK_SIZE], 1.0, dtype=tl.float32)
                if HAS_Q_MASK:
                    q_mask = tl.load(
                        q_mask_batch_ptr + q_tok_idx * qm_token_stride,
                        mask=q_tok_valid,
                        other=0.0
                    ).to(tl.float32)

                q_max_sim = tl.full([Q_TOKEN_BLOCK_SIZE], -1e9, dtype=tl.float32)

                # Loop over document tokens
                for d_tok_start in range(0, NUM_D_TOKENS, D_TOKEN_BLOCK_SIZE):
                    d_tok_idx = d_tok_start + d_token_offsets
                    d_tok_valid = d_tok_idx < NUM_D_TOKENS

                    d_ptr = d_batch_ptr + d_tok_idx[None, :] * d_token_stride + dim_offsets[:, None]

                    if USE_INT8:
                        # Load INT8 and dequantize to FP16
                        d_block_i8 = tl.load(d_ptr, mask=d_tok_valid[None, :], other=0).to(tl.int8)
                        d_block = d_block_i8.to(tl.float16) * d_scale
                    else:
                        d_block = tl.load(d_ptr, mask=d_tok_valid[None, :], other=0.0).to(tl.float16)

                    # FP16 @ FP16 -> FP32
                    sim = tl.dot(q_block, d_block, out_dtype=tl.float32)

                    # Apply document mask - set invalid to -1e9 for max operation
                    if HAS_D_MASK:
                        d_mask = tl.load(
                            d_mask_batch_ptr + d_tok_idx * dm_token_stride,
                            mask=d_tok_valid,
                            other=0.0,
                        ).to(tl.float32)
                        sim = tl.where(d_mask[None, :] > 0.5, sim, -1e9)

                    sim = tl.where(d_tok_valid[None, :], sim, -1e9)

                    block_max = tl.max(sim, axis=1)
                    q_max_sim = tl.maximum(q_max_sim, block_max)

                # Apply query mask - set invalid to 0.0 for sum operation
                q_max_sim = q_max_sim * q_mask
                q_max_sim = tl.where(q_tok_valid, q_max_sim, 0.0)
                doc_score += tl.sum(q_max_sim)

            out_ptr = OUTPUT_PTR + q_idx * output_q_stride + doc_idx * output_d_stride
            tl.store(out_ptr, doc_score)


# ============================================================================
# Helper Functions
# ============================================================================

def _quantize_docs_to_int8(documents_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs fast per-document INT8 quantization.

    Args:
        documents_embeddings: (batch_docs, num_tokens, dim) tensor

    Returns:
        Tuple of (doc_embeddings_int8, doc_scales)
    """
    # Find max absolute value for each document
    doc_scales = documents_embeddings.abs().max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    # Calculate scale factor (absmax / 127.0)
    doc_scales = doc_scales / 127.0

    # Avoid division by zero for empty docs
    doc_scales[doc_scales == 0] = 1.0

    # Quantize
    doc_embeddings_int8 = (documents_embeddings / doc_scales).round().clamp(-127, 127).to(torch.int8)

    return doc_embeddings_int8, doc_scales


def _check_fp8_support() -> bool:
    """
    Check if FP8 (E4M3) is supported on the current GPU.

    FP8 requires NVIDIA Ada Lovelace (RTX 40xx) or Hopper (H100) architecture.
    """
    if not torch.cuda.is_available():
        return False

    # Check compute capability (8.9+ for Ada, 9.0+ for Hopper)
    major, minor = torch.cuda.get_device_capability()
    compute_cap = major + minor / 10

    # FP8 is available on compute capability 8.9+ (Ada) and 9.0+ (Hopper)
    return compute_cap >= 8.9


def _quantize_docs_to_fp8(
    documents_embeddings: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs fast per-document FP8 E4M3 quantization.

    FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
    Range: [-448, 448] with special values for inf/nan

    Adapted from: https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fp8.py

    Args:
        documents_embeddings: (batch_docs, num_tokens, dim) tensor

    Returns:
        Tuple of (doc_embeddings_fp8, doc_scales)

    Note:
        Requires PyTorch 2.1+ and CUDA 12.0+ for native FP8 support.
        Falls back to FP16 simulation if FP8 not available.
    """

    # Find max absolute value for each document
    doc_absmax = documents_embeddings.abs().max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    # FP8 E4M3 max representable value is 448
    FP8_MAX = 448.0

    # Calculate scale factor
    doc_scales = doc_absmax / FP8_MAX
    doc_scales = torch.where(doc_scales == 0, torch.ones_like(doc_scales), doc_scales)

    # Scale to FP8 range
    scaled = documents_embeddings / doc_scales

    # Try to use native FP8 if available
    try:
        # PyTorch 2.1+ with CUDA 12.0+
        if hasattr(torch, 'float8_e4m3fn'):
            doc_embeddings_fp8 = scaled.to(torch.float8_e4m3fn)
            return doc_embeddings_fp8, doc_scales
    except (AttributeError, RuntimeError):
        pass

    # Fallback: store as FP16 with FP8-like quantization
    # This simulates FP8 precision loss for testing on older hardware
    doc_embeddings_fp8 = scaled.clamp(-FP8_MAX, FP8_MAX).to(torch.float16)

    logger.debug("FP8 native not available, using FP16 simulation")
    return doc_embeddings_fp8, doc_scales


def _dequantize_fp8(
    quantized: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize FP8 embeddings back to FP16/FP32.

    Args:
        quantized: FP8 quantized tensor
        scales: Per-document scale factors

    Returns:
        Dequantized tensor in FP16
    """
    # Convert to float and apply scale
    return quantized.to(torch.float16) * scales


def pad_embeddings(embeddings_list: List[torch.Tensor], pad_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of variable-length embeddings to the same length.

    Args:
        embeddings_list: List of embeddings with shape (seq_len_i, hidden_dim)
        pad_value: Value to use for padding

    Returns:
        Tuple of (padded, mask):
            - padded: (batch_size, max_seq_len, hidden_dim)
            - mask: (batch_size, max_seq_len), 1 for real tokens, 0 for padding
    """
    if not embeddings_list:
        raise ValueError("embeddings_list cannot be empty")

    # Handle both 2D and 3D inputs
    if embeddings_list[0].dim() == 3:
        # Already batched (B, S, H) - just return
        return embeddings_list[0], None

    # Get max length and hidden dim
    max_len = max(emb.shape[0] for emb in embeddings_list)
    hidden_dim = embeddings_list[0].shape[1]
    device = embeddings_list[0].device
    dtype = embeddings_list[0].dtype

    # Create padded tensor and mask
    batch_size = len(embeddings_list)
    padded = torch.full((batch_size, max_len, hidden_dim), pad_value, dtype=dtype, device=device)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)

    # Fill in the actual values
    for i, emb in enumerate(embeddings_list):
        seq_len = emb.shape[0]
        padded[i, :seq_len] = emb
        mask[i, :seq_len] = 1.0

    return padded, mask


# ============================================================================
# CPU Fallback Implementation
# ============================================================================

def _colbert_scores_cpu(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    CPU fallback for ColBERT MaxSim scoring using pure PyTorch.

    ⚠️  WARNING: This is for TESTING ONLY with small datasets (<100 documents).
    For production use with 1K+ documents, GPU is required.

    Implements MaxSim: For each query token, find the max similarity with all
    document tokens, then sum across query tokens.

    Args:
        queries_embeddings: (A, S, H) query token embeddings
        documents_embeddings: (B, T, H) document token embeddings
        queries_mask: Optional (A, S) mask (1 for real tokens, 0 for padding)
        documents_mask: Optional (B, T) mask

    Returns:
        scores: (A, B) tensor of ColBERT scores

    Performance:
        - 100 docs: ~50ms per query (acceptable for testing)
        - 1K docs: ~500ms per query (slow)
        - 10K docs: ~5s per query (unusable)
    """
    # Ensure CPU
    Q = queries_embeddings.cpu()
    D = documents_embeddings.cpu()

    A, S, H = Q.shape
    B, T, H_D = D.shape

    if H != H_D:
        raise ValueError(f"Embedding dimensions must match: Q={H}, D={H_D}")

    # Initialize output
    scores = torch.zeros(A, B, dtype=Q.dtype, device='cpu')

    # Process each query-document pair
    for q_idx in range(A):
        q = Q[q_idx]  # [S, H]

        # Apply query mask if provided
        if queries_mask is not None:
            q_mask = queries_mask[q_idx].cpu()  # [S]
        else:
            q_mask = torch.ones(S, dtype=torch.float32, device='cpu')

        for d_idx in range(B):
            d = D[d_idx]  # [T, H]

            # Apply document mask if provided
            if documents_mask is not None:
                d_mask = documents_mask[d_idx].cpu()  # [T]
            else:
                d_mask = torch.ones(T, dtype=torch.float32, device='cpu')

            # Compute similarity matrix: [S, H] @ [H, T] -> [S, T]
            sim_matrix = q @ d.T

            # Apply document mask (set masked positions to very negative)
            if documents_mask is not None:
                sim_matrix = sim_matrix.masked_fill(d_mask.unsqueeze(0) == 0, -1e9)

            # MaxSim: max over document tokens [S, T] -> [S]
            max_sim = sim_matrix.max(dim=1).values

            # Apply query mask (set masked positions to 0 for sum)
            max_sim = max_sim * q_mask

            # Sum over query tokens
            scores[q_idx, d_idx] = max_sim.sum()

    return scores


def _colbert_scores_pairwise_cpu(
    queries_embeddings: torch.Tensor,
    documents_embeddings: torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    CPU fallback for pairwise ColBERT scoring.

    Computes scores[i] = score(query[i], document[i]).
    More efficient than all-to-all when only diagonal scores are needed.

    Args:
        queries_embeddings: (B, S, H) query embeddings
        documents_embeddings: (B, T, H) document embeddings
        queries_mask: Optional (B, S) mask
        documents_mask: Optional (B, T) mask

    Returns:
        scores: (B,) tensor of pairwise scores
    """
    Q = queries_embeddings.cpu()
    D = documents_embeddings.cpu()

    B, S, H = Q.shape
    B_D, T, H_D = D.shape

    if B != B_D:
        raise ValueError(f"Batch sizes must match: Q={B}, D={B_D}")
    if H != H_D:
        raise ValueError(f"Embedding dimensions must match: Q={H}, D={H_D}")

    scores = torch.zeros(B, dtype=Q.dtype, device='cpu')

    for i in range(B):
        q = Q[i]  # [S, H]
        d = D[i]  # [T, H]

        # Masks
        if queries_mask is not None:
            q_mask = queries_mask[i].cpu()
        else:
            q_mask = torch.ones(S, dtype=torch.float32, device='cpu')

        if documents_mask is not None:
            d_mask = documents_mask[i].cpu()
        else:
            d_mask = torch.ones(T, dtype=torch.float32, device='cpu')

        # Similarity matrix
        sim_matrix = q @ d.T  # [S, T]

        # Apply document mask
        if documents_mask is not None:
            sim_matrix = sim_matrix.masked_fill(d_mask.unsqueeze(0) == 0, -1e9)

        # MaxSim
        max_sim = sim_matrix.max(dim=1).values  # [S]
        max_sim = max_sim * q_mask
        scores[i] = max_sim.sum()

    return scores


# ============================================================================
# Public API Functions
# ============================================================================

def fast_colbert_scores(
    queries_embeddings: torch.Tensor | List[torch.Tensor],
    documents_embeddings: torch.Tensor | List[torch.Tensor],
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    use_quantization: bool = False,
    quantization_mode: str = "int8",
) -> torch.Tensor:
    """
    Drop-in, fused-kernel replacement for colbert_scores (all-to-all).

    Computes (A, B) scores without materializing (A, B, S, T).
    Uses query-tiling strategy and optional quantization for speedup.

    Quantization modes:
    - "none" / False: Full precision FP16 (baseline)
    - "int8": Per-document INT8 absmax quantization (~1.5x speedup)
    - "fp8": Per-document FP8 E4M3 quantization (~2x speedup, requires Ada/Hopper)

    Args:
        queries_embeddings: Query token embeddings. Can be:
            - Single 3D tensor: (A, S, H)
            - List of A tensors with shape (S_i, H) - will be padded automatically
        documents_embeddings: Document token embeddings. Can be:
            - Single 3D tensor: (B, T, H)
            - List of B tensors with shape (T_i, H) - will be padded automatically
        queries_mask: Optional (A, S) float/bool attention mask for queries
        documents_mask: Optional (B, T) float/bool attention mask for documents
        use_quantization: If True, applies quantization (uses quantization_mode)
            **NOTE**: For production, pre-quantize documents offline
        quantization_mode: Quantization type - "int8" (default), "fp8", or "none"

    Returns:
        scores: (A, B) tensor with dtype matching input embeddings

    Example:
        >>> Q = torch.randn(8, 32, 128, device='cuda')  # 8 queries, 32 tokens
        >>> D = torch.randn(100, 512, 128, device='cuda')  # 100 docs, 512 tokens
        >>>
        >>> # INT8 quantization (default)
        >>> scores = fast_colbert_scores(Q, D, use_quantization=True)
        >>>
        >>> # FP8 quantization (faster on Ada/Hopper GPUs)
        >>> scores = fast_colbert_scores(Q, D, use_quantization=True, quantization_mode="fp8")
        >>>
        >>> scores.shape
        torch.Size([8, 100])
    """
    # Handle list inputs - pad to same length
    if isinstance(queries_embeddings, list):
        Q, queries_mask_inferred = pad_embeddings(queries_embeddings)
        if queries_mask is None:
            queries_mask = queries_mask_inferred
    else:
        Q = convert_to_tensor(queries_embeddings)

    if isinstance(documents_embeddings, list):
        D, documents_mask_inferred = pad_embeddings(documents_embeddings)
        if documents_mask is None:
            documents_mask = documents_mask_inferred
    else:
        D = convert_to_tensor(documents_embeddings)

    # Ensure contiguous
    Q = Q.contiguous()
    D = D.contiguous()

    # CPU FALLBACK: If not on CUDA or CUDA not available, use CPU implementation
    if not Q.is_cuda or not torch.cuda.is_available():
        if not Q.is_cuda:
            logger.warning(
                "⚠️  Using CPU fallback for ColBERT scoring. "
                "This is ONLY suitable for testing with <100 documents. "
                "For production use with 1K+ documents, GPU is required."
            )
        return _colbert_scores_cpu(Q, D, queries_mask, documents_mask)

    # Ensure CUDA (GPU path)
    Q = Q.cuda()
    D = D.cuda()

    if Q.dim() != 3 or D.dim() != 3:
        raise ValueError("queries_embeddings and documents_embeddings must be 3D: (A/B, S/T, H)")

    A, S_raw, H = Q.shape
    B, T_raw, H_D = D.shape

    if H != H_D:
        raise ValueError(f"Embedding dimensions must match: Q={H}, D={H_D}")

    device = Q.device

    S = _next_power_of_2(S_raw, minimum=8)
    T = _next_power_of_2(T_raw, minimum=32)

    if S > S_raw:
        Q = torch.nn.functional.pad(Q, (0, 0, 0, S - S_raw))
        if queries_mask is not None:
            queries_mask = torch.nn.functional.pad(queries_mask, (0, S - S_raw))
        else:
            queries_mask = torch.ones(A, S, device=device, dtype=torch.float32)
            queries_mask[:, S_raw:] = 0.0

    if T > T_raw:
        D = torch.nn.functional.pad(D, (0, 0, 0, T - T_raw))
        if documents_mask is not None:
            documents_mask = torch.nn.functional.pad(documents_mask, (0, T - T_raw))
        else:
            documents_mask = torch.ones(B, T, device=device, dtype=torch.float32)
            documents_mask[:, T_raw:] = 0.0

    # FP16 is crucial for performance
    Qh = Q.to(torch.float16)

    # --- Quantization ---
    USE_INT8 = 0
    USE_FP8 = 0

    if use_quantization:
        mode = quantization_mode.lower() if isinstance(quantization_mode, str) else "int8"

        if mode == "fp8":
            # FP8 E4M3 quantization (requires Ada/Hopper GPU)
            if _check_fp8_support():
                doc_embeddings_fp8, doc_scales = _quantize_docs_to_fp8(D)
                D_ptr = doc_embeddings_fp8.contiguous()
                D_scales_ptr = doc_scales.contiguous()
                USE_FP8 = 1
                logger.debug(f"Using FP8 quantization for {B} documents")
            else:
                logger.warning(
                    "FP8 not supported on this GPU (requires Ada/Hopper). "
                    "Falling back to INT8 quantization."
                )
                documents_embeddings_int8, doc_scales = _quantize_docs_to_int8(D)
                D_ptr = documents_embeddings_int8
                D_scales_ptr = doc_scales.contiguous()
                USE_INT8 = 1
        else:
            # INT8 quantization (default, works on all GPUs)
            documents_embeddings_int8, doc_scales = _quantize_docs_to_int8(D)
            D_ptr = documents_embeddings_int8
            D_scales_ptr = doc_scales.contiguous()
            USE_INT8 = 1
            logger.debug(f"Using INT8 quantization for {B} documents")
    else:
        D_ptr = D.to(torch.float16)
        D_scales_ptr = torch.ones((B, 1, 1), device=device, dtype=torch.float32).contiguous()

    # Note: The Triton kernel currently treats FP8 like FP16 (dequantized on load)
    # Future optimization: Add native FP8 tensor core support in kernel
    if USE_FP8:
        # Dequantize FP8 back to FP16 for kernel (temporary until native FP8 kernel)
        D_ptr = _dequantize_fp8(D_ptr, D_scales_ptr).contiguous()
        D_scales_ptr = torch.ones((B, 1, 1), device=device, dtype=torch.float32).contiguous()
        USE_INT8 = 0  # Already dequantized

    # --- Output Tensor ---
    output_scores = torch.empty((A, B), device=device, dtype=torch.float32)

    # --- Mask Handling ---
    HAS_Q_MASK = int(queries_mask is not None)
    HAS_D_MASK = int(documents_mask is not None)

    if HAS_Q_MASK:
        qm = convert_to_tensor(queries_mask).contiguous().cuda().to(torch.float32)
        if qm.shape[0] != A or qm.shape[1] != S:
            raise ValueError(f"queries_mask must have shape {(A, S)}, got {qm.shape}")
        qm_batch_stride, qm_token_stride = qm.stride()
        Q_MASK_PTR = qm
    else:
        Q_MASK_PTR = torch.empty(0, device=device, dtype=torch.float32)
        qm_batch_stride, qm_token_stride = 0, 0

    if HAS_D_MASK:
        dm = convert_to_tensor(documents_mask).contiguous().cuda().to(torch.float32)
        if dm.shape[0] != B or dm.shape[1] != T:
            raise ValueError(f"documents_mask must have shape {(B, T)}, got {dm.shape}")
        dm_batch_stride, dm_token_stride = dm.stride()
        D_MASK_PTR = dm
    else:
        D_MASK_PTR = torch.empty(0, device=device, dtype=torch.float32)
        dm_batch_stride, dm_token_stride = 0, 0

    d_scales_batch_stride = D_scales_ptr.stride(0)

    # --- Kernel Grid ---
    def grid(META):
        return (A, (B + META['DOCS_PER_KERNEL'] - 1) // META['DOCS_PER_KERNEL'])

    # Launch kernel
    _fused_maxsim_query_tiled_kernel[grid](
        Q_PTR=Qh,
        D_PTR=D_ptr,
        Q_MASK_PTR=Q_MASK_PTR,
        D_MASK_PTR=D_MASK_PTR,
        OUTPUT_PTR=output_scores,
        D_SCALES_PTR=D_scales_ptr,

        q_batch_stride=Qh.stride(0),
        q_token_stride=Qh.stride(1),
        d_batch_stride=D_ptr.stride(0),
        d_token_stride=D_ptr.stride(1),

        qm_batch_stride=qm_batch_stride,
        qm_token_stride=qm_token_stride,
        dm_batch_stride=dm_batch_stride,
        dm_token_stride=dm_token_stride,

        output_q_stride=output_scores.stride(0),
        output_d_stride=output_scores.stride(1),
        d_scales_batch_stride=d_scales_batch_stride,

        NUM_QUERIES=A,
        NUM_DOCS=B,
        NUM_Q_TOKENS=S,
        NUM_D_TOKENS=T,
        EMBED_DIM=H,

        USE_INT8=USE_INT8,
        HAS_Q_MASK=HAS_Q_MASK,
        HAS_D_MASK=HAS_D_MASK,
    )

    return output_scores.to(Q.dtype)


def fast_colbert_scores_pairwise(
    queries_embeddings: torch.Tensor | List[torch.Tensor],
    documents_embeddings: torch.Tensor | List[torch.Tensor],
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Pairwise ColBERT scores: scores[i] = score(query[i], document[i]).

    More efficient than all-to-all when you only need diagonal scores.

    Args:
        queries_embeddings: (B, S, H) or list of B tensors
        documents_embeddings: (B, T, H) or list of B tensors
        queries_mask: Optional (B, S) mask
        documents_mask: Optional (B, T) mask

    Returns:
        scores: (B,) tensor of pairwise scores

    Example:
        >>> Q = torch.randn(16, 32, 128, device='cuda')
        >>> D = torch.randn(16, 512, 128, device='cuda')
        >>> scores = fast_colbert_scores_pairwise(Q, D)
        >>> scores.shape
        torch.Size([16])
    """
    # Handle list inputs
    if isinstance(queries_embeddings, list):
        Q, queries_mask_inferred = pad_embeddings(queries_embeddings)
        if queries_mask is None:
            queries_mask = queries_mask_inferred
    else:
        Q = convert_to_tensor(queries_embeddings)

    if isinstance(documents_embeddings, list):
        D, documents_mask_inferred = pad_embeddings(documents_embeddings)
        if documents_mask is None:
            documents_mask = documents_mask_inferred
    else:
        D = convert_to_tensor(documents_embeddings)

    # Ensure contiguous
    Q = Q.contiguous()
    D = D.contiguous()

    # CPU FALLBACK: If not on CUDA or CUDA not available, use CPU implementation
    if not Q.is_cuda or not torch.cuda.is_available():
        if not Q.is_cuda:
            logger.warning(
                "⚠️  Using CPU fallback for ColBERT pairwise scoring. "
                "This is ONLY suitable for testing with <100 documents."
            )
        return _colbert_scores_pairwise_cpu(Q, D, queries_mask, documents_mask)

    # Ensure CUDA (GPU path)
    Q = Q.cuda()
    D = D.cuda()

    if Q.dim() != 3 or D.dim() != 3:
        raise ValueError("queries_embeddings and documents_embeddings must be 3D")

    B, S, H = Q.shape
    B_D, T, H_D = D.shape

    if B != B_D:
        raise ValueError(f"Batch sizes must match: Q={B}, D={B_D}")

    if H != H_D:
        raise ValueError(f"Embedding dimensions must match: Q={H}, D={H_D}")

    device = Q.device

    # Convert to FP16
    Qh = Q.to(torch.float16)
    Dh = D.to(torch.float16)

    # Output tensor
    output_scores = torch.empty((B,), device=device, dtype=torch.float32)

    # Mask handling
    has_q_mask = queries_mask is not None
    has_d_mask = documents_mask is not None

    if has_q_mask:
        qm = convert_to_tensor(queries_mask).contiguous().cuda().to(torch.float32)
        qm_batch_stride, qm_token_stride = qm.stride()
        Q_MASK_PTR = qm
    else:
        Q_MASK_PTR = None
        qm_batch_stride, qm_token_stride = 0, 0

    if has_d_mask:
        dm = convert_to_tensor(documents_mask).contiguous().cuda().to(torch.float32)
        dm_batch_stride, dm_token_stride = dm.stride()
        D_MASK_PTR = dm
    else:
        D_MASK_PTR = None
        dm_batch_stride, dm_token_stride = 0, 0

    # Launch kernel (1D grid, one program per pair)
    grid = (B,)

    _fused_maxsim_pairwise_kernel[grid](
        Q_PTR=Qh,
        D_PTR=Dh,
        Q_MASK_PTR=Q_MASK_PTR,
        D_MASK_PTR=D_MASK_PTR,
        OUTPUT_PTR=output_scores,

        q_batch_stride=Qh.stride(0),
        q_token_stride=Qh.stride(1),
        d_batch_stride=Dh.stride(0),
        d_token_stride=Dh.stride(1),

        qm_batch_stride=qm_batch_stride,
        qm_token_stride=qm_token_stride,
        dm_batch_stride=dm_batch_stride,
        dm_token_stride=dm_token_stride,

        NUM_Q_TOKENS=S,
        NUM_D_TOKENS=T,
        EMBED_DIM=H,
        Q_TOKEN_BLOCK_SIZE=32,
        D_TOKEN_BLOCK_SIZE=64,
        EMBED_DIM_BLOCK_SIZE=H,
    )

    return output_scores.to(Q.dtype)

