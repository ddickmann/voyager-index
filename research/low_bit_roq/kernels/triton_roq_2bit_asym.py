"""
Asymmetric 2-bit MaxSim kernel: high-bit query × 2-bit doc.

Plan reference: Phase A2. The existing
``_roq_maxsim_2bit_kernel`` is *symmetric* (2-bit query × 2-bit doc) which
is one of the reasons existing 2-bit ROQ is unusable. This kernel mirrors
the 1-bit asymmetric architecture in
``voyager_index/_internal/kernels/triton_roq.py`` for 2-bit docs.

Doc encoding (matches ``RotationalQuantizer.quantize`` for ``num_bits=2``,
grouped layout):

- Codes ∈ {0, 1, 2, 3} → split offline into two 1-bit planes (low / high)
  packed into int32 words, persisted alongside the existing 4-codes-per-
  byte stream. Hot-path inner loop is then popcount-only.
- Per-group (scale, offset, code_sum) (float32 each).

Reconstruction: ``d_hat[i] = scale_{g(i)} · code[i] + offset_{g(i)}``
with ``code[i] = bit0[i] + 2 · bit1[i]``.

Estimated dot per (q, d), four-term affine:

    dot_qd = Σ_g [
        q_offset · scale_g · code_sum_g
      + q_offset · offset_g · group_size
      + q_scale  · scale_g · plane_dot_g
      + q_scale  · offset_g · q_code_sum_g
    ]

where
    plane_dot_g  = Σ_{i∈g} c_q[i] · code[i]
                 = Σ_k 2^k · ( popc(q_k & bit0)_g + 2·popc(q_k & bit1)_g )
    q_code_sum_g = Σ_{i∈g} c_q[i]      (precomputed by the launcher,
                                        passed in as Q_GROUP_SUM)
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _popc(x):
    return tl.inline_asm_elementwise(
        "popc.b32 $0, $1;",
        "=r,r",
        [x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 128}, num_warps=8, num_stages=2),
    ],
    key=["n_d_tokens", "dim", "QUERY_BITS"],
)
@triton.jit
def _roq_maxsim_2bit_asym_kernel(
    Q_PLANES_PTR,
    Q_GROUP_SUM_PTR,
    D_BIT0_PTR,
    D_BIT1_PTR,
    D_SCALES_PTR,
    D_OFFSETS_PTR,
    D_CODE_SUM_PTR,
    Q_META_PTR,
    Q_MASK_PTR,
    D_MASK_PTR,
    OUTPUT_PTR,
    q_batch_stride,
    q_token_stride,
    q_plane_stride,
    q_word_stride,
    qgs_batch_stride,
    qgs_token_stride,
    qgs_group_stride,
    d_batch_stride,
    d_token_stride,
    d_word_stride,
    d_scales_batch_stride,
    d_scales_token_stride,
    d_scales_group_stride,
    q_meta_token_stride,
    q_mask_batch_stride,
    q_mask_token_stride,
    d_mask_batch_stride,
    d_mask_token_stride,
    output_q_stride,
    output_d_stride,
    n_queries,
    n_docs,
    n_q_tokens,
    n_d_tokens,
    dim,
    n_groups,
    group_size,
    N_WORDS: tl.constexpr,
    GROUP_WORDS: tl.constexpr,
    QUERY_BITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_idx = tl.program_id(0)
    d_idx = tl.program_id(1)

    q_ptr_base = Q_PLANES_PTR + q_idx * q_batch_stride
    qgs_base = Q_GROUP_SUM_PTR + q_idx * qgs_batch_stride
    d_bit0_base = D_BIT0_PTR + d_idx * d_batch_stride
    d_bit1_base = D_BIT1_PTR + d_idx * d_batch_stride
    d_scales_base = D_SCALES_PTR + d_idx * d_scales_batch_stride
    d_offsets_base = D_OFFSETS_PTR + d_idx * d_scales_batch_stride
    d_code_sum_base = D_CODE_SUM_PTR + d_idx * d_scales_batch_stride
    q_meta_ptr_base = Q_META_PTR + q_idx * n_q_tokens * q_meta_token_stride

    total_score = 0.0

    for i in range(n_q_tokens):
        q_token_active = tl.load(
            Q_MASK_PTR + q_idx * q_mask_batch_stride + i * q_mask_token_stride
        ) > 0
        if q_token_active:
            q_meta_ptr = q_meta_ptr_base + i * q_meta_token_stride
            q_scale = tl.load(q_meta_ptr + 0)
            q_offset = tl.load(q_meta_ptr + 1)

            max_sim = -1.0e9
            for j_start in range(0, n_d_tokens, BLOCK_D):
                j_offsets = j_start + tl.arange(0, BLOCK_D)
                j_mask = j_offsets < n_d_tokens
                d_token_active = tl.load(
                    D_MASK_PTR
                    + d_idx * d_mask_batch_stride
                    + j_offsets * d_mask_token_stride,
                    mask=j_mask,
                    other=0,
                ) > 0

                est = tl.zeros([BLOCK_D], dtype=tl.float32)

                q_token_base = q_ptr_base + i * q_token_stride
                qgs_token_base = qgs_base + i * qgs_token_stride
                d_bit0_token = d_bit0_base + j_offsets * d_token_stride
                d_bit1_token = d_bit1_base + j_offsets * d_token_stride

                for grp in range(n_groups):
                    d_scale_g = tl.load(
                        d_scales_base
                        + j_offsets * d_scales_token_stride
                        + grp * d_scales_group_stride,
                        mask=j_mask,
                        other=0.0,
                    )
                    d_offset_g = tl.load(
                        d_offsets_base
                        + j_offsets * d_scales_token_stride
                        + grp * d_scales_group_stride,
                        mask=j_mask,
                        other=0.0,
                    )
                    d_code_sum_g = tl.load(
                        d_code_sum_base
                        + j_offsets * d_scales_token_stride
                        + grp * d_scales_group_stride,
                        mask=j_mask,
                        other=0.0,
                    )
                    q_code_sum_g = tl.load(
                        qgs_token_base + grp * qgs_group_stride
                    )

                    plane_dot_g = tl.zeros([BLOCK_D], dtype=tl.float32)
                    base_word = grp * GROUP_WORDS
                    for w in range(GROUP_WORDS):
                        word_idx = base_word + w
                        d_b0 = tl.load(
                            d_bit0_token + word_idx * d_word_stride,
                            mask=j_mask,
                            other=0,
                        ).to(tl.int32)
                        d_b1 = tl.load(
                            d_bit1_token + word_idx * d_word_stride,
                            mask=j_mask,
                            other=0,
                        ).to(tl.int32)
                        weight = 1.0
                        for k in tl.static_range(0, QUERY_BITS):
                            q_planek = tl.load(
                                q_token_base
                                + k * q_plane_stride
                                + word_idx * q_word_stride
                            ).to(tl.int32)
                            p0 = _popc(q_planek & d_b0).to(tl.float32)
                            p1 = _popc(q_planek & d_b1).to(tl.float32)
                            plane_dot_g += weight * (p0 + 2.0 * p1)
                            weight *= 2.0

                    est += (
                        q_offset * d_scale_g * d_code_sum_g
                        + q_offset * d_offset_g * group_size
                        + q_scale * d_scale_g * plane_dot_g
                        + q_scale * d_offset_g * q_code_sum_g
                    )

                sim = tl.where(j_mask & d_token_active, est, -1.0e9)
                block_max = tl.max(sim, axis=0)
                if block_max > max_sim:
                    max_sim = block_max
            total_score += max_sim

    out_ptr = OUTPUT_PTR + q_idx * output_q_stride + d_idx * output_d_stride
    tl.store(out_ptr, total_score)


def roq_maxsim_2bit_asym(
    queries_planes: torch.Tensor,
    queries_meta: torch.Tensor,
    queries_group_sum: torch.Tensor,
    docs_bit0: torch.Tensor,
    docs_bit1: torch.Tensor,
    docs_scales: torch.Tensor,
    docs_offsets: torch.Tensor,
    docs_code_sum: torch.Tensor,
    *,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    n_groups: int | None = None,
) -> torch.Tensor:
    """Asymmetric 2-bit MaxSim launcher.

    Args:
      queries_planes:    (A, S, query_bits, n_words) int32
      queries_meta:      (A, S, M) float32 — first two columns must be
                         ``[scale, offset]``.
      queries_group_sum: (A, S, n_groups) float32 — Σ_{i∈g} c_q[i]
                         precomputed by the host.
      docs_bit0:         (B, T, n_words) int32 — packed low bit
      docs_bit1:         (B, T, n_words) int32 — packed high bit
      docs_scales:       (B, T, n_groups) float32
      docs_offsets:      (B, T, n_groups) float32
      docs_code_sum:     (B, T, n_groups) float32 — Σ_{i∈g} code[i]
    """
    A, S = queries_planes.shape[:2]
    B, T = docs_bit0.shape[:2]
    n_words = docs_bit0.shape[-1]
    query_bits = queries_planes.shape[2]
    dim = n_words * 32
    if n_groups is None:
        n_groups = docs_scales.shape[-1]
    if n_words % n_groups != 0:
        raise ValueError(
            f"n_words={n_words} must be divisible by n_groups={n_groups}"
        )
    group_words = n_words // n_groups
    group_size = group_words * 32

    if queries_mask is None:
        queries_mask = torch.ones(
            (A, S), dtype=torch.float32, device=queries_planes.device
        )
    else:
        queries_mask = queries_mask.to(
            device=queries_planes.device, dtype=torch.float32
        )
    if documents_mask is None:
        documents_mask = torch.ones(
            (B, T), dtype=torch.float32, device=docs_bit0.device
        )
    else:
        documents_mask = documents_mask.to(
            device=docs_bit0.device, dtype=torch.float32
        )

    scores = torch.empty((A, B), dtype=torch.float32, device=queries_planes.device)
    grid = (A, B)
    _roq_maxsim_2bit_asym_kernel[grid](
        Q_PLANES_PTR=queries_planes,
        Q_GROUP_SUM_PTR=queries_group_sum,
        D_BIT0_PTR=docs_bit0,
        D_BIT1_PTR=docs_bit1,
        D_SCALES_PTR=docs_scales,
        D_OFFSETS_PTR=docs_offsets,
        D_CODE_SUM_PTR=docs_code_sum,
        Q_META_PTR=queries_meta,
        Q_MASK_PTR=queries_mask,
        D_MASK_PTR=documents_mask,
        OUTPUT_PTR=scores,
        q_batch_stride=queries_planes.stride(0),
        q_token_stride=queries_planes.stride(1),
        q_plane_stride=queries_planes.stride(2),
        q_word_stride=queries_planes.stride(3),
        qgs_batch_stride=queries_group_sum.stride(0),
        qgs_token_stride=queries_group_sum.stride(1),
        qgs_group_stride=queries_group_sum.stride(2),
        d_batch_stride=docs_bit0.stride(0),
        d_token_stride=docs_bit0.stride(1),
        d_word_stride=docs_bit0.stride(2),
        d_scales_batch_stride=docs_scales.stride(0),
        d_scales_token_stride=docs_scales.stride(1),
        d_scales_group_stride=docs_scales.stride(2),
        q_meta_token_stride=queries_meta.stride(1),
        q_mask_batch_stride=queries_mask.stride(0),
        q_mask_token_stride=queries_mask.stride(1),
        d_mask_batch_stride=documents_mask.stride(0),
        d_mask_token_stride=documents_mask.stride(1),
        output_q_stride=scores.stride(0),
        output_d_stride=scores.stride(1),
        n_queries=A,
        n_docs=B,
        n_q_tokens=S,
        n_d_tokens=T,
        dim=dim,
        n_groups=n_groups,
        group_size=group_size,
        N_WORDS=n_words,
        GROUP_WORDS=group_words,
        QUERY_BITS=query_bits,
    )
    return scores


# ---------------------------------------------------------------------------
# Encoder helpers (used by integration.persist_2bit_asym_layout)
# ---------------------------------------------------------------------------


def split_2bit_codes_to_bit_planes(
    packed_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the existing 4-codes-per-byte 2-bit RoQ packing into two
    int32-word bit-planes (low / high) suitable for the asymmetric kernel.

    Input: ``(N, n_bytes)`` uint8 with 4 codes per byte (high-pos first
    packing matching ``RotationalQuantizer._pack_lowbit_codes``).
    Output: ``(low_words, high_words)``, each ``(N, n_bytes // 4)`` int32.
    """
    if packed_codes.dtype != np.uint8:
        packed_codes = packed_codes.astype(np.uint8)
    n, n_bytes = packed_codes.shape
    codes = np.zeros((n, n_bytes * 4), dtype=np.uint8)
    codes[:, 0::4] = (packed_codes >> 6) & 0x3
    codes[:, 1::4] = (packed_codes >> 4) & 0x3
    codes[:, 2::4] = (packed_codes >> 2) & 0x3
    codes[:, 3::4] = packed_codes & 0x3
    bit0 = (codes & 0x1).astype(np.uint8)
    bit1 = ((codes >> 1) & 0x1).astype(np.uint8)
    return _pack_bits_into_int32(bit0), _pack_bits_into_int32(bit1)


def _pack_bits_into_int32(bits: np.ndarray) -> np.ndarray:
    n, d = bits.shape
    if d % 32 != 0:
        pad = (-d) % 32
        bits = np.pad(bits, ((0, 0), (0, pad)))
        d = bits.shape[1]
    packed_bytes = np.packbits(bits, axis=1, bitorder="little")
    return packed_bytes.view(np.int32).copy()


def encode_query_for_2bit_asym(
    rotated_queries: np.ndarray,
    *,
    query_bits: int,
    group_size: int,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the (planes, meta, group_sum) tensors the kernel consumes.

    ``rotated_queries`` shape (N, dim), already FWHT-rotated.
    """
    if query_bits not in (4, 6, 8):
        raise ValueError("query_bits ∈ {4, 6, 8}")
    levels = float((1 << query_bits) - 1)
    min_vals = rotated_queries.min(axis=1)
    max_vals = rotated_queries.max(axis=1)
    ranges = np.where((max_vals - min_vals) < 1e-6, 1.0, max_vals - min_vals)
    scales = ranges / levels
    quant = np.round(
        (rotated_queries - min_vals[:, None]) / scales[:, None]
    ).clip(0, levels).astype(np.uint8)
    planes = []
    for k in range(query_bits):
        plane = ((quant >> k) & 0x01).astype(np.uint8)
        words = _pack_bits_into_int32(plane)
        planes.append(words)
    planes_stack = np.stack(planes, axis=1)
    quant_grouped = quant.reshape(quant.shape[0], n_groups, group_size).astype(np.float32)
    group_sum = quant_grouped.sum(axis=2)
    code_sum = quant.astype(np.float32).sum(axis=1)
    meta = np.stack(
        [scales.astype(np.float32), min_vals.astype(np.float32), code_sum],
        axis=1,
    )
    return planes_stack, meta, group_sum
