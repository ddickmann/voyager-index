"""
rroq4_riem fused MaxSim kernel — Riemannian-aware 4-bit asymmetric ROQ.

Companion to ``triton_roq_rroq158.py`` but with 4-bit asymmetric per-group
residuals instead of ternary 1.58-bit. ~1.9× larger on disk than rroq158
and within ≈ 0.5% NDCG@10 of fp16 on production multi-vector data — the
"no-degradation" lane.

Score formula per (q_token, d_token):

    sim(q, d) = cos_norm[d] * <q_amb, c_d>
              + sin_norm[d] * <q_rot, dequant_4b(r_d)>

with cos_norm / sin_norm pre-baked to fold ``norm_d``, and the inner
4-bit dot product expanded as

    <q_rot, dequant(r)> = Σ_g ( min[g] * Σ_{i ∈ g} q_rot[i]
                              + delta[g] * <q_rot[g, :], code[g, :]> )

We feed the kernel ``q_group_sums`` already reduced (per-query, host-side)
so the inner per-(q_tok, d_tok) work is

    Σ_g (delta[g] * dot4bit(q_rot[g], code[g]) + min[g] * q_group_sums[g])

which vectorises cleanly: BLOCK_D doc tokens × n_groups groups × group_size
unpacked nibbles, all FMA-friendly fp32 against an int8 → fp32 dequantized
code.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 128}, num_warps=8, num_stages=2),
    ],
    key=["n_d_tokens", "dim", "K", "GROUP_SIZE"],
)
@triton.jit
def _roq_maxsim_rroq4_riem_kernel(
    Q_ROT_PTR,           # (A, S, dim) float32 — FWHT(q_amb)
    Q_GROUP_SUMS_PTR,    # (A, S, n_groups) float32 — per-group Σ q_rot
    QC_TABLE_PTR,        # (A, S, K) float32 — q_amb @ centroids.T
    D_CENTROID_PTR,      # (B, T) int32
    D_COS_NORM_PTR,      # (B, T) float32 — cos(||r||) * norm_d
    D_SIN_NORM_PTR,      # (B, T) float32 — sinc(||r||) * norm_d
    D_CODES_PTR,         # (B, T, dim/2) uint8 — packed nibble pairs
    D_MINS_PTR,          # (B, T, n_groups) float32 — per-group offset
    D_DELTAS_PTR,        # (B, T, n_groups) float32 — per-group scale
    Q_MASK_PTR,          # (A, S)
    D_MASK_PTR,          # (B, T)
    OUTPUT_PTR,          # (A, B) float32
    q_rot_batch_stride, q_rot_token_stride, q_rot_dim_stride,
    qgs_batch_stride, qgs_token_stride, qgs_group_stride,
    qc_batch_stride, qc_token_stride,
    d_codes_batch_stride, d_codes_token_stride, d_codes_byte_stride,
    d_mins_batch_stride, d_mins_token_stride, d_mins_group_stride,
    d_deltas_batch_stride, d_deltas_token_stride, d_deltas_group_stride,
    d_centroid_batch_stride, d_centroid_token_stride,
    d_cos_batch_stride, d_cos_token_stride,
    d_sin_batch_stride, d_sin_token_stride,
    q_mask_batch_stride, q_mask_token_stride,
    d_mask_batch_stride, d_mask_token_stride,
    output_q_stride, output_d_stride,
    n_q_tokens, n_d_tokens, dim, n_groups, K,
    GROUP_SIZE: tl.constexpr,
    HALF_GROUP: tl.constexpr,            # GROUP_SIZE // 2 — number of bytes per group
    BLOCK_D: tl.constexpr,
):
    """One CTA per (query, doc); BLOCK_D doc tokens per inner block.

    The hot path per doc-token is:
      - per group g of GROUP_SIZE coords:
          - load HALF_GROUP packed nibble bytes
          - unpack to fp32 codes (low nibble + high nibble)
          - dot product against the matching slice of q_rot
          - FMA into a running fp32 ``inner_sum``
      - scale by per-group ``delta`` and add ``min * q_group_sum``
      - combine cos*qc + sin*resi
    """
    q_idx = tl.program_id(0)
    d_idx = tl.program_id(1)

    q_rot_qa = Q_ROT_PTR + q_idx * q_rot_batch_stride
    qgs_qa = Q_GROUP_SUMS_PTR + q_idx * qgs_batch_stride
    qc_qa = QC_TABLE_PTR + q_idx * qc_batch_stride
    d_codes_d = D_CODES_PTR + d_idx * d_codes_batch_stride
    d_mins_d = D_MINS_PTR + d_idx * d_mins_batch_stride
    d_deltas_d = D_DELTAS_PTR + d_idx * d_deltas_batch_stride
    d_cid_d = D_CENTROID_PTR + d_idx * d_centroid_batch_stride
    d_cos_d = D_COS_NORM_PTR + d_idx * d_cos_batch_stride
    d_sin_d = D_SIN_NORM_PTR + d_idx * d_sin_batch_stride

    total_score = 0.0

    for i in range(n_q_tokens):
        q_tok_active = tl.load(
            Q_MASK_PTR + q_idx * q_mask_batch_stride + i * q_mask_token_stride
        ) > 0
        if q_tok_active:
            q_rot_row = q_rot_qa + i * q_rot_token_stride
            qgs_row = qgs_qa + i * qgs_token_stride
            qc_row = qc_qa + i * qc_token_stride

            max_sim = -1.0e9
            for j_start in range(0, n_d_tokens, BLOCK_D):
                j_offsets = j_start + tl.arange(0, BLOCK_D)
                j_mask = j_offsets < n_d_tokens
                d_tok_active = tl.load(
                    D_MASK_PTR
                    + d_idx * d_mask_batch_stride
                    + j_offsets * d_mask_token_stride,
                    mask=j_mask, other=0,
                ) > 0

                resi = tl.zeros([BLOCK_D], dtype=tl.float32)
                # Walk the n_groups groups of GROUP_SIZE coords. Each
                # group contributes
                #     delta_g * <q_rot[g], code[g]>  +  min_g * q_group_sum[g]
                # to the per-doc-token residual.
                for grp in range(n_groups):
                    # Per-group scale + offset (both broadcast over BLOCK_D)
                    d_min_g = tl.load(
                        d_mins_d
                        + j_offsets * d_mins_token_stride
                        + grp * d_mins_group_stride,
                        mask=j_mask, other=0.0,
                    )
                    d_delta_g = tl.load(
                        d_deltas_d
                        + j_offsets * d_deltas_token_stride
                        + grp * d_deltas_group_stride,
                        mask=j_mask, other=0.0,
                    )
                    qgs_g = tl.load(qgs_row + grp * qgs_group_stride)

                    # Inner 4-bit dot for this group: HALF_GROUP bytes,
                    # each byte = (high<<4) | low — two coords per byte.
                    inner = tl.zeros([BLOCK_D], dtype=tl.float32)
                    base_byte = grp * HALF_GROUP
                    base_q = grp * GROUP_SIZE
                    for b in range(HALF_GROUP):
                        byte_v = tl.load(
                            d_codes_d
                            + j_offsets * d_codes_token_stride
                            + (base_byte + b) * d_codes_byte_stride,
                            mask=j_mask, other=0,
                        ).to(tl.int32)
                        low = (byte_v & 0xF).to(tl.float32)
                        high = ((byte_v >> 4) & 0xF).to(tl.float32)
                        q_lo = tl.load(
                            q_rot_row + (base_q + 2 * b) * q_rot_dim_stride
                        )
                        q_hi = tl.load(
                            q_rot_row + (base_q + 2 * b + 1) * q_rot_dim_stride
                        )
                        inner += q_lo * low + q_hi * high

                    resi += d_delta_g * inner + d_min_g * qgs_g

                cid = tl.load(
                    d_cid_d + j_offsets * d_centroid_token_stride,
                    mask=j_mask, other=0,
                ).to(tl.int32)
                qc = tl.load(qc_row + cid, mask=j_mask, other=0.0)
                cos_n = tl.load(
                    d_cos_d + j_offsets * d_cos_token_stride,
                    mask=j_mask, other=0.0,
                )
                sin_n = tl.load(
                    d_sin_d + j_offsets * d_sin_token_stride,
                    mask=j_mask, other=0.0,
                )

                est = cos_n * qc + sin_n * resi
                sim = tl.where(j_mask & d_tok_active, est, -1.0e9)
                block_max = tl.max(sim, axis=0)
                if block_max > max_sim:
                    max_sim = block_max
            total_score += max_sim

    out_ptr = OUTPUT_PTR + q_idx * output_q_stride + d_idx * output_d_stride
    tl.store(out_ptr, total_score)


def roq_maxsim_rroq4_riem(
    queries_rot: torch.Tensor,           # (A, S, dim) float32
    queries_group_sums: torch.Tensor,    # (A, S, n_groups) float32
    qc_table: torch.Tensor,              # (A, S, K) float32
    docs_centroid_id: torch.Tensor,      # (B, T) int32
    docs_cos_norm: torch.Tensor,         # (B, T) float32
    docs_sin_norm: torch.Tensor,         # (B, T) float32
    docs_codes_packed: torch.Tensor,     # (B, T, dim/2) uint8
    docs_mins: torch.Tensor,             # (B, T, n_groups) float32
    docs_deltas: torch.Tensor,           # (B, T, n_groups) float32
    *,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    group_size: int | None = None,
) -> torch.Tensor:
    """rroq4_riem fused MaxSim launcher.

    Returns scores of shape (A, B) float32.
    """
    if queries_rot.dtype != torch.float32:
        queries_rot = queries_rot.float()
    if queries_group_sums.dtype != torch.float32:
        queries_group_sums = queries_group_sums.float()
    if qc_table.dtype != torch.float32:
        qc_table = qc_table.float()
    if docs_codes_packed.dtype != torch.uint8:
        raise TypeError(
            f"docs_codes_packed must be uint8, got {docs_codes_packed.dtype}"
        )
    if docs_mins.dtype != torch.float32:
        docs_mins = docs_mins.float()
    if docs_deltas.dtype != torch.float32:
        docs_deltas = docs_deltas.float()
    if docs_cos_norm.dtype != torch.float32:
        docs_cos_norm = docs_cos_norm.float()
    if docs_sin_norm.dtype != torch.float32:
        docs_sin_norm = docs_sin_norm.float()
    if docs_centroid_id.dtype != torch.int32:
        docs_centroid_id = docs_centroid_id.to(torch.int32)

    A, S, dim = queries_rot.shape
    B, T = docs_codes_packed.shape[:2]
    n_bytes = docs_codes_packed.shape[-1]
    if n_bytes * 2 != dim:
        raise ValueError(
            f"docs_codes_packed last dim ({n_bytes}) must be dim/2 = {dim // 2}"
        )
    if group_size is None:
        n_groups = docs_mins.shape[-1]
        if dim % n_groups != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by n_groups ({n_groups})"
            )
        group_size = dim // n_groups
    else:
        n_groups = dim // group_size
    if group_size % 2 != 0:
        raise ValueError(
            f"group_size ({group_size}) must be even so 4-bit codes pack into bytes"
        )
    if queries_group_sums.shape[-1] != n_groups:
        raise ValueError(
            f"queries_group_sums n_groups ({queries_group_sums.shape[-1]}) "
            f"!= docs n_groups ({n_groups})"
        )
    half_group = group_size // 2
    K = qc_table.shape[-1]
    if qc_table.stride(-1) != 1:
        qc_table = qc_table.contiguous()

    if queries_mask is None:
        queries_mask = torch.ones((A, S), dtype=torch.float32, device=queries_rot.device)
    else:
        queries_mask = queries_mask.to(device=queries_rot.device, dtype=torch.float32)
    if documents_mask is None:
        documents_mask = torch.ones((B, T), dtype=torch.float32, device=docs_codes_packed.device)
    else:
        documents_mask = documents_mask.to(device=docs_codes_packed.device, dtype=torch.float32)

    scores = torch.empty((A, B), dtype=torch.float32, device=queries_rot.device)
    grid = (A, B)
    _roq_maxsim_rroq4_riem_kernel[grid](
        Q_ROT_PTR=queries_rot,
        Q_GROUP_SUMS_PTR=queries_group_sums,
        QC_TABLE_PTR=qc_table,
        D_CENTROID_PTR=docs_centroid_id,
        D_COS_NORM_PTR=docs_cos_norm,
        D_SIN_NORM_PTR=docs_sin_norm,
        D_CODES_PTR=docs_codes_packed,
        D_MINS_PTR=docs_mins,
        D_DELTAS_PTR=docs_deltas,
        Q_MASK_PTR=queries_mask,
        D_MASK_PTR=documents_mask,
        OUTPUT_PTR=scores,
        q_rot_batch_stride=queries_rot.stride(0),
        q_rot_token_stride=queries_rot.stride(1),
        q_rot_dim_stride=queries_rot.stride(2),
        qgs_batch_stride=queries_group_sums.stride(0),
        qgs_token_stride=queries_group_sums.stride(1),
        qgs_group_stride=queries_group_sums.stride(2),
        qc_batch_stride=qc_table.stride(0),
        qc_token_stride=qc_table.stride(1),
        d_codes_batch_stride=docs_codes_packed.stride(0),
        d_codes_token_stride=docs_codes_packed.stride(1),
        d_codes_byte_stride=docs_codes_packed.stride(2),
        d_mins_batch_stride=docs_mins.stride(0),
        d_mins_token_stride=docs_mins.stride(1),
        d_mins_group_stride=docs_mins.stride(2),
        d_deltas_batch_stride=docs_deltas.stride(0),
        d_deltas_token_stride=docs_deltas.stride(1),
        d_deltas_group_stride=docs_deltas.stride(2),
        d_centroid_batch_stride=docs_centroid_id.stride(0),
        d_centroid_token_stride=docs_centroid_id.stride(1),
        d_cos_batch_stride=docs_cos_norm.stride(0),
        d_cos_token_stride=docs_cos_norm.stride(1),
        d_sin_batch_stride=docs_sin_norm.stride(0),
        d_sin_token_stride=docs_sin_norm.stride(1),
        q_mask_batch_stride=queries_mask.stride(0),
        q_mask_token_stride=queries_mask.stride(1),
        d_mask_batch_stride=documents_mask.stride(0),
        d_mask_token_stride=documents_mask.stride(1),
        output_q_stride=scores.stride(0),
        output_d_stride=scores.stride(1),
        n_q_tokens=S,
        n_d_tokens=T,
        dim=dim,
        n_groups=n_groups,
        K=K,
        GROUP_SIZE=group_size,
        HALF_GROUP=half_group,
    )
    return scores


# ---------------------------------------------------------------------------
# Numpy reference scorer for parity tests.
# ---------------------------------------------------------------------------


def reference_score_rroq4_riem(
    queries_rot,
    queries_group_sums,
    qc_table,
    docs_centroid_id,
    docs_cos_norm,
    docs_sin_norm,
    docs_codes_packed,
    docs_mins,
    docs_deltas,
    *,
    queries_mask=None,
    documents_mask=None,
    group_size: int | None = None,
):
    """Pure numpy reimplementation of the kernel inner loop. Slow.

    Used by tests/test_rroq4_riem_kernel.py for parity validation against
    both the Triton kernel and the Rust SIMD kernel.
    """
    import numpy as np

    qr = queries_rot.cpu().numpy() if hasattr(queries_rot, "cpu") else queries_rot
    qgs = queries_group_sums.cpu().numpy() if hasattr(queries_group_sums, "cpu") else queries_group_sums
    qc_t = qc_table.cpu().numpy() if hasattr(qc_table, "cpu") else qc_table
    cid = docs_centroid_id.cpu().numpy() if hasattr(docs_centroid_id, "cpu") else docs_centroid_id
    cos_n = docs_cos_norm.cpu().numpy() if hasattr(docs_cos_norm, "cpu") else docs_cos_norm
    sin_n = docs_sin_norm.cpu().numpy() if hasattr(docs_sin_norm, "cpu") else docs_sin_norm
    codes_p = docs_codes_packed.cpu().numpy() if hasattr(docs_codes_packed, "cpu") else docs_codes_packed
    dm = docs_mins.cpu().numpy() if hasattr(docs_mins, "cpu") else docs_mins
    dd = docs_deltas.cpu().numpy() if hasattr(docs_deltas, "cpu") else docs_deltas

    A, S, dim = qr.shape
    B, T = codes_p.shape[:2]
    n_groups = dm.shape[-1]
    if group_size is None:
        group_size = dim // n_groups

    # Unpack codes once: (B, T, dim) uint8
    codes_full = np.empty((B, T, dim), dtype=np.uint8)
    codes_full[:, :, 0::2] = codes_p & 0x0F
    codes_full[:, :, 1::2] = (codes_p >> 4) & 0x0F

    out = np.zeros((A, B), dtype=np.float32)
    for q_idx in range(A):
        for d_idx in range(B):
            total = 0.0
            for i in range(S):
                if queries_mask is not None and queries_mask[q_idx, i] <= 0:
                    continue
                max_sim = -1e9
                for j in range(T):
                    if documents_mask is not None and documents_mask[d_idx, j] <= 0:
                        continue
                    resi = 0.0
                    for grp in range(n_groups):
                        delta_g = float(dd[d_idx, j, grp])
                        min_g = float(dm[d_idx, j, grp])
                        qgs_g = float(qgs[q_idx, i, grp])
                        inner = 0.0
                        for k in range(group_size):
                            d_idx_dim = grp * group_size + k
                            inner += float(qr[q_idx, i, d_idx_dim]) * float(
                                codes_full[d_idx, j, d_idx_dim]
                            )
                        resi += delta_g * inner + min_g * qgs_g
                    cidv = int(cid[d_idx, j])
                    qc = float(qc_t[q_idx, i, cidv])
                    est = float(cos_n[d_idx, j]) * qc + float(sin_n[d_idx, j]) * resi
                    if est > max_sim:
                        max_sim = est
                total += max_sim
            out[q_idx, d_idx] = total
    return out


__all__ = [
    "roq_maxsim_rroq4_riem",
    "reference_score_rroq4_riem",
]
