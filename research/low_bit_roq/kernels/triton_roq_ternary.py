"""
Ternary (1.58-bit) asymmetric MaxSim kernel.

Plan reference: Phase A2.5. Mirrors the existing
``_roq_maxsim_1bit_asymmetric_kernel`` architecture in
``voyager_index/_internal/kernels/triton_roq.py`` so the implementation
lift is small and the launcher / Python encoder share the same packed
layout.

Doc encoding (2 bit-planes per coord, dim/8 bytes each → dim/4 bytes
total per token before per-group scale meta):

- ``sign_plane[i]    = 1`` if ``x[i] > 0``  else 0
- ``nonzero_plane[i] = 1`` if ``|x[i]| > τ_group`` else 0
- per-group ``scale`` (float32). Reconstruction:
  ``d_hat[i] = scale_{g(i)} · (2*sign_plane[i] - 1) · nonzero_plane[i]`` ∈
  ``{-scale, 0, +scale}``.

Query encoding: identical to the existing 1-bit asymmetric path —
``QUERY_BITS`` ∈ {4, 6, 8} bit-planes over the scalar-quantized query.

Estimated dot per (q, d):

    dot = q_offset · Σ_g scale_g · S_g + q_scale · Σ_g scale_g · D_g
          where
            S_g = popc(d_sign & d_nz)_g - popc(~d_sign & d_nz)_g   (∈ [-gs, gs])
            D_g = Σ_k 2^k · ( popc(q_k & d_sign & d_nz)_g
                             - popc(q_k & ~d_sign & d_nz)_g )

Both terms are popcount-only — no fp coordinate-wise multiplication.
"""

from __future__ import annotations

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
def _roq_maxsim_ternary_kernel(
    Q_PLANES_PTR,
    D_SIGN_PTR,
    D_NZ_PTR,
    D_SCALES_PTR,
    Q_META_PTR,
    Q_MASK_PTR,
    D_MASK_PTR,
    OUTPUT_PTR,
    q_batch_stride,
    q_token_stride,
    q_plane_stride,
    q_word_stride,
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
    N_WORDS: tl.constexpr,
    GROUP_WORDS: tl.constexpr,
    QUERY_BITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One CTA per (query, doc) pair, BLOCK_D doc tokens per block."""
    q_idx = tl.program_id(0)
    d_idx = tl.program_id(1)

    q_ptr_base = Q_PLANES_PTR + q_idx * q_batch_stride
    d_sign_base = D_SIGN_PTR + d_idx * d_batch_stride
    d_nz_base = D_NZ_PTR + d_idx * d_batch_stride
    d_scales_base = D_SCALES_PTR + d_idx * d_scales_batch_stride
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

                offset_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
                scale_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

                q_token_base = q_ptr_base + i * q_token_stride
                d_sign_token = d_sign_base + j_offsets * d_token_stride
                d_nz_token = d_nz_base + j_offsets * d_token_stride

                for grp in range(n_groups):
                    d_scale = tl.load(
                        d_scales_base
                        + j_offsets * d_scales_token_stride
                        + grp * d_scales_group_stride,
                        mask=j_mask,
                        other=0.0,
                    )

                    s_g = tl.zeros([BLOCK_D], dtype=tl.int32)
                    d_g = tl.zeros([BLOCK_D], dtype=tl.float32)

                    base_word = grp * GROUP_WORDS
                    for w in range(GROUP_WORDS):
                        word_idx = base_word + w
                        d_sign_word = tl.load(
                            d_sign_token + word_idx * d_word_stride,
                            mask=j_mask,
                            other=0,
                        ).to(tl.int32)
                        d_nz_word = tl.load(
                            d_nz_token + word_idx * d_word_stride,
                            mask=j_mask,
                            other=0,
                        ).to(tl.int32)

                        # signed-sum-per-group contribution from this word
                        pos = _popc(d_sign_word & d_nz_word).to(tl.int32)
                        neg = _popc((~d_sign_word) & d_nz_word).to(tl.int32)
                        s_g += pos - neg

                        # weighted dot-per-group contribution from this word
                        q_plane0 = tl.load(
                            q_token_base + 0 * q_plane_stride + word_idx * q_word_stride
                        ).to(tl.int32)
                        m0 = _popc(d_sign_word & q_plane0 & d_nz_word).to(tl.int32)
                        c0 = _popc((~d_sign_word) & q_plane0 & d_nz_word).to(tl.int32)
                        d_g += (m0 - c0).to(tl.float32)
                        if QUERY_BITS > 1:
                            q_planek = tl.load(
                                q_token_base + 1 * q_plane_stride + word_idx * q_word_stride
                            ).to(tl.int32)
                            mk = _popc(d_sign_word & q_planek & d_nz_word).to(tl.int32)
                            ck = _popc((~d_sign_word) & q_planek & d_nz_word).to(tl.int32)
                            d_g += 2.0 * (mk - ck).to(tl.float32)
                        if QUERY_BITS > 2:
                            q_planek = tl.load(
                                q_token_base + 2 * q_plane_stride + word_idx * q_word_stride
                            ).to(tl.int32)
                            mk = _popc(d_sign_word & q_planek & d_nz_word).to(tl.int32)
                            ck = _popc((~d_sign_word) & q_planek & d_nz_word).to(tl.int32)
                            d_g += 4.0 * (mk - ck).to(tl.float32)
                        if QUERY_BITS > 3:
                            q_planek = tl.load(
                                q_token_base + 3 * q_plane_stride + word_idx * q_word_stride
                            ).to(tl.int32)
                            mk = _popc(d_sign_word & q_planek & d_nz_word).to(tl.int32)
                            ck = _popc((~d_sign_word) & q_planek & d_nz_word).to(tl.int32)
                            d_g += 8.0 * (mk - ck).to(tl.float32)
                        if QUERY_BITS > 4:
                            q_planek = tl.load(
                                q_token_base + 4 * q_plane_stride + word_idx * q_word_stride
                            ).to(tl.int32)
                            mk = _popc(d_sign_word & q_planek & d_nz_word).to(tl.int32)
                            ck = _popc((~d_sign_word) & q_planek & d_nz_word).to(tl.int32)
                            d_g += 16.0 * (mk - ck).to(tl.float32)
                        if QUERY_BITS > 5:
                            q_planek = tl.load(
                                q_token_base + 5 * q_plane_stride + word_idx * q_word_stride
                            ).to(tl.int32)
                            mk = _popc(d_sign_word & q_planek & d_nz_word).to(tl.int32)
                            ck = _popc((~d_sign_word) & q_planek & d_nz_word).to(tl.int32)
                            d_g += 32.0 * (mk - ck).to(tl.float32)
                        if QUERY_BITS > 6:
                            q_planek = tl.load(
                                q_token_base + 6 * q_plane_stride + word_idx * q_word_stride
                            ).to(tl.int32)
                            mk = _popc(d_sign_word & q_planek & d_nz_word).to(tl.int32)
                            ck = _popc((~d_sign_word) & q_planek & d_nz_word).to(tl.int32)
                            d_g += 64.0 * (mk - ck).to(tl.float32)
                        if QUERY_BITS > 7:
                            q_planek = tl.load(
                                q_token_base + 7 * q_plane_stride + word_idx * q_word_stride
                            ).to(tl.int32)
                            mk = _popc(d_sign_word & q_planek & d_nz_word).to(tl.int32)
                            ck = _popc((~d_sign_word) & q_planek & d_nz_word).to(tl.int32)
                            d_g += 128.0 * (mk - ck).to(tl.float32)

                    offset_acc += d_scale * s_g.to(tl.float32)
                    scale_acc += d_scale * d_g

                est_dot = q_offset * offset_acc + q_scale * scale_acc
                sim = tl.where(j_mask & d_token_active, est_dot, -1.0e9)
                block_max = tl.max(sim, axis=0)
                if block_max > max_sim:
                    max_sim = block_max
            total_score += max_sim

    out_ptr = OUTPUT_PTR + q_idx * output_q_stride + d_idx * output_d_stride
    tl.store(out_ptr, total_score)


def roq_maxsim_ternary(
    queries_planes: torch.Tensor,
    queries_meta: torch.Tensor,
    docs_sign: torch.Tensor,
    docs_nz: torch.Tensor,
    docs_scales: torch.Tensor,
    *,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    n_groups: int | None = None,
) -> torch.Tensor:
    """Ternary asymmetric MaxSim launcher.

    Args:
      queries_planes:  (A, S, query_bits, n_words) int32 — bit-planes
      queries_meta:    (A, S, M) float32 — first two columns must be
                       [scale, offset] (further columns ignored, kept for
                       layout parity with the existing 1-bit kernel)
      docs_sign:       (B, T, n_words) int32 — packed sign plane
      docs_nz:         (B, T, n_words) int32 — packed nonzero plane
      docs_scales:     (B, T, n_groups) float32 — per-group scales
    """
    A, S = queries_planes.shape[:2]
    B, T = docs_sign.shape[:2]
    n_words = docs_sign.shape[-1]
    query_bits = queries_planes.shape[2]
    dim = n_words * 32
    if n_groups is None:
        n_groups = docs_scales.shape[-1]
    if n_words % n_groups != 0:
        raise ValueError(
            f"n_words={n_words} must be divisible by n_groups={n_groups}"
        )
    group_words = n_words // n_groups

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
            (B, T), dtype=torch.float32, device=docs_sign.device
        )
    else:
        documents_mask = documents_mask.to(
            device=docs_sign.device, dtype=torch.float32
        )

    scores = torch.empty((A, B), dtype=torch.float32, device=queries_planes.device)
    grid = (A, B)
    _roq_maxsim_ternary_kernel[grid](
        Q_PLANES_PTR=queries_planes,
        D_SIGN_PTR=docs_sign,
        D_NZ_PTR=docs_nz,
        D_SCALES_PTR=docs_scales,
        Q_META_PTR=queries_meta,
        Q_MASK_PTR=queries_mask,
        D_MASK_PTR=documents_mask,
        OUTPUT_PTR=scores,
        q_batch_stride=queries_planes.stride(0),
        q_token_stride=queries_planes.stride(1),
        q_plane_stride=queries_planes.stride(2),
        q_word_stride=queries_planes.stride(3),
        d_batch_stride=docs_sign.stride(0),
        d_token_stride=docs_sign.stride(1),
        d_word_stride=docs_sign.stride(2),
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
        N_WORDS=n_words,
        GROUP_WORDS=group_words,
        QUERY_BITS=query_bits,
    )
    return scores


def reference_score_ternary(
    queries_planes,
    queries_meta,
    docs_sign,
    docs_nz,
    docs_scales,
    *,
    queries_mask=None,
    documents_mask=None,
):
    """Numpy reference implementation for parity testing.

    Used by ``tests/test_kernels.py`` to verify the Triton kernel matches
    a transparent Python implementation. Slow — do not use in the hot
    path.
    """
    import numpy as np

    A, S = queries_planes.shape[:2]
    B, T = docs_sign.shape[:2]
    n_words = docs_sign.shape[-1]
    query_bits = queries_planes.shape[2]
    n_groups = docs_scales.shape[-1]
    group_words = n_words // n_groups
    dim = n_words * 32

    qp = queries_planes.cpu().numpy().view(np.int32) if hasattr(queries_planes, "cpu") else queries_planes
    qm = queries_meta.cpu().numpy() if hasattr(queries_meta, "cpu") else queries_meta
    ds = docs_sign.cpu().numpy().view(np.int32) if hasattr(docs_sign, "cpu") else docs_sign
    dn = docs_nz.cpu().numpy().view(np.int32) if hasattr(docs_nz, "cpu") else docs_nz
    dsc = docs_scales.cpu().numpy() if hasattr(docs_scales, "cpu") else docs_scales

    out = np.zeros((A, B), dtype=np.float32)
    for q_idx in range(A):
        for d_idx in range(B):
            total = 0.0
            for i in range(S):
                if queries_mask is not None and queries_mask[q_idx, i] <= 0:
                    continue
                q_scale = float(qm[q_idx, i, 0])
                q_offset = float(qm[q_idx, i, 1])
                max_sim = -1e9
                for j in range(T):
                    if documents_mask is not None and documents_mask[d_idx, j] <= 0:
                        continue
                    est = 0.0
                    for grp in range(n_groups):
                        d_scale_g = float(dsc[d_idx, j, grp])
                        s_g = 0
                        d_g = 0.0
                        for w in range(group_words):
                            word_idx = grp * group_words + w
                            ds_w = int(ds[d_idx, j, word_idx]) & 0xFFFFFFFF
                            dn_w = int(dn[d_idx, j, word_idx]) & 0xFFFFFFFF
                            pos = _python_popc(ds_w & dn_w)
                            neg = _python_popc((~ds_w & 0xFFFFFFFF) & dn_w)
                            s_g += pos - neg
                            for k in range(query_bits):
                                qk = int(qp[q_idx, i, k, word_idx]) & 0xFFFFFFFF
                                m = _python_popc(ds_w & qk & dn_w)
                                c = _python_popc((~ds_w & 0xFFFFFFFF) & qk & dn_w)
                                d_g += (1 << k) * (m - c)
                        est += d_scale_g * (q_offset * s_g + q_scale * d_g)
                    if est > max_sim:
                        max_sim = est
                total += max_sim
            out[q_idx, d_idx] = total
    return out


def _python_popc(x: int) -> int:
    return bin(x & 0xFFFFFFFF).count("1")
