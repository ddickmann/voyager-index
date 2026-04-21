"""
rroq158 fused MaxSim kernel — Riemannian-aware 1.58-bit (ternary) ROQ.

Plan reference: research plan Phase B3 + B3-kernel.

Score formula per (q_token, d_token):

    sim(q, d) = norm_d * ( cos(||r_d||) * <q, c_d>
                          + sinc(||r_d||) * <q, r_d_ambient> )

where
    c_d           is the unit-norm spherical centroid (lookup via centroid_id[d])
    r_d_ambient   is the tangent residual in ambient space (encoded ternary
                  in FWHT-rotated space; FWHT is orthogonal so the dot
                  product against q can be computed as <FWHT(q), r_d_rotated>)
    norm_d        is the original token norm (so the score sums up multi-vector
                  MaxSim without renormalising every doc token)

Two-stage launch (the memory optimisation):

    Stage 1 (host-side, a single tiny matmul):
        qc_table = q_amb @ centroids.T            shape (n_q_tok, K) fp32
                                                  K=1024, n_q_tok≈32  →  128 KB
        q_rot = FWHT(q_amb)                       once per query
        q_planes, q_meta = ternary_query_encode(q_rot)

    Stage 2 (this Triton kernel):
        per (query, doc) CTA, BLOCK_D parallel over doc tokens
        each d-token contributes:
          - a gather into qc_table at centroid_id[d]
          - a popcount-based <q_rot, r_rotated> using the existing
            ternary asymmetric inner loop (mirrored from
            ``research/low_bit_roq/kernels/triton_roq_ternary.py``)
          - a cos·qc + sinc·resi combine, multiplied by norm_d

Per-doc-token storage on disk (dim=128, group_size=16, n_groups=8):

    centroid_id   2 B   uint16  (K ≤ 65k)
    sign_plane   16 B   dim/8
    nz_plane     16 B   dim/8
    group_scales 16 B   8 × fp16
    cos_norm      2 B   fp16   (= cos(||r||) * norm_d)
    sin_norm      2 B   fp16   (= sinc(||r||) * norm_d)
    -------------------------
    total       54 B/token   vs 256 fp16, vs 64 ROQ4

Plus per-shard centroid table: K × dim × 4 B = 512 KB at K=1024 (kept on
GPU resident; one-time per shard).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# (shape_key, device_key) -> Tensor of ones, fp32. Used as a per-query
# mask scratchpad for the rroq158 launcher when the caller doesn't pass
# its own mask. See ``roq_maxsim_rroq158`` for the rationale.
_ONES_CACHE: "dict[tuple, torch.Tensor]" = {}


def _ones_cached(shape: tuple, device: torch.device) -> torch.Tensor:
    key = (tuple(shape), str(device))
    cached = _ONES_CACHE.get(key)
    if cached is None:
        cached = torch.ones(shape, dtype=torch.float32, device=device)
        _ONES_CACHE[key] = cached
    return cached


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
        # ── Short-T tier (T=32, dominant in skewed corpora like quora) ──
        # BLOCK_D=32 makes the inner j-loop run once → no wasted half-tile,
        # and num_warps=1/2 keeps occupancy high when there are 500K+ CTAs.
        triton.Config({"BLOCK_D": 32}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_D": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_D": 32}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_D": 32}, num_warps=4, num_stages=2),
        # ── Mid tier (T=64 / 128) ──
        triton.Config({"BLOCK_D": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 64}, num_warps=4, num_stages=3),
        # ── Long-T tier (T>=128, the rare tail) ──
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_D": 128}, num_warps=8, num_stages=3),
    ],
    key=["n_d_tokens", "dim", "QUERY_BITS", "K"],
)
@triton.jit
def _roq_maxsim_rroq158_kernel(
    Q_PLANES_PTR,        # (A, S, query_bits, n_words) int32 — FWHT-rotated query bit-planes
    Q_META_PTR,          # (A, S, 2)               float32 — [scale, offset] per query token
    QC_TABLE_PTR,        # (A, S, K)               float32 — <q_amb, centroid_k> precomputed
    D_CENTROID_PTR,      # (B, T)                  int32   — centroid_id per doc token
    D_COS_NORM_PTR,      # (B, T)                  float32 — cos(||r||) * norm_d
    D_SIN_NORM_PTR,      # (B, T)                  float32 — sinc(||r||) * norm_d
    D_SIGN_PTR,          # (B, T, n_words)         int32   — packed ternary sign plane
    D_NZ_PTR,            # (B, T, n_words)         int32   — packed ternary nonzero plane
    D_SCALES_PTR,        # (B, T, n_groups)        float32 — per-group residual scale
    Q_MASK_PTR,          # (A, S)
    D_MASK_PTR,          # (B, T)
    OUTPUT_PTR,          # (A, B)                  float32 — output scores
    q_batch_stride, q_token_stride, q_plane_stride, q_word_stride,
    qc_batch_stride, qc_token_stride,                # K dim is contig (stride=1 enforced)
    d_batch_stride, d_token_stride, d_word_stride,
    d_centroid_batch_stride, d_centroid_token_stride,
    d_cos_batch_stride, d_cos_token_stride,
    d_sin_batch_stride, d_sin_token_stride,
    d_scales_batch_stride, d_scales_token_stride, d_scales_group_stride,
    q_meta_token_stride,
    q_mask_batch_stride, q_mask_token_stride,
    d_mask_batch_stride, d_mask_token_stride,
    output_q_stride, output_d_stride,
    n_queries, n_docs, n_q_tokens, n_d_tokens, dim, n_groups, K,
    N_WORDS: tl.constexpr,
    GROUP_WORDS: tl.constexpr,
    QUERY_BITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One CTA per (query, doc) pair, BLOCK_D doc tokens per inner block.

    Mirrors the ternary asymmetric kernel structure but adds two
    per-d-token gathers (centroid_id → qc_table, plus the cos/sin caches)
    and the cos·qc + sinc·resi combine.
    """
    d_idx = tl.program_id(0)
    q_idx = tl.program_id(1)

    q_ptr_base = Q_PLANES_PTR + q_idx * q_batch_stride
    qc_base = QC_TABLE_PTR + q_idx * qc_batch_stride
    d_sign_base = D_SIGN_PTR + d_idx * d_batch_stride
    d_nz_base = D_NZ_PTR + d_idx * d_batch_stride
    d_scales_base = D_SCALES_PTR + d_idx * d_scales_batch_stride
    d_centroid_base = D_CENTROID_PTR + d_idx * d_centroid_batch_stride
    d_cos_base = D_COS_NORM_PTR + d_idx * d_cos_batch_stride
    d_sin_base = D_SIN_NORM_PTR + d_idx * d_sin_batch_stride
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
            qc_row_base = qc_base + i * qc_token_stride

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

                # ---- Stage A: ternary residual popcount  --------------------
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
                            mask=j_mask, other=0,
                        ).to(tl.int32)
                        d_nz_word = tl.load(
                            d_nz_token + word_idx * d_word_stride,
                            mask=j_mask, other=0,
                        ).to(tl.int32)

                        pos = _popc(d_sign_word & d_nz_word).to(tl.int32)
                        neg = _popc((~d_sign_word) & d_nz_word).to(tl.int32)
                        s_g += pos - neg

                        q_plane0 = tl.load(
                            q_token_base + 0 * q_plane_stride + word_idx * q_word_stride
                        ).to(tl.int32)
                        m0 = _popc(d_sign_word & q_plane0 & d_nz_word).to(tl.int32)
                        c0 = _popc((~d_sign_word) & q_plane0 & d_nz_word).to(tl.int32)
                        d_g += (m0 - c0).to(tl.float32)
                        if QUERY_BITS > 1:
                            qp = tl.load(
                                q_token_base + 1 * q_plane_stride + word_idx * q_word_stride
                            ).to(tl.int32)
                            m = _popc(d_sign_word & qp & d_nz_word).to(tl.int32)
                            c = _popc((~d_sign_word) & qp & d_nz_word).to(tl.int32)
                            d_g += 2.0 * (m - c).to(tl.float32)
                        if QUERY_BITS > 2:
                            qp = tl.load(
                                q_token_base + 2 * q_plane_stride + word_idx * q_word_stride
                            ).to(tl.int32)
                            m = _popc(d_sign_word & qp & d_nz_word).to(tl.int32)
                            c = _popc((~d_sign_word) & qp & d_nz_word).to(tl.int32)
                            d_g += 4.0 * (m - c).to(tl.float32)
                        if QUERY_BITS > 3:
                            qp = tl.load(
                                q_token_base + 3 * q_plane_stride + word_idx * q_word_stride
                            ).to(tl.int32)
                            m = _popc(d_sign_word & qp & d_nz_word).to(tl.int32)
                            c = _popc((~d_sign_word) & qp & d_nz_word).to(tl.int32)
                            d_g += 8.0 * (m - c).to(tl.float32)
                        if QUERY_BITS > 4:
                            qp = tl.load(
                                q_token_base + 4 * q_plane_stride + word_idx * q_word_stride
                            ).to(tl.int32)
                            m = _popc(d_sign_word & qp & d_nz_word).to(tl.int32)
                            c = _popc((~d_sign_word) & qp & d_nz_word).to(tl.int32)
                            d_g += 16.0 * (m - c).to(tl.float32)
                        if QUERY_BITS > 5:
                            qp = tl.load(
                                q_token_base + 5 * q_plane_stride + word_idx * q_word_stride
                            ).to(tl.int32)
                            m = _popc(d_sign_word & qp & d_nz_word).to(tl.int32)
                            c = _popc((~d_sign_word) & qp & d_nz_word).to(tl.int32)
                            d_g += 32.0 * (m - c).to(tl.float32)
                        if QUERY_BITS > 6:
                            qp = tl.load(
                                q_token_base + 6 * q_plane_stride + word_idx * q_word_stride
                            ).to(tl.int32)
                            m = _popc(d_sign_word & qp & d_nz_word).to(tl.int32)
                            c = _popc((~d_sign_word) & qp & d_nz_word).to(tl.int32)
                            d_g += 64.0 * (m - c).to(tl.float32)
                        if QUERY_BITS > 7:
                            qp = tl.load(
                                q_token_base + 7 * q_plane_stride + word_idx * q_word_stride
                            ).to(tl.int32)
                            m = _popc(d_sign_word & qp & d_nz_word).to(tl.int32)
                            c = _popc((~d_sign_word) & qp & d_nz_word).to(tl.int32)
                            d_g += 128.0 * (m - c).to(tl.float32)

                    offset_acc += d_scale * s_g.to(tl.float32)
                    scale_acc += d_scale * d_g

                resi = q_offset * offset_acc + q_scale * scale_acc

                # ---- Stage B: centroid lookup + cos/sinc combine ------------
                cid = tl.load(
                    d_centroid_base + j_offsets * d_centroid_token_stride,
                    mask=j_mask, other=0,
                ).to(tl.int32)
                qc = tl.load(qc_row_base + cid, mask=j_mask, other=0.0)
                cos_n = tl.load(
                    d_cos_base + j_offsets * d_cos_token_stride,
                    mask=j_mask, other=0.0,
                )
                sin_n = tl.load(
                    d_sin_base + j_offsets * d_sin_token_stride,
                    mask=j_mask, other=0.0,
                )

                est_dot = cos_n * qc + sin_n * resi
                sim = tl.where(j_mask & d_token_active, est_dot, -1.0e9)
                block_max = tl.max(sim, axis=0)
                if block_max > max_sim:
                    max_sim = block_max
            total_score += max_sim

    out_ptr = OUTPUT_PTR + q_idx * output_q_stride + d_idx * output_d_stride
    tl.store(out_ptr, total_score)


def roq_maxsim_rroq158(
    queries_planes: torch.Tensor,        # (A, S, query_bits, n_words) int32
    queries_meta: torch.Tensor,          # (A, S, 2) float32 [scale, offset]
    qc_table: torch.Tensor,              # (A, S, K) float32 — q_amb @ centroids.T
    docs_centroid_id: torch.Tensor,      # (B, T) int32
    docs_cos_norm: torch.Tensor,         # (B, T) float32
    docs_sin_norm: torch.Tensor,         # (B, T) float32
    docs_sign: torch.Tensor,             # (B, T, n_words) int32
    docs_nz: torch.Tensor,               # (B, T, n_words) int32
    docs_scales: torch.Tensor,           # (B, T, n_groups) float32
    *,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    n_groups: int | None = None,
) -> torch.Tensor:
    """rroq158 fused MaxSim launcher.

    Returns scores of shape (A, B) float32.
    """
    A, S = queries_planes.shape[:2]
    B, T = docs_sign.shape[:2]
    n_words = docs_sign.shape[-1]
    query_bits = queries_planes.shape[2]
    dim = n_words * 32
    if n_groups is None:
        n_groups = docs_scales.shape[-1]
    if n_words % n_groups != 0:
        raise ValueError(f"n_words={n_words} not divisible by n_groups={n_groups}")
    group_words = n_words // n_groups
    K = qc_table.shape[-1]
    if qc_table.stride(-1) != 1:
        qc_table = qc_table.contiguous()

    # Hot-path masks: cache an "all-ones" mask per (device, shape) so we
    # don't allocate + memset a fresh torch.ones((A,S)) on every query
    # (measured ~50 μs per call on H100, ~5% of the rroq158 fast-path
    # budget at A=1, S<=64). Cache lives on the launcher function as a
    # module-level dict; eviction is by shape key, no LRU needed because
    # there are only a handful of (qt, dt) shapes in any one process
    # after Triton autotune warmup pinned the corpus-padded extents.
    if queries_mask is None:
        queries_mask = _ones_cached(
            (A, S), device=queries_planes.device
        )
    elif queries_mask.device != queries_planes.device or queries_mask.dtype != torch.float32:
        queries_mask = queries_mask.to(
            device=queries_planes.device, dtype=torch.float32
        )
    if documents_mask is None:
        documents_mask = _ones_cached((B, T), device=docs_sign.device)
    elif documents_mask.device != docs_sign.device or documents_mask.dtype != torch.float32:
        documents_mask = documents_mask.to(
            device=docs_sign.device, dtype=torch.float32
        )

    scores = torch.empty((A, B), dtype=torch.float32, device=queries_planes.device)
    # Grid = (B, A): B in gridX (limit 2^31-1, accommodates million-doc corpora),
    # A in gridY (limit 65535, A is typically 1). The previous (A, B) layout
    # crashed with "Triton Error [CUDA]: invalid argument" on quora-class
    # corpora (B > 65535) because gridY has the lower limit.
    grid = (B, A)
    _roq_maxsim_rroq158_kernel[grid](
        Q_PLANES_PTR=queries_planes,
        Q_META_PTR=queries_meta,
        QC_TABLE_PTR=qc_table,
        D_CENTROID_PTR=docs_centroid_id,
        D_COS_NORM_PTR=docs_cos_norm,
        D_SIN_NORM_PTR=docs_sin_norm,
        D_SIGN_PTR=docs_sign,
        D_NZ_PTR=docs_nz,
        D_SCALES_PTR=docs_scales,
        Q_MASK_PTR=queries_mask,
        D_MASK_PTR=documents_mask,
        OUTPUT_PTR=scores,
        q_batch_stride=queries_planes.stride(0),
        q_token_stride=queries_planes.stride(1),
        q_plane_stride=queries_planes.stride(2),
        q_word_stride=queries_planes.stride(3),
        qc_batch_stride=qc_table.stride(0),
        qc_token_stride=qc_table.stride(1),
        d_batch_stride=docs_sign.stride(0),
        d_token_stride=docs_sign.stride(1),
        d_word_stride=docs_sign.stride(2),
        d_centroid_batch_stride=docs_centroid_id.stride(0),
        d_centroid_token_stride=docs_centroid_id.stride(1),
        d_cos_batch_stride=docs_cos_norm.stride(0),
        d_cos_token_stride=docs_cos_norm.stride(1),
        d_sin_batch_stride=docs_sin_norm.stride(0),
        d_sin_token_stride=docs_sin_norm.stride(1),
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
        K=K,
        N_WORDS=n_words,
        GROUP_WORDS=group_words,
        QUERY_BITS=query_bits,
    )
    return scores


# ---------------------------------------------------------------------------
# Numpy reference scorer for parity tests.
# ---------------------------------------------------------------------------


def reference_score_rroq158(
    queries_planes,
    queries_meta,
    qc_table,
    docs_centroid_id,
    docs_cos_norm,
    docs_sin_norm,
    docs_sign,
    docs_nz,
    docs_scales,
    *,
    queries_mask=None,
    documents_mask=None,
):
    """Pure numpy reimplementation of the kernel inner loop. Slow.

    Used by tests/test_rroq158_kernel.py for parity validation.
    """
    import numpy as np

    A, S = queries_planes.shape[:2]
    B, T = docs_sign.shape[:2]
    n_words = docs_sign.shape[-1]
    query_bits = queries_planes.shape[2]
    n_groups = docs_scales.shape[-1]
    group_words = n_words // n_groups

    qp = queries_planes.cpu().numpy().view(np.int32) if hasattr(queries_planes, "cpu") else queries_planes
    qm = queries_meta.cpu().numpy() if hasattr(queries_meta, "cpu") else queries_meta
    qc_t = qc_table.cpu().numpy() if hasattr(qc_table, "cpu") else qc_table
    cid = docs_centroid_id.cpu().numpy() if hasattr(docs_centroid_id, "cpu") else docs_centroid_id
    cos_n = docs_cos_norm.cpu().numpy() if hasattr(docs_cos_norm, "cpu") else docs_cos_norm
    sin_n = docs_sin_norm.cpu().numpy() if hasattr(docs_sin_norm, "cpu") else docs_sin_norm
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
                    resi = 0.0
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
                        resi += d_scale_g * (q_offset * s_g + q_scale * d_g)
                    cidv = int(cid[d_idx, j])
                    qc = float(qc_t[q_idx, i, cidv])
                    est = float(cos_n[d_idx, j]) * qc + float(sin_n[d_idx, j]) * resi
                    if est > max_sim:
                        max_sim = est
                total += max_sim
            out[q_idx, d_idx] = total
    return out


def _python_popc(x: int) -> int:
    return bin(x & 0xFFFFFFFF).count("1")
