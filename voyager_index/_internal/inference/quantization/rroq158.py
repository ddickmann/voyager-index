"""
Riemannian-aware 1.58-bit (ternary) ROQ encoder + kernel input builders.

Lifted from ``research/low_bit_roq/rroq.py`` and trimmed to the production
hot path. The research version supports {2-bit, ternary} residuals and
{identity, FWHT, per-cluster PCA} bases; the production version is fixed
at the configuration the offline bench validated: ternary residual + FWHT
basis. K is configurable.

Score formula per (q_token, d_token):

    sim(q, d) = norm_d * ( cos(||r_d||) * <q, c_d>
                          + sinc(||r_d||) * <q, r_d_ambient> )

Encode pipeline (all numpy, CPU, run at index-build time):

    1. Spherical k-means to fit K unit-norm centroids c_k from a
       sub-sample of L2-normalized doc tokens.
    2. Per d-token x:
         - assign centroid c = c_{argmax_k <x_n, c_k>}
         - tangent r_amb = log_c(x_n)         (in ambient space)
         - r_rot = FWHT(r_amb)                (rotated, kernel-friendly)
         - ternary-encode r_rot               (sign + nz planes + scales)
         - cache cos(||r_rot||) * norm_d, sinc(||r_rot||) * norm_d as fp16
    3. Persist {centroids, centroid_id, sign, nz, scales, cos_norm,
       sin_norm} per shard; centroid table is shared.

Memory: encode is chunked (default 32k tokens per chunk). Centroid fit
runs on a 100k-token subsample. Validated to fit the 24 GB CPU envelope on
scidocs (4.84M tokens).
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Rroq158Config:
    K: int = 1024
    """Centroid codebook size. Disk overhead = ceil(log2(K))/dim bits per coord;
    K=1024 is the offline-validated sweet spot."""

    group_size: int = 32
    """Ternary group size in coords. Must be a multiple of 32 (one popcount word
    per group). At dim=128 → n_groups=4."""

    spherical_kmeans_iter: int = 15
    fit_sample_cap: int = 100_000
    """Max tokens used for centroid fitting (subsampled)."""

    encode_chunk: int = 32_768
    """Tokens per encode chunk to keep peak RAM in budget."""

    seed: int = 42


def _fwht_rotator(dim: int, seed: int):
    """Match the FWHT rotation used by ``ternary.TernaryQuantizer`` /
    ``RotationalQuantizer`` so the residual lives in the same rotated frame
    the asymmetric kernel expects."""
    from voyager_index._internal.inference.quantization.rotational import FastWalshHadamard

    block_size = 1
    while block_size < dim:
        block_size *= 2
    return FastWalshHadamard(dim=dim, num_rounds=3, block_size=block_size, seed=seed)


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def _log_map_unit_sphere(c: np.ndarray, q: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    cos_theta = np.clip((c * q).sum(axis=-1, keepdims=True), -1.0 + eps, 1.0 - eps)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    direction = q - cos_theta * c
    safe = np.where(sin_theta > eps, theta / sin_theta, 0.0)
    return direction * safe


def _spherical_kmeans(
    x: np.ndarray, k: int, n_iter: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical Lloyd with k-means++ init. Returns (centroids, assign).
    ``x`` is expected L2-normalized. Memory: O(n*k*4) for the sims matrix
    each iter, so call on a sub-sample if n*k is large."""
    rng = np.random.default_rng(seed)
    n, _d = x.shape
    if k > n:
        raise ValueError(f"K={k} > n={n}")
    idx = [int(rng.integers(0, n))]
    closest_sq = np.full(n, np.inf, dtype=np.float32)
    for _ in range(1, k):
        last = x[idx[-1]]
        d_sq = ((x - last) ** 2).sum(axis=1)
        closest_sq = np.minimum(closest_sq, d_sq)
        probs = closest_sq / closest_sq.sum()
        idx.append(int(rng.choice(n, p=probs)))
    centroids = _l2(x[np.asarray(idx)])
    assign = np.zeros(n, dtype=np.int64)

    for it in range(n_iter):
        sims = x @ centroids.T
        assign_new = sims.argmax(axis=1)
        if it > 0 and (assign_new != assign).sum() < 1e-4 * n:
            assign = assign_new
            break
        assign = assign_new
        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            mask = assign == c
            if mask.sum() < 1:
                far = (1.0 - sims[np.arange(n), assign]).argmax()
                new_centroids[c] = x[far]
                assign[far] = c
            else:
                new_centroids[c] = x[mask].sum(axis=0)
        centroids = _l2(new_centroids)
    return centroids, assign.astype(np.int32)


def _ternary_encode_rotated(rotated: np.ndarray, group_size: int) -> dict:
    """Encode an already-FWHT-rotated residual to ternary planes + scales.
    Mirrors ``research/low_bit_roq/ternary.py:quantize`` (tau_frac=0.5,
    fit_method='tau_frac' — anisotropic is opt-in but did not pay off on
    real BEIR). Uses bitorder='little' to match the Triton kernel."""
    n, dim = rotated.shape
    if dim % group_size != 0:
        raise ValueError(f"dim={dim} not divisible by group_size={group_size}")
    n_groups = dim // group_size
    grouped = rotated.reshape(n, n_groups, group_size)
    std_per_group = grouped.std(axis=2, ddof=0) + 1e-8
    tau = std_per_group * 0.5
    abs_g = np.abs(grouped)
    mask_init = abs_g > tau[..., None]
    num = (abs_g * mask_init).sum(axis=2)
    den = mask_init.sum(axis=2).clip(min=1)
    scales = (num / den).astype(np.float32)

    sign = (rotated > 0).astype(np.uint8)
    nonzero = (np.abs(rotated) > np.repeat(tau, group_size, axis=1)).astype(np.uint8)
    sign_packed = np.packbits(sign, axis=1, bitorder="little")
    nonzero_packed = np.packbits(nonzero, axis=1, bitorder="little")
    return {
        "sign_plane": sign_packed,           # (n, dim/8) uint8
        "nonzero_plane": nonzero_packed,     # (n, dim/8) uint8
        "scales": scales,                    # (n, n_groups) float32
    }


@dataclass
class Rroq158Encoded:
    """Per-shard encoded payload."""

    centroids: np.ndarray          # (K, dim) float32, unit-norm
    centroid_id: np.ndarray        # (n_tok,) uint16
    sign_plane: np.ndarray         # (n_tok, dim/8) uint8
    nonzero_plane: np.ndarray      # (n_tok, dim/8) uint8
    scales: np.ndarray             # (n_tok, n_groups) float16
    cos_norm: np.ndarray           # (n_tok,) float16  = cos(||r||) * norm_d
    sin_norm: np.ndarray           # (n_tok,) float16  = sinc(||r||) * norm_d
    fwht_seed: int                 # for re-creating the rotator at query time
    dim: int
    group_size: int


def encode_rroq158(
    tokens: np.ndarray, cfg: Rroq158Config | None = None,
) -> Rroq158Encoded:
    """Encode a corpus of multi-vector tokens with rroq158.

    ``tokens`` has shape (N, D) and may be pre-flattened across docs (the
    kernel only needs per-token arrays; doc offsets are tracked by the
    shard store separately).
    """
    cfg = cfg or Rroq158Config()
    n, dim = tokens.shape
    rng = np.random.default_rng(cfg.seed)

    # ---- centroid fit on subsample --------------------------------------
    fit_idx = rng.choice(n, size=min(cfg.fit_sample_cap, n), replace=False)
    fit_tokens = _l2(tokens[fit_idx].astype(np.float32))
    log.info("rroq158 fit: %d/%d tokens, K=%d", fit_tokens.shape[0], n, cfg.K)
    centroids, _ = _spherical_kmeans(
        fit_tokens, k=cfg.K, n_iter=cfg.spherical_kmeans_iter, seed=cfg.seed
    )
    del fit_tokens; gc.collect()

    rotator = _fwht_rotator(dim=dim, seed=cfg.seed)
    norms_full = np.linalg.norm(tokens, axis=1).astype(np.float32)

    # ---- chunked assign + ternary encode --------------------------------
    centroid_id = np.empty(n, dtype=np.uint16)
    sign_planes = []
    nonzero_planes = []
    scales_all = []
    cos_norm = np.empty(n, dtype=np.float16)
    sin_norm = np.empty(n, dtype=np.float16)
    log.info("rroq158 encode: chunked, chunk=%d", cfg.encode_chunk)
    for s in range(0, n, cfg.encode_chunk):
        e = min(s + cfg.encode_chunk, n)
        chunk = tokens[s:e].astype(np.float32)
        chunk_n = _l2(chunk)
        sims_chunk = chunk_n @ centroids.T
        cid_chunk = sims_chunk.argmax(axis=1).astype(np.uint16)
        centroid_id[s:e] = cid_chunk
        del sims_chunk

        c_per_tok = centroids[cid_chunk]
        tangent_amb = _log_map_unit_sphere(c_per_tok, chunk_n)

        import torch
        tangent_rot = (
            rotator.forward(torch.from_numpy(tangent_amb.astype(np.float32)))
            .cpu().numpy()
        )
        # Pad/trim to original dim — FWHT may pad up to next power-of-two.
        if tangent_rot.shape[1] != dim:
            tangent_rot = tangent_rot[:, :dim]

        enc = _ternary_encode_rotated(tangent_rot, group_size=cfg.group_size)
        sign_planes.append(enc["sign_plane"])
        nonzero_planes.append(enc["nonzero_plane"])
        scales_all.append(enc["scales"].astype(np.float16))

        # Cache cos/sinc multiplied by original norm — kernel-side combine
        # is then a single fp16 fmadd per (q_tok, d_tok) pair.
        r_norm = np.linalg.norm(tangent_rot, axis=1) + 1e-12
        cos_t = np.cos(r_norm).astype(np.float32)
        sin_t = np.sinc(r_norm / np.pi).astype(np.float32)   # sin(x)/x
        cos_norm[s:e] = (cos_t * norms_full[s:e]).astype(np.float16)
        sin_norm[s:e] = (sin_t * norms_full[s:e]).astype(np.float16)

    return Rroq158Encoded(
        centroids=centroids.astype(np.float32),
        centroid_id=centroid_id,
        sign_plane=np.concatenate(sign_planes, axis=0),
        nonzero_plane=np.concatenate(nonzero_planes, axis=0),
        scales=np.concatenate(scales_all, axis=0),
        cos_norm=cos_norm,
        sin_norm=sin_norm,
        fwht_seed=cfg.seed,
        dim=dim,
        group_size=cfg.group_size,
    )


# ---------------------------------------------------------------------------
# Query-side encoding + kernel-input builders
# ---------------------------------------------------------------------------


def encode_query_for_rroq158(
    queries: np.ndarray,
    centroids: np.ndarray,
    *,
    fwht_seed: int,
    query_bits: int = 4,
):
    """Build the Stage-1 host-side tensors the kernel consumes.

    Returns a dict of fp32 / int32 numpy arrays:

        - q_planes     : (S, query_bits, n_words) int32
        - q_meta       : (S, 2) float32  [scale, offset]
        - qc_table     : (S, K) float32  = q_amb @ centroids.T

    ``queries`` is (S, dim) — we treat each query token independently
    (the wrapping launcher in scorer.py adds the batch dim).
    """
    if queries.ndim != 2:
        raise ValueError(f"expected (S, dim) queries, got {queries.shape}")
    s, dim = queries.shape
    rotator = _fwht_rotator(dim=dim, seed=fwht_seed)

    # Centroid table: <q_amb, c_k>
    qc_table = (queries.astype(np.float32) @ centroids.T).astype(np.float32)

    # Rotated query for the residual term
    import torch
    q_rot = rotator.forward(torch.from_numpy(queries.astype(np.float32))).cpu().numpy()
    if q_rot.shape[1] != dim:
        q_rot = q_rot[:, :dim]

    # Asymmetric scalar quantization, per-token bit-planes
    if query_bits not in (4, 6, 8):
        raise ValueError("query_bits ∈ {4, 6, 8}")
    levels = float((1 << query_bits) - 1)
    min_v = q_rot.min(axis=1)
    max_v = q_rot.max(axis=1)
    rng_ = np.where((max_v - min_v) < 1e-6, 1.0, max_v - min_v)
    scale = rng_ / levels
    quant = np.round((q_rot - min_v[:, None]) / scale[:, None]).clip(0, levels).astype(np.uint8)

    n_words = (dim + 31) // 32
    planes = np.zeros((s, query_bits, n_words), dtype=np.int32)
    for k in range(query_bits):
        bit = ((quant >> k) & 0x01).astype(np.uint8)
        # Pad to multiple of 32 if needed
        if bit.shape[1] % 32 != 0:
            pad = 32 - bit.shape[1] % 32
            bit = np.pad(bit, ((0, 0), (0, pad)))
        packed = np.packbits(bit, axis=1, bitorder="little").view(np.int32)
        planes[:, k, :packed.shape[1]] = packed
    meta = np.stack([scale, min_v], axis=1).astype(np.float32)
    return {"q_planes": planes, "q_meta": meta, "qc_table": qc_table}


def pack_doc_codes_to_int32_words(sign_planes_u8: np.ndarray) -> np.ndarray:
    """Reinterpret packed-uint8 sign/nonzero planes as int32 words for the
    Triton kernel. The kernel uses 32-bit popcount, so we feed it the
    same bytes viewed as int32. ``sign_planes_u8`` shape is
    ``(n_tok, dim/8)`` uint8; output is ``(n_tok, dim/32)`` int32."""
    n, nb = sign_planes_u8.shape
    if nb % 4 != 0:
        sign_planes_u8 = np.pad(sign_planes_u8, ((0, 0), (0, 4 - nb % 4)))
    return sign_planes_u8.view(np.int32).copy()


__all__ = [
    "Rroq158Config",
    "Rroq158Encoded",
    "encode_rroq158",
    "encode_query_for_rroq158",
    "pack_doc_codes_to_int32_words",
]
