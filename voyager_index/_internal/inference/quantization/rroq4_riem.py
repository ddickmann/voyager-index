"""
Riemannian-aware 4-bit asymmetric ROQ encoder + kernel input builders.

This is the production "safe fallback" codec — same Riemannian +
FWHT-rotated residual structure as ``rroq158.py``, but the residual is
quantized at **4 bits per dimension asymmetric per-group** instead of
1.58-bit ternary. ~1.9× larger on disk than rroq158, ~3× smaller than
fp16, and the recall floor is ≈ fp16 (the 4-bit quantization noise on a
unit-radius residual sits below the per-token sinc weighting on real
multi-vector data — empirically within 0.5% NDCG@10 on hard BEIR tasks).

Score formula per (q_token, d_token), identical in shape to rroq158:

    sim(q, d) = norm_d * ( cos(||r_d||) * <q, c_d>
                         + sinc(||r_d||) * <q_rot, dequant_4b(r_d_rot)> )

The 4-bit dequantization expands as

    dequant_4b(r)[d] = min[grp(d)] + code[d] * delta[grp(d)]

so the inner product against the FWHT-rotated query becomes

    <q_rot, dequant(r)> = Σ_g ( min[g] * Σ_q q_rot[g, :]
                              + delta[g] * <q_rot[g, :], code[g, :]> )

The first sum is a per-group reduction of q_rot that can be computed
*once per query* (fed to the kernel as ``q_group_sums``), so the per-
(q_token, d_token) cost on the doc side is one 4-bit dot product per
group + 2 FMAs per group. That is clean enough to vectorize on AVX2 and
match Triton's GPU throughput within the same launch overhead budget.

Encode pipeline (numpy-only, CPU, runs at index-build time):

    1. Spherical k-means to fit K unit-norm centroids c_k (same as rroq158).
    2. Per d-token x:
         - assign centroid c = c_{argmax_k <x_n, c_k>}
         - tangent r_amb = log_c(x_n)
         - r_rot = FWHT(r_amb)                  (rotated, kernel-friendly)
         - per-group asymmetric 4-bit encode    (min, delta, codes ∈ [0, 15])
         - cache cos(||r_rot||) * norm_d, sinc(||r_rot||) * norm_d as fp16
    3. Persist {centroids, centroid_id, codes_packed, mins, deltas,
       cos_norm, sin_norm}; centroid table is shared.

Per-token disk overhead (dim=128, group_size=32, n_groups=4):

    centroid_id   2 B   uint16
    codes        64 B   dim/2 (4-bit packed)
    mins          8 B   n_groups × fp16
    deltas        8 B   n_groups × fp16
    cos_norm      2 B   fp16
    sin_norm      2 B   fp16
    -----------------------------
    total        86 B/token   vs 256 fp16, vs 46 rroq158, vs 64 ROQ4

Memory at build time: same envelope as rroq158 (centroid fit on a 100 k
subsample; encode is chunked).
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import numpy as np

# Re-use the spherical kmeans + log-map + FWHT rotator caching from rroq158
# to avoid duplicating the Riemannian primitives across codecs. They are
# the codec-independent half of the Riemannian-aware pipeline.
from voyager_index._internal.inference.quantization.rroq158 import (
    _l2,
    _log_map_unit_sphere,
    _spherical_kmeans,
    get_cached_fwht_rotator,
)

log = logging.getLogger(__name__)


@dataclass
class Rroq4RiemConfig:
    K: int = 8192
    """Centroid codebook size. Same defaults as rroq158 — 8192 closes the
    K=1024 quality gap. Must be a power of two and ``>= group_size``."""

    group_size: int = 32
    """4-bit asymmetric quantization group size in coords. Must divide ``dim``.
    Smaller groups → finer scale resolution → less quantization noise → more
    storage (per-group min+delta = 4 B per group). 32 is the rroq158 default
    and gives ≈ fp16 quality on production multi-vector embeddings."""

    spherical_kmeans_iter: int = 15
    fit_sample_cap: int = 100_000
    encode_chunk: int = 32_768
    seed: int = 42

    def __post_init__(self) -> None:
        if self.K < self.group_size:
            raise ValueError(
                f"Rroq4RiemConfig.K ({self.K}) must be >= group_size "
                f"({self.group_size})"
            )
        if (self.K & (self.K - 1)) != 0:
            raise ValueError(
                f"Rroq4RiemConfig.K must be a power of two, got {self.K}"
            )
        if self.group_size <= 0 or self.group_size % 2 != 0:
            raise ValueError(
                f"Rroq4RiemConfig.group_size must be a positive even "
                f"integer (4-bit codes pack two coords per byte), got "
                f"{self.group_size}"
            )
        if self.fit_sample_cap < self.K:
            raise ValueError(
                f"Rroq4RiemConfig.fit_sample_cap ({self.fit_sample_cap}) "
                f"must be >= K ({self.K}) for k-means to converge"
            )
        if self.encode_chunk < 1:
            raise ValueError(
                f"Rroq4RiemConfig.encode_chunk must be positive, got "
                f"{self.encode_chunk}"
            )
        if self.K < 1024:
            log.debug(
                "Rroq4RiemConfig.K=%d is below the production floor of "
                "1024. Quality drops fast for K<1024; production indexes "
                "should use K >= 8192.",
                self.K,
            )


def choose_effective_rroq4_riem_k(
    n_tokens: int, requested_k: int, group_size: int = 32
) -> int:
    """Pick the largest power-of-two K ≤ ``requested_k`` that the corpus can
    actually train.

    Mirrors :func:`choose_effective_rroq158_k` but with the laxer
    rroq4_riem ``group_size`` constraint (positive even integer rather
    than multiple of 32). Returns ``group_size`` if the corpus is too
    small for any larger power-of-two; raises ``ValueError`` if even
    ``group_size`` is too large for the corpus.
    """
    if n_tokens <= 0:
        raise ValueError(f"n_tokens must be > 0, got {n_tokens}")
    if requested_k <= 0 or (requested_k & (requested_k - 1)) != 0:
        raise ValueError(
            f"requested_k must be a positive power of two, got {requested_k}"
        )
    if group_size <= 0 or group_size % 2 != 0:
        raise ValueError(
            f"group_size must be a positive even integer, got {group_size}"
        )
    if n_tokens < group_size:
        raise ValueError(
            f"choose_effective_rroq4_riem_k: corpus has {n_tokens} tokens "
            f"but group_size={group_size}; need at least one full group "
            f"of tokens. Caller should fall back to FP16."
        )
    if n_tokens >= requested_k:
        return requested_k
    # Largest power of two ≤ n_tokens, but never below group_size — the
    # encoder cannot produce a codebook smaller than one group.
    k = 1
    while k * 2 <= n_tokens:
        k *= 2
    return max(k, group_size)


def _pack_4bit(codes_u8: np.ndarray) -> np.ndarray:
    """Pack a (n, dim) uint8 array with values in [0, 15] into a
    (n, dim/2) uint8 array of nibble pairs.

    Layout: byte ``b`` holds ``codes[2*b]`` in the low nibble and
    ``codes[2*b + 1]`` in the high nibble. This is the same convention
    the Triton/Rust kernels expect (so ``code[d] = (byte >> ((d&1)*4))
    & 0xF``).
    """
    if codes_u8.dtype != np.uint8:
        raise TypeError(
            f"_pack_4bit expects uint8 codes, got dtype={codes_u8.dtype}"
        )
    if codes_u8.ndim != 2:
        raise ValueError(
            f"_pack_4bit expects a 2-D (n_tok, dim) code matrix, got "
            f"shape {codes_u8.shape}"
        )
    n, d = codes_u8.shape
    if d % 2 != 0:
        raise ValueError(
            f"_pack_4bit requires an even dim (got {d}); group_size and "
            "dim must keep dim%2==0 so 4-bit nibbles pack into bytes"
        )
    if codes_u8.max(initial=0) > 15:
        raise ValueError(
            "_pack_4bit input contains values > 15; encoder bug — "
            "asymmetric 4-bit must clip to [0, 15] before packing"
        )
    low = codes_u8[:, 0::2]
    high = codes_u8[:, 1::2]
    return (low | (high << 4)).astype(np.uint8)


def unpack_4bit(packed_u8: np.ndarray, dim: int) -> np.ndarray:
    """Inverse of :func:`_pack_4bit` — exposed for parity tests + the
    pure-python reference scorer."""
    if packed_u8.dtype != np.uint8:
        raise TypeError(f"unpack_4bit expects uint8 packed bytes, got {packed_u8.dtype}")
    if packed_u8.ndim != 2:
        raise ValueError(f"unpack_4bit expects 2-D, got shape {packed_u8.shape}")
    n, nb = packed_u8.shape
    if nb * 2 != dim:
        raise ValueError(
            f"unpack_4bit shape mismatch: packed has {nb} bytes => "
            f"dim={nb*2} but caller asked for dim={dim}"
        )
    out = np.empty((n, dim), dtype=np.uint8)
    out[:, 0::2] = packed_u8 & 0x0F
    out[:, 1::2] = (packed_u8 >> 4) & 0x0F
    return out


def _asym_4bit_encode_rotated(
    rotated: np.ndarray, group_size: int
) -> dict:
    """Per-group asymmetric 4-bit quantization of an FWHT-rotated residual.

    Returns a dict with ``codes_packed`` (n, dim/2) uint8, ``mins`` and
    ``deltas`` (n, n_groups) float32. Caller may downcast scales/offsets
    to fp16 for storage.
    """
    n, dim = rotated.shape
    if dim % group_size != 0:
        raise ValueError(
            f"_asym_4bit_encode_rotated: dim={dim} not divisible by "
            f"group_size={group_size}"
        )
    n_groups = dim // group_size
    grouped = rotated.reshape(n, n_groups, group_size)
    mins = grouped.min(axis=2)
    maxs = grouped.max(axis=2)
    rng = maxs - mins
    # Guard against zero-range groups (constant residual in the group);
    # store delta=1 so dequant returns the same constant ``min`` value
    # without dividing by zero in the encoder.
    safe_rng = np.where(rng < 1e-7, 1.0, rng).astype(np.float32)
    deltas = (safe_rng / 15.0).astype(np.float32)
    quant = np.round(
        (grouped - mins[..., None]) / deltas[..., None]
    ).clip(0, 15).astype(np.uint8)
    codes_packed = _pack_4bit(quant.reshape(n, dim))
    return {
        "codes_packed": codes_packed,
        "mins": mins.astype(np.float32),
        "deltas": deltas.astype(np.float32),
    }


@dataclass
class Rroq4RiemEncoded:
    """Per-shard encoded payload."""

    centroids: np.ndarray          # (K, dim) float32, unit-norm
    centroid_id: np.ndarray        # (n_tok,) uint16
    codes_packed: np.ndarray       # (n_tok, dim/2) uint8 — 4-bit nibble pairs
    mins: np.ndarray               # (n_tok, n_groups) float16 — per-group offset
    deltas: np.ndarray             # (n_tok, n_groups) float16 — per-group scale
    cos_norm: np.ndarray           # (n_tok,) float16  = cos(||r||) * norm_d
    sin_norm: np.ndarray           # (n_tok,) float16  = sinc(||r||) * norm_d
    fwht_seed: int                 # for re-creating the rotator at query time
    dim: int
    group_size: int


def encode_rroq4_riem(
    tokens: np.ndarray, cfg: Rroq4RiemConfig | None = None,
) -> Rroq4RiemEncoded:
    """Encode a corpus of multi-vector tokens with rroq4_riem.

    ``tokens`` has shape ``(N, D)`` and may be pre-flattened across docs;
    ``D`` must be a positive multiple of ``cfg.group_size`` (and even, so
    4-bit nibbles pack into bytes).
    """
    cfg = cfg or Rroq4RiemConfig()
    if tokens.ndim != 2:
        raise ValueError(
            f"encode_rroq4_riem expects a (N, D) token matrix, got shape "
            f"{tokens.shape}"
        )
    n, dim = tokens.shape
    if n == 0:
        raise ValueError("encode_rroq4_riem received an empty token matrix")
    if dim < cfg.group_size:
        raise ValueError(
            f"encode_rroq4_riem: dim={dim} is smaller than group_size="
            f"{cfg.group_size}; reduce Rroq4RiemConfig.group_size to a "
            f"divisor of dim that is even (so 4-bit codes pack into bytes)"
        )
    if dim % cfg.group_size != 0:
        raise ValueError(
            f"encode_rroq4_riem: dim={dim} is not divisible by group_size="
            f"{cfg.group_size}; pick a group_size that divides dim"
        )
    if dim % 2 != 0:
        raise ValueError(
            f"encode_rroq4_riem: dim={dim} must be even so the 4-bit "
            "nibble plane packs into bytes"
        )
    if n < cfg.K:
        raise ValueError(
            f"encode_rroq4_riem: only {n} tokens available but K={cfg.K} "
            f"centroids requested. Either lower Rroq4RiemConfig.K (must "
            f"remain a power of two and >= {cfg.group_size}) or feed more "
            "tokens."
        )
    rng = np.random.default_rng(cfg.seed)

    fit_idx = rng.choice(n, size=min(cfg.fit_sample_cap, n), replace=False)
    fit_tokens = _l2(tokens[fit_idx].astype(np.float32))
    log.info("rroq4_riem fit: %d/%d tokens, K=%d", fit_tokens.shape[0], n, cfg.K)
    centroids, _ = _spherical_kmeans(
        fit_tokens, k=cfg.K, n_iter=cfg.spherical_kmeans_iter, seed=cfg.seed
    )
    del fit_tokens
    gc.collect()

    import torch  # local import keeps cold-start light

    rotator = get_cached_fwht_rotator(dim=dim, seed=cfg.seed)
    norms_full = np.linalg.norm(tokens, axis=1).astype(np.float32)

    centroid_id = np.empty(n, dtype=np.uint16)
    n_groups = dim // cfg.group_size
    codes_chunks: list[np.ndarray] = []
    mins_chunks: list[np.ndarray] = []
    deltas_chunks: list[np.ndarray] = []
    cos_norm = np.empty(n, dtype=np.float16)
    sin_norm = np.empty(n, dtype=np.float16)
    log.info("rroq4_riem encode: chunked, chunk=%d", cfg.encode_chunk)
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

        with torch.no_grad():
            tangent_rot = (
                rotator.forward(torch.from_numpy(tangent_amb.astype(np.float32)))
                .cpu().numpy()
            )
        if tangent_rot.shape[1] != dim:
            tangent_rot = tangent_rot[:, :dim]

        enc = _asym_4bit_encode_rotated(tangent_rot, group_size=cfg.group_size)
        codes_chunks.append(enc["codes_packed"])
        mins_chunks.append(enc["mins"].astype(np.float16))
        deltas_chunks.append(enc["deltas"].astype(np.float16))

        r_norm = np.linalg.norm(tangent_rot, axis=1) + 1e-12
        cos_t = np.cos(r_norm).astype(np.float32)
        sin_t = np.sinc(r_norm / np.pi).astype(np.float32)
        cos_norm[s:e] = (cos_t * norms_full[s:e]).astype(np.float16)
        sin_norm[s:e] = (sin_t * norms_full[s:e]).astype(np.float16)

    return Rroq4RiemEncoded(
        centroids=centroids.astype(np.float32),
        centroid_id=centroid_id,
        codes_packed=np.concatenate(codes_chunks, axis=0),
        mins=np.concatenate(mins_chunks, axis=0),
        deltas=np.concatenate(deltas_chunks, axis=0),
        cos_norm=cos_norm,
        sin_norm=sin_norm,
        fwht_seed=cfg.seed,
        dim=dim,
        group_size=cfg.group_size,
    )


# ---------------------------------------------------------------------------
# Query-side helpers (kernel input builders)
# ---------------------------------------------------------------------------


def encode_query_for_rroq4_riem(
    queries: np.ndarray,
    centroids: np.ndarray | None,
    *,
    fwht_seed: int,
    group_size: int,
    rotator: object | None = None,
    skip_qc_table: bool = False,
):
    """Build the host-side tensors the rroq4_riem kernel consumes.

    Returns a dict with:

        - q_rot         : (S, dim) float32  — FWHT(query)
        - q_group_sums  : (S, n_groups) float32 — Σ_d q_rot[g*GS + d] per group
                          (kernel multiplies this by per-group ``min`` directly)
        - qc_table      : (S, K) float32    — q_amb @ centroids.T
                          (omitted when ``skip_qc_table=True``)

    No bit-plane packing — the asymmetric 4-bit lives entirely on the doc
    side, the query stays full-precision.

    ``group_size`` is implicit through ``q_group_sums``'s second dim,
    which the caller must match to the index-side ``group_size``.
    """
    if queries.ndim != 2:
        raise ValueError(f"expected (S, dim) queries, got {queries.shape}")
    s, dim = queries.shape
    if s == 0:
        raise ValueError("encode_query_for_rroq4_riem: zero query tokens")
    if dim % 2 != 0:
        raise ValueError(
            f"encode_query_for_rroq4_riem: dim={dim} must be even to "
            "match the 4-bit nibble packing on the doc side"
        )
    if group_size <= 0 or group_size % 2 != 0:
        raise ValueError(
            f"encode_query_for_rroq4_riem: group_size={group_size} must be "
            "a positive even integer"
        )
    if dim % group_size != 0:
        raise ValueError(
            f"encode_query_for_rroq4_riem: dim={dim} not divisible by "
            f"group_size={group_size}"
        )
    if rotator is None:
        rotator = get_cached_fwht_rotator(dim=dim, seed=fwht_seed)

    queries_f32 = queries.astype(np.float32, copy=False)
    if skip_qc_table:
        qc_table = None
    else:
        if centroids is None:
            raise ValueError("centroids required when skip_qc_table is False")
        qc_table = (queries_f32 @ centroids.T).astype(np.float32, copy=False)

    dense = getattr(rotator, "_dense_matrix_np", None)
    if dense is not None and dense.shape == (dim, dim):
        q_rot = (queries_f32 @ dense).astype(np.float32, copy=False)
    else:
        import torch
        with torch.no_grad():
            q_rot = rotator.forward(torch.from_numpy(queries_f32)).cpu().numpy()
        if q_rot.shape[1] != dim:
            q_rot = q_rot[:, :dim]
    q_rot = np.ascontiguousarray(q_rot, dtype=np.float32)

    # ``q_group_sums[s, g] = Σ_{d ∈ group g} q_rot[s, d]`` is consumed by
    # the rroq4_riem kernel as the contribution of the per-group ``min``
    # offset to the final inner product. Pre-reducing it here removes a
    # GROUP_SIZE-wide accumulator from every (q_token, d_token) pair in
    # the inner loop, which is the dominant cost on small batches.
    n_groups = dim // group_size
    q_group_sums = q_rot.reshape(s, n_groups, group_size).sum(
        axis=2, dtype=np.float32
    )
    q_group_sums = np.ascontiguousarray(q_group_sums, dtype=np.float32)

    out = {"q_rot": q_rot, "q_group_sums": q_group_sums}
    if qc_table is not None:
        out["qc_table"] = qc_table
    return out


__all__ = [
    "Rroq4RiemConfig",
    "Rroq4RiemEncoded",
    "encode_rroq4_riem",
    "encode_query_for_rroq4_riem",
    "choose_effective_rroq4_riem_k",
    "unpack_4bit",
]
