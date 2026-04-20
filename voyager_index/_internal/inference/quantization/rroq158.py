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
    K: int = 8192
    """Centroid codebook size. Disk overhead = ceil(log2(K))/dim bits per coord;
    K=8192 is the production default — closes the K=1024 quality gap on hard
    BEIR datasets (scidocs / arguana) at ~2 MB centroid table cost. Must be
    a power of two and ``>= 256``."""

    group_size: int = 128
    """Ternary group size in coords. Production SOTA default ``128`` (one scale
    per token at dim=128, the most-tested production dim). Must be a positive
    multiple of 32 (one popcount word per group). For dims that are not a
    multiple of the requested ``group_size``, the encoder transparently falls
    back to the largest compatible value in ``{128, 64, 32}`` and logs a
    warning — so dim=64 / 96 / 160 corpora still build cleanly without
    requiring callers to override the default. Set explicitly (typically to
    ``64``) when (a) you serve a corpus with high intra-token magnitude
    variance such that one scale per token is too coarse — see
    ``docs/guides/quantization-tuning.md`` for the per-dim recipe table —
    or (b) you want to pin behaviour across dim variations."""

    spherical_kmeans_iter: int = 15
    fit_sample_cap: int = 100_000
    """Max tokens used for centroid fitting (subsampled)."""

    encode_chunk: int = 32_768
    """Tokens per encode chunk to keep peak RAM in budget."""

    seed: int = 42

    def __post_init__(self) -> None:
        if self.K < self.group_size:
            raise ValueError(
                f"Rroq158Config.K ({self.K}) must be >= group_size "
                f"({self.group_size})"
            )
        if (self.K & (self.K - 1)) != 0:
            raise ValueError(
                f"Rroq158Config.K must be a power of two, got {self.K}"
            )
        if self.group_size < 32 or self.group_size % 32 != 0:
            raise ValueError(
                f"Rroq158Config.group_size must be a positive multiple of 32, "
                f"got {self.group_size}"
            )
        if self.fit_sample_cap < self.K:
            raise ValueError(
                f"Rroq158Config.fit_sample_cap ({self.fit_sample_cap}) must "
                f"be >= K ({self.K}) for k-means to converge"
            )
        if self.encode_chunk < 1:
            raise ValueError(
                f"Rroq158Config.encode_chunk must be positive, got "
                f"{self.encode_chunk}"
            )
        if self.K < 1024:
            log.debug(
                "Rroq158Config.K=%d is below the production floor of 1024. "
                "Quality drops fast for K<1024; production indexes should "
                "use K >= 8192.",
                self.K,
            )


def choose_effective_rroq158_k(
    n_tokens: int, requested_k: int, group_size: int = 32
) -> int:
    """Pick the largest power-of-two K ≤ requested_k that the corpus can
    actually train.

    The encoder hard-fails when ``n_tokens < K`` (you can't fit K centroids
    from < K samples). In production the corpus is millions of tokens so the
    requested K is always feasible; in tests / demos / tiny shards it can
    happen that ``n_tokens < requested_k``. Returning a clamped K here lets
    the build orchestration (``lifecycle.py`` / ``pipeline.py``) keep the
    user's choice of the rroq158 codec while shrinking the codebook to fit —
    same score scale, no silent fp16 fallback, just a smaller K.

    Returns the chosen K (always a power of two and ``>= group_size``).
    Raises ``ValueError`` if even ``group_size`` is too large for the corpus.
    """
    if n_tokens <= 0:
        raise ValueError(f"n_tokens must be > 0, got {n_tokens}")
    if requested_k <= 0 or (requested_k & (requested_k - 1)) != 0:
        raise ValueError(
            f"requested_k must be a positive power of two, got {requested_k}"
        )
    if group_size <= 0 or group_size % 32 != 0:
        raise ValueError(
            f"group_size must be a positive multiple of 32, got {group_size}"
        )
    if n_tokens < group_size:
        raise ValueError(
            f"corpus has only {n_tokens} tokens but rroq158 requires at least "
            f"group_size={group_size} tokens to train a single centroid"
        )
    if n_tokens >= requested_k:
        return requested_k
    # Largest power of two ≤ n_tokens
    k = 1 << (int(n_tokens).bit_length() - 1)
    if k < group_size:
        k = group_size
    log.warning(
        "rroq158: corpus has %d tokens but requested K=%d; clamping K to %d "
        "(largest power-of-two ≤ n_tokens). For production-grade quality "
        "feed >= %d tokens or pass compression=Compression.FP16 for tiny "
        "corpora.",
        n_tokens, requested_k, k, requested_k,
    )
    return k


def _fwht_rotator(dim: int, seed: int):
    """Match the FWHT rotation used by ``ternary.TernaryQuantizer`` /
    ``RotationalQuantizer`` so the residual lives in the same rotated frame
    the asymmetric kernel expects."""
    from voyager_index._internal.inference.quantization.rotational import FastWalshHadamard

    block_size = 1
    while block_size < dim:
        block_size *= 2
    return FastWalshHadamard(dim=dim, num_rounds=3, block_size=block_size, seed=seed)


# Module-level rotator cache. Building the rotator costs ~0.3-0.5 ms (k random
# perms + sign vectors); caching by (dim, seed) eliminates this from the
# per-query hot path. Bounded to a handful of entries because we expect at
# most one (dim, seed) pair per index. The bound stops a multi-tenant server
# that hosts many distinct (dim, seed) indexes from leaking unbounded memory:
# at the cap, the oldest entry is evicted (FIFO insertion order).
_FWHT_ROTATOR_CACHE_CAP = 32
_FWHT_ROTATOR_CACHE: dict[tuple[int, int], object] = {}


def clear_fwht_rotator_cache() -> None:
    """Drop all cached FWHT rotators. Useful for tests and for multi-tenant
    servers that need to free GPU/CPU memory tied to a tenant being unloaded.
    """
    _FWHT_ROTATOR_CACHE.clear()


def get_cached_fwht_rotator(dim: int, seed: int):
    """Return a process-cached FWHT rotator for (dim, seed).

    Safe to call from many threads: dict insertion is atomic in CPython and
    a duplicate construction race is harmless (we just replace the entry).

    The returned rotator is also augmented with a ``_dense_matrix_np`` cache:
    a ``(dim, dim)`` numpy float32 view of the rotator's linear operator. The
    operator is fixed by ``(dim, seed)``, so we can extract it once and
    replace the per-query 7-stage PyTorch dispatch (which has a >30 ms p95
    tail under GIL contention with an active CUDA context) with a single
    BLAS GEMM. See ``encode_query_for_rroq158`` for the consumer.
    """
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(
            f"get_cached_fwht_rotator: dim must be a positive int, got {dim!r}"
        )
    key = (dim, seed)
    rot = _FWHT_ROTATOR_CACHE.get(key)
    if rot is None:
        rot = _fwht_rotator(dim, seed)
        if len(_FWHT_ROTATOR_CACHE) >= _FWHT_ROTATOR_CACHE_CAP:
            # FIFO eviction: drop the oldest insertion. dict preserves insert
            # order in CPython 3.7+, so next(iter(...)) gives the oldest key.
            _FWHT_ROTATOR_CACHE.pop(next(iter(_FWHT_ROTATOR_CACHE)), None)
        _FWHT_ROTATOR_CACHE[key] = rot
    if not hasattr(rot, "_dense_matrix_np"):
        import torch
        with torch.no_grad():
            # `rot.forward` expects inputs with shape[-1] == rot.dim and pads
            # internally to padded_dim. Feed an `eye(dim)` so each row is the
            # operator applied to a one-hot of length `dim`, then slice to
            # `dim` rotated coords. This is correct for non-power-of-2 dims
            # (e.g. dim=96 with padded_dim=128) — see
            # ``test_rroq158_dense_matrix_fwht_path``.
            eye = torch.eye(int(dim), dtype=torch.float32)
            full = rot.forward(eye).cpu().numpy().astype(np.float32, copy=False)
        rot._dense_matrix_np = np.ascontiguousarray(full[:dim, :dim])
    return rot


def _resolve_group_size(requested: int, dim: int) -> int:
    """Pick the largest ``group_size`` in ``{requested, 64, 32}`` that the
    encoder can actually use for this ``dim``.

    The dim divisibility constraint cannot be checked at config-construction
    time (``Rroq158Config.__post_init__``) because ``dim`` only becomes
    known when the corpus is handed to ``encode_rroq158``. Resolving here
    lets the production default ``group_size=128`` (one scale per token at
    dim=128 — the SOTA path) work transparently on dim=64 / 96 / 160
    production corpora by stepping down to gs=64 or gs=32 with a warning,
    instead of raising and forcing every caller to special-case dim.

    Returns the resolved ``group_size`` (always a positive multiple of 32
    that divides ``dim``). Raises ``ValueError`` if neither the requested
    value nor any of {64, 32} divides ``dim`` (e.g. dim=48).
    """
    if requested <= 0 or requested % 32 != 0:
        raise ValueError(
            f"_resolve_group_size: requested={requested} must be a positive "
            "multiple of 32"
        )
    candidates = (requested, 64, 32)
    seen: set[int] = set()
    for gs in candidates:
        if gs in seen:
            continue
        seen.add(gs)
        if dim % gs == 0 and gs % 32 == 0:
            if gs != requested:
                log.warning(
                    "rroq158: dim=%d not divisible by requested group_size=%d; "
                    "falling back to gs=%d (largest divisor in {128, 64, 32}). "
                    "Override Rroq158Config.group_size explicitly to silence "
                    "this warning.",
                    dim, requested, gs,
                )
            return gs
    raise ValueError(
        f"rroq158: dim={dim} cannot fit any of {{128, 64, 32}} as group_size; "
        "production dims (64, 96, 128, 160, 256, 384, 768, 1024) all support "
        "at least one of these. Provide a dim that is a multiple of 32."
    )


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
    if tokens.ndim != 2:
        raise ValueError(
            f"encode_rroq158 expects a (N, D) token matrix, got shape "
            f"{tokens.shape}"
        )
    n, dim = tokens.shape
    if n == 0:
        raise ValueError("encode_rroq158 received an empty token matrix")
    if dim % 32 != 0:
        raise ValueError(
            f"encode_rroq158: dim={dim} must be a multiple of 32 so the "
            "ternary planes pack into int32 words for the popcount kernel"
        )
    # Resolve dim-aware group_size: the SOTA default (128) covers dim=128 /
    # 256 / 384 / 512 / 768 / 1024; for dim=64 / 96 / 160 we transparently
    # step down to 64 or 32. Manifest below records the resolved value.
    resolved_gs = _resolve_group_size(cfg.group_size, dim)
    if n < cfg.K:
        raise ValueError(
            f"encode_rroq158: only {n} tokens available but K={cfg.K} centroids "
            f"requested. Either lower Rroq158Config.K (must remain a power of "
            f"two and >= {resolved_gs}) or feed more tokens."
        )
    rng = np.random.default_rng(cfg.seed)

    # ---- centroid fit on subsample --------------------------------------
    fit_idx = rng.choice(n, size=min(cfg.fit_sample_cap, n), replace=False)
    fit_tokens = _l2(tokens[fit_idx].astype(np.float32))
    log.info("rroq158 fit: %d/%d tokens, K=%d", fit_tokens.shape[0], n, cfg.K)
    centroids, _ = _spherical_kmeans(
        fit_tokens, k=cfg.K, n_iter=cfg.spherical_kmeans_iter, seed=cfg.seed
    )
    del fit_tokens
    gc.collect()

    import torch  # local import keeps top-level cold-start light

    rotator = get_cached_fwht_rotator(dim=dim, seed=cfg.seed)
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

        with torch.no_grad():
            tangent_rot = (
                rotator.forward(torch.from_numpy(tangent_amb.astype(np.float32)))
                .cpu().numpy()
            )
        # Pad/trim to original dim — FWHT may pad up to next power-of-two.
        if tangent_rot.shape[1] != dim:
            tangent_rot = tangent_rot[:, :dim]

        enc = _ternary_encode_rotated(tangent_rot, group_size=resolved_gs)
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
        group_size=resolved_gs,
    )


# ---------------------------------------------------------------------------
# Query-side encoding + kernel-input builders
# ---------------------------------------------------------------------------


def encode_query_for_rroq158(
    queries: np.ndarray,
    centroids: np.ndarray | None,
    *,
    fwht_seed: int,
    query_bits: int = 4,
    rotator: object | None = None,
    skip_qc_table: bool = False,
):
    """Build the Stage-1 host-side tensors the kernel consumes.

    Returns a dict of fp32 / int32 numpy arrays:

        - q_planes     : (S, query_bits, n_words) int32
        - q_meta       : (S, 2) float32  [scale, offset]
        - qc_table     : (S, K) float32  = q_amb @ centroids.T  (or absent if
                         ``skip_qc_table`` is True; pass ``centroids=None``
                         in that case)

    ``queries`` is (S, dim) — we treat each query token independently
    (the wrapping launcher in scorer.py adds the batch dim).

    ``centroids`` should be a CPU-resident float32 numpy array of shape
    ``(K, dim)`` when ``skip_qc_table`` is False. When ``skip_qc_table`` is
    True the caller is expected to compute qc_table itself on its preferred
    device (e.g. on GPU, which is much cheaper for large K). ``rotator`` is
    an optional pre-built ``FastWalshHadamard`` instance; when omitted we
    hit the process-wide ``(dim, seed)`` cache so we don't pay the
    construction cost (~0.5 ms) on every query.
    """
    if queries.ndim != 2:
        raise ValueError(f"expected (S, dim) queries, got {queries.shape}")
    s, dim = queries.shape
    if s == 0:
        raise ValueError("encode_query_for_rroq158: zero query tokens")
    if dim % 32 != 0:
        raise ValueError(
            f"encode_query_for_rroq158: dim={dim} must be a multiple of 32 "
            "to match the index-side popcount packing"
        )
    if rotator is None:
        rotator = get_cached_fwht_rotator(dim=dim, seed=fwht_seed)

    queries_f32 = queries.astype(np.float32, copy=False)

    # OpenBLAS / MKL spawn ``cpu_count`` worker threads on every GEMM
    # (default 64 on a 128-core box). The two query-side GEMMs here
    # (``q @ centroids.T`` and ``q @ dense``) are tiny — a 32x128 row
    # against an 8192x128 centroid matrix is ~32M ops, ~0.2 ms with the
    # full pool but ~1.2 ms single-threaded. A 1 ms loss is fine; the
    # win is that we don't leave 64 OpenBLAS workers spinning on cores
    # the immediately-following ``rroq158_score_batch`` rayon pool is
    # about to use. In the rroq158 CPU audit (2026-04-19) the kernel
    # p50 dropped from 97.7 ms to 6.9 ms once the BLAS pool stopped
    # fighting rayon for the same 16 cores. ``threadpoolctl`` is
    # already a transitive dependency via scikit-learn / numpy, so we
    # try to import it and fall back gracefully.
    try:
        from threadpoolctl import threadpool_limits

        _blas_cap = threadpool_limits(limits=1, user_api="blas")
    except ImportError:  # pragma: no cover — best-effort
        from contextlib import nullcontext

        _blas_cap = nullcontext()

    with _blas_cap:
        if skip_qc_table:
            qc_table = None
        else:
            if centroids is None:
                raise ValueError("centroids required when skip_qc_table is False")
            qc_table = (queries_f32 @ centroids.T).astype(np.float32, copy=False)

        dense = getattr(rotator, "_dense_matrix_np", None)
        if dense is not None and dense.shape == (dim, dim):
            # Fast path: precomputed (dim, dim) linear operator. Single
            # numpy GEMM, ~50 µs for S=32, dim=128 vs the 7-stage
            # PyTorch dispatch path which has a 30+ ms p95 tail
            # (rroq158 wrapper benchmarking 2026-04-19).
            q_rot = (queries_f32 @ dense).astype(np.float32, copy=False)
        else:
            import torch
            with torch.no_grad():
                q_rot = rotator.forward(torch.from_numpy(queries_f32)).cpu().numpy()
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
        if bit.shape[1] % 32 != 0:
            pad = 32 - bit.shape[1] % 32
            bit = np.pad(bit, ((0, 0), (0, pad)))
        packed = np.packbits(bit, axis=1, bitorder="little").view(np.int32)
        planes[:, k, :packed.shape[1]] = packed
    meta = np.stack([scale, min_v], axis=1).astype(np.float32)
    out = {"q_planes": planes, "q_meta": meta}
    if qc_table is not None:
        out["qc_table"] = qc_table
    return out


def pack_doc_codes_to_int32_words(sign_planes_u8: np.ndarray) -> np.ndarray:
    """Reinterpret packed-uint8 sign/nonzero planes as int32 words for the
    Triton kernel. The kernel uses 32-bit popcount, so we feed it the
    same bytes viewed as int32. ``sign_planes_u8`` shape is
    ``(n_tok, dim/8)`` uint8; output is ``(n_tok, dim/32)`` int32.

    ``dim`` is required by the encoder to be a multiple of 32 already
    (``Rroq158Config.group_size`` floor + ``dim % group_size == 0``), so a
    misshaped input here means an upstream contract was violated. We raise
    rather than assert because Python ``-O`` strips asserts and silently
    miscompiled int32 reinterpretations would be impossible to debug.
    """
    if sign_planes_u8.dtype != np.uint8:
        raise TypeError(
            f"pack_doc_codes_to_int32_words expects uint8 planes, got "
            f"dtype={sign_planes_u8.dtype}"
        )
    if sign_planes_u8.ndim != 2:
        raise ValueError(
            f"pack_doc_codes_to_int32_words expects a 2-D (n_tok, dim/8) "
            f"plane, got shape {sign_planes_u8.shape}"
        )
    _n, nb = sign_planes_u8.shape
    if nb % 4 != 0:
        raise ValueError(
            f"pack_doc_codes_to_int32_words received nb={nb} bytes "
            f"(dim={nb*8} bits); expected a multiple of 4 bytes (dim must "
            "be a multiple of 32). This indicates a misconfigured "
            "Rroq158Config or upstream encoder bug."
        )
    if not sign_planes_u8.flags["C_CONTIGUOUS"]:
        sign_planes_u8 = np.ascontiguousarray(sign_planes_u8)
    return sign_planes_u8.view(np.int32).copy()


__all__ = [
    "Rroq158Config",
    "Rroq158Encoded",
    "encode_rroq158",
    "encode_query_for_rroq158",
    "pack_doc_codes_to_int32_words",
    "get_cached_fwht_rotator",
    "clear_fwht_rotator_cache",
]
