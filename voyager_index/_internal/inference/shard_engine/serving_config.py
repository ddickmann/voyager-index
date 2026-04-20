"""Configuration dataclasses for shard-engine build and serving paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Compression(str, Enum):
    FP16 = "fp16"
    INT8 = "int8"
    ROQ4 = "roq4"
    RROQ158 = "rroq158"
    """Riemannian-aware 1.58-bit (ternary) ROQ. ~5.5x smaller than fp16,
    ~30% smaller than ROQ4, and matches fp16 R@10 on offline brute-force
    nfcorpus (0.345 vs 0.345). See research/low_bit_roq/PROGRESS.md."""

    RROQ4_RIEM = "rroq4_riem"
    """Riemannian-aware 4-bit asymmetric ROQ — the no-quality-loss lane.
    Same Riemannian + FWHT-rotated structure as RROQ158, but the residual
    is 4-bit asymmetric per-group instead of 1.58-bit ternary. ~3x smaller
    than fp16 and within ≈ 0.5% NDCG@10 of fp16 on production BEIR — the
    pick for callers who cannot accept any quality loss while still
    wanting smaller indexes than fp16.

    Currently slower than fp16 in absolute latency on the BEIR sweep
    (~2-3x on GPU, ~10x on CPU on small / short-query corpora — see the
    BEIR 2026-Q2 sweep verdict in ``reports/beir_2026q2/``); the win is
    storage, not throughput. Use RROQ158 when latency parity with fp16
    matters more than zero-degradation quality."""


class StorageLayout(str, Enum):
    RANDOM = "random"
    CENTROID_GROUPED = "centroid_grouped"
    TOKEN_BALANCED = "token_balanced"
    PROXY_GROUPED = "proxy_grouped"


class TransferMode(str, Enum):
    PAGEABLE = "pageable"
    PINNED = "pinned"
    DOUBLE_BUFFERED = "double_buffered"


class RouterType(str, Enum):
    CENTROID = "centroid"
    LEMUR = "lemur"


class AnnBackend(str, Enum):
    FAISS_FLAT_IP = "faiss_flat_ip"
    FAISS_HNSW_IP = "faiss_flat_ip"  # Legacy alias preserved for compatibility.
    FAISS_IVFPQ_IP = "faiss_ivfpq_ip"
    TORCH_EXACT_IP = "torch_exact_ip"


@dataclass
class PoolingConfig:
    enabled: bool = False
    method: str = "hierarchical"
    pool_factor: int = 2
    protected_tokens: int = 0


@dataclass
class LemurConfig:
    enabled: bool = False
    device: str = "cuda"
    ann_backend: AnnBackend = AnnBackend.FAISS_FLAT_IP
    epochs: int = 10
    k_candidates: int = 2000
    search_k_cap: int | None = 2048
    nprobe: int = 10
    retrain_every_ops: int = 50_000
    retrain_every_hours: int = 24
    retrain_dirty_doc_ratio: float = 0.05
    retrain_dirty_shard_ratio: float = 0.10


@dataclass
class BuildConfig:
    """Build-time configuration for offline shard index construction.

    The default codec is :class:`Compression.RROQ158` (Riemannian-aware
    1.58-bit ternary ROQ at K=8192). On the BEIR 2026-Q2 production
    sweep it averages **−1.43 pt NDCG@10** vs fp16 with **−0.48 pt
    R@100** at **~5.5x smaller doc-token storage**, and is at
    **GPU p95 parity (1.13× avg)**. Documented quality cost: up to
    −2.69 pt NDCG@10 on hard datasets (worst: arguana). The
    brute-force top-K codec-fidelity overlap with fp16 is ~82% on
    arguana — relevant docs are still recovered (R@100 within −1.4 pt
    on the same brute-force path), but ~18% of top-10 positions are
    displaced relative to fp16. See ``benchmarks/topk_overlap_sweep.py``.
    Workloads requiring exact top-10 rank fidelity should opt into
    ``Compression.RROQ4_RIEM`` or use rroq158 with an FP16 rerank on
    the shortlist (``benchmarks/diag_rroq158_rescue.py``).

    Set ``compression=Compression.RROQ4_RIEM`` to opt into the
    no-quality-loss lane: avg ΔNDCG@10 = +0.02 pt vs fp16 (max ±0.05 pt
    across BEIR-6), avg ΔR@100 = +0.23 pt, ~3x smaller than fp16. Note:
    currently slower than fp16 in absolute latency on the BEIR sweep
    (~5.0x on GPU, ~12.7x on CPU at the production batch shape); the
    win is **storage with zero quality regression**, not throughput.

    Set ``compression=Compression.FP16`` to use the legacy
    full-precision lane (no compression, baseline latency).

    Existing indexes on disk continue to load regardless of this
    default — the manifest carries the build-time codec; this default
    only affects newly built indexes.
    """

    corpus_size: int = 100_000
    n_centroids: int = 1024
    n_shards: int = 256
    dim: int = 128
    compression: Compression = Compression.RROQ158
    layout: StorageLayout = StorageLayout.CENTROID_GROUPED
    kmeans_sample_fraction: float = 0.1
    max_kmeans_iter: int = 50
    seed: int = 42
    uniform_shard_tokens: bool = True
    router_type: RouterType = RouterType.CENTROID
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    lemur: LemurConfig = field(default_factory=LemurConfig)
    rroq158_k: int = 8192
    """Centroid codebook size for RROQ158. Must be a power of two and ``>=
    256``. K=8192 is the default — closes the K=1024 quality gap on hard
    BEIR datasets at ~2 MB centroid table cost."""

    rroq158_seed: int = 42
    """Seed for the FWHT rotation and spherical k-means init."""

    rroq158_group_size: int = 32
    """Ternary group size in coordinates. Must divide ``dim`` and be a
    multiple of 32 (one popcount word per group)."""

    rroq4_riem_k: int = 8192
    """Centroid codebook size for RROQ4_RIEM. Same shape rules as
    ``rroq158_k`` (power of two, ``>= rroq4_riem_group_size``)."""

    rroq4_riem_seed: int = 42
    """Seed for the FWHT rotation and spherical k-means init for RROQ4_RIEM."""

    rroq4_riem_group_size: int = 32
    """4-bit asymmetric group size in coordinates. Must divide ``dim`` and
    be even so the nibble plane packs into bytes."""


@dataclass
class SearchConfig:
    """Runtime shard search parameters.

    ``batch_size`` is reserved for future batched query processing and is
    not currently used by the shard search path.
    """

    max_docs_exact: int = 10_000
    n_full_scores: int = 4096
    n_centroid_approx: int = 0
    residual_nbits: int = 0
    transfer_mode: TransferMode = TransferMode.PINNED
    pinned_pool_buffers: int = 3
    pinned_buffer_max_tokens: int = 50_000
    batch_size: int = 1  # Compatibility placeholder for future batched search paths.
    k_candidates: int = 2000
    lemur_search_k_cap: int | None = 2048
    use_colbandit: bool = False
    quantization_mode: str = ""
    variable_length_strategy: str = "bucketed"
    gpu_corpus_rerank_topn: int = 16
    distill_rerank: bool = False
    """Apply the MV-distill rerank head (numpy MLP, ~1.2k params) on top of
    the rroq158 top-K shortlist. Disabled by default — see PROGRESS.md
    [2026-04-19]; current version regresses real BEIR qrels even though it
    helps offline NN50*. Opt-in for experimentation only."""


__all__ = [
    "AnnBackend",
    "BuildConfig",
    "Compression",
    "LemurConfig",
    "PoolingConfig",
    "RouterType",
    "SearchConfig",
    "StorageLayout",
    "TransferMode",
]
