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
    """Build-time configuration for offline shard index construction."""

    corpus_size: int = 100_000
    n_centroids: int = 1024
    n_shards: int = 256
    dim: int = 128
    compression: Compression = Compression.FP16
    layout: StorageLayout = StorageLayout.CENTROID_GROUPED
    kmeans_sample_fraction: float = 0.1
    max_kmeans_iter: int = 50
    seed: int = 42
    uniform_shard_tokens: bool = True
    router_type: RouterType = RouterType.CENTROID
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    lemur: LemurConfig = field(default_factory=LemurConfig)


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
