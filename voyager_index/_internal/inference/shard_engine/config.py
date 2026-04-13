"""Configuration dataclasses for the shard engine and benchmark sweep."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class Compression(str, Enum):
    FP16 = "fp16"
    INT8 = "int8"
    ROQ4 = "roq4"


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
    FAISS_HNSW_IP = "faiss_flat_ip"  # alias (was misnamed; actual impl is flat)
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
    nprobe: int = 10
    retrain_every_ops: int = 50_000
    retrain_every_hours: int = 24
    retrain_dirty_doc_ratio: float = 0.05
    retrain_dirty_shard_ratio: float = 0.10


@dataclass
class BuildConfig:
    """Build-time configuration for benchmarking and offline index construction.

    Defaults here are tuned for benchmark sweeps (centroid-grouped layout,
    centroid router).  For production serving, use ``ShardEngineConfig``
    which defaults to LEMUR routing and proxy-grouped layout.
    """

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
    """Runtime search parameters.

    ``batch_size`` is reserved for future batched query processing and is
    not currently used by the shard search path.
    """

    max_docs_exact: int = 10_000
    n_full_scores: int = 4096
    transfer_mode: TransferMode = TransferMode.PINNED
    pinned_pool_buffers: int = 3
    pinned_buffer_max_tokens: int = 50_000
    batch_size: int = 1
    k_candidates: int = 2000
    use_colbandit: bool = False
    quantization_mode: str = ""


@dataclass
class BenchmarkConfig:
    build: BuildConfig = field(default_factory=BuildConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    n_eval_queries: int = 300
    n_stress_queries: int = 1_000
    top_k: int = 10
    top_k_recall: int = 100
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "shard-bench")
    results_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "results")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cache_dir"] = str(self.cache_dir)
        d["results_dir"] = str(self.results_dir)
        return d


SWEEP_CORPUS_SIZES = [100_000, 300_000, 1_000_000]
SWEEP_TOP_SHARDS = [4, 8, 16, 32]
SWEEP_MAX_DOCS_EXACT = [1_000, 5_000, 10_000, 25_000]
SWEEP_K_CANDIDATES = [100, 200, 300, 500, 750, 1000, 1500, 2000]
SWEEP_COMPRESSION = [Compression.FP16, Compression.INT8]
SWEEP_LAYOUT = [StorageLayout.RANDOM, StorageLayout.CENTROID_GROUPED, StorageLayout.PROXY_GROUPED]
SWEEP_TRANSFER = [TransferMode.PAGEABLE, TransferMode.PINNED, TransferMode.DOUBLE_BUFFERED]
