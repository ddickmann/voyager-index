"""Sweep configuration for shard-engine experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .serving_config import BuildConfig, Compression, SearchConfig, StorageLayout, TransferMode


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
        data = asdict(self)
        data["cache_dir"] = str(self.cache_dir)
        data["results_dir"] = str(self.results_dir)
        return data


SWEEP_CORPUS_SIZES = [100_000, 300_000, 1_000_000]
SWEEP_TOP_SHARDS = [4, 8, 16, 32]
SWEEP_MAX_DOCS_EXACT = [1_000, 5_000, 10_000, 25_000]
SWEEP_K_CANDIDATES = [100, 200, 300, 500, 750, 1000, 1500, 2000]
SWEEP_COMPRESSION = [Compression.RROQ158, Compression.FP16, Compression.INT8]
SWEEP_LAYOUT = [StorageLayout.RANDOM, StorageLayout.CENTROID_GROUPED, StorageLayout.PROXY_GROUPED]
SWEEP_TRANSFER = [TransferMode.PAGEABLE, TransferMode.PINNED, TransferMode.DOUBLE_BUFFERED]


__all__ = [
    "BenchmarkConfig",
    "SWEEP_COMPRESSION",
    "SWEEP_CORPUS_SIZES",
    "SWEEP_K_CANDIDATES",
    "SWEEP_LAYOUT",
    "SWEEP_MAX_DOCS_EXACT",
    "SWEEP_TOP_SHARDS",
    "SWEEP_TRANSFER",
]
