"""Shared imports and configuration for shard-engine manager internals."""
from __future__ import annotations

import gc
import json
import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

try:
    from voyager_index._internal.inference.index_core.io_utils import (
        FileLock as _FileLock,
    )
    from voyager_index._internal.inference.index_core.io_utils import (
        atomic_json_write as _atomic_json_write,
    )
except ImportError:
    _FileLock = None
    _atomic_json_write = None

from ..checkpoint import ShardCheckpointManager
from ..colbandit_reranker import ColBanditReranker
from ..config import (
    AnnBackend,
    BuildConfig,
    Compression,
    LemurConfig,
    RouterType,
    SearchConfig,
    StorageLayout,
    TransferMode,
)
from ..fetch_pipeline import FetchPipeline, PinnedBufferPool
from ..lemur_router import CandidatePlan, LemurRouter
from ..memtable import MemTable
from ..profiler import Timer
from ..scorer import (
    PreloadedGpuCorpus,
    proxy_score_candidates,
    score_all_docs_topk,
    score_roq4_topk,
    warmup_maxsim,
)
from ..shard_store import ShardStore
from ..wal import WalOp, WalReader, WalWriter

logger = logging.getLogger(__name__)

def atomic_json_write(path: Path, data: Any) -> None:
    """Atomic JSON write with graceful fallback to plain json.dump."""
    if _atomic_json_write is not None:
        _atomic_json_write(path, data)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

class ShardEngineConfig:
    """Production configuration for the shard engine.

    Wraps BuildConfig + SearchConfig with sensible defaults
    for the LEMUR-routed path.
    """

    def __init__(
        self,
        *,
        n_shards: int = 256,
        dim: int = 128,
        compression: Compression = Compression.RROQ158,
        layout: StorageLayout = StorageLayout.PROXY_GROUPED,
        router_type: RouterType = RouterType.LEMUR,
        ann_backend: AnnBackend = AnnBackend.FAISS_FLAT_IP,
        lemur_epochs: int = 10,
        k_candidates: int = 2000,
        max_docs_exact: int = 10_000,
        lemur_search_k_cap: int | None = 2048,
        n_full_scores: int = 4096,
        transfer_mode: TransferMode = TransferMode.PINNED,
        pinned_pool_buffers: int = 3,
        pinned_buffer_max_tokens: int = 50_000,
        use_colbandit: bool = False,
        uniform_shard_tokens: bool = True,
        quantization_mode: str = "",
        variable_length_strategy: str = "bucketed",
        gpu_corpus_rerank_topn: int = 16,
        n_centroids: int = 1024,
        n_centroid_approx: int = 0,
        router_device: str | None = "cpu",
        seed: int = 42,
        rroq158_k: int = 8192,
        rroq158_seed: int = 42,
        rroq158_group_size: int = 32,
        rroq4_riem_k: int = 8192,
        rroq4_riem_seed: int = 42,
        rroq4_riem_group_size: int = 32,
    ):
        # Validate mode-specific knobs upfront so misconfiguration surfaces
        # at config-construction time (in the user's stack frame) instead of
        # midway through a multi-minute index build.
        if dim <= 0:
            raise ValueError(f"ShardEngineConfig.dim must be > 0, got {dim}")
        if n_shards <= 0:
            raise ValueError(
                f"ShardEngineConfig.n_shards must be > 0, got {n_shards}"
            )
        if rroq158_k <= 0 or (rroq158_k & (rroq158_k - 1)) != 0:
            raise ValueError(
                f"rroq158_k must be a positive power of two, got {rroq158_k}"
            )
        if rroq158_group_size <= 0 or rroq158_group_size % 32 != 0:
            raise ValueError(
                f"rroq158_group_size must be a positive multiple of 32, "
                f"got {rroq158_group_size}"
            )
        # rroq4_riem has looser group_size constraints than rroq158 (the
        # 4-bit codes pack two coords per byte, so the rule is "positive
        # even integer" rather than "multiple of 32"). K still needs to be
        # a power of two for the spherical k-means structure.
        if rroq4_riem_k <= 0 or (rroq4_riem_k & (rroq4_riem_k - 1)) != 0:
            raise ValueError(
                f"rroq4_riem_k must be a positive power of two, got {rroq4_riem_k}"
            )
        if rroq4_riem_group_size <= 0 or rroq4_riem_group_size % 2 != 0:
            raise ValueError(
                f"rroq4_riem_group_size must be a positive even integer, "
                f"got {rroq4_riem_group_size}"
            )
        if rroq4_riem_k < rroq4_riem_group_size:
            raise ValueError(
                f"rroq4_riem_k ({rroq4_riem_k}) must be >= "
                f"rroq4_riem_group_size ({rroq4_riem_group_size})"
            )
        # Coerce string-typed enum args to their canonical enum values so the
        # public IndexBuilder (`with_shard(compression="rroq158")`) and the
        # HTTP service hydration path (manifest stores enum.value strings)
        # both work without surprising the user. Enums are tolerated as-is.
        if isinstance(compression, str):
            compression = Compression(compression)
        if isinstance(layout, str):
            layout = StorageLayout(layout)
        if isinstance(router_type, str):
            router_type = RouterType(router_type)
        if isinstance(ann_backend, str):
            ann_backend = AnnBackend(ann_backend)
        if isinstance(transfer_mode, str):
            transfer_mode = TransferMode(transfer_mode)
        self.n_shards = n_shards
        self.dim = dim
        self.compression = compression
        self.layout = layout
        self.router_type = router_type
        self.ann_backend = ann_backend
        self.lemur_epochs = lemur_epochs
        self.k_candidates = k_candidates
        self.max_docs_exact = max_docs_exact
        self.lemur_search_k_cap = lemur_search_k_cap
        self.n_full_scores = n_full_scores
        self.transfer_mode = transfer_mode
        self.pinned_pool_buffers = pinned_pool_buffers
        self.pinned_buffer_max_tokens = pinned_buffer_max_tokens
        self.use_colbandit = use_colbandit
        self.uniform_shard_tokens = uniform_shard_tokens
        self.quantization_mode = quantization_mode
        self.variable_length_strategy = variable_length_strategy
        self.gpu_corpus_rerank_topn = gpu_corpus_rerank_topn
        self.n_centroids = n_centroids
        self.n_centroid_approx = n_centroid_approx
        self.router_device = router_device
        self.seed = seed
        self.rroq158_k = int(rroq158_k)
        self.rroq158_seed = int(rroq158_seed)
        self.rroq158_group_size = int(rroq158_group_size)
        self.rroq4_riem_k = int(rroq4_riem_k)
        self.rroq4_riem_seed = int(rroq4_riem_seed)
        self.rroq4_riem_group_size = int(rroq4_riem_group_size)

    def to_build_config(self, corpus_size: int) -> BuildConfig:
        cfg = BuildConfig(
            corpus_size=corpus_size,
            n_shards=self.n_shards,
            dim=self.dim,
            compression=self.compression,
            layout=self.layout,
            router_type=self.router_type,
            uniform_shard_tokens=self.uniform_shard_tokens,
            seed=self.seed,
            rroq158_k=self.rroq158_k,
            rroq158_seed=self.rroq158_seed,
            rroq158_group_size=self.rroq158_group_size,
            rroq4_riem_k=self.rroq4_riem_k,
            rroq4_riem_seed=self.rroq4_riem_seed,
            rroq4_riem_group_size=self.rroq4_riem_group_size,
        )
        cfg.lemur = LemurConfig(
            enabled=self.router_type == RouterType.LEMUR,
            device=self.router_device or "cuda",
            ann_backend=self.ann_backend,
            epochs=self.lemur_epochs,
            k_candidates=self.k_candidates,
            search_k_cap=self.lemur_search_k_cap,
        )
        return cfg

    def to_search_config(self) -> SearchConfig:
        return SearchConfig(
            k_candidates=self.k_candidates,
            max_docs_exact=self.max_docs_exact,
            lemur_search_k_cap=self.lemur_search_k_cap,
            n_full_scores=self.n_full_scores,
            n_centroid_approx=self.n_centroid_approx,
            transfer_mode=self.transfer_mode,
            pinned_pool_buffers=self.pinned_pool_buffers,
            pinned_buffer_max_tokens=self.pinned_buffer_max_tokens,
            use_colbandit=self.use_colbandit,
            quantization_mode=self.quantization_mode,
            variable_length_strategy=self.variable_length_strategy,
            gpu_corpus_rerank_topn=self.gpu_corpus_rerank_topn,
        )

