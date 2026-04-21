"""Engine selection and manager construction for the public `Index` facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def _check_gem_available() -> bool:
    try:
        from colsearch._internal.inference.index_core.gem_manager import (
            GemNativeSegmentManager,
        )

        if GemNativeSegmentManager is None:
            return False
        from latence_gem_index import GemSegment, PyMutableGemSegment

        return GemSegment is not None and PyMutableGemSegment is not None
    except ImportError:
        return False


def _check_hnsw_available() -> bool:
    try:
        from colsearch._internal.inference.index_core.hnsw_manager import (
            HnswSegmentManager,
        )

        return HnswSegmentManager is not None
    except ImportError:
        return False


def _check_shard_available() -> bool:
    try:
        from colsearch._internal.inference.shard_engine.manager import (
            ShardSegmentManager,
        )

        return ShardSegmentManager is not None
    except ImportError:
        return False


def resolve_engine(engine: str, mode: Optional[str], kwargs: dict[str, Any]) -> str:
    resolved_engine = engine
    if mode in ("colbert", "colpali"):
        resolved_engine = "gem"
        kwargs.setdefault("enable_shortcuts", mode == "colpali")
    if engine == "auto":
        if _check_gem_available():
            resolved_engine = "gem"
        elif _check_shard_available():
            resolved_engine = "shard"
        else:
            resolved_engine = "hnsw"
    return resolved_engine


def create_index_manager(
    path: Path,
    dim: int,
    *,
    engine: str,
    mode: Optional[str],
    n_fine: int,
    n_coarse: int,
    max_degree: int,
    ef_construction: int,
    n_probes: int,
    enable_wal: bool,
    kwargs: dict[str, Any],
) -> tuple[str, Any]:
    mgr_kwargs = dict(kwargs)
    resolved_engine = resolve_engine(engine, mode, mgr_kwargs)

    if resolved_engine == "gem":
        from colsearch._internal.inference.index_core.gem_manager import (
            GemNativeSegmentManager,
        )

        mgr_kwargs["enable_wal"] = enable_wal
        manager = GemNativeSegmentManager(
            shard_path=str(path / "shard"),
            dim=dim,
            n_fine=n_fine,
            n_coarse=n_coarse,
            max_degree=max_degree,
            gem_ef_construction=ef_construction,
            n_probes=n_probes,
            **mgr_kwargs,
        )
        return resolved_engine, manager

    if resolved_engine == "shard":
        from colsearch._internal.inference.shard_engine.manager import (
            ShardEngineConfig,
            ShardSegmentManager,
        )

        shard_kwargs = {
            key: value
            for key, value in mgr_kwargs.items()
            if key
            in {
                "n_shards",
                "compression",
                "layout",
                "router_type",
                "ann_backend",
                "lemur_epochs",
                "k_candidates",
                "max_docs_exact",
                "lemur_search_k_cap",
                "n_full_scores",
                "transfer_mode",
                "pinned_pool_buffers",
                "pinned_buffer_max_tokens",
                "use_colbandit",
                "quantization_mode",
                "variable_length_strategy",
                "gpu_corpus_rerank_topn",
                "n_centroid_approx",
                "router_device",
                "uniform_shard_tokens",
                "seed",
            }
        }
        shard_config = mgr_kwargs.get("shard_config") or ShardEngineConfig(
            dim=dim,
            **shard_kwargs,
        )
        device = mgr_kwargs.get("device", "cuda")
        manager = ShardSegmentManager(
            path=path / "shard",
            config=shard_config,
            device=device,
        )
        return resolved_engine, manager

    if resolved_engine == "hnsw":
        from colsearch._internal.inference.index_core.hnsw_manager import (
            HnswSegmentManager,
        )

        manager = HnswSegmentManager(
            shard_path=str(path / "shard"),
            dim=dim,
            **mgr_kwargs,
        )
        return resolved_engine, manager

    raise ValueError(f"Unknown engine: {engine!r}. Use 'gem', 'hnsw', 'shard', or 'auto'.")


__all__ = [
    "create_index_manager",
    "resolve_engine",
]
