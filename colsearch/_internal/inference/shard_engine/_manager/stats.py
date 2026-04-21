"""Metrics and statistics helpers for the shard manager."""
from __future__ import annotations

from .common import *  # noqa: F401,F403
from ..capabilities import detect_runtime_capabilities

class ShardSegmentManagerStatsMixin:
    def set_metrics_hook(self, hook: Callable[[str, float], None]) -> None:
        """Register a callback for internal search metrics (e.g. Prometheus)."""
        self._metrics_hook = hook

    def _emit_metric(self, name: str, value: Any) -> None:
        if self._metrics_hook:
            try:
                self._metrics_hook(name, float(value))
            except Exception as exc:
                logger.debug("Metrics hook error: %s", exc)

    def get_statistics(self) -> Dict[str, Any]:
        n_memtable = self._memtable.size if self._memtable else 0
        n_tombstones = self._memtable.tombstone_count if self._memtable else 0
        n_live = self._compute_live_count()

        stats: Dict[str, Any] = {
            "engine": "shard",
            "path": str(self._path),
            "dim": self._dim,
            "is_built": self._is_built,
            "device": self._device,
            "gpu_corpus_loaded": self._gpu_corpus is not None,
            "total_vectors": n_live,
            "active": n_memtable,
            "n_live": n_live,
            "n_tombstones": n_tombstones,
        }
        if self._store and self._store.manifest:
            m = self._store.manifest
            stats.update(
                {
                    "num_docs": m.num_docs,
                    "sealed_docs": m.num_docs,
                    "num_shards": m.num_shards,
                    "total_tokens": m.total_tokens,
                    "compression": m.compression,
                    "avg_tokens_per_doc": m.avg_tokens_per_chunk,
                }
            )
        if self._router:
            stats["router_type"] = "lemur"
        if self._store:
            pc = self._store.page_cache_residency()
            if pc is not None:
                stats["page_cache"] = pc
        router_caps = self._router.capability_snapshot() if self._router else {}
        pipeline_caps = self._pipeline.capability_snapshot() if self._pipeline else {}
        stats["runtime_capabilities"] = {
            **detect_runtime_capabilities(),
            "router_backend": router_caps.get("backend_name"),
            "router_ann_backend": router_caps.get("ann_backend"),
            "pinned_staging_enabled": pipeline_caps.get("pinned_staging_enabled"),
            "pinned_staging_mode": pipeline_caps.get("pinned_staging_mode"),
            "native_exact_available": self._native_backend_available,
            "native_exact_ready": self._rust_index is not None,
            "native_exact_merged_ready": self._rust_merged_ready,
            "native_exact_reason": self._native_backend_reason,
            "last_exact_path": self._last_exact_path,
            "last_prune_path": self._last_prune_path,
        }
        return stats

