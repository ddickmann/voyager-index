"""
Per-query profiler with stage-level latency, memory, and transfer tracking.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class QueryProfile:
    routing_ms: float = 0.0
    prune_ms: float = 0.0
    fetch_ms: float = 0.0
    exact_ms: float = 0.0
    h2d_ms: float = 0.0
    maxsim_ms: float = 0.0
    topk_ms: float = 0.0
    total_ms: float = 0.0

    h2d_bytes: int = 0
    num_shards_fetched: int = 0
    num_docs_scored: int = 0
    num_tokens_scored: int = 0

    retrieved_ids: List[int] = field(default_factory=list)
    retrieved_scores: List[float] = field(default_factory=list)


def aggregate_profiles(profiles: List[QueryProfile]) -> dict:
    """Compute p50/p95/p99 over a list of query profiles."""
    if not profiles:
        return {}

    fields = [
        "routing_ms", "prune_ms", "fetch_ms", "exact_ms", "h2d_ms", "maxsim_ms",
        "topk_ms", "total_ms", "h2d_bytes",
        "num_shards_fetched", "num_docs_scored", "num_tokens_scored",
    ]
    agg = {}
    for f in fields:
        vals = np.array([getattr(p, f) for p in profiles], dtype=np.float64)
        agg[f"mean_{f}"] = float(np.mean(vals))
        agg[f"p50_{f}"] = float(np.percentile(vals, 50))
        agg[f"p95_{f}"] = float(np.percentile(vals, 95))
        agg[f"p99_{f}"] = float(np.percentile(vals, 99))

    return agg


class Timer:
    """Context manager for precise timing with optional CUDA sync."""

    def __init__(self, sync_cuda: bool = False):
        self._sync = sync_cuda
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        if self._sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self._sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def memory_snapshot() -> dict:
    """Capture current CPU and GPU memory usage."""
    snap: Dict[str, float] = {}

    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        snap["cpu_rss_gb"] = pages * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        snap["cpu_rss_gb"] = -1.0

    if torch.cuda.is_available():
        snap["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        snap["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
    else:
        snap["gpu_allocated_gb"] = 0.0
        snap["gpu_reserved_gb"] = 0.0

    return snap
