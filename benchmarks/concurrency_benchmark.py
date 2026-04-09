"""
Concurrency benchmark: measure GEM search throughput and tail latency
under parallel load with 1, 2, 4, 8 threads.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ── Configuration ──────────────────────────────────────────
N_DOCS = 100
DIM = 32
VECS_PER_DOC = 16
N_QUERIES = 200
K = 10
EF = 64
N_PROBES = 2
THREAD_COUNTS = [1, 2, 4, 8]

GEM_PARAMS = dict(
    n_fine=16, n_coarse=4, max_degree=8, ef_construction=64,
    max_kmeans_iter=10, ctop_r=2,
)


# ── Data generation ────────────────────────────────────────

def generate_corpus(
    n_docs: int, vecs_per_doc: int, dim: int, seed: int = 42,
) -> Tuple[np.ndarray, list, list]:
    rng = np.random.RandomState(seed)
    all_vectors = rng.randn(n_docs * vecs_per_doc, dim).astype(np.float32)
    doc_ids = list(range(n_docs))
    offsets = [(i * vecs_per_doc, (i + 1) * vecs_per_doc) for i in range(n_docs)]
    return all_vectors, doc_ids, offsets


def generate_queries(
    n_queries: int, vecs_per_query: int, dim: int, seed: int = 99,
) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    return [
        rng.randn(vecs_per_query, dim).astype(np.float32)
        for _ in range(n_queries)
    ]


# ── Benchmark core ─────────────────────────────────────────

def run_concurrent_searches(
    seg, queries: List[np.ndarray], n_threads: int,
) -> dict:
    """Submit all queries to a thread pool and collect per-query latencies."""
    latencies: List[float] = []

    def search_one(query: np.ndarray) -> float:
        t0 = time.perf_counter()
        seg.search(query, k=K, ef=EF, n_probes=N_PROBES)
        return time.perf_counter() - t0

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(search_one, q) for q in queries]
        for fut in as_completed(futures):
            latencies.append(fut.result())
    wall_elapsed = time.perf_counter() - wall_start

    latencies_ms = np.array(latencies) * 1000.0
    throughput = len(queries) / wall_elapsed

    return {
        "n_threads": n_threads,
        "n_queries": len(queries),
        "wall_time_s": round(wall_elapsed, 4),
        "throughput_qps": round(throughput, 2),
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 3),
        "mean_ms": round(float(latencies_ms.mean()), 3),
        "min_ms": round(float(latencies_ms.min()), 3),
        "max_ms": round(float(latencies_ms.max()), 3),
    }


# ── Output formatting ─────────────────────────────────────

def print_table(all_metrics: list):
    cols = [
        ("Threads",    "n_threads",      "{:>8d}",   8),
        ("QPS",        "throughput_qps",  "{:>10.1f}", 10),
        ("p50 (ms)",   "p50_ms",         "{:>10.3f}", 10),
        ("p95 (ms)",   "p95_ms",         "{:>10.3f}", 10),
        ("p99 (ms)",   "p99_ms",         "{:>10.3f}", 10),
        ("mean (ms)",  "mean_ms",        "{:>10.3f}", 10),
        ("max (ms)",   "max_ms",         "{:>10.3f}", 10),
        ("wall (s)",   "wall_time_s",    "{:>10.4f}", 10),
    ]

    header = "  ".join(f"{label:>{w}s}" for label, _, _, w in cols)
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        row = "  ".join(fmt.format(m[key]) for _, key, fmt, _ in cols)
        print(row)


# ── Main ───────────────────────────────────────────────────

def main():
    from latence_gem_index import GemSegment

    print("=" * 80)
    print("GEM Concurrency Benchmark — throughput & tail latency")
    print("=" * 80)
    print(f"  N_DOCS={N_DOCS}  DIM={DIM}  VECS_PER_DOC={VECS_PER_DOC}")
    print(f"  N_QUERIES={N_QUERIES}  K={K}  EF={EF}  N_PROBES={N_PROBES}")
    print(f"  THREAD_COUNTS={THREAD_COUNTS}")
    print(f"  GEM_PARAMS={GEM_PARAMS}")
    print()

    print("Generating corpus...", flush=True)
    all_vectors, doc_ids, offsets = generate_corpus(N_DOCS, VECS_PER_DOC, DIM)
    print(f"  Total vectors: {all_vectors.shape[0]:,}")

    print("Generating queries...", flush=True)
    queries = generate_queries(N_QUERIES, VECS_PER_DOC, DIM)
    print(f"  Queries: {len(queries)}")
    print()

    print("Building GemSegment...", flush=True)
    seg = GemSegment()
    t0 = time.perf_counter()
    seg.build(all_vectors, doc_ids, offsets, **GEM_PARAMS)
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"  Build time: {build_ms:.1f} ms")
    print()

    # Warmup: single-threaded pass to stabilize caches
    print("Warmup (single-threaded, 10 queries)...", flush=True)
    for q in queries[:10]:
        seg.search(q, k=K, ef=EF, n_probes=N_PROBES)
    print()

    all_metrics = []
    for n_threads in THREAD_COUNTS:
        print(f"Running with {n_threads} thread(s)...", flush=True)
        metrics = run_concurrent_searches(seg, queries, n_threads)
        all_metrics.append(metrics)
        print(f"  QPS={metrics['throughput_qps']:.1f}  "
              f"p50={metrics['p50_ms']:.3f}ms  "
              f"p95={metrics['p95_ms']:.3f}ms  "
              f"p99={metrics['p99_ms']:.3f}ms")

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print_table(all_metrics)

    # Scaling summary
    if len(all_metrics) >= 2:
        baseline_qps = all_metrics[0]["throughput_qps"]
        print()
        print("Scaling vs single-thread:")
        for m in all_metrics:
            speedup = m["throughput_qps"] / max(baseline_qps, 0.01)
            print(f"  {m['n_threads']}T: {speedup:.2f}x throughput")

    # Save results
    results = {
        "config": {
            "n_docs": N_DOCS,
            "dim": DIM,
            "vecs_per_doc": VECS_PER_DOC,
            "n_queries": N_QUERIES,
            "k": K,
            "ef": EF,
            "n_probes": N_PROBES,
            "gem_params": GEM_PARAMS,
        },
        "thread_results": all_metrics,
    }

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "concurrency.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
