"""
Focused benchmark: GEM vs HNSW at 1024 token sequence length.

Measures index build time and search latency (p50/p95/p99) for both
backends with a statistically significant sample.
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

# ── Configuration ──────────────────────────────────────────
DIM = 128
SEQ_LEN = 1024
N_DOCS = 100
N_QUERIES = 100
K = 10

GEM_PARAMS = dict(n_fine=64, n_coarse=8, max_degree=16, ef_construction=100, ctop_r=3)
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100


def generate_data(rng, n_docs, seq_len, dim):
    docs = []
    ids = list(range(n_docs))
    for _ in range(n_docs):
        docs.append(rng.randn(seq_len, dim).astype(np.float32))
    return docs, ids


def percentile(data, p):
    data_sorted = sorted(data)
    idx = int(len(data_sorted) * p / 100)
    return data_sorted[min(idx, len(data_sorted) - 1)]


def benchmark_gem(docs, doc_ids, queries, dim):
    from latence_gem_index import GemSegment

    all_vecs = np.vstack(docs)
    offsets = []
    pos = 0
    for d in docs:
        n = d.shape[0]
        offsets.append((pos, pos + n))
        pos += n

    seg = GemSegment()
    t0 = time.perf_counter()
    seg.build(all_vecs, doc_ids, offsets, **GEM_PARAMS)
    build_ms = (time.perf_counter() - t0) * 1000

    for i in range(min(5, len(queries))):
        seg.search(queries[i], k=K, ef=64, n_probes=4)

    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        seg.search(q, k=K, ef=64, n_probes=4)
        latencies.append((time.perf_counter() - t0) * 1e6)

    return {
        "build_ms": round(build_ms, 1),
        "n_docs": seg.n_docs(),
        "n_edges": seg.n_edges(),
        "search_p50_us": round(percentile(latencies, 50), 1),
        "search_p95_us": round(percentile(latencies, 95), 1),
        "search_p99_us": round(percentile(latencies, 99), 1),
        "search_mean_us": round(statistics.mean(latencies), 1),
    }


def benchmark_hnsw(docs, doc_ids, queries, dim, n_query_tokens=32):
    """HNSW benchmark with per-token search + MaxSim aggregation.

    Since HNSW is a single-vector index, multi-vector search requires
    searching with each query token independently and aggregating via
    MaxSim (max over tokens of best per-doc similarity). We use a
    representative subset of query tokens to keep it tractable.
    """
    from latence_hnsw import HnswSegment

    with tempfile.TemporaryDirectory() as td:
        seg = HnswSegment(
            path=td, dim=dim, distance_metric="cosine",
            m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT,
            on_disk=False, is_appendable=True,
            multivector_comparator="max_sim",
        )

        t0 = time.perf_counter()
        seg.add_multidense(docs, ids=doc_ids)
        build_ms = (time.perf_counter() - t0) * 1000

        # Warmup with per-token search
        for i in range(min(3, len(queries))):
            q = queries[i][:n_query_tokens]
            for token in q:
                seg.search(token.astype(np.float32), K * 5)

        latencies = []
        for q_full in queries:
            q = q_full[:n_query_tokens]
            t0 = time.perf_counter()
            doc_scores: dict = {}
            for token in q:
                results = seg.search(token.astype(np.float32), K * 5)
                for r_id, r_score in results:
                    if r_id not in doc_scores:
                        doc_scores[r_id] = 0.0
                    doc_scores[r_id] = max(doc_scores[r_id], r_score)
            latencies.append((time.perf_counter() - t0) * 1e6)

        return {
            "build_ms": round(build_ms, 1),
            "n_items": seg.len(),
            "n_query_tokens_used": n_query_tokens,
            "note": f"per-token search x {n_query_tokens} tokens + MaxSim aggregation",
            "search_p50_us": round(percentile(latencies, 50), 1),
            "search_p95_us": round(percentile(latencies, 95), 1),
            "search_p99_us": round(percentile(latencies, 99), 1),
            "search_mean_us": round(statistics.mean(latencies), 1),
        }


def main():
    print("=" * 70)
    print("GEM vs HNSW Benchmark — Focused @ 1024 tokens")
    print("=" * 70)
    print(f"  Documents: {N_DOCS}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Dimension: {DIM}")
    print(f"  Queries: {N_QUERIES}")
    print(f"  k={K}")
    print()

    rng = np.random.RandomState(42)
    print("Generating data...", flush=True)
    docs, doc_ids = generate_data(rng, N_DOCS, SEQ_LEN, DIM)
    queries = [rng.randn(SEQ_LEN, DIM).astype(np.float32) for _ in range(N_QUERIES)]
    print(f"  Total vectors: {N_DOCS * SEQ_LEN:,}")
    print()

    print("Benchmarking GEM...", flush=True)
    gem_results = benchmark_gem(docs, doc_ids, queries, DIM)
    print(f"  Build: {gem_results['build_ms']:,.0f} ms")
    print(f"  Search p50: {gem_results['search_p50_us']:,.0f} us")
    print(f"  Search p95: {gem_results['search_p95_us']:,.0f} us")
    print(f"  Search p99: {gem_results['search_p99_us']:,.0f} us")
    print()

    print("Benchmarking HNSW...", flush=True)
    hnsw_results = benchmark_hnsw(docs, doc_ids, queries, DIM)
    print(f"  Build: {hnsw_results['build_ms']:,.0f} ms")
    print(f"  Search p50: {hnsw_results['search_p50_us']:,.0f} us")
    print(f"  Search p95: {hnsw_results['search_p95_us']:,.0f} us")
    print(f"  Search p99: {hnsw_results['search_p99_us']:,.0f} us")
    print()

    # Batch search benchmark (GEM only — reuse segment from benchmark_gem)
    print("Benchmarking GEM batch search...", flush=True)
    try:
        from latence_gem_index import GemSegment as _GS

        seg_b = _GS()
        all_vecs_b = np.vstack(docs)
        offsets_b = []
        pos_b = 0
        for d in docs:
            n = d.shape[0]
            offsets_b.append((pos_b, pos_b + n))
            pos_b += n
        seg_b.build(all_vecs_b, doc_ids, offsets_b, **GEM_PARAMS)

        batch_size = min(16, N_QUERIES)
        batch_q = queries[:batch_size]

        seg_b.search_batch(batch_q, k=K, ef=64, n_probes=4)

        t0 = time.perf_counter()
        seg_b.search_batch(batch_q, k=K, ef=64, n_probes=4)
        batch_total_us = (time.perf_counter() - t0) * 1e6

        seq_total = 0.0
        for q in batch_q:
            t0 = time.perf_counter()
            seg_b.search(q, k=K, ef=64, n_probes=4)
            seq_total += (time.perf_counter() - t0) * 1e6

        gem_results["batch_total_us"] = round(batch_total_us, 1)
        gem_results["batch_per_query_us"] = round(batch_total_us / batch_size, 1)
        gem_results["sequential_total_us"] = round(seq_total, 1)
        gem_results["batch_speedup"] = round(seq_total / max(batch_total_us, 0.1), 2)
        print(f"  Batch ({batch_size} queries): {batch_total_us:,.0f} us total, {batch_total_us / batch_size:,.0f} us/query")
        print(f"  Sequential ({batch_size} queries): {seq_total:,.0f} us total")
        print(f"  Batch speedup: {seq_total / max(batch_total_us, 0.1):.1f}x")
        print()
    except Exception as e:
        print(f"  Batch benchmark skipped: {e}")
        print()

    sp50 = hnsw_results["search_p50_us"] / max(gem_results["search_p50_us"], 0.1)
    sp99 = hnsw_results["search_p99_us"] / max(gem_results["search_p99_us"], 0.1)

    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'GEM':>15} {'HNSW':>15} {'Speedup':>12}")
    print("-" * 70)
    print(f"{'Build (ms)':<25} {gem_results['build_ms']:>15,.0f} {hnsw_results['build_ms']:>15,.0f}")
    print(f"{'Search p50 (us)':<25} {gem_results['search_p50_us']:>15,.0f} {hnsw_results['search_p50_us']:>15,.0f} {sp50:>11.1f}x")
    print(f"{'Search p95 (us)':<25} {gem_results['search_p95_us']:>15,.0f} {hnsw_results['search_p95_us']:>15,.0f}")
    print(f"{'Search p99 (us)':<25} {gem_results['search_p99_us']:>15,.0f} {hnsw_results['search_p99_us']:>15,.0f} {sp99:>11.1f}x")
    print(f"{'Search mean (us)':<25} {gem_results['search_mean_us']:>15,.0f} {hnsw_results['search_mean_us']:>15,.0f}")
    print("=" * 70)

    if sp50 > 1.5:
        print(f"\nGEM is {sp50:.1f}x FASTER than HNSW at p50 search latency.")
    elif sp50 < 0.67:
        print(f"\nHNSW is {1/sp50:.1f}x faster than GEM at p50.")
    else:
        print(f"\nComparable performance (speedup: {sp50:.1f}x).")

    results = {
        "config": {"dim": DIM, "seq_len": SEQ_LEN, "n_docs": N_DOCS, "n_queries": N_QUERIES, "k": K},
        "gem": gem_results,
        "hnsw": hnsw_results,
        "speedup_p50": round(sp50, 2),
        "speedup_p99": round(sp99, 2),
    }

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gem_vs_hnsw_1024.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
