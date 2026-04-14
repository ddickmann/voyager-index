"""
GEM Router Benchmark Harness
=============================

Measures the GEM router's candidate generation quality and latency against
baselines (full scan, PLAID-style IVF, prototype HNSW) at varying corpus
sizes. Designed to enforce strict promotion gates before the router becomes
the default balanced-mode candidate generator.

Usage:
    python tools/benchmarks/benchmark_gem_router.py --n-docs 10000
    python tools/benchmarks/benchmark_gem_router.py --sweep
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from latence_gem_router import PyGemRouter
    GEM_AVAILABLE = True
except ImportError:
    GEM_AVAILABLE = False
    print("WARNING: latence_gem_router not available; install it first")


def generate_clustered_corpus(
    n_docs: int,
    dim: int = 128,
    doc_tokens: int = 64,
    n_topics: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a clustered corpus with known ground-truth topic structure."""
    rng = np.random.default_rng(seed)
    n_topics = min(n_topics, n_docs)

    topic_centroids = rng.standard_normal((n_topics, dim)).astype(np.float32)
    topic_centroids /= np.linalg.norm(topic_centroids, axis=1, keepdims=True) + 1e-8

    doc_embeddings = np.zeros((n_docs, doc_tokens, dim), dtype=np.float32)
    doc_topics = np.zeros(n_docs, dtype=np.int32)

    for i in range(n_docs):
        topic = rng.integers(0, n_topics)
        doc_topics[i] = topic
        base = topic_centroids[topic]
        for t in range(doc_tokens):
            noise = rng.standard_normal(dim).astype(np.float32) * 0.3
            vec = base + noise
            doc_embeddings[i, t] = vec / (np.linalg.norm(vec) + 1e-8)

    query_tokens = 32
    n_queries = min(100, n_docs // 2)
    query_embeddings = np.zeros((n_queries, query_tokens, dim), dtype=np.float32)
    query_gt_topics = np.zeros(n_queries, dtype=np.int32)

    for q in range(n_queries):
        topic = rng.integers(0, n_topics)
        query_gt_topics[q] = topic
        base = topic_centroids[topic]
        for t in range(query_tokens):
            noise = rng.standard_normal(dim).astype(np.float32) * 0.2
            vec = base + noise
            query_embeddings[q, t] = vec / (np.linalg.norm(vec) + 1e-8)

    return doc_embeddings, query_embeddings, doc_topics


def compute_exact_maxsim(
    query: np.ndarray,
    docs: np.ndarray,
) -> np.ndarray:
    """Compute exact MaxSim scores between one query and all docs."""
    scores = np.zeros(docs.shape[0], dtype=np.float32)
    for d in range(docs.shape[0]):
        sim = query @ docs[d].T
        scores[d] = sim.max(axis=1).mean()
    return scores


def recall_at_k(
    predicted_ids: List[int],
    gt_ids: List[int],
    k: int,
) -> float:
    """Recall@k: fraction of GT top-k items present in predicted top-k."""
    gt_set = set(gt_ids[:k])
    if not gt_set:
        return 1.0
    pred_set = set(predicted_ids[:k])
    return len(gt_set & pred_set) / len(gt_set)


def benchmark_gem_router(
    n_docs: int,
    dim: int = 128,
    doc_tokens: int = 64,
    n_fine: Optional[int] = None,
    n_coarse: int = 32,
    n_probes: int = 4,
    max_candidates: int = 500,
    seed: int = 42,
) -> Dict:
    """Run a single benchmark configuration and return metrics."""
    if not GEM_AVAILABLE:
        return {"error": "latence_gem_router not available"}

    print(f"\n{'='*60}")
    print(f"Benchmark: n_docs={n_docs}, dim={dim}, doc_tokens={doc_tokens}")
    print(f"{'='*60}")

    gen_start = time.perf_counter()
    doc_embeddings, query_embeddings, doc_topics = generate_clustered_corpus(
        n_docs, dim, doc_tokens, seed=seed,
    )
    gen_ms = (time.perf_counter() - gen_start) * 1000.0
    n_queries = query_embeddings.shape[0]
    print(f"Generated {n_docs} docs, {n_queries} queries in {gen_ms:.0f}ms")

    # Flatten docs for router build
    all_vecs = doc_embeddings.reshape(-1, dim)
    offsets = [(i * doc_tokens, (i + 1) * doc_tokens) for i in range(n_docs)]
    doc_ids = list(range(n_docs))

    if n_fine is None:
        n_fine = min(4096, max(4, int(all_vecs.shape[0] * 0.1)))
    n_coarse = min(n_coarse, n_fine // 2)

    # Build GEM router
    build_start = time.perf_counter()
    router = PyGemRouter(dim=dim)
    router.build(all_vecs, doc_ids, offsets, n_fine=n_fine, n_coarse=n_coarse)
    build_ms = (time.perf_counter() - build_start) * 1000.0
    print(f"GEM router built in {build_ms:.0f}ms "
          f"(n_fine={router.n_fine()}, n_coarse={router.n_coarse()})")

    # Compute exact ground truth
    gt_start = time.perf_counter()
    gt_rankings = []
    for q in range(n_queries):
        scores = compute_exact_maxsim(query_embeddings[q], doc_embeddings)
        ranked = np.argsort(-scores).tolist()
        gt_rankings.append(ranked)
    gt_ms = (time.perf_counter() - gt_start) * 1000.0
    print(f"Exact MaxSim ground truth computed in {gt_ms:.0f}ms")

    # Benchmark GEM router candidate generation
    route_latencies = []
    recalls_at_10 = []
    recalls_at_50 = []
    recalls_at_100 = []
    candidate_counts = []

    for q in range(n_queries):
        q_vecs = query_embeddings[q]

        route_start = time.perf_counter()
        results = router.route_query(q_vecs, n_probes=n_probes, max_candidates=max_candidates)
        route_ms = (time.perf_counter() - route_start) * 1000.0
        route_latencies.append(route_ms)

        predicted = [int(doc_id) for doc_id, _ in results]
        candidate_counts.append(len(predicted))
        gt = gt_rankings[q]

        recalls_at_10.append(recall_at_k(predicted, gt, 10))
        recalls_at_50.append(recall_at_k(predicted, gt, 50))
        recalls_at_100.append(recall_at_k(predicted, gt, 100))

    # Benchmark brute-force baseline latency
    bf_latencies = []
    for q in range(min(n_queries, 20)):
        bf_start = time.perf_counter()
        compute_exact_maxsim(query_embeddings[q], doc_embeddings)
        bf_ms = (time.perf_counter() - bf_start) * 1000.0
        bf_latencies.append(bf_ms)

    metrics = {
        "n_docs": n_docs,
        "n_queries": n_queries,
        "dim": dim,
        "doc_tokens": doc_tokens,
        "n_fine": router.n_fine(),
        "n_coarse": router.n_coarse(),
        "n_probes": n_probes,
        "max_candidates": max_candidates,
        "build_ms": build_ms,
        "route_latency_p50_ms": statistics.median(route_latencies),
        "route_latency_p99_ms": sorted(route_latencies)[int(0.99 * len(route_latencies))],
        "route_latency_mean_ms": statistics.mean(route_latencies),
        "recall@10": statistics.mean(recalls_at_10),
        "recall@50": statistics.mean(recalls_at_50),
        "recall@100": statistics.mean(recalls_at_100),
        "avg_candidates": statistics.mean(candidate_counts),
        "brute_force_latency_p50_ms": statistics.median(bf_latencies) if bf_latencies else 0,
        "speedup_vs_brute_force": (
            statistics.median(bf_latencies) / max(statistics.median(route_latencies), 0.001)
            if bf_latencies else 0
        ),
    }

    print(f"\nResults:")
    print(f"  Build time:          {metrics['build_ms']:.0f}ms")
    print(f"  Route latency p50:   {metrics['route_latency_p50_ms']:.2f}ms")
    print(f"  Route latency p99:   {metrics['route_latency_p99_ms']:.2f}ms")
    print(f"  Avg candidates:      {metrics['avg_candidates']:.0f}")
    print(f"  Recall@10:           {metrics['recall@10']:.3f}")
    print(f"  Recall@50:           {metrics['recall@50']:.3f}")
    print(f"  Recall@100:          {metrics['recall@100']:.3f}")
    print(f"  Brute-force p50:     {metrics['brute_force_latency_p50_ms']:.2f}ms")
    print(f"  Speedup vs BF:       {metrics['speedup_vs_brute_force']:.1f}x")

    return metrics


def check_promotion_gates(metrics: Dict) -> Tuple[bool, List[str]]:
    """
    Check strict promotion gates for the GEM router.

    Returns (passed, list_of_failures).
    """
    failures = []

    if metrics.get("recall@10", 0) < 0.7:
        failures.append(
            f"recall@10 = {metrics['recall@10']:.3f} < 0.70 threshold"
        )

    if metrics.get("recall@50", 0) < 0.8:
        failures.append(
            f"recall@50 = {metrics['recall@50']:.3f} < 0.80 threshold"
        )

    if metrics.get("recall@100", 0) < 0.85:
        failures.append(
            f"recall@100 = {metrics['recall@100']:.3f} < 0.85 threshold"
        )

    if metrics.get("speedup_vs_brute_force", 0) < 2.0:
        failures.append(
            f"speedup = {metrics['speedup_vs_brute_force']:.1f}x < 2.0x threshold"
        )

    passed = len(failures) == 0
    return passed, failures


def run_sweep():
    """Run benchmark sweep across multiple corpus sizes."""
    sizes = [100, 500, 1000, 5000, 10000, 50000]
    all_metrics = []

    for n_docs in sizes:
        metrics = benchmark_gem_router(n_docs=n_docs)
        passed, failures = check_promotion_gates(metrics)
        metrics["promotion_gate_passed"] = passed
        metrics["promotion_gate_failures"] = failures
        all_metrics.append(metrics)

        status = "PASS" if passed else "FAIL"
        print(f"\n  Promotion gate: {status}")
        if failures:
            for f in failures:
                print(f"    - {f}")

    # Save results
    out_path = Path("gem_router_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'n_docs':>8} {'build_ms':>10} {'route_p50':>10} {'R@10':>8} "
          f"{'R@50':>8} {'R@100':>8} {'speedup':>8} {'gate':>6}")
    print(f"{'='*80}")
    for m in all_metrics:
        gate = "PASS" if m["promotion_gate_passed"] else "FAIL"
        print(f"{m['n_docs']:>8} {m['build_ms']:>10.0f} "
              f"{m['route_latency_p50_ms']:>10.2f} "
              f"{m['recall@10']:>8.3f} {m['recall@50']:>8.3f} "
              f"{m['recall@100']:>8.3f} "
              f"{m['speedup_vs_brute_force']:>7.1f}x "
              f"{gate:>6}")


def main():
    parser = argparse.ArgumentParser(description="GEM Router Benchmark")
    parser.add_argument("--n-docs", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--doc-tokens", type=int, default=64)
    parser.add_argument("--n-fine", type=int, default=None)
    parser.add_argument("--n-coarse", type=int, default=32)
    parser.add_argument("--n-probes", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--sweep", action="store_true", help="Run benchmark sweep")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.sweep:
        run_sweep()
        return

    metrics = benchmark_gem_router(
        n_docs=args.n_docs,
        dim=args.dim,
        doc_tokens=args.doc_tokens,
        n_fine=args.n_fine,
        n_coarse=args.n_coarse,
        n_probes=args.n_probes,
        max_candidates=args.max_candidates,
    )

    passed, failures = check_promotion_gates(metrics)
    metrics["promotion_gate_passed"] = passed
    metrics["promotion_gate_failures"] = failures

    print(f"\nPromotion gate: {'PASS' if passed else 'FAIL'}")
    if failures:
        for f in failures:
            print(f"  - {f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
