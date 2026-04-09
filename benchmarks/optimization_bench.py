#!/usr/bin/env python3
"""
Combined benchmark: validates that optimizations in the ROQ v2 kernel and
qEMD construction (f32 Sinkhorn, rayon, ScoreCache) preserve recall while
improving latency and build speed.

Usage:
    python benchmarks/optimization_bench.py [--n_docs 500] [--dim 128] [--tpd 32] [--n_queries 50]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: torch is required.")
    sys.exit(1)

try:
    from latence_gem_index import GemSegment
except ImportError:
    print("ERROR: latence_gem_index is required.")
    sys.exit(1)

try:
    from voyager_index._internal.inference.index_core.gem_manager import (
        GemNativeSegmentManager,
        _MAXSIM_AVAILABLE,
        _ROQ_AVAILABLE,
    )
except ImportError:
    print("ERROR: voyager_index gem_manager is required.")
    sys.exit(1)

try:
    from voyager_index._internal.kernels.kernel_warmup import warmup_triton_kernels
    _WARMUP_AVAILABLE = True
except ImportError:
    _WARMUP_AVAILABLE = False

HAS_CUDA = torch.cuda.is_available()


def _sync():
    if HAS_CUDA:
        torch.cuda.synchronize()


def generate_data(n_docs, dim, tpd, seed=42):
    rng = np.random.default_rng(seed)
    vectors, ids = [], list(range(1, n_docs + 1))
    for _ in range(n_docs):
        v = rng.standard_normal((tpd, dim)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        vectors.append(v)
    return vectors, ids


def generate_queries(n_queries, dim, n_tokens=8, seed=999):
    rng = np.random.default_rng(seed)
    return [
        (lambda q: q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8))(
            rng.standard_normal((n_tokens, dim)).astype(np.float32)
        )
        for _ in range(n_queries)
    ]


def brute_force_maxsim(queries, doc_vectors, doc_ids, k):
    results = []
    for q in queries:
        q_t = torch.from_numpy(q)
        scores = []
        for doc_id, doc_vecs in zip(doc_ids, doc_vectors):
            d_t = torch.from_numpy(doc_vecs)
            sim = (q_t @ d_t.T).max(dim=1).values.sum().item()
            scores.append((doc_id, sim))
        scores.sort(key=lambda x: -x[1])
        results.append([s[0] for s in scores[:k]])
    return results


def recall_at_k(predicted, ground_truth, k):
    recalls = []
    for pred, gt in zip(predicted, ground_truth):
        pred_set = set(pred[:k])
        gt_set = set(gt[:k])
        if gt_set:
            recalls.append(len(pred_set & gt_set) / len(gt_set))
    return np.mean(recalls) if recalls else 0.0


def build_gem_segment_raw(vectors, dim, tpd, *, use_emd=False, dual_graph=False):
    """Build a raw GemSegment to measure Rust-level build time."""
    n_docs = len(vectors)
    all_vecs = np.concatenate(vectors, axis=0).astype(np.float32)
    offsets = []
    pos = 0
    for v in vectors:
        offsets.append((pos, pos + v.shape[0]))
        pos += v.shape[0]
    n_fine = max(8, min(64, int(np.sqrt(pos))))
    n_coarse = max(4, n_fine // 4)

    seg = GemSegment()
    t0 = time.perf_counter()
    seg.build(
        all_vecs, list(range(n_docs)), offsets,
        n_coarse=n_coarse, n_fine=n_fine, max_degree=16,
        ef_construction=64, max_kmeans_iter=10, ctop_r=2,
        use_emd=use_emd, dual_graph=dual_graph,
    )
    build_s = time.perf_counter() - t0
    return seg, build_s


def measure_search_latency(mgr, queries, k, ef, n_probes, n_warmup=3):
    for i in range(min(n_warmup, len(queries))):
        mgr.search_multivector(queries[i], k=k, ef=ef, n_probes=n_probes)
    _sync()
    latencies = []
    for q in queries:
        _sync()
        t0 = time.perf_counter()
        mgr.search_multivector(q, k=k, ef=ef, n_probes=n_probes)
        _sync()
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def build_manager(tmp_dir, dim, vectors, ids, *, rerank_device=None, roq_rerank=False,
                  use_emd=False, dual_graph=False, warmup_kernels=True):
    n_docs = len(ids)
    n_fine = max(8, min(64, int(np.sqrt(n_docs))))
    n_coarse = max(4, n_fine // 4)
    mgr = GemNativeSegmentManager(
        str(tmp_dir), dim,
        n_fine=n_fine, n_coarse=n_coarse,
        max_degree=16, gem_ef_construction=64,
        max_kmeans_iter=10, ctop_r=2,
        seed_batch_size=n_docs,
        enable_wal=True,
        rerank_device=rerank_device,
        roq_rerank=roq_rerank,
        use_emd=use_emd,
        dual_graph=dual_graph,
        warmup_kernels=warmup_kernels,
    )
    mgr.add_multidense(vectors, ids)
    mgr.seal_active_segment()
    return mgr


def run_benchmark(args):
    device = "cuda" if HAS_CUDA else "cpu"
    print(f"\n{'='*70}")
    print(f"  GEM Optimization Benchmark")
    print(f"  Vectorized data-prep + shape bucketing + kernel warmup")
    print(f"{'='*70}")
    print(f"  Device:     {device}")
    print(f"  Docs:       {args.n_docs}")
    print(f"  Dim:        {args.dim}")
    print(f"  Tokens/doc: {args.tpd}")
    print(f"  Queries:    {args.n_queries}")
    print(f"  k:          {args.k}")
    print(f"  ef:         {args.ef}")
    print(f"  FP32 MaxSim: {_MAXSIM_AVAILABLE}")
    print(f"  ROQ Triton:  {_ROQ_AVAILABLE}")
    print(f"  Warmup:      {_WARMUP_AVAILABLE}")
    print(f"{'='*70}\n")

    print("[1/7] Generating synthetic data...")
    vectors, ids = generate_data(args.n_docs, args.dim, args.tpd)
    queries = generate_queries(args.n_queries, args.dim)

    print("[2/7] Computing brute-force ground truth...")
    gt = brute_force_maxsim(queries, vectors, ids, args.k)

    tmp_dirs = []
    try:
        # ----------------------------------------------------------------
        # Part A: Warmup Effect (run FIRST to get clean Triton state)
        # ----------------------------------------------------------------
        print(f"\n  {'='*66}")
        print(f"  Part A: Warmup Effect (first-query latency)")
        print(f"  {'='*66}")

        if device == "cuda" and _WARMUP_AVAILABLE:
            warmup_configs = []
            if _MAXSIM_AVAILABLE:
                warmup_configs.append(("FP32 MaxSim", dict(rerank_device=device)))
            if _ROQ_AVAILABLE:
                warmup_configs.append(("ROQ 4-bit", dict(roq_rerank=True)))

            for label, kwargs in warmup_configs:
                td = tempfile.mkdtemp()
                tmp_dirs.append(td)
                print(f"\n  Building {label} (warmup_kernels=True)...")
                t_build = time.perf_counter()
                mgr_warm = build_manager(
                    Path(td), args.dim, vectors, ids,
                    warmup_kernels=True, **kwargs,
                )
                t_build = time.perf_counter() - t_build

                _sync()
                t0 = time.perf_counter()
                mgr_warm.search_multivector(queries[0], k=args.k, ef=args.ef, n_probes=4)
                _sync()
                first_ms = (time.perf_counter() - t0) * 1000

                steady_lats = []
                for q in queries[1:]:
                    _sync()
                    t0 = time.perf_counter()
                    mgr_warm.search_multivector(q, k=args.k, ef=args.ef, n_probes=4)
                    _sync()
                    steady_lats.append((time.perf_counter() - t0) * 1000)

                p50 = np.percentile(steady_lats, 50)
                p95 = np.percentile(steady_lats, 95)
                print(f"    init:       {t_build:.2f}s")
                print(f"    1st query:  {first_ms:.2f}ms")
                print(f"    steady p50: {p50:.2f}ms  p95: {p95:.2f}ms")
                print(f"    1st/p50:    {first_ms/max(p50, 0.001):.1f}x")
                mgr_warm.close()
        else:
            print("  (skipped — CUDA or warmup not available)")

        # ----------------------------------------------------------------
        # Part B: qEMD vs qCH Build Speed
        # ----------------------------------------------------------------
        print(f"\n  {'='*66}")
        print(f"  Part B: Build Speed")
        print(f"  {'='*66}")

        print("\n  [qCH] Measuring build time...")
        _, build_ch = build_gem_segment_raw(vectors, args.dim, args.tpd, use_emd=False, dual_graph=True)
        print(f"    qCH build: {build_ch:.3f}s")

        print("  [qEMD] Measuring build time (f32 Sinkhorn + rayon + cache)...")
        _, build_emd = build_gem_segment_raw(vectors, args.dim, args.tpd, use_emd=True, dual_graph=True)
        print(f"    qEMD build: {build_emd:.3f}s  (ratio: {build_emd/max(build_ch, 0.001):.2f}x)")

        # ----------------------------------------------------------------
        # Part C: Search Quality + Latency
        # ----------------------------------------------------------------
        print(f"\n  {'='*66}")
        print(f"  Part C: Search Quality + Latency")
        print(f"  {'='*66}")

        configs = [
            ("Proxy only (qCH)", dict()),
            ("FP32 MaxSim", dict(rerank_device=device)),
        ]
        if _ROQ_AVAILABLE:
            configs.append(("ROQ 4-bit", dict(roq_rerank=True)))

        managers = {}
        search_results = {}

        for label, kwargs in configs:
            td = tempfile.mkdtemp()
            tmp_dirs.append(td)
            print(f"\n  Building: {label}...")
            t0 = time.perf_counter()
            mgr = build_manager(Path(td), args.dim, vectors, ids, **kwargs)
            bt = time.perf_counter() - t0
            managers[label] = mgr
            print(f"    Built in {bt:.2f}s")

        print("\n  Running searches...")
        for label, mgr in managers.items():
            results = []
            for q in queries:
                r = mgr.search_multivector(q, k=args.k, ef=args.ef, n_probes=4)
                results.append([x[0] for x in r])
            search_results[label] = results

        print("  Measuring latencies...")
        latencies = {}
        for label, mgr in managers.items():
            latencies[label] = measure_search_latency(mgr, queries, args.k, args.ef, 4)

        # ----------------------------------------------------------------
        # Part D: Results Summary
        # ----------------------------------------------------------------
        print(f"\n{'='*70}")
        print(f"  RESULTS")
        print(f"{'='*70}\n")

        print("  --- Build Speed ---")
        print(f"  qCH dual-graph:   {build_ch:.3f}s")
        print(f"  qEMD dual-graph:  {build_emd:.3f}s  ({build_emd/max(build_ch,0.001):.2f}x slower)")

        print(f"\n  --- Recall@k ---")
        header = "  {:20s}".format("")
        for ks in [1, 5, args.k]:
            header += f"  R@{ks:<3d}"
        print(header)
        for label in search_results:
            row = f"  {label:20s}"
            for ks in [1, 5, args.k]:
                r = recall_at_k(search_results[label], gt, ks)
                row += f"  {r:.3f}"
            print(row)

        print(f"\n  --- Latency (ms/query) ---")
        print(f"  {'':20s}  {'p50':>6s}  {'p95':>6s}  {'mean':>6s}")
        for label in latencies:
            lat = latencies[label]
            print(f"  {label:20s}  {np.percentile(lat,50):6.2f}  {np.percentile(lat,95):6.2f}  {np.mean(lat):6.2f}")

        if _ROQ_AVAILABLE and "FP32 MaxSim" in managers:
            print(f"\n  --- Memory (rerank vectors) ---")
            fp32_bytes = sum(v.nbytes for v in managers["FP32 MaxSim"]._doc_vectors.values())
            roq_bytes = sum(c.nbytes + m.nbytes for c, m in managers["ROQ 4-bit"]._doc_roq.values())
            n_tok = args.n_docs * args.tpd
            print(f"  FP32:  {fp32_bytes/1024/1024:.2f} MB  ({fp32_bytes/n_tok:.1f} bytes/token)")
            print(f"  ROQ4:  {roq_bytes/1024/1024:.2f} MB  ({roq_bytes/n_tok:.1f} bytes/token)")
            print(f"  Ratio: {fp32_bytes/max(roq_bytes,1):.1f}x compression")

        print(f"\n{'='*70}\n")

        for mgr in managers.values():
            mgr.close()

    finally:
        for td in tmp_dirs:
            shutil.rmtree(td, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="GEM Optimization Benchmark")
    parser.add_argument("--n_docs", type=int, default=500)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--tpd", type=int, default=32, help="tokens per document")
    parser.add_argument("--n_queries", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef", type=int, default=64)
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear Triton cache before running (for CI cold-start test)")
    args = parser.parse_args()

    if args.clear_cache:
        import os
        cache_dir = os.path.expanduser("~/.triton/cache")
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Cleared Triton cache: {cache_dir}")

    run_benchmark(args)


if __name__ == "__main__":
    main()
