#!/usr/bin/env python3
"""
Comprehensive GEM benchmark: recall curves, build-time extrapolation,
latency breakdown, memory estimation, and scaling projections to 1M docs.

Usage:
    python benchmarks/comprehensive_bench.py
"""

from __future__ import annotations

import math
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

HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"


def _sync():
    if HAS_CUDA:
        torch.cuda.synchronize()


def gen_data(n, dim, tpd, seed=42):
    rng = np.random.default_rng(seed)
    vecs, ids = [], list(range(1, n + 1))
    for _ in range(n):
        v = rng.standard_normal((tpd, dim)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        vecs.append(v)
    return vecs, ids


def gen_queries(n, dim, nt=8, seed=999):
    rng = np.random.default_rng(seed)
    return [
        (lambda q: q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8))(
            rng.standard_normal((nt, dim)).astype(np.float32)
        )
        for _ in range(n)
    ]


def brute_force(queries, dvecs, dids, k):
    results = []
    for q in queries:
        qt = torch.from_numpy(q)
        scores = []
        for did, dv in zip(dids, dvecs):
            dt = torch.from_numpy(dv)
            sim = (qt @ dt.T).max(dim=1).values.sum().item()
            scores.append((did, sim))
        scores.sort(key=lambda x: -x[1])
        results.append([s[0] for s in scores[:k]])
    return results


def recall_at_k(pred, gt, k):
    recs = []
    for p, g in zip(pred, gt):
        ps, gs = set(p[:k]), set(g[:k])
        if gs:
            recs.append(len(ps & gs) / len(gs))
    return np.mean(recs) if recs else 0.0


def build_raw_segment(vecs, *, use_emd=False, dual_graph=True):
    n_docs = len(vecs)
    all_v = np.concatenate(vecs).astype(np.float32)
    offs, pos = [], 0
    for v in vecs:
        offs.append((pos, pos + v.shape[0]))
        pos += v.shape[0]
    nf = max(8, min(64, int(np.sqrt(pos))))
    nc = max(4, nf // 4)
    seg = GemSegment()
    t0 = time.perf_counter()
    seg.build(
        all_v, list(range(n_docs)), offs,
        n_coarse=nc, n_fine=nf, max_degree=16,
        ef_construction=64, max_kmeans_iter=10, ctop_r=2,
        use_emd=use_emd, dual_graph=dual_graph,
    )
    return seg, time.perf_counter() - t0


def build_manager(td, dim, vecs, ids, **kw):
    n = len(ids)
    nf = max(8, min(64, int(np.sqrt(n))))
    nc = max(4, nf // 4)
    mgr = GemNativeSegmentManager(
        str(td), dim,
        n_fine=nf, n_coarse=nc, max_degree=16,
        gem_ef_construction=64, max_kmeans_iter=10, ctop_r=2,
        seed_batch_size=n, enable_wal=True, warmup_kernels=True,
        **kw,
    )
    mgr.add_multidense(vecs, ids)
    mgr.seal_active_segment()
    return mgr


def main():
    sep = "=" * 70
    sep2 = "-" * 66
    dim = 128
    k = 10

    print(f"\n{sep}")
    print("  COMPREHENSIVE GEM BENCHMARK")
    print(f"  Device: {DEVICE}  |  MaxSim: {_MAXSIM_AVAILABLE}  |  ROQ: {_ROQ_AVAILABLE}")
    print(f"{sep}\n")

    # ------------------------------------------------------------------
    # 1. Build time at multiple scales for extrapolation
    # ------------------------------------------------------------------
    print(f"  {sep2}")
    print("  1. Build Time (qCH dual-graph)")
    print(f"  {sep2}\n")

    build_points = {}
    for n_docs, tpd in [(200, 128), (500, 128), (500, 32)]:
        vecs, _ = gen_data(n_docs, dim, tpd, seed=7)
        _, bt = build_raw_segment(vecs)
        build_points[(n_docs, tpd)] = bt
        total_tok = n_docs * tpd
        print(f"    {n_docs:>5d} docs x {tpd:>4d} tok  -> {bt:.3f}s  ({total_tok:>7,d} total tokens)")

    # Extrapolation: T(N,t) ~ N * t * log(N)
    t_200 = build_points[(200, 128)]
    t_500 = build_points[(500, 128)]
    # Derive scaling constant from the 500-doc measurement
    a = t_500 / (500 * 128 * math.log(500))

    N_1m = 1_000_000
    tpd_target = 1024
    t_1m = a * N_1m * tpd_target * math.log(N_1m)

    print(f"\n    Extrapolation (T ~ a*N*t*log(N), a={a:.2e}):")
    print(f"    1M docs x 1024 tok  -> ~{t_1m / 3600:.1f} hours")
    print(f"    1M docs x  128 tok  -> ~{a * N_1m * 128 * math.log(N_1m) / 3600:.1f} hours")

    # ------------------------------------------------------------------
    # 2. Recall@k at varying ef (500 docs, 32 tpd, FP32 rerank)
    # ------------------------------------------------------------------
    print(f"\n  {sep2}")
    print("  2. Recall@10 vs ef (500 docs, 32 tok, FP32 rerank)")
    print(f"  {sep2}\n")

    vecs500, ids500 = gen_data(500, dim, 32)
    queries = gen_queries(100, dim)
    gt = brute_force(queries, vecs500, ids500, k)

    td = tempfile.mkdtemp()
    mgr = build_manager(Path(td), dim, vecs500, ids500, rerank_device=DEVICE)

    print(f"    {'ef':>6s}  {'R@1':>6s}  {'R@5':>6s}  {'R@10':>6s}  {'p50 ms':>8s}  {'p95 ms':>8s}")
    for ef_val in [16, 32, 64, 100, 200]:
        preds, lats = [], []
        for q in queries:
            _sync()
            t0 = time.perf_counter()
            r = mgr.search_multivector(q, k=k, ef=ef_val, n_probes=4)
            _sync()
            lats.append((time.perf_counter() - t0) * 1000)
            preds.append([x[0] for x in r])
        r1 = recall_at_k(preds, gt, 1)
        r5 = recall_at_k(preds, gt, 5)
        r10 = recall_at_k(preds, gt, 10)
        p50 = np.percentile(lats, 50)
        p95 = np.percentile(lats, 95)
        print(f"    {ef_val:>6d}  {r1:>6.3f}  {r5:>6.3f}  {r10:>6.3f}  {p50:>8.2f}  {p95:>8.2f}")
    mgr.close()
    shutil.rmtree(td, ignore_errors=True)

    # ------------------------------------------------------------------
    # 3. Latency breakdown by profile (proxy, FP32 MaxSim, ROQ 4-bit)
    # ------------------------------------------------------------------
    print(f"\n  {sep2}")
    print("  3. Latency Breakdown by Reranking Profile")
    print(f"  {sep2}\n")

    configs = [
        ("Proxy only (qCH)", {}),
        ("FP32 MaxSim", {"rerank_device": DEVICE}),
    ]
    if _ROQ_AVAILABLE:
        configs.append(("ROQ 4-bit", {"roq_rerank": True}))

    managers = {}
    tmp_dirs = []
    for label, kw in configs:
        td = tempfile.mkdtemp()
        tmp_dirs.append(td)
        mgr = build_manager(Path(td), dim, vecs500, ids500, **kw)
        managers[label] = mgr

    print(f"    {'Profile':>20s}  {'p50':>7s}  {'p95':>7s}  {'mean':>7s}")
    for label, mgr in managers.items():
        # warmup
        for q in queries[:5]:
            mgr.search_multivector(q, k=k, ef=64, n_probes=4)
        _sync()
        lats = []
        for q in queries:
            _sync()
            t0 = time.perf_counter()
            mgr.search_multivector(q, k=k, ef=64, n_probes=4)
            _sync()
            lats.append((time.perf_counter() - t0) * 1000)
        p50 = np.percentile(lats, 50)
        p95 = np.percentile(lats, 95)
        mean = np.mean(lats)
        print(f"    {label:>20s}  {p50:>7.2f}  {p95:>7.2f}  {mean:>7.2f}")

    # ------------------------------------------------------------------
    # 4. Memory per document
    # ------------------------------------------------------------------
    print(f"\n  {sep2}")
    print("  4. Memory per Document")
    print(f"  {sep2}\n")

    if _ROQ_AVAILABLE and "FP32 MaxSim" in managers:
        fp32_bytes = sum(v.nbytes for v in managers["FP32 MaxSim"]._doc_vectors.values())
        roq_bytes = sum(c.nbytes + m.nbytes for c, m in managers["ROQ 4-bit"]._doc_roq.values())
        n_tok = 500 * 32
        print(f"    FP32:  {fp32_bytes / 1024 / 1024:.2f} MB  ({fp32_bytes / n_tok:.1f} bytes/token)")
        print(f"    ROQ4:  {roq_bytes / 1024 / 1024:.2f} MB  ({roq_bytes / n_tok:.1f} bytes/token)")
        print(f"    Ratio: {fp32_bytes / max(roq_bytes, 1):.1f}x compression")

    print(f"\n    Scaling to 1M docs:")
    for tpd_sc in [128, 512, 1024]:
        fp32 = N_1m * tpd_sc * dim * 4
        roq4 = N_1m * tpd_sc * (dim // 2 + 16)
        print(f"      {tpd_sc:>5d} tok:  FP32 = {fp32 / 1e9:.1f} GB   ROQ4 = {roq4 / 1e9:.1f} GB   ratio = {fp32 / roq4:.1f}x")

    for mgr in managers.values():
        mgr.close()
    for td in tmp_dirs:
        shutil.rmtree(td, ignore_errors=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("  BENCHMARK COMPLETE")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
