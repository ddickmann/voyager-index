"""
Benchmarks for Elite Innovation Layer:
  - GPU vs CPU qCH latency (Triton vs PyTorch fallback vs naive CPU)
  - Filter routing speedup
  - Self-healing effectiveness
  - Ensemble RRF recall
"""

from __future__ import annotations

import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    print("PyTorch not available, GPU benchmarks will be skipped")
    CUDA_AVAILABLE = False

try:
    from voyager_index._internal.inference.index_core.triton_qch_kernel import (
        qch_max_gather_gpu,
        qch_max_gather_torch,
        TRITON_AVAILABLE,
    )
    QCH_AVAILABLE = True
except ImportError:
    QCH_AVAILABLE = False
    TRITON_AVAILABLE = False

try:
    from latence_gem_index import GemSegment, PyMutableGemSegment, PyEnsembleGemSegment
    GEM_AVAILABLE = True
except ImportError:
    GEM_AVAILABLE = False


def _header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _naive_cpu_qch(scores_flat, codes, offsets, lengths, n_query, n_fine):
    n_docs = len(offsets)
    out = np.empty(n_docs, dtype=np.float32)
    scores_2d = scores_flat.reshape(n_query, n_fine)
    for d in range(n_docs):
        off = offsets[d]
        ln = lengths[d]
        total = 0.0
        for qi in range(n_query):
            max_val = -1e30
            for ci in range(ln):
                c = codes[off + ci]
                s = scores_2d[qi, c]
                if s > max_val:
                    max_val = s
            total += max_val
        out[d] = 1.0 - total / n_query
    return out


def _synthetic_qch_data(n_docs, n_query, n_fine, max_doc_len, seed=42):
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal(n_query * n_fine).astype(np.float32)
    lengths = rng.integers(max_doc_len // 2, max_doc_len + 1, size=n_docs).astype(np.int32)
    offsets = np.zeros(n_docs, dtype=np.int32)
    offsets[1:] = np.cumsum(lengths[:-1])
    total_codes = int(offsets[-1] + lengths[-1]) if n_docs > 0 else 0
    codes = rng.integers(0, n_fine, size=total_codes).astype(np.int32)
    return scores, codes, offsets, lengths


def _synthetic_corpus(n_docs, dim, vecs_per_doc, seed=42):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_docs * vecs_per_doc, dim)).astype(np.float32)
    doc_ids = list(range(1, n_docs + 1))
    offsets = [(i * vecs_per_doc, (i + 1) * vecs_per_doc) for i in range(n_docs)]
    return data, doc_ids, offsets


# ===========================================================================
# Benchmark 1: GPU vs CPU qCH Latency
# ===========================================================================

def bench_gpu_qch():
    _header("GPU vs CPU qCH Latency")

    if not QCH_AVAILABLE:
        print("  SKIPPED: qCH kernel not available")
        return
    if not CUDA_AVAILABLE:
        print("  SKIPPED: No CUDA GPU")
        return

    configs = [
        (50_000, 16, 256, 512),
        (50_000, 16, 256, 2048),
        (10_000, 32, 512, 1024),
    ]

    for n_docs, n_query, n_fine, max_doc_len in configs:
        print(f"\n  n_docs={n_docs}, n_query={n_query}, n_fine={n_fine}, max_doc_len={max_doc_len}")
        scores, codes, offsets, lengths = _synthetic_qch_data(
            n_docs, n_query, n_fine, max_doc_len,
        )
        t_s = torch.from_numpy(scores).cuda()
        t_c = torch.from_numpy(codes).cuda()
        t_o = torch.from_numpy(offsets).cuda()
        t_l = torch.from_numpy(lengths).cuda()

        # Warmup
        _ = qch_max_gather_torch(t_s, t_c, t_o, t_l, n_query, n_fine)
        torch.cuda.synchronize()

        # PyTorch fallback
        n_runs = 5
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = qch_max_gather_torch(t_s, t_c, t_o, t_l, n_query, n_fine)
            torch.cuda.synchronize()
        pytorch_ms = (time.perf_counter() - t0) / n_runs * 1000

        # Triton
        if TRITON_AVAILABLE:
            _ = qch_max_gather_gpu(t_s, t_c, t_o, t_l, n_query, n_fine)
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                _ = qch_max_gather_gpu(t_s, t_c, t_o, t_l, n_query, n_fine)
                torch.cuda.synchronize()
            triton_ms = (time.perf_counter() - t0) / n_runs * 1000
        else:
            triton_ms = float("nan")

        # CPU reference (use smaller subset)
        cpu_n = min(500, n_docs)
        cpu_scores = scores
        cpu_codes = codes[:int(offsets[cpu_n - 1] + lengths[cpu_n - 1])] if cpu_n > 0 else codes
        cpu_offsets = offsets[:cpu_n]
        cpu_lengths = lengths[:cpu_n]
        t0 = time.perf_counter()
        _naive_cpu_qch(cpu_scores, cpu_codes, cpu_offsets, cpu_lengths, n_query, n_fine)
        cpu_ms = (time.perf_counter() - t0) * 1000
        cpu_ms_projected = cpu_ms * (n_docs / cpu_n)

        print(f"    Triton:          {triton_ms:8.2f} ms")
        print(f"    PyTorch fallback:{pytorch_ms:8.2f} ms")
        print(f"    CPU (projected): {cpu_ms_projected:8.2f} ms")
        if not np.isnan(triton_ms):
            print(f"    Triton speedup vs CPU:     {cpu_ms_projected / triton_ms:6.1f}x")
            print(f"    Triton speedup vs PyTorch: {pytorch_ms / triton_ms:6.1f}x")


# ===========================================================================
# Benchmark 2: Filter Routing Speedup
# ===========================================================================

def bench_filter_routing():
    _header("Filter Routing Speedup")

    if not GEM_AVAILABLE:
        print("  SKIPPED: latence_gem_index not available")
        return

    n_docs, dim, vecs_per_doc = 2000, 32, 8
    data, ids, offsets = _synthetic_corpus(n_docs, dim, vecs_per_doc)

    seg = GemSegment()
    seg.build(data, ids, offsets, n_fine=64, n_coarse=8, max_degree=16,
              ef_construction=64, max_kmeans_iter=10, ctop_r=3)

    payloads = []
    for i, doc_id in enumerate(ids):
        cat = "A" if i < n_docs // 2 else "B"
        payloads.append((doc_id, [("category", cat)]))
    seg.set_doc_payloads(payloads)

    query = np.random.default_rng(42).standard_normal((4, dim)).astype(np.float32)
    n_runs = 20

    # Warmup
    seg.search(query, k=10, ef=50, n_probes=3)

    # Unfiltered
    t0 = time.perf_counter()
    for _ in range(n_runs):
        seg.search(query, k=10, ef=50, n_probes=3)
    unfiltered_ms = (time.perf_counter() - t0) / n_runs * 1000

    # Filtered (50% selectivity)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        seg.search(query, k=10, ef=50, n_probes=3, filter=[("category", "A")])
    filtered_ms = (time.perf_counter() - t0) / n_runs * 1000

    print(f"\n  n_docs={n_docs}, filter_selectivity=50%")
    print(f"    Unfiltered:  {unfiltered_ms:8.2f} ms")
    print(f"    Filtered:    {filtered_ms:8.2f} ms")
    print(f"    Overhead:    {filtered_ms / unfiltered_ms:6.2f}x")


# ===========================================================================
# Benchmark 3: Self-Healing Effectiveness
# ===========================================================================

def bench_heal_effectiveness():
    _header("Self-Healing Effectiveness")

    if not GEM_AVAILABLE:
        print("  SKIPPED: latence_gem_index not available")
        return

    n_docs, dim, vecs_per_doc = 1000, 32, 8
    data, ids, offsets = _synthetic_corpus(n_docs, dim, vecs_per_doc)

    seg = PyMutableGemSegment()
    seg.build(data, ids, offsets, n_fine=32, n_coarse=8, max_degree=16,
              ef_construction=64, max_kmeans_iter=10, ctop_r=3)

    metrics_initial = seg.graph_quality_metrics()
    print(f"\n  Initial metrics (n_docs={n_docs}):")
    print(f"    delete_ratio:      {metrics_initial[0]:.4f}")
    print(f"    avg_degree:        {metrics_initial[1]:.2f}")
    print(f"    isolated_ratio:    {metrics_initial[2]:.4f}")
    print(f"    stale_rep_ratio:   {metrics_initial[3]:.4f}")

    # Delete 20%
    n_delete = n_docs // 5
    for doc_id in ids[:n_delete]:
        seg.delete(doc_id)

    metrics_post_delete = seg.graph_quality_metrics()
    print(f"\n  After {n_delete} deletes ({100*n_delete/n_docs:.0f}%):")
    print(f"    delete_ratio:      {metrics_post_delete[0]:.4f}")
    print(f"    avg_degree:        {metrics_post_delete[1]:.2f}")
    print(f"    isolated_ratio:    {metrics_post_delete[2]:.4f}")
    print(f"    stale_rep_ratio:   {metrics_post_delete[3]:.4f}")
    print(f"    needs_healing:     {seg.needs_healing()}")

    # Heal
    t0 = time.perf_counter()
    seg.heal()
    heal_ms = (time.perf_counter() - t0) * 1000

    metrics_post_heal = seg.graph_quality_metrics()
    print(f"\n  After heal() ({heal_ms:.1f} ms):")
    print(f"    delete_ratio:      {metrics_post_heal[0]:.4f}")
    print(f"    avg_degree:        {metrics_post_heal[1]:.2f}")
    print(f"    isolated_ratio:    {metrics_post_heal[2]:.4f}")
    print(f"    stale_rep_ratio:   {metrics_post_heal[3]:.4f}")
    print(f"    needs_healing:     {seg.needs_healing()}")

    # Search quality check
    query = np.random.default_rng(99).standard_normal((4, dim)).astype(np.float32)
    results = seg.search(query, k=10, ef=50)
    deleted_set = set(ids[:n_delete])
    n_deleted_in_results = sum(1 for d, _ in results if d in deleted_set)
    print(f"\n  Search after heal: {len(results)} results, {n_deleted_in_results} deleted docs leaked")


# ===========================================================================
# Benchmark 4: Ensemble RRF Recall
# ===========================================================================

def bench_ensemble_recall():
    _header("Ensemble RRF Recall")

    if not GEM_AVAILABLE:
        print("  SKIPPED: latence_gem_index not available")
        return

    n_docs, dim, vecs_per_doc = 500, 32, 8
    n_modalities = 2
    rng = np.random.default_rng(42)
    total_vecs = n_docs * vecs_per_doc
    all_vectors = rng.standard_normal((total_vecs, dim)).astype(np.float32)
    doc_ids = list(range(1, n_docs + 1))
    doc_offsets = [(i * vecs_per_doc, (i + 1) * vecs_per_doc) for i in range(n_docs)]
    modality_tags = [i % n_modalities for i in range(total_vecs)]

    # Build ensemble
    ens = PyEnsembleGemSegment()
    t0 = time.perf_counter()
    ens.build(all_vectors, doc_ids, doc_offsets, modality_tags, n_modalities,
              n_fine=32, n_coarse=8, max_degree=16, ef_construction=64,
              max_kmeans_iter=10, ctop_r=3)
    build_ms = (time.perf_counter() - t0) * 1000

    # Build single-modality baseline (modality 0 only)
    seg0 = GemSegment()
    seg0.build(all_vectors, doc_ids, doc_offsets, n_fine=32, n_coarse=8,
               max_degree=16, ef_construction=64, max_kmeans_iter=10, ctop_r=3)

    print(f"\n  n_docs={n_docs}, n_modalities={n_modalities}, dim={dim}")
    print(f"  Ensemble build time: {build_ms:.1f} ms")

    n_queries = 20
    ens_results_all = []
    single_results_all = []
    for qi in range(n_queries):
        q = rng.standard_normal((4, dim)).astype(np.float32)
        q_tags = [0, 0, 1, 1]
        ens_r = ens.search(q, q_tags, k=10, ef=100, n_probes=4)
        single_r = seg0.search(q, k=10, ef=100, n_probes=4)
        ens_results_all.append(set(d for d, _ in ens_r))
        single_results_all.append(set(d for d, _ in single_r))

    # Overlap between ensemble and single-modality
    overlaps = []
    for e, s in zip(ens_results_all, single_results_all):
        if len(e) > 0 and len(s) > 0:
            overlap = len(e & s) / max(len(e), len(s))
            overlaps.append(overlap)

    avg_overlap = np.mean(overlaps) if overlaps else 0
    ens_avg_k = np.mean([len(r) for r in ens_results_all])
    single_avg_k = np.mean([len(r) for r in single_results_all])

    # Latency
    q = rng.standard_normal((4, dim)).astype(np.float32)
    q_tags = [0, 0, 1, 1]
    n_runs = 20

    t0 = time.perf_counter()
    for _ in range(n_runs):
        ens.search(q, q_tags, k=10, ef=100, n_probes=4)
    ens_ms = (time.perf_counter() - t0) / n_runs * 1000

    t0 = time.perf_counter()
    for _ in range(n_runs):
        seg0.search(q, k=10, ef=100, n_probes=4)
    single_ms = (time.perf_counter() - t0) / n_runs * 1000

    print(f"  Ensemble avg results@10: {ens_avg_k:.1f}")
    print(f"  Single-mod avg results@10: {single_avg_k:.1f}")
    print(f"  Avg overlap@10: {avg_overlap:.3f}")
    print(f"  Ensemble search latency:  {ens_ms:.2f} ms")
    print(f"  Single-mod search latency: {single_ms:.2f} ms")
    print(f"  Latency ratio: {ens_ms / single_ms:.2f}x")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    bench_gpu_qch()
    bench_filter_routing()
    bench_heal_effectiveness()
    bench_ensemble_recall()
    print(f"\n{'='*70}")
    print("  All benchmarks complete.")
    print(f"{'='*70}")
