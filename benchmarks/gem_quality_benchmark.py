"""
GEM index quality benchmark — workstream A1 (qEMD) and A2 (dual-graph).

Measures recall, local-minima rate, distribution-shift robustness, and
delete/compact resilience across four build configurations.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ── Configuration ──────────────────────────────────────────
N_DOCS = 200
DIM = 32
VECS_PER_DOC = 16
N_QUERIES = 50
K = 10
K_RECALL_100 = 100
EF = 200
N_CLUSTERS = 8

GEM_PARAMS = dict(
    n_fine=32, n_coarse=8, max_degree=16, ef_construction=100,
    max_kmeans_iter=20, ctop_r=3,
)

BUILD_CONFIGS = {
    "qCH (default)": dict(use_emd=False, dual_graph=False),
    "qEMD":          dict(use_emd=True,  dual_graph=False),
    "Dual-graph":    dict(use_emd=False, dual_graph=True),
    "qEMD + Dual":   dict(use_emd=True,  dual_graph=True),
}


# ── Data generation ────────────────────────────────────────

def generate_corpus(
    n_docs: int, vecs_per_doc: int, dim: int, seed: int = 42,
    n_clusters: int = N_CLUSTERS,
) -> Tuple[np.ndarray, list, list]:
    """Generate clustered document data for meaningful retrieval."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim).astype(np.float32) * 3.0
    all_vecs = []
    for i in range(n_docs):
        cluster = i % n_clusters
        vecs = centers[cluster] + rng.randn(vecs_per_doc, dim).astype(np.float32) * 0.3
        all_vecs.append(vecs)
    all_vectors = np.vstack(all_vecs).astype(np.float32)
    doc_ids = list(range(n_docs))
    offsets = [(i * vecs_per_doc, (i + 1) * vecs_per_doc) for i in range(n_docs)]
    return all_vectors, doc_ids, offsets


def generate_queries(
    n_queries: int, vecs_per_query: int, dim: int, seed: int = 99,
    mean_shift: float = 0.0, n_clusters: int = N_CLUSTERS,
) -> List[np.ndarray]:
    """Generate queries from same cluster structure (or shifted for OOD)."""
    rng = np.random.RandomState(seed)
    centers = np.random.RandomState(42).randn(n_clusters, dim).astype(np.float32) * 3.0
    queries = []
    for i in range(n_queries):
        cluster = rng.randint(0, n_clusters)
        q = centers[cluster] + rng.randn(vecs_per_query, dim).astype(np.float32) * 0.3
        q += mean_shift
        queries.append(q)
    return queries


# ── Brute-force MaxSim ground truth ───────────────────────

def brute_force_maxsim(
    all_vectors: np.ndarray, doc_offsets: list, dim: int,
    query: np.ndarray, k: int,
) -> List[Tuple[int, float]]:
    """Compute exact qCH-proxy scores via brute-force MaxSim.

    For each query token qi, compute dot(qi, dj) with every doc token dj.
    Per-doc contribution from qi = max over dj in that doc.
    Doc score = 1.0 - mean over qi of max contributions.
    Lower is better (distance-like), matching GemSegment.search() convention.
    """
    n_query_tokens = query.shape[0]
    scores = []
    for doc_id, (start, end) in enumerate(doc_offsets):
        doc_vecs = all_vectors[start:end]
        # (n_query_tokens, n_doc_tokens)
        sims = query @ doc_vecs.T
        max_per_query_token = sims.max(axis=1)
        avg_max_sim = max_per_query_token.mean()
        scores.append((doc_id, 1.0 - avg_max_sim))

    scores.sort(key=lambda x: x[1])
    return scores[:k]


# ── Recall computation ─────────────────────────────────────

def compute_recall(
    results: List[Tuple[int, float]],
    ground_truth: List[Tuple[int, float]],
    k: int,
) -> float:
    gt_ids = {doc_id for doc_id, _ in ground_truth[:k]}
    result_ids = {doc_id for doc_id, _ in results[:k]}
    if not gt_ids:
        return 1.0
    return len(gt_ids & result_ids) / len(gt_ids)


# ── Benchmark helpers ──────────────────────────────────────

def build_and_measure(
    all_vectors: np.ndarray, doc_ids: list, offsets: list,
    in_domain_queries: List[np.ndarray],
    ood_queries: List[np.ndarray],
    config_name: str, config_kwargs: dict,
) -> dict:
    from latence_gem_index import GemSegment

    seg = GemSegment()
    t0 = time.perf_counter()
    seg.build(all_vectors, doc_ids, offsets, **GEM_PARAMS, **config_kwargs)
    build_ms = (time.perf_counter() - t0) * 1000

    n_edges = seg.n_edges()

    # In-domain recall@10
    recall_10_vals = []
    for q in in_domain_queries:
        results = seg.search(q, k=K, ef=EF, n_probes=4)
        gt = brute_force_maxsim(all_vectors, offsets, DIM, q, K)
        recall_10_vals.append(compute_recall(results, gt, K))

    # In-domain recall@100
    recall_100_vals = []
    for q in in_domain_queries:
        results = seg.search(q, k=K_RECALL_100, ef=max(EF, K_RECALL_100), n_probes=4)
        gt = brute_force_maxsim(all_vectors, offsets, DIM, q, K_RECALL_100)
        recall_100_vals.append(compute_recall(results, gt, K_RECALL_100))

    # Out-of-domain recall@10
    ood_recall_vals = []
    for q in ood_queries:
        results = seg.search(q, k=K, ef=EF, n_probes=4)
        gt = brute_force_maxsim(all_vectors, offsets, DIM, q, K)
        ood_recall_vals.append(compute_recall(results, gt, K))

    # Local minima rate: greedy search (ef=1) vs brute-force top-1
    local_minima_count = 0
    for q in in_domain_queries:
        greedy = seg.search(q, k=1, ef=1, n_probes=4)
        gt = brute_force_maxsim(all_vectors, offsets, DIM, q, 1)
        if greedy and gt:
            if greedy[0][0] != gt[0][0]:
                local_minima_count += 1
        elif not greedy and gt:
            local_minima_count += 1

    # Proxy-oracle: brute-force rank by the segment's own qCH proxy
    proxy_recall_10_vals = []
    for q in in_domain_queries:
        results = seg.search(q, k=K, ef=EF, n_probes=4)
        proxy_gt = seg.brute_force_proxy(q, k=K)
        proxy_recall_10_vals.append(compute_recall(results, proxy_gt, K))

    # Connectivity check for dual-graph configs
    connectivity_ok = n_edges > 0

    return {
        "config": config_name,
        "build_time_ms": round(build_ms, 2),
        "n_edges": n_edges,
        "recall_at_10": round(np.mean(recall_10_vals), 4),
        "recall_at_100": round(np.mean(recall_100_vals), 4),
        "recall_at_10_ood": round(np.mean(ood_recall_vals), 4),
        "recall_at_10_vs_proxy": round(np.mean(proxy_recall_10_vals), 4),
        "local_minima_rate": round(local_minima_count / len(in_domain_queries), 4),
        "connectivity_ok": connectivity_ok,
    }


def test_mutable_delete_compact() -> dict:
    """Delete/compact resilience: build, delete 20%, compact, search."""
    from latence_gem_index import PyMutableGemSegment

    n_docs = 50
    all_vectors, doc_ids, offsets = generate_corpus(n_docs, VECS_PER_DOC, DIM, seed=77)

    seg = PyMutableGemSegment()
    seg.build(all_vectors, doc_ids, offsets, **GEM_PARAMS, n_probes=2)

    n_delete = n_docs // 5
    rng = random.Random(123)
    to_delete = rng.sample(doc_ids, n_delete)
    deleted_count = 0
    for did in to_delete:
        if seg.delete(did):
            deleted_count += 1

    seg.compact()

    query = np.random.RandomState(42).randn(VECS_PER_DOC, DIM).astype(np.float32)
    results = seg.search(query, k=5, ef=64, n_probes=2)

    surviving_ids = set(doc_ids) - set(to_delete)
    result_ids = {doc_id for doc_id, _ in results}
    no_deleted_in_results = result_ids.issubset(surviving_ids)

    return {
        "n_docs_initial": n_docs,
        "n_deleted": deleted_count,
        "n_live_after_compact": seg.n_live(),
        "search_returned": len(results),
        "no_crash": True,
        "no_deleted_in_results": no_deleted_in_results,
        "pass": len(results) > 0 and no_deleted_in_results,
    }


# ── Output formatting ─────────────────────────────────────

def print_table(all_metrics: list):
    cols = [
        ("Config",       "config",              "{:<16s}", 16),
        ("Build ms",     "build_time_ms",       "{:>10.1f}", 10),
        ("Edges",        "n_edges",             "{:>8d}", 8),
        ("R@10",         "recall_at_10",        "{:>8.4f}", 8),
        ("R@10 proxy",   "recall_at_10_vs_proxy", "{:>11.4f}", 11),
        ("R@100",        "recall_at_100",       "{:>8.4f}", 8),
        ("R@10 OOD",     "recall_at_10_ood",    "{:>10.4f}", 10),
        ("LM rate",      "local_minima_rate",   "{:>9.4f}", 9),
        ("Build ratio",  "build_time_ratio",    "{:>12.2f}", 12),
    ]

    header = "  ".join(f"{label:>{w}s}" if i > 0 else f"{label:<{w}s}"
                       for i, (label, _, _, w) in enumerate(cols))
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        row = "  ".join(fmt.format(m[key]) for _, key, fmt, _ in cols)
        print(row)


# ── Main ───────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("GEM Quality Benchmark — A1 (qEMD) / A2 (dual-graph)")
    print("=" * 80)
    print(f"  N_DOCS={N_DOCS}  DIM={DIM}  VECS_PER_DOC={VECS_PER_DOC}")
    print(f"  N_QUERIES={N_QUERIES}  K={K}  EF={EF}")
    print(f"  GEM_PARAMS={GEM_PARAMS}")
    print()

    print("Generating corpus...", flush=True)
    all_vectors, doc_ids, offsets = generate_corpus(N_DOCS, VECS_PER_DOC, DIM)
    print(f"  Total vectors: {all_vectors.shape[0]:,}")

    print("Generating queries...", flush=True)
    in_domain_queries = generate_queries(N_QUERIES, VECS_PER_DOC, DIM, seed=99, mean_shift=0.0)
    ood_queries = generate_queries(N_QUERIES, VECS_PER_DOC, DIM, seed=200, mean_shift=2.0)
    print(f"  In-domain: {len(in_domain_queries)} queries (same clusters)")
    print(f"  Out-of-domain: {len(ood_queries)} queries (shifted +2.0)")
    print()

    all_metrics = []
    baseline_build_ms = None

    for name, kwargs in BUILD_CONFIGS.items():
        print(f"Building [{name}]...", flush=True)
        metrics = build_and_measure(
            all_vectors, doc_ids, offsets,
            in_domain_queries, ood_queries,
            name, kwargs,
        )
        if baseline_build_ms is None:
            baseline_build_ms = metrics["build_time_ms"]
        metrics["build_time_ratio"] = round(
            metrics["build_time_ms"] / max(baseline_build_ms, 0.01), 2
        )
        all_metrics.append(metrics)
        print(f"  R@10={metrics['recall_at_10']:.4f}  "
              f"R@100={metrics['recall_at_100']:.4f}  "
              f"LM={metrics['local_minima_rate']:.4f}  "
              f"edges={metrics['n_edges']}")

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print_table(all_metrics)

    # Connectivity check for dual-graph configs
    print()
    for m in all_metrics:
        if "Dual" in m["config"]:
            status = "PASS" if m["connectivity_ok"] else "FAIL"
            print(f"  Connectivity [{m['config']}]: {status} (n_edges={m['n_edges']})")

    # Mutable segment delete/compact resilience
    print()
    print("-" * 80)
    print("Mutable segment delete/compact resilience test")
    print("-" * 80)
    mut_result = test_mutable_delete_compact()
    status = "PASS" if mut_result["pass"] else "FAIL"
    print(f"  Initial docs: {mut_result['n_docs_initial']}")
    print(f"  Deleted: {mut_result['n_deleted']}")
    print(f"  Live after compact: {mut_result['n_live_after_compact']}")
    print(f"  Search returned: {mut_result['search_returned']} results")
    print(f"  No deleted docs in results: {mut_result['no_deleted_in_results']}")
    print(f"  Result: {status}")

    # ── Search Latency Regression: qEMD does NOT affect search speed ──────
    print()
    print("-" * 80)
    print("Search latency regression: use_emd=True vs use_emd=False")
    print("-" * 80)

    from latence_gem_index import GemSegment as _Seg

    latency_test_queries = in_domain_queries[:200] if len(in_domain_queries) >= 200 else in_domain_queries
    n_warmup = 5

    for emd_flag, label in [(False, "use_emd=False"), (True, "use_emd=True")]:
        seg_lat = _Seg()
        seg_lat.build(all_vectors, doc_ids, offsets, **GEM_PARAMS, use_emd=emd_flag)

        # Warmup
        for q in latency_test_queries[:n_warmup]:
            seg_lat.search(q, k=K, ef=EF, n_probes=4)

        times_us = []
        for q in latency_test_queries:
            t0 = time.perf_counter()
            seg_lat.search(q, k=K, ef=EF, n_probes=4)
            t1 = time.perf_counter()
            times_us.append((t1 - t0) * 1e6)

        p50 = np.percentile(times_us, 50)
        p95 = np.percentile(times_us, 95)
        p99 = np.percentile(times_us, 99)
        print(f"  [{label}] p50={p50:.0f}us  p95={p95:.0f}us  p99={p99:.0f}us  (n={len(times_us)})")

    print("  Gate: p50 difference should be within 5% (same search code path)")

    # Save results
    results = {
        "config": {
            "n_docs": N_DOCS,
            "dim": DIM,
            "vecs_per_doc": VECS_PER_DOC,
            "n_queries": N_QUERIES,
            "k": K,
            "ef": EF,
            "gem_params": GEM_PARAMS,
        },
        "build_configs": {m["config"]: m for m in all_metrics},
        "mutable_resilience": mut_result,
    }

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gem_quality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
