"""
Multimodal evaluation runner.

Usage:
    python -m benchmarks.multimodal.eval_runner --dataset synthetic --n_docs 500
    python -m benchmarks.multimodal.eval_runner --dataset okvqa --data_dir /path/to/data
    python -m benchmarks.multimodal.eval_runner --dataset evqa --data_dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from benchmarks.multimodal.dataset_loader import (
    EVQADataset,
    MultimodalDataset,
    OKVQADataset,
    SyntheticMultimodalDataset,
)

# ---------------------------------------------------------------------------
# GEM build parameters (reasonable defaults for benchmarking)
# ---------------------------------------------------------------------------
GEM_PARAMS = dict(
    n_fine=64,
    n_coarse=8,
    max_degree=16,
    ef_construction=100,
    ctop_r=3,
)

N_PROBES = 4
EF_VALUES = [32, 64, 128, 256]
K_VALUES = [1, 10, 100]


# ---------------------------------------------------------------------------
# Brute-force MaxSim ground truth
# ---------------------------------------------------------------------------

def brute_force_maxsim_topk(
    all_vectors: np.ndarray,
    offsets: List[Tuple[int, int]],
    query: np.ndarray,
    k: int,
) -> List[Tuple[int, float]]:
    """Exact MaxSim scoring: for each query token, max similarity across doc
    tokens, then average over query tokens.  Returns (doc_id, distance) sorted
    ascending (lower = closer), matching GemSegment convention."""

    scores = []
    for doc_id, (start, end) in enumerate(offsets):
        doc_vecs = all_vectors[start:end]
        sims = query @ doc_vecs.T                     # (n_qtok, n_dtok)
        avg_max_sim = sims.max(axis=1).mean()
        scores.append((doc_id, 1.0 - avg_max_sim))    # distance-like

    scores.sort(key=lambda x: x[1])
    return scores[:k]


def build_ground_truth_cache(
    all_vectors: np.ndarray,
    offsets: List[Tuple[int, int]],
    queries: List[np.ndarray],
    max_k: int,
) -> List[List[Tuple[int, float]]]:
    """Compute brute-force top-k for every query (cached for re-use across ef)."""
    gt = []
    for i, q in enumerate(queries):
        gt.append(brute_force_maxsim_topk(all_vectors, offsets, q, max_k))
        if (i + 1) % 50 == 0:
            print(f"  Ground truth: {i + 1}/{len(queries)}", flush=True)
    return gt


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------

def recall_at_k(
    results: List[Tuple[int, float]],
    ground_truth: List[Tuple[int, float]],
    k: int,
) -> float:
    gt_ids = {doc_id for doc_id, _ in ground_truth[:k]}
    result_ids = {doc_id for doc_id, _ in results[:k]}
    if not gt_ids:
        return 1.0
    return len(gt_ids & result_ids) / len(gt_ids)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    dataset: MultimodalDataset,
    dataset_name: str,
) -> Dict:
    print("Loading dataset...", flush=True)
    dataset.load()

    queries = dataset.queries()
    all_vectors, doc_ids, offsets = dataset.documents()

    print(f"  Documents: {len(doc_ids):,}")
    print(f"  Total vectors: {all_vectors.shape[0]:,}  dim={all_vectors.shape[1]}")
    print(f"  Queries: {len(queries)}")
    print()

    # -- Build GemSegment ------------------------------------------------
    from latence_gem_index import GemSegment

    seg = GemSegment()
    print("Building GEM index...", flush=True)
    t0 = time.perf_counter()
    seg.build(all_vectors, doc_ids, offsets, **GEM_PARAMS)
    build_s = time.perf_counter() - t0
    n_edges = seg.n_edges()
    print(f"  Build time: {build_s:.2f}s  |  Edges: {n_edges:,}")
    print()

    # -- Brute-force ground truth ----------------------------------------
    max_k = max(K_VALUES)
    print("Computing brute-force ground truth...", flush=True)
    gt_cache = build_ground_truth_cache(all_vectors, offsets, queries, max_k)
    print()

    # -- Warmup ----------------------------------------------------------
    for q in queries[:min(5, len(queries))]:
        seg.search(q, k=10, ef=64, n_probes=N_PROBES)

    # -- Sweep ef values -------------------------------------------------
    results_by_ef: Dict[int, Dict] = {}

    for ef in EF_VALUES:
        print(f"Evaluating ef={ef}...", flush=True)
        recalls: Dict[int, List[float]] = {k: [] for k in K_VALUES}
        latencies_us: List[float] = []

        for qi, q in enumerate(queries):
            t0 = time.perf_counter()
            res = seg.search(q, k=max_k, ef=ef, n_probes=N_PROBES)
            lat = (time.perf_counter() - t0) * 1e6
            latencies_us.append(lat)

            gt = gt_cache[qi]
            for k in K_VALUES:
                recalls[k].append(recall_at_k(res, gt, k))

        mean_recalls = {k: float(np.mean(v)) for k, v in recalls.items()}
        lat_sorted = sorted(latencies_us)

        entry = {
            "ef": ef,
            "recall@1": round(mean_recalls[1], 4),
            "recall@10": round(mean_recalls[10], 4),
            "recall@100": round(mean_recalls[100], 4),
            "latency_mean_us": round(statistics.mean(latencies_us), 1),
            "latency_p50_us": round(lat_sorted[len(lat_sorted) // 2], 1),
            "latency_p95_us": round(lat_sorted[int(len(lat_sorted) * 0.95)], 1),
            "latency_p99_us": round(lat_sorted[int(len(lat_sorted) * 0.99)], 1),
        }
        results_by_ef[ef] = entry

        print(
            f"  R@1={entry['recall@1']:.4f}  "
            f"R@10={entry['recall@10']:.4f}  "
            f"R@100={entry['recall@100']:.4f}  "
            f"lat_p50={entry['latency_p50_us']:.0f}us"
        )

    # -- Assemble report -------------------------------------------------
    report = {
        "dataset": dataset_name,
        "n_docs": len(doc_ids),
        "n_queries": len(queries),
        "dim": int(all_vectors.shape[1]),
        "total_vectors": int(all_vectors.shape[0]),
        "build_time_s": round(build_s, 3),
        "n_edges": n_edges,
        "gem_params": GEM_PARAMS,
        "n_probes": N_PROBES,
        "ef_sweep": {str(ef): results_by_ef[ef] for ef in EF_VALUES},
    }

    return report


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_report(report: Dict) -> None:
    print()
    print("=" * 85)
    print(f"Multimodal Benchmark Report — {report['dataset']}")
    print("=" * 85)
    print(f"  Documents: {report['n_docs']:,}  |  Queries: {report['n_queries']:,}")
    print(f"  Dim: {report['dim']}  |  Total vectors: {report['total_vectors']:,}")
    print(f"  Build time: {report['build_time_s']:.2f}s  |  Edges: {report['n_edges']:,}")
    print()

    header = (
        f"{'ef':>6}  {'R@1':>8}  {'R@10':>8}  {'R@100':>8}  "
        f"{'p50 (us)':>10}  {'p95 (us)':>10}  {'p99 (us)':>10}  {'mean (us)':>10}"
    )
    print(header)
    print("-" * len(header))

    for ef_str in sorted(report["ef_sweep"], key=int):
        e = report["ef_sweep"][ef_str]
        print(
            f"{e['ef']:>6}  {e['recall@1']:>8.4f}  {e['recall@10']:>8.4f}  "
            f"{e['recall@100']:>8.4f}  {e['latency_p50_us']:>10.0f}  "
            f"{e['latency_p95_us']:>10.0f}  {e['latency_p99_us']:>10.0f}  "
            f"{e['latency_mean_us']:>10.0f}"
        )

    print("=" * 85)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multimodal evaluation runner")
    p.add_argument("--dataset", required=True, choices=["synthetic", "okvqa", "evqa"])
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--n_docs", type=int, default=500)
    p.add_argument("--n_queries", type=int, default=50)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--vecs_per_doc", type=int, default=32)
    p.add_argument("--vecs_per_query", type=int, default=32)
    p.add_argument("--max_queries", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset == "synthetic":
        ds: MultimodalDataset = SyntheticMultimodalDataset(
            n_docs=args.n_docs,
            n_queries=args.n_queries,
            dim=args.dim,
            vecs_per_doc=args.vecs_per_doc,
            vecs_per_query=args.vecs_per_query,
            seed=args.seed,
        )
    elif args.dataset == "okvqa":
        ds = OKVQADataset(
            data_dir=args.data_dir,
            split="val",
            max_queries=args.max_queries,
        )
    elif args.dataset == "evqa":
        ds = EVQADataset(
            data_dir=args.data_dir,
            split="val",
            max_queries=args.max_queries,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    report = evaluate(ds, args.dataset)
    print_report(report)

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"multimodal_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
