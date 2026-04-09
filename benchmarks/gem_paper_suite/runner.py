"""
GEM Paper Reproduction Suite.

Runs recall/latency/size benchmarks matching GEM paper conditions.
Supports synthetic data for testing the pipeline and real datasets
(MS MARCO, LoTTE) when embeddings are available.

Usage:
    # Synthetic run (always available, for CI testing):
    python -m benchmarks.gem_paper_suite.runner --mode synthetic --n_docs 1000

    # Full run with pre-computed ColBERT embeddings:
    python -m benchmarks.gem_paper_suite.runner --config benchmarks/gem_paper_suite/configs/msmarco.json --data_dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ── Synthetic defaults ─────────────────────────────────────

SYNTHETIC_DEFAULTS = {
    "name": "synthetic",
    "dim": 128,
    "n_docs": 1000,
    "vecs_per_doc": 32,
    "n_queries": 100,
    "query_vecs": 16,
    "gem_params": {
        "n_fine": 64,
        "n_coarse": 16,
        "max_degree": 16,
        "ef_construction": 100,
        "max_kmeans_iter": 20,
        "ctop_r": 3,
    },
    "search_params": {
        "ef_values": [32, 64, 128],
        "n_probes_values": [2, 4, 8],
        "k_values": [1, 10, 100],
    },
}


# ── Data generation / loading ──────────────────────────────

def generate_synthetic_corpus(
    n_docs: int, vecs_per_doc: int, dim: int, seed: int = 42,
) -> Tuple[np.ndarray, List[int], List[Tuple[int, int]]]:
    rng = np.random.RandomState(seed)
    all_vectors = rng.randn(n_docs * vecs_per_doc, dim).astype(np.float32)
    doc_ids = list(range(n_docs))
    offsets = [(i * vecs_per_doc, (i + 1) * vecs_per_doc) for i in range(n_docs)]
    return all_vectors, doc_ids, offsets


def generate_synthetic_queries(
    n_queries: int, query_vecs: int, dim: int, seed: int = 99,
) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    return [
        rng.randn(query_vecs, dim).astype(np.float32)
        for _ in range(n_queries)
    ]


def load_real_corpus(
    data_dir: str, config: dict,
) -> Tuple[np.ndarray, List[int], List[Tuple[int, int]]]:
    """Load pre-computed embeddings from numpy files.

    Expected layout under data_dir/<documents path>:
        vectors.npy   – (total_tokens, dim) float32
        doc_ids.npy   – (n_docs,) int64
        offsets.npy   – (n_docs, 2) int64  (start, end per doc)
    """
    doc_path = Path(data_dir) / config["data_paths"]["documents"]
    all_vectors = np.load(doc_path / "vectors.npy").astype(np.float32)
    doc_ids = np.load(doc_path / "doc_ids.npy").tolist()
    offsets_arr = np.load(doc_path / "offsets.npy")
    offsets = [(int(s), int(e)) for s, e in offsets_arr]
    return all_vectors, doc_ids, offsets


def load_real_queries(data_dir: str, config: dict) -> List[np.ndarray]:
    """Load query embeddings.

    Expected layout: queries/ directory with query_vectors.npy (n_queries, query_vecs, dim).
    """
    q_path = Path(data_dir) / config["data_paths"]["queries"] / "query_vectors.npy"
    raw = np.load(q_path).astype(np.float32)
    return [raw[i] for i in range(raw.shape[0])]


# ── Brute-force MaxSim ground truth ───────────────────────

def brute_force_maxsim(
    all_vectors: np.ndarray,
    doc_offsets: List[Tuple[int, int]],
    query: np.ndarray,
    k: int,
) -> List[Tuple[int, float]]:
    """Exact top-k via MaxSim (lower distance = better).

    score(q, d) = 1 - mean_over_qi(max_over_dj(dot(qi, dj)))
    """
    scores: List[Tuple[int, float]] = []
    for doc_id, (start, end) in enumerate(doc_offsets):
        doc_vecs = all_vectors[start:end]
        sims = query @ doc_vecs.T          # (n_q_tokens, n_d_tokens)
        avg_max_sim = sims.max(axis=1).mean()
        scores.append((doc_id, 1.0 - avg_max_sim))

    scores.sort(key=lambda x: x[1])
    return scores[:k]


# ── Recall computation ────────────────────────────────────

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


# ── Percentile helper ─────────────────────────────────────

def percentile(data: List[float], p: int) -> float:
    s = sorted(data)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


# ── Build index ───────────────────────────────────────────

def build_index(
    all_vectors: np.ndarray,
    doc_ids: List[int],
    offsets: List[Tuple[int, int]],
    gem_params: dict,
):
    from latence_gem_index import GemSegment

    seg = GemSegment()
    t0 = time.perf_counter()
    seg.build(all_vectors, doc_ids, offsets, **gem_params)
    build_s = time.perf_counter() - t0
    return seg, build_s


# ── Sweep search parameters ──────────────────────────────

def sweep(
    seg,
    queries: List[np.ndarray],
    all_vectors: np.ndarray,
    offsets: List[Tuple[int, int]],
    search_params: dict,
) -> List[Dict[str, Any]]:
    ef_values = search_params["ef_values"]
    n_probes_values = search_params["n_probes_values"]
    k_values = search_params["k_values"]

    max_k = max(k_values)
    gt_cache: Dict[int, List[Tuple[int, float]]] = {}
    print("  Computing brute-force ground truth...", flush=True)
    for qi, q in enumerate(queries):
        gt_cache[qi] = brute_force_maxsim(all_vectors, offsets, q, max_k)

    rows: List[Dict[str, Any]] = []
    combos = list(product(ef_values, n_probes_values, k_values))
    total = len(combos)

    for idx, (ef, n_probes, k) in enumerate(combos, 1):
        print(f"  [{idx}/{total}] ef={ef}  n_probes={n_probes}  k={k}", flush=True)

        # Warmup
        for q in queries[:min(3, len(queries))]:
            seg.search(q, k=k, ef=ef, n_probes=n_probes)

        latencies_us: List[float] = []
        recall_vals: List[float] = []

        for qi, q in enumerate(queries):
            t0 = time.perf_counter()
            results = seg.search(q, k=k, ef=ef, n_probes=n_probes)
            latencies_us.append((time.perf_counter() - t0) * 1e6)

            gt = gt_cache[qi][:k]
            recall_vals.append(compute_recall(results, gt, k))

        rows.append({
            "ef": ef,
            "n_probes": n_probes,
            "k": k,
            "recall": round(statistics.mean(recall_vals), 6),
            "latency_p50_us": round(percentile(latencies_us, 50), 1),
            "latency_p95_us": round(percentile(latencies_us, 95), 1),
            "latency_p99_us": round(percentile(latencies_us, 99), 1),
        })

    return rows


# ── Output formatting ─────────────────────────────────────

def print_results_table(rows: List[Dict[str, Any]]):
    cols = [
        ("ef",       "ef",              "{:>4d}"),
        ("probes",   "n_probes",        "{:>6d}"),
        ("k",        "k",               "{:>5d}"),
        ("recall",   "recall",          "{:>8.4f}"),
        ("p50 (us)", "latency_p50_us",  "{:>10.1f}"),
        ("p95 (us)", "latency_p95_us",  "{:>10.1f}"),
        ("p99 (us)", "latency_p99_us",  "{:>10.1f}"),
    ]

    header = "  ".join(f"{label:>{len(fmt.format(0))}s}" for label, _, fmt in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = "  ".join(fmt.format(r[key]) for _, key, fmt in cols)
        print(line)


def print_summary(
    config_name: str,
    build_s: float,
    n_edges: int,
    n_docs: int,
    rows: List[Dict[str, Any]],
):
    print()
    print("=" * 78)
    print(f"SUMMARY — {config_name}")
    print("=" * 78)
    print(f"  Build time:   {build_s:.2f} s")
    print(f"  Index edges:  {n_edges:,}")
    print(f"  Documents:    {n_docs:,}")
    print()

    for k_val in sorted({r["k"] for r in rows}):
        subset = [r for r in rows if r["k"] == k_val]
        best = max(subset, key=lambda r: r["recall"])
        print(f"  Best recall@{k_val}: {best['recall']:.4f}  "
              f"(ef={best['ef']}, probes={best['n_probes']}, "
              f"p50={best['latency_p50_us']:.0f} us)")


# ── Main ───────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GEM Paper Reproduction Suite")
    p.add_argument("--mode", choices=["synthetic", "full"], default="synthetic",
                   help="synthetic = random data smoke test; full = real embeddings")
    p.add_argument("--config", type=str, default=None,
                   help="Path to JSON config (required for --mode full)")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Root directory for real data (required for --mode full)")
    p.add_argument("--n_docs", type=int, default=None,
                   help="Override n_docs for synthetic mode")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "full":
        if not args.config:
            raise SystemExit("--config is required for --mode full")
        if not args.data_dir:
            raise SystemExit("--data_dir is required for --mode full")

        with open(args.config) as f:
            config = json.load(f)

        config_name = config["name"]
        dim = config["dim"]
        gem_params = config["gem_params"]
        search_params = config["search_params"]

        print("=" * 78)
        print(f"GEM Paper Reproduction — {config_name}")
        print("=" * 78)
        print(f"  Config: {args.config}")
        print(f"  Data:   {args.data_dir}")
        print()

        print("Loading corpus...", flush=True)
        all_vectors, doc_ids, offsets = load_real_corpus(args.data_dir, config)
        print(f"  Documents: {len(doc_ids):,}  Vectors: {all_vectors.shape[0]:,}  Dim: {dim}")

        print("Loading queries...", flush=True)
        queries = load_real_queries(args.data_dir, config)
        print(f"  Queries: {len(queries)}")
        print()

    else:
        synth = dict(SYNTHETIC_DEFAULTS)
        if args.n_docs is not None:
            synth["n_docs"] = args.n_docs

        config_name = "synthetic"
        dim = synth["dim"]
        gem_params = synth["gem_params"]
        search_params = synth["search_params"]
        n_docs = synth["n_docs"]
        vecs_per_doc = synth["vecs_per_doc"]
        n_queries = synth["n_queries"]
        query_vecs = synth["query_vecs"]

        print("=" * 78)
        print("GEM Paper Reproduction — Synthetic Smoke Test")
        print("=" * 78)
        print(f"  n_docs={n_docs}  dim={dim}  vecs_per_doc={vecs_per_doc}")
        print(f"  n_queries={n_queries}  query_vecs={query_vecs}")
        print(f"  gem_params={gem_params}")
        print()

        print("Generating synthetic corpus...", flush=True)
        all_vectors, doc_ids, offsets = generate_synthetic_corpus(
            n_docs, vecs_per_doc, dim,
        )
        print(f"  Total vectors: {all_vectors.shape[0]:,}")

        print("Generating synthetic queries...", flush=True)
        queries = generate_synthetic_queries(n_queries, query_vecs, dim)
        print(f"  Queries: {len(queries)}")
        print()

    # Build
    print("Building GEM index...", flush=True)
    seg, build_s = build_index(all_vectors, doc_ids, offsets, gem_params)
    n_edges = seg.n_edges()
    n_docs_total = seg.n_docs()
    print(f"  Build time: {build_s:.2f} s")
    print(f"  Edges: {n_edges:,}   Documents: {n_docs_total:,}")
    print()

    # Parameter sweep
    print("Running parameter sweep...", flush=True)
    rows = sweep(seg, queries, all_vectors, offsets, search_params)
    print()

    # Print table
    print("=" * 78)
    print("DETAILED RESULTS")
    print("=" * 78)
    print_results_table(rows)
    print_summary(config_name, build_s, n_edges, n_docs_total, rows)

    # Save results
    results = {
        "config_name": config_name,
        "gem_params": gem_params,
        "search_params": search_params,
        "build_time_s": round(build_s, 4),
        "n_edges": n_edges,
        "n_docs": n_docs_total,
        "n_vectors": int(all_vectors.shape[0]),
        "n_queries": len(queries),
        "dim": dim,
        "sweep": rows,
    }

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = config_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out_path = out_dir / f"gem_paper_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
