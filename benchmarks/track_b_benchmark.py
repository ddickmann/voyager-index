"""
Track B Phase 2 Benchmark Suite
================================

Produces hard evidence comparing Python vs Rust shard engine components.

Micro-benchmarks:
  E. Proxy scoring: Python (numpy BLAS) vs Rust SIMD at various candidate counts
  F. WAL throughput: Rust WAL write + replay vs Python JSON
  G. Top-k merge: Python sorted vs numpy argpartition vs Rust heap
  H. Concurrent QPS: 1/2/4 threads (tests GIL release via proxy_score)
  I. SQLite filter vs dict scan (selective + bulk)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "shard_bench" / "results"
BASELINE_FILE = RESULTS_DIR / "bench_100000.jsonl"


def _load_baseline() -> Optional[dict]:
    if not BASELINE_FILE.exists():
        return None
    with open(BASELINE_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                if data.get("pipeline") == "shard_routed_lemur":
                    return data
    return None


def _percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0}
    arr = sorted(values)
    n = len(arr)
    return {
        "p50": arr[int(n * 0.5)],
        "p95": arr[min(int(n * 0.95), n - 1)],
        "p99": arr[min(int(n * 0.99), n - 1)],
        "mean": statistics.mean(arr),
    }


# ======================================================================
# E: Proxy Scoring — Python numpy vs Rust SIMD
# ======================================================================

def bench_proxy_scoring(n_docs: int = 100_000, dim: int = 128, n_q_tokens: int = 32, n_full: int = 4096, n_iter: int = 100) -> dict:
    log.info("=== Micro E: Proxy Scoring (n_docs=%d, dim=%d, n_q=%d) ===", n_docs, dim, n_q_tokens)
    results: Dict[str, Any] = {"benchmark": "E_proxy_scoring", "n_docs": n_docs, "dim": dim, "n_q_tokens": n_q_tokens}

    rng = np.random.default_rng(42)
    query = rng.standard_normal((n_q_tokens, dim)).astype(np.float32)
    doc_means = rng.standard_normal((n_docs, dim)).astype(np.float32)
    doc_ids_arr = np.arange(n_docs, dtype=np.uint64)

    # --- Python numpy path (uses BLAS matmul) ---
    def python_proxy_score():
        scores = (query @ doc_means.T).max(axis=0)
        top_idx = np.argpartition(-scores, n_full)[:n_full]
        return doc_ids_arr[top_idx[np.argsort(-scores[top_idx])]]

    for _ in range(3):
        python_proxy_score()

    py_times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        python_proxy_score()
        py_times.append((time.perf_counter() - t0) * 1000)

    results["python_ms"] = _percentiles(py_times)
    log.info("  Python (numpy BLAS): p50=%.3fms p95=%.3fms", results["python_ms"]["p50"], results["python_ms"]["p95"])

    # --- Rust SIMD path ---
    try:
        from latence_shard_engine import ShardIndex
        has_rust = True
    except ImportError:
        log.warning("  latence_shard_engine not importable — skipping Rust proxy")
        has_rust = False

    if has_rust:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            idx = ShardIndex(td, dim)
            idx.set_doc_means(doc_means, doc_ids_arr.tolist())

            for _ in range(5):
                idx.proxy_score(query, n_full=n_full)

            rust_times = []
            for _ in range(n_iter):
                t0 = time.perf_counter()
                idx.proxy_score(query, n_full=n_full)
                rust_times.append((time.perf_counter() - t0) * 1000)

            results["rust_ms"] = _percentiles(rust_times)
            log.info("  Rust SIMD: p50=%.3fms p95=%.3fms", results["rust_ms"]["p50"], results["rust_ms"]["p95"])

            py_p50 = results["python_ms"]["p50"]
            rs_p50 = results["rust_ms"]["p50"]
            if py_p50 > 0 and rs_p50 > 0:
                results["speedup"] = py_p50 / rs_p50
                log.info("  Speedup: %.2fx", results["speedup"])

            # Also test with smaller candidate sets (realistic LEMUR scenario)
            for n_cand in [5000, 10000]:
                candidates = rng.choice(doc_ids_arr, size=n_cand, replace=False).tolist()
                cand_idx = np.array(candidates, dtype=np.intp)

                # Pre-extract sub-means for Python benchmark (avoid re-indexing overhead)
                sub_means = doc_means[cand_idx].copy()
                top_n = min(n_full, n_cand)

                # Warmup both paths separately
                for _ in range(5):
                    idx.proxy_score(query, candidate_ids=candidates, n_full=top_n)
                for _ in range(5):
                    sc = (query @ sub_means.T).max(axis=0)
                    np.argpartition(-sc, top_n)[:top_n]

                # Measure Rust
                cand_rust_times = []
                for _ in range(n_iter):
                    t0 = time.perf_counter()
                    idx.proxy_score(query, candidate_ids=candidates, n_full=top_n)
                    cand_rust_times.append((time.perf_counter() - t0) * 1000)

                # Measure Python (separate loop to avoid cache interference)
                cand_py_times = []
                for _ in range(n_iter):
                    t0 = time.perf_counter()
                    sub_scores = (query @ sub_means.T).max(axis=0)
                    np.argpartition(-sub_scores, top_n)[:top_n]
                    cand_py_times.append((time.perf_counter() - t0) * 1000)

                rs_stats = _percentiles(cand_rust_times)
                py_stats = _percentiles(cand_py_times)
                sp = py_stats["p50"] / rs_stats["p50"] if rs_stats["p50"] > 0 else 0
                results[f"rust_{n_cand}_ms"] = rs_stats
                results[f"python_{n_cand}_ms"] = py_stats
                results[f"speedup_{n_cand}"] = sp
                log.info("  %d candidates: Rust=%.3fms Python=%.3fms (%.2fx)", n_cand, rs_stats["p50"], py_stats["p50"], sp)

            idx.close()

    return results


# ======================================================================
# F: WAL Throughput
# ======================================================================

def bench_wal_throughput(n_inserts: int = 10_000, dim: int = 128) -> dict:
    log.info("=== Micro F: WAL Throughput (n_inserts=%d) ===", n_inserts)
    results: Dict[str, Any] = {"benchmark": "F_wal_throughput", "n_inserts": n_inserts}

    rng = np.random.default_rng(42)

    try:
        from latence_shard_engine import ShardIndex
        has_rust = True
    except ImportError:
        log.warning("  latence_shard_engine not importable")
        has_rust = False

    if has_rust:
        import tempfile
        embeddings_list = [rng.standard_normal((3, dim)).astype(np.float32) for _ in range(n_inserts)]

        # Rust WAL writes (includes state management, which is real production cost)
        with tempfile.TemporaryDirectory() as td:
            idx = ShardIndex(td, dim)
            t0 = time.perf_counter()
            for i in range(n_inserts):
                idx.insert(i, embeddings_list[i], json.dumps({"i": i}))
            idx.sync()
            insert_ms = (time.perf_counter() - t0) * 1000
            results["rust_insert_total_ms"] = insert_ms
            results["rust_insert_ops_per_sec"] = n_inserts / (insert_ms / 1000)
            log.info("  Rust insert+state: %d inserts in %.1fms (%.0f ops/s)", n_inserts, insert_ms, results["rust_insert_ops_per_sec"])
            idx.close()

            # Replay benchmark
            t0 = time.perf_counter()
            idx2 = ShardIndex(td, dim)
            replay_ms = (time.perf_counter() - t0) * 1000
            results["rust_replay_ms"] = replay_ms
            results["rust_replay_doc_count"] = idx2.doc_count
            log.info("  Rust WAL replay: %.1fms (%d docs)", replay_ms, idx2.doc_count)
            idx2.close()

    # Python JSON WAL (fair comparison: JSON serialization + write + fsync)
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        fpath = os.path.join(td, "wal.jsonl")
        payloads = [json.dumps({"i": i}) for i in range(n_inserts)]
        vecs = [rng.standard_normal((3, dim)).astype(np.float32) for _ in range(n_inserts)]

        t0 = time.perf_counter()
        with open(fpath, "w") as f:
            for i in range(n_inserts):
                f.write(json.dumps({"op": "insert", "id": i, "payload": payloads[i], "n_vecs": 3}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        py_write_ms = (time.perf_counter() - t0) * 1000
        results["python_wal_write_ms"] = py_write_ms
        log.info("  Python JSON WAL write: %.1fms (%.0f ops/s)", py_write_ms, n_inserts / (py_write_ms / 1000))

        # Python replay (JSON parse)
        t0 = time.perf_counter()
        with open(fpath) as f:
            replayed = [json.loads(line) for line in f]
        py_replay_ms = (time.perf_counter() - t0) * 1000
        results["python_replay_ms"] = py_replay_ms
        log.info("  Python JSON replay: %.1fms (%d entries)", py_replay_ms, len(replayed))

    if "rust_replay_ms" in results and results["python_replay_ms"] > 0:
        results["replay_speedup"] = results["python_replay_ms"] / results["rust_replay_ms"]
        log.info("  Replay speedup: %.1fx", results["replay_speedup"])

    return results


# ======================================================================
# G: Top-k Merge
# ======================================================================

def bench_topk(n_scored: int = 100_000, k: int = 100, n_iter: int = 100) -> dict:
    log.info("=== Micro G: Top-k (%d scored docs, k=%d) ===", n_scored, k)
    results: Dict[str, Any] = {"benchmark": "G_topk", "n_scored": n_scored, "k": k}

    rng = np.random.default_rng(42)
    scores = rng.standard_normal(n_scored).astype(np.float32)
    doc_ids = np.arange(n_scored, dtype=np.uint64)

    def python_sorted_topk():
        paired = sorted(zip(scores.tolist(), doc_ids.tolist()), key=lambda x: -x[0])
        return paired[:k]

    python_sorted_topk()
    py_times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        python_sorted_topk()
        py_times.append((time.perf_counter() - t0) * 1000)
    results["python_sorted_ms"] = _percentiles(py_times)
    log.info("  Python sorted: p50=%.3fms", results["python_sorted_ms"]["p50"])

    def numpy_topk():
        top_idx = np.argpartition(-scores, k)[:k]
        return top_idx[np.argsort(-scores[top_idx])]

    numpy_topk()
    np_times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        numpy_topk()
        np_times.append((time.perf_counter() - t0) * 1000)
    results["numpy_ms"] = _percentiles(np_times)
    log.info("  Numpy argpartition: p50=%.3fms", results["numpy_ms"]["p50"])

    return results


# ======================================================================
# H: Concurrent QPS (GIL release via proxy_score)
# ======================================================================

def bench_concurrent_qps(n_queries: int = 500, n_docs: int = 50_000, dim: int = 128) -> dict:
    log.info("=== Micro H: Concurrent QPS (n_docs=%d) ===", n_docs)
    results: Dict[str, Any] = {"benchmark": "H_concurrent_qps", "n_docs": n_docs}

    try:
        from latence_shard_engine import ShardIndex
    except ImportError:
        log.warning("  latence_shard_engine not importable")
        return results

    import tempfile
    rng = np.random.default_rng(42)

    with tempfile.TemporaryDirectory() as td:
        idx = ShardIndex(td, dim)
        doc_means = rng.standard_normal((n_docs, dim)).astype(np.float32)
        doc_ids_list = list(range(n_docs))
        idx.set_doc_means(doc_means, doc_ids_list)

        query = rng.standard_normal((32, dim)).astype(np.float32)

        def do_proxy_score(_):
            idx.proxy_score(query, n_full=1000)

        # Warmup
        for _ in range(5):
            do_proxy_score(0)

        for n_threads in [1, 2, 4]:
            t0 = time.perf_counter()
            if n_threads == 1:
                for i in range(n_queries):
                    do_proxy_score(i)
            else:
                with ThreadPoolExecutor(max_workers=n_threads) as pool:
                    list(pool.map(do_proxy_score, range(n_queries)))
            elapsed = time.perf_counter() - t0

            qps = n_queries / elapsed
            results[f"threads_{n_threads}_qps"] = qps
            results[f"threads_{n_threads}_elapsed_s"] = elapsed
            log.info("  %d thread(s): %.1f QPS (%.2fs for %d queries)", n_threads, qps, elapsed, n_queries)

        t1 = results.get("threads_1_qps", 0)
        if t1 > 0:
            for n in [2, 4]:
                results[f"threads_{n}_scaling"] = results.get(f"threads_{n}_qps", 0) / t1

        idx.close()

    return results


# ======================================================================
# I: SQLite Filter vs Dict Scan
# ======================================================================

def bench_sqlite_filter(n_docs: int = 100_000) -> dict:
    log.info("=== Micro I: SQLite Filter (n_docs=%d) ===", n_docs)
    results: Dict[str, Any] = {"benchmark": "I_sqlite_filter", "n_docs": n_docs}

    rng = np.random.default_rng(42)
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "white", "black", "gray"]
    categories = [f"cat_{i}" for i in range(100)]

    payloads = {}
    for i in range(n_docs):
        payloads[i] = {"color": colors[i % len(colors)], "category": categories[i % len(categories)], "score": int(rng.integers(0, 1000))}

    n_iter = 100

    # ---- Selective filter: 1% of docs (1 category out of 100) ----
    target_cat = "cat_42"
    def dict_filter_selective():
        return [did for did, p in payloads.items() if p.get("category") == target_cat]

    dict_filter_selective()
    py_sel_times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        py_sel_result = dict_filter_selective()
        py_sel_times.append((time.perf_counter() - t0) * 1000)
    results["python_selective_ms"] = _percentiles(py_sel_times)
    results["python_selective_n"] = len(py_sel_result)
    log.info("  Python selective (1%%): p50=%.3fms (%d results)", results["python_selective_ms"]["p50"], len(py_sel_result))

    # ---- Broad filter: 10% of docs (1 color out of 10) ----
    def dict_filter_broad():
        return [did for did, p in payloads.items() if p.get("color") == "red"]

    dict_filter_broad()
    py_broad_times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        py_broad_result = dict_filter_broad()
        py_broad_times.append((time.perf_counter() - t0) * 1000)
    results["python_broad_ms"] = _percentiles(py_broad_times)
    results["python_broad_n"] = len(py_broad_result)
    log.info("  Python broad (10%%): p50=%.3fms (%d results)", results["python_broad_ms"]["p50"], len(py_broad_result))

    # ---- SQLite ----
    try:
        from latence_shard_engine import MetadataStore
        has_rust = True
    except ImportError:
        log.warning("  MetadataStore not importable")
        has_rust = False

    if has_rust:
        store = MetadataStore(":memory:")

        t0 = time.perf_counter()
        items = [(i, json.dumps(payloads[i]), f"doc {i} {payloads[i]['color']} {payloads[i]['category']}") for i in range(n_docs)]
        store.set_payloads_bulk(items)
        load_ms = (time.perf_counter() - t0) * 1000
        results["sqlite_load_ms"] = load_ms
        log.info("  SQLite bulk load: %.1fms", load_ms)

        store.create_field_index("category")
        store.create_field_index("color")
        log.info("  Created expression indexes")

        # Selective filter
        sq_sel_times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            sq_sel_result = store.filter_by_field("category", target_cat, limit=n_docs)
            sq_sel_times.append((time.perf_counter() - t0) * 1000)
        results["sqlite_selective_ms"] = _percentiles(sq_sel_times)
        results["sqlite_selective_n"] = len(sq_sel_result)
        if results["python_selective_ms"]["p50"] > 0 and results["sqlite_selective_ms"]["p50"] > 0:
            results["selective_speedup"] = results["python_selective_ms"]["p50"] / results["sqlite_selective_ms"]["p50"]
        log.info("  SQLite selective (1%%): p50=%.3fms (%d results) speedup=%.1fx",
                 results["sqlite_selective_ms"]["p50"], len(sq_sel_result), results.get("selective_speedup", 0))

        # Broad filter
        sq_broad_times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            sq_broad_result = store.filter_by_field("color", "red", limit=n_docs)
            sq_broad_times.append((time.perf_counter() - t0) * 1000)
        results["sqlite_broad_ms"] = _percentiles(sq_broad_times)
        results["sqlite_broad_n"] = len(sq_broad_result)
        if results["python_broad_ms"]["p50"] > 0 and results["sqlite_broad_ms"]["p50"] > 0:
            results["broad_speedup"] = results["python_broad_ms"]["p50"] / results["sqlite_broad_ms"]["p50"]
        log.info("  SQLite broad (10%%): p50=%.3fms (%d results) speedup=%.1fx",
                 results["sqlite_broad_ms"]["p50"], len(sq_broad_result), results.get("broad_speedup", 0))

        # FTS5 benchmark
        fts_times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            fts_result = store.search_text("red", limit=1000)
            fts_times.append((time.perf_counter() - t0) * 1000)
        results["sqlite_fts_ms"] = _percentiles(fts_times)
        results["sqlite_fts_n"] = len(fts_result)
        log.info("  SQLite FTS5: p50=%.3fms (%d results)", results["sqlite_fts_ms"]["p50"], len(fts_result))

    return results


# ======================================================================
# Summary table
# ======================================================================

def _print_summary(baseline: Optional[dict], micro_results: List[dict]) -> str:
    lines = []
    w = 90
    lines.append("")
    lines.append("=" * w)
    lines.append("TRACK B PHASE 2 — BENCHMARK RESULTS")
    lines.append("=" * w)

    if baseline:
        lines.append("")
        lines.append("End-to-End 100K Baseline (Python + GPU corpus, April 12):")
        lines.append(f"  p50={baseline.get('p50_total_ms', 0):.2f}ms  "
                     f"p95={baseline.get('p95_total_ms', 0):.2f}ms  "
                     f"QPS={baseline.get('qps', 0):.1f}  "
                     f"R@10={baseline.get('recall_at_10', 0):.3f}  "
                     f"R@100={baseline.get('recall_at_100', 0):.3f}")

    lines.append("")
    lines.append("-" * w)
    lines.append("COMPONENT MICRO-BENCHMARKS")
    lines.append("-" * w)

    for r in micro_results:
        bname = r.get("benchmark", "?")
        lines.append("")
        lines.append(f"  [{bname}]")

        if bname == "E_proxy_scoring":
            py = r.get("python_ms", {})
            rs = r.get("rust_ms", {})
            lines.append(f"    100K docs (all): Python(numpy BLAS)={py.get('p50',0):.3f}ms  Rust(SIMD)={rs.get('p50',0):.3f}ms  "
                         f"({'%.1fx' % r['speedup'] if 'speedup' in r else 'N/A'})")
            for nc in [5000, 10000]:
                rk = f"rust_{nc}_ms"
                pk = f"python_{nc}_ms"
                sk = f"speedup_{nc}"
                if rk in r:
                    lines.append(f"    {nc:,} candidates:  Python={r[pk].get('p50',0):.3f}ms  Rust={r[rk].get('p50',0):.3f}ms  "
                                 f"({r.get(sk,0):.1f}x)")

        elif bname == "F_wal_throughput":
            lines.append(f"    Rust WAL: {r.get('rust_insert_total_ms',0):.0f}ms for {r.get('n_inserts',0)} inserts "
                         f"({r.get('rust_insert_ops_per_sec',0):.0f} ops/s)")
            lines.append(f"    Rust replay: {r.get('rust_replay_ms',0):.1f}ms → {r.get('rust_replay_doc_count',0)} docs")
            lines.append(f"    Python JSON write: {r.get('python_wal_write_ms',0):.0f}ms")
            lines.append(f"    Python JSON replay: {r.get('python_replay_ms',0):.1f}ms")
            if "replay_speedup" in r:
                lines.append(f"    Replay speedup: {r['replay_speedup']:.1f}x")

        elif bname == "G_topk":
            py = r.get("python_sorted_ms", {})
            np_ = r.get("numpy_ms", {})
            lines.append(f"    Python sorted: p50={py.get('p50',0):.3f}ms")
            lines.append(f"    Numpy argpartition: p50={np_.get('p50',0):.3f}ms")
            if py.get("p50", 0) > 0 and np_.get("p50", 0) > 0:
                lines.append(f"    Numpy speedup: {py['p50']/np_['p50']:.0f}x over sorted")

        elif bname == "H_concurrent_qps":
            for n in [1, 2, 4]:
                qps = r.get(f"threads_{n}_qps", 0)
                if n == 1:
                    lines.append(f"    {n} thread:  {qps:.1f} QPS")
                else:
                    sc = r.get(f"threads_{n}_scaling", 0)
                    lines.append(f"    {n} threads: {qps:.1f} QPS ({sc:.2f}x linear)")

        elif bname == "I_sqlite_filter":
            lines.append(f"    Selective (1% match):")
            lines.append(f"      Python dict: {r.get('python_selective_ms',{}).get('p50',0):.3f}ms ({r.get('python_selective_n',0)} results)")
            if "sqlite_selective_ms" in r:
                lines.append(f"      SQLite idx:  {r['sqlite_selective_ms'].get('p50',0):.3f}ms ({r.get('sqlite_selective_n',0)} results) "
                             f"({r.get('selective_speedup',0):.1f}x)")
            lines.append(f"    Broad (10% match):")
            lines.append(f"      Python dict: {r.get('python_broad_ms',{}).get('p50',0):.3f}ms ({r.get('python_broad_n',0)} results)")
            if "sqlite_broad_ms" in r:
                lines.append(f"      SQLite idx:  {r['sqlite_broad_ms'].get('p50',0):.3f}ms ({r.get('sqlite_broad_n',0)} results) "
                             f"({r.get('broad_speedup',0):.1f}x)")
            if "sqlite_fts_ms" in r:
                lines.append(f"    FTS5 keyword: {r['sqlite_fts_ms'].get('p50',0):.3f}ms ({r.get('sqlite_fts_n',0)} results)")

    lines.append("")
    lines.append("=" * w)
    output = "\n".join(lines)
    print(output)
    return output


def main():
    parser = argparse.ArgumentParser(description="Track B Phase 2 Benchmark Suite")
    parser.add_argument("--n-docs-proxy", type=int, default=100_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "track_b_results.jsonl"))
    args = parser.parse_args()

    baseline = _load_baseline()
    if baseline:
        log.info("Baseline: p50=%.2fms QPS=%.1f R@10=%.3f",
                 baseline.get("p50_total_ms", 0), baseline.get("qps", 0), baseline.get("recall_at_10", 0))

    all_results: List[dict] = []

    log.info("=" * 60)
    log.info("TRACK B MICRO-BENCHMARKS")
    log.info("=" * 60)

    all_results.append(bench_proxy_scoring(n_docs=args.n_docs_proxy, dim=args.dim))
    all_results.append(bench_wal_throughput())
    all_results.append(bench_topk())
    all_results.append(bench_concurrent_qps())
    all_results.append(bench_sqlite_filter())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")
    log.info("Results written to %s", output_path)

    _print_summary(baseline, all_results)


if __name__ == "__main__":
    main()
