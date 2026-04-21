"""
Latency Opportunity Experiments for the stable 100k path.

Four isolated experiments run in sequence, sharing the same loaded index:

  Exp-1  Phase Accounting + Concurrency Sweep
         Breaks latency into routing / exact-scoring stages and measures
         throughput at 1, 4, 8, and 16 concurrent workers.

  Exp-2  Frozen-Candidate Oracle
         Freezes routed candidate IDs, then replays exact scoring only.
         Tells you the maximum gain available by eliminating routing cost.

  Exp-3  Exact-Shape Microbench
         Calls score_all_docs_topk() directly with uniform lengths, the
         legacy padded fallback, and the current bucketed fallback under
         the same token budget.
         Tells you whether bucketing removes the old variable-length penalty.

  Exp-4  Cache and Residency Upper-Bound
         Measures cold vs warm shard fetch, and candidate-cache replay.
         Tells you whether reducing data-movement is high-leverage.

Usage:
  python benchmarks/latency_opportunity_bench.py [--device cpu|cuda] [--k-candidates 2000]

Each experiment prints a summary section. A final ranking table tells you
which opportunity clears the productization threshold (>=1.3x QPS or >=25%
p50 reduction at the same recall).
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colsearch._internal.inference.shard_engine.config import (
    AnnBackend,
    Compression,
    RouterType,
    StorageLayout,
    TransferMode,
)
from colsearch._internal.inference.shard_engine.fetch_pipeline import FetchPipeline
from colsearch._internal.inference.shard_engine.manager import (
    ShardEngineConfig,
    ShardSegmentManager,
)
from colsearch._internal.inference.shard_engine.profiler import Timer
from colsearch._internal.inference.shard_engine.scorer import (
    score_all_docs_topk,
    warmup_maxsim,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument("--index-dir", default="/workspace/.cache/shard-bench/index_100000_fp16_proxy_grouped_lemur_uniform")
_p.add_argument("--npz",       default="/workspace/.cache/voyager-qa/beir_100k.npz")
_p.add_argument("--gt",        default="/workspace/.cache/shard-bench/gt_100000.npz")
_p.add_argument("--n-eval",    type=int, default=100)
_p.add_argument("--n-warmup",  type=int, default=10)
_p.add_argument("--k",         type=int, default=10)
_p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
_p.add_argument("--k-candidates", type=int, default=2000)
_p.add_argument("--max-docs-exact", type=int, default=0, help="0 = same as k-candidates")
_p.add_argument("--n-shards",  type=int, default=256)
_p.add_argument("--workers",   type=int, nargs="*", default=[1, 4, 8, 16])
_p.add_argument("--exp",       nargs="*", default=["all"],
                choices=["all", "1", "2", "3", "4"],
                help="Which experiments to run (default: all)")
_args = _p.parse_args()

INDEX_DIR = Path(_args.index_dir)
NPZ_PATH  = Path(_args.npz)
GT_CACHE  = Path(_args.gt)
N_EVAL    = _args.n_eval
N_WARMUP  = _args.n_warmup
K         = _args.k
DEVICE    = _args.device
K_CAND    = _args.k_candidates
MAX_EXACT = _args.max_docs_exact or K_CAND
WORKER_COUNTS = _args.workers
RUN_EXP   = set(_args.exp)
if "all" in RUN_EXP:
    RUN_EXP = {"1", "2", "3", "4"}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_queries(n: int) -> List[np.ndarray]:
    data = np.load(str(NPZ_PATH), allow_pickle=True)
    q_vecs    = data["query_vectors"].astype(np.float32)
    q_offsets = data["query_offsets"]
    queries   = []
    for i in range(min(n, len(q_offsets))):
        s, e = int(q_offsets[i, 0]), int(q_offsets[i, 1])
        queries.append(q_vecs[s:e])
    return queries


def load_ground_truth(n: int) -> List[List[int]]:
    data = np.load(str(GT_CACHE), allow_pickle=True)
    return [row[row >= 0].tolist() for row in data["gt_ids"][:n]]


def recall_at_k(preds: List[List[int]], gts: List[List[int]], k: int) -> float:
    scores = []
    for pred, gt in zip(preds, gts):
        gt_set = set(gt[:k])
        if gt_set:
            scores.append(len(set(pred[:k]) & gt_set) / len(gt_set))
    return float(np.mean(scores)) if scores else 0.0


def build_manager(n_centroid_approx: int = 0) -> ShardSegmentManager:
    cfg = ShardEngineConfig(
        dim=128,
        n_shards=_args.n_shards,
        compression=Compression.FP16,
        layout=StorageLayout.PROXY_GROUPED,
        router_type=RouterType.LEMUR,
        ann_backend=AnnBackend.FAISS_FLAT_IP,
        k_candidates=K_CAND,
        max_docs_exact=MAX_EXACT,
        lemur_search_k_cap=2048,
        n_full_scores=4096,
        n_centroid_approx=n_centroid_approx,
        transfer_mode=TransferMode.PINNED,
        router_device="cpu",
    )
    mgr = ShardSegmentManager(INDEX_DIR, config=cfg, device=DEVICE)
    mgr.load()
    return mgr

# ---------------------------------------------------------------------------
# Shared search primitive that returns stage breakdowns
# ---------------------------------------------------------------------------

def timed_search(mgr: ShardSegmentManager, q: np.ndarray) -> dict:
    """Return the full per-stage trace for one query."""
    return mgr.inspect_query_pipeline(q, k=K)


def serial_eval(
    mgr: ShardSegmentManager,
    queries: List[np.ndarray],
    gts: List[List[int]],
    label: str,
) -> dict:
    all_results = []
    route_ms_arr = []
    prune_ms_arr = []
    fetch_ms_arr = []
    exact_ms_arr = []
    maxsim_ms_arr = []
    topk_ms_arr = []
    total_ms_arr = []
    for q in queries[:N_WARMUP]:
        mgr.inspect_query_pipeline(q, k=K)
    for q, _gt in zip(queries[:N_EVAL], gts[:N_EVAL]):
        trace = timed_search(mgr, q)
        all_results.append(trace["result_ids"])
        route_ms_arr.append(float(trace["route_ms"]))
        prune_ms_arr.append(float(trace.get("prune_ms", 0.0)))
        fetch_ms_arr.append(float(trace.get("fetch_ms", 0.0)))
        exact_ms_arr.append(float(trace.get("exact_ms", 0.0)))
        maxsim_ms_arr.append(float(trace.get("maxsim_ms", 0.0)))
        topk_ms_arr.append(float(trace.get("topk_ms", 0.0)))
        total_ms_arr.append(float(trace.get("total_ms", 0.0)))

    recall = recall_at_k(all_results, gts[:N_EVAL], K)
    total_arr = np.array(total_ms_arr)
    route_arr = np.array(route_ms_arr)
    prune_arr = np.array(prune_ms_arr)
    fetch_arr = np.array(fetch_ms_arr)
    exact_arr = np.array(exact_ms_arr)
    maxsim_arr = np.array(maxsim_ms_arr)
    topk_arr = np.array(topk_ms_arr)
    qps = N_EVAL / (total_arr.sum() / 1000.0)
    return {
        "label": label,
        "recall@10": round(recall, 4),
        "qps": round(qps, 1),
        "p50_ms":  round(float(np.percentile(total_arr, 50)), 2),
        "p95_ms":  round(float(np.percentile(total_arr, 95)), 2),
        "route_p50_ms": round(float(np.percentile(route_arr, 50)), 2),
        "prune_p50_ms": round(float(np.percentile(prune_arr, 50)), 2),
        "fetch_p50_ms": round(float(np.percentile(fetch_arr, 50)), 2),
        "exact_p50_ms": round(float(np.percentile(exact_arr, 50)), 2),
        "maxsim_p50_ms": round(float(np.percentile(maxsim_arr, 50)), 2),
        "topk_p50_ms": round(float(np.percentile(topk_arr, 50)), 2),
        "route_frac":   round(float(np.mean(route_arr) / np.mean(total_arr)), 3),
        "prune_frac":   round(float(np.mean(prune_arr) / np.mean(total_arr)), 3),
        "fetch_frac":   round(float(np.mean(fetch_arr) / np.mean(total_arr)), 3),
        "exact_frac":   round(float(np.mean(exact_arr) / np.mean(total_arr)), 3),
    }

# ============================================================================
# EXP 1: Phase Accounting + Concurrency Sweep
# ============================================================================

def exp1_phase_and_concurrency(
    mgr: ShardSegmentManager,
    queries: List[np.ndarray],
    gts: List[List[int]],
) -> dict:
    log.info("=== EXP 1: Phase Accounting + Concurrency Sweep ===")

    # Single-threaded breakdown first
    baseline = serial_eval(mgr, queries, gts, "serial_baseline")
    log.info(
        "Serial  recall=%.4f  QPS=%.1f  p50=%.1fms  route=%.1fms  prune=%.1fms  "
        "fetch=%.1fms  exact=%.1fms  maxsim=%.1fms  topk=%.1fms",
        baseline["recall@10"], baseline["qps"], baseline["p50_ms"],
        baseline["route_p50_ms"], baseline["prune_p50_ms"],
        baseline["fetch_p50_ms"], baseline["exact_p50_ms"],
        baseline["maxsim_p50_ms"], baseline["topk_p50_ms"],
    )

    # Concurrency sweep
    concurrency_results = []
    for n_workers in WORKER_COUNTS:
        latencies: List[float] = []
        results_list: List[List[int]] = [[] for _ in range(N_EVAL)]
        route_ms: List[float] = []
        prune_ms: List[float] = []
        fetch_ms: List[float] = []
        exact_ms: List[float] = []
        maxsim_ms: List[float] = []

        # Warmup
        for q in queries[:N_WARMUP]:
            mgr.inspect_query_pipeline(q, k=K)

        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            idx_query = list(enumerate(queries[:N_EVAL]))
            future_map = {
                pool.submit(mgr.inspect_query_pipeline, q, K): (i, q)
                for i, q in idx_query
            }
            t_query_start = {f: time.perf_counter() for f in future_map}
            for future in as_completed(future_map):
                i, _q = future_map[future]
                t_end = time.perf_counter()
                trace = future.result()
                results_list[i] = trace["result_ids"]
                latencies.append((t_end - t_query_start[future]) * 1000.0)
                route_ms.append(float(trace["route_ms"]))
                prune_ms.append(float(trace.get("prune_ms", 0.0)))
                fetch_ms.append(float(trace.get("fetch_ms", 0.0)))
                exact_ms.append(float(trace.get("exact_ms", 0.0)))
                maxsim_ms.append(float(trace.get("maxsim_ms", 0.0)))
        wall_elapsed = time.perf_counter() - wall_start

        throughput = N_EVAL / wall_elapsed
        lat_arr = np.array(latencies)
        route_arr = np.array(route_ms)
        prune_arr = np.array(prune_ms)
        fetch_arr = np.array(fetch_ms)
        exact_arr = np.array(exact_ms)
        maxsim_arr = np.array(maxsim_ms)
        r = recall_at_k(results_list, gts[:N_EVAL], K)
        row = {
            "n_workers": n_workers,
            "recall@10": round(r, 4),
            "throughput_qps": round(throughput, 1),
            "p50_ms": round(float(np.percentile(lat_arr, 50)), 2),
            "p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
            "p99_ms": round(float(np.percentile(lat_arr, 99)), 2),
            "route_p50_ms": round(float(np.percentile(route_arr, 50)), 2),
            "prune_p50_ms": round(float(np.percentile(prune_arr, 50)), 2),
            "fetch_p50_ms": round(float(np.percentile(fetch_arr, 50)), 2),
            "exact_p50_ms": round(float(np.percentile(exact_arr, 50)), 2),
            "maxsim_p50_ms": round(float(np.percentile(maxsim_arr, 50)), 2),
        }
        concurrency_results.append(row)
        log.info(
            "  Workers=%2d  recall=%.4f  QPS=%6.1f  p50=%6.1fms  route=%5.1f  prune=%5.1f  fetch=%5.1f  exact=%5.1f",
            n_workers, r, throughput,
            float(np.percentile(lat_arr, 50)),
            float(np.percentile(route_arr, 50)),
            float(np.percentile(prune_arr, 50)),
            float(np.percentile(fetch_arr, 50)),
            float(np.percentile(exact_arr, 50)),
        )

    return {"baseline": baseline, "concurrency": concurrency_results}


# ============================================================================
# EXP 2: Frozen-Candidate Oracle
# ============================================================================

def exp2_frozen_candidate_oracle(
    mgr: ShardSegmentManager,
    queries: List[np.ndarray],
    gts: List[List[int]],
) -> dict:
    log.info("=== EXP 2: Frozen-Candidate Oracle ===")

    # Step A: collect frozen candidate IDs for each query (full pipeline warmup)
    for q in queries[:N_WARMUP]:
        mgr.inspect_query_pipeline(q, k=K)

    frozen_candidates: List[List[int]] = []
    full_route_ms: List[float] = []
    full_total_ms: List[float] = []

    for q in queries[:N_EVAL]:
        trace = mgr.inspect_query_pipeline(q, k=K)
        frozen_candidates.append(trace["exact_candidate_ids"])
        full_route_ms.append(float(trace["route_ms"]))
        full_total_ms.append(float(trace.get("total_ms", 0.0)))

    full_route_arr = np.array(full_route_ms)
    full_total_arr = np.array(full_total_ms)

    log.info(
        "Full pipeline: route_p50=%.1fms  total_p50=%.1fms  route%%=%.0f%%",
        float(np.percentile(full_route_arr, 50)),
        float(np.percentile(full_total_arr, 50)),
        100.0 * float(np.mean(full_route_arr)) / float(np.mean(full_total_arr)),
    )

    # Step B: replay exact scoring only, no routing
    rust_idx = mgr._rust_index
    replay_ms: List[float] = []
    replay_results: List[List[int]] = []

    for q, cands in zip(queries[:N_EVAL], frozen_candidates):
        if not cands:
            replay_results.append([])
            replay_ms.append(0.0)
            continue
        q_np = q.astype(np.float32)
        with Timer(sync_cuda=False) as t:
            if rust_idx is not None:
                try:
                    ids_arr, _ = rust_idx.score_candidates_exact(q_np, cands, K)
                    result_ids = ids_arr.tolist()
                except Exception:
                    result_ids = []
            else:
                result_ids = []
        replay_ms.append(t.elapsed_ms)
        replay_results.append(result_ids)

    replay_arr = np.array(replay_ms)
    recall_full   = recall_at_k([mgr.inspect_query_pipeline(q, k=K)["result_ids"] for q in queries[:N_EVAL]], gts[:N_EVAL], K) \
        if N_EVAL <= 20 else None  # skip re-run for large N
    recall_replay = recall_at_k(replay_results, gts[:N_EVAL], K)

    speedup_potential = float(np.mean(full_total_arr)) / max(float(np.mean(replay_arr)), 0.01)

    log.info(
        "Frozen replay: exact_p50=%.1fms  exact_p95=%.1fms  replay_recall=%.4f  "
        "speedup_potential=%.1fx (if routing were free)",
        float(np.percentile(replay_arr, 50)),
        float(np.percentile(replay_arr, 95)),
        recall_replay,
        speedup_potential,
    )

    return {
        "full_route_p50_ms": round(float(np.percentile(full_route_arr, 50)), 2),
        "full_total_p50_ms": round(float(np.percentile(full_total_arr, 50)), 2),
        "route_fraction":    round(float(np.mean(full_route_arr)) / float(np.mean(full_total_arr)), 3),
        "replay_exact_p50_ms": round(float(np.percentile(replay_arr, 50)), 2),
        "replay_exact_p95_ms": round(float(np.percentile(replay_arr, 95)), 2),
        "recall_frozen_replay": round(recall_replay, 4),
        "speedup_if_routing_free": round(speedup_potential, 2),
    }


# ============================================================================
# EXP 3: Exact-Shape Microbench
# ============================================================================

def _make_uniform_chunks(
    n_docs: int, tokens_per_doc: int, dim: int, rng: np.random.RandomState,
) -> List[Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]]:
    """Build a single shard_chunk where every doc has the same token count."""
    total = n_docs * tokens_per_doc
    emb   = torch.from_numpy(rng.randn(total, dim).astype(np.float16))
    offsets = [(i * tokens_per_doc, (i + 1) * tokens_per_doc) for i in range(n_docs)]
    ids     = list(range(n_docs))
    return [(emb, offsets, ids)]


def _make_variable_chunks(
    n_docs: int, token_budget: int, dim: int, rng: np.random.RandomState,
    lengths: Optional[List[int]] = None,
) -> List[Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]]:
    """Build a shard_chunk with variable-length docs (same total token budget)."""
    if lengths is None:
        # Draw lengths uniformly in [32, token_budget * 2 // n_docs]
        max_len = max(64, token_budget * 2 // n_docs)
        lengths = [int(rng.randint(32, max_len + 1)) for _ in range(n_docs)]
    emb_pieces = [rng.randn(l, dim).astype(np.float16) for l in lengths]
    flat = torch.from_numpy(np.concatenate(emb_pieces, axis=0))
    offsets: List[Tuple[int, int]] = []
    pos = 0
    for l in lengths:
        offsets.append((pos, pos + l))
        pos += l
    ids = list(range(n_docs))
    return [(flat, offsets, ids)]


def exp3_exact_shape_microbench(
    device_str: str,
    n_docs: int = 2000,
    dim: int = 128,
    tokens_per_doc: int = 128,
    n_reps: int = 50,
    n_warmup: int = 10,
) -> dict:
    log.info("=== EXP 3: Exact-Shape Microbench ===")
    dev = torch.device(device_str if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(42)

    warmup_maxsim(dim=dim, doc_token_counts=[tokens_per_doc], device=str(dev))

    q_np = rng.randn(32, dim).astype(np.float32)
    q    = torch.from_numpy(q_np).float()

    token_budget = n_docs * tokens_per_doc
    uniform_chunks  = _make_uniform_chunks(n_docs, tokens_per_doc, dim, rng)
    variable_chunks = _make_variable_chunks(n_docs, token_budget, dim, rng)

    # Warmup
    for _ in range(n_warmup):
        score_all_docs_topk(q, uniform_chunks, k=10, device=dev)
        score_all_docs_topk(
            q, variable_chunks, k=10, device=dev, variable_length_strategy="padded",
        )
        score_all_docs_topk(
            q, variable_chunks, k=10, device=dev, variable_length_strategy="bucketed",
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Uniform timing
    uni_times = []
    for _ in range(n_reps):
        with Timer(sync_cuda=True) as t:
            score_all_docs_topk(q, uniform_chunks, k=10, device=dev)
        uni_times.append(t.elapsed_ms)

    # Legacy padded timing
    padded_times = []
    for _ in range(n_reps):
        with Timer(sync_cuda=True) as t:
            score_all_docs_topk(
                q, variable_chunks, k=10, device=dev, variable_length_strategy="padded",
            )
        padded_times.append(t.elapsed_ms)

    # Current bucketed timing
    bucketed_times = []
    for _ in range(n_reps):
        with Timer(sync_cuda=True) as t:
            score_all_docs_topk(
                q, variable_chunks, k=10, device=dev, variable_length_strategy="bucketed",
            )
        bucketed_times.append(t.elapsed_ms)

    uni_arr = np.array(uni_times)
    padded_arr = np.array(padded_times)
    bucketed_arr = np.array(bucketed_times)
    padded_over_bucketed = float(np.mean(padded_arr)) / max(float(np.mean(bucketed_arr)), 0.01)
    bucketed_over_uniform = float(np.mean(bucketed_arr)) / max(float(np.mean(uni_arr)), 0.01)

    log.info(
        "Uniform  path: mean=%.2fms  p50=%.2fms  p95=%.2fms  (is_uniform=True, no pad)",
        np.mean(uni_arr), np.percentile(uni_arr, 50), np.percentile(uni_arr, 95),
    )
    log.info(
        "Padded   path: mean=%.2fms  p50=%.2fms  p95=%.2fms  (legacy NumPy pad fallback)",
        np.mean(padded_arr), np.percentile(padded_arr, 50), np.percentile(padded_arr, 95),
    )
    log.info(
        "Bucketed path: mean=%.2fms  p50=%.2fms  p95=%.2fms  (current packed fallback)",
        np.mean(bucketed_arr), np.percentile(bucketed_arr, 50), np.percentile(bucketed_arr, 95),
    )
    log.info(
        "Padded/Bucketed ratio = %.2fx  Bucketed/Uniform ratio = %.2fx",
        padded_over_bucketed,
        bucketed_over_uniform,
    )

    return {
        "n_docs":           n_docs,
        "tokens_per_doc":   tokens_per_doc,
        "token_budget":     token_budget,
        "device":           str(dev),
        "uniform_mean_ms":  round(float(np.mean(uni_arr)),  2),
        "uniform_p50_ms":   round(float(np.percentile(uni_arr, 50)), 2),
        "uniform_p95_ms":   round(float(np.percentile(uni_arr, 95)), 2),
        "padded_mean_ms":   round(float(np.mean(padded_arr)),  2),
        "padded_p50_ms":    round(float(np.percentile(padded_arr, 50)), 2),
        "padded_p95_ms":    round(float(np.percentile(padded_arr, 95)), 2),
        "bucketed_mean_ms": round(float(np.mean(bucketed_arr)),  2),
        "bucketed_p50_ms":  round(float(np.percentile(bucketed_arr, 50)), 2),
        "bucketed_p95_ms":  round(float(np.percentile(bucketed_arr, 95)), 2),
        "padded_over_bucketed_ratio": round(padded_over_bucketed, 2),
        "bucketed_over_uniform_ratio": round(bucketed_over_uniform, 2),
        "passes_threshold": padded_over_bucketed >= 1.25,
    }


# ============================================================================
# EXP 4: Cache and Residency Upper-Bound
# ============================================================================

def _drop_page_cache_best_effort() -> None:
    """Best-effort OS page-cache drop (Linux only, requires sudo or vm.drop_caches)."""
    try:
        import subprocess
        subprocess.run(
            ["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            check=True, capture_output=True,
        )
        log.info("Page cache dropped successfully.")
    except Exception as e:
        log.warning("Could not drop page cache (need root): %s. Cold run is warm-cache instead.", e)


def _measure_fetch_ms(
    pipeline: FetchPipeline,
    queries_by_shard: List[Dict[int, List[int]]],
    max_docs: int,
    label: str,
) -> List[float]:
    """Run fetch_candidate_docs() for each query's routed shard groups and record fetch_ms."""
    fetch_times = []
    for docs_by_shard in queries_by_shard:
        _, stats = pipeline.fetch_candidate_docs(docs_by_shard, max_docs=max_docs)
        fetch_times.append(float(stats.get("fetch_ms", 0.0)))
    return fetch_times


def exp4_cache_residency(
    mgr: ShardSegmentManager,
    queries: List[np.ndarray],
    gts: List[List[int]],
) -> dict:
    log.info("=== EXP 4: Cache and Residency Upper-Bound ===")

    store = mgr._store
    if store is None:
        log.warning("No ShardStore available — skipping Exp 4.")
        return {"skipped": True, "reason": "no_store"}

    pipeline = mgr._pipeline
    if pipeline is None:
        pipeline = FetchPipeline(
            store=store,
            mode=mgr._config.transfer_mode,
            pinned_pool=None,
            device=DEVICE,
        )

    # Step A: collect routed shard groups for each query (no scoring)
    for q in queries[:N_WARMUP]:
        mgr.inspect_query_pipeline(q, k=K)

    shard_groups: List[Dict[int, List[int]]] = []
    for q in queries[:N_EVAL]:
        trace = mgr.inspect_query_pipeline(q, k=K)
        cands = trace["exact_candidate_ids"]
        groups = mgr._group_candidate_ids_by_shard(cands)
        shard_groups.append(groups)

    # Step B: Warm-cache fetch (current OS page cache is warm after the above runs)
    warm_times = _measure_fetch_ms(pipeline, shard_groups, MAX_EXACT, "warm")
    warm_arr = np.array(warm_times)

    # Step C: Cold-cache fetch (best effort; may not be truly cold without root)
    _drop_page_cache_best_effort()
    cold_times = _measure_fetch_ms(pipeline, shard_groups, MAX_EXACT, "cold")
    cold_arr = np.array(cold_times)

    # Warm again for a second warm measurement (confirms reproducibility)
    warm2_times = _measure_fetch_ms(pipeline, shard_groups, MAX_EXACT, "warm2")
    warm2_arr = np.array(warm2_times)

    # Step D: Candidate-cache upper bound – build a tiny in-memory dict
    # keyed by doc_id -> tensor slice, then "fetch" from it (zero I/O cost)
    candidate_ids_all = set()
    for groups in shard_groups:
        for _, dids in groups.items():
            candidate_ids_all.update(dids)

    log.info("Building candidate cache for %d unique IDs…", len(candidate_ids_all))
    t0 = time.perf_counter()
    candidate_cache: Dict[int, torch.Tensor] = {}
    if store._doc_index:
        doc_ids_by_shard: Dict[int, List[int]] = {}
        for did in candidate_ids_all:
            dmeta = store._doc_index.get(int(did))
            if dmeta is not None:
                doc_ids_by_shard.setdefault(dmeta.shard_id, []).append(int(did))
        for sid, dids in doc_ids_by_shard.items():
            try:
                emb, offsets, loaded_ids = store.load_docs_from_shard(sid, dids)
                for lid, (s, e) in zip(loaded_ids, offsets):
                    candidate_cache[lid] = emb[s:e]
            except Exception as exc:
                log.warning("Cache load failed for shard %d: %s", sid, exc)
    cache_build_s = time.perf_counter() - t0
    log.info("Candidate cache built in %.1fs for %d IDs", cache_build_s, len(candidate_cache))

    # "Fetch" from cache
    cache_replay_ms: List[float] = []
    for groups in shard_groups:
        cand_ids = [did for dids in groups.values() for did in dids]
        with Timer() as t:
            _ = [candidate_cache.get(did) for did in cand_ids]
        cache_replay_ms.append(t.elapsed_ms)
    cache_arr = np.array(cache_replay_ms)

    log.info(
        "Fetch warm:   p50=%.2fms  p95=%.2fms",
        np.percentile(warm_arr, 50), np.percentile(warm_arr, 95),
    )
    log.info(
        "Fetch cold:   p50=%.2fms  p95=%.2fms",
        np.percentile(cold_arr, 50), np.percentile(cold_arr, 95),
    )
    log.info(
        "Candidate cache (zero I/O upper bound): p50=%.2fms  p95=%.2fms",
        np.percentile(cache_arr, 50), np.percentile(cache_arr, 95),
    )

    cold_warm_ratio = float(np.mean(cold_arr)) / max(float(np.mean(warm_arr)), 0.01)
    residency_gain  = float(np.mean(warm_arr)) / max(float(np.mean(cache_arr)), 0.01)

    log.info(
        "Cold/warm ratio=%.2fx   warm/cache_floor ratio=%.2fx",
        cold_warm_ratio, residency_gain,
    )

    return {
        "warm_fetch_p50_ms":  round(float(np.percentile(warm_arr, 50)),  2),
        "warm_fetch_p95_ms":  round(float(np.percentile(warm_arr, 95)),  2),
        "cold_fetch_p50_ms":  round(float(np.percentile(cold_arr, 50)),  2),
        "cold_fetch_p95_ms":  round(float(np.percentile(cold_arr, 95)),  2),
        "warm2_fetch_p50_ms": round(float(np.percentile(warm2_arr, 50)), 2),
        "cache_floor_p50_ms": round(float(np.percentile(cache_arr, 50)), 2),
        "cache_floor_p95_ms": round(float(np.percentile(cache_arr, 95)), 2),
        "cold_over_warm_ratio":    round(cold_warm_ratio, 2),
        "warm_over_cache_ratio":   round(residency_gain, 2),
        "cache_build_seconds":     round(cache_build_s, 1),
        "n_cached_ids":            len(candidate_cache),
    }


# ============================================================================
# Final ranking / recommendation
# ============================================================================

def rank_and_recommend(
    baseline_qps: float,
    baseline_p50: float,
    exp1: dict,
    exp2: dict,
    exp3: dict,
    exp4: dict,
) -> None:
    """Print a ranked summary and productization recommendation."""

    print("\n" + "=" * 72)
    print("LATENCY OPPORTUNITY RANKING")
    print("=" * 72)
    print(f"  Stable baseline: QPS={baseline_qps:.1f}  p50={baseline_p50:.1f}ms\n")

    THRESHOLD_QPS_RATIO   = 1.30
    THRESHOLD_P50_REDUCE  = 0.25   # 25% reduction

    opportunities = []

    # --- Routing removal potential (from Exp 2) ---
    if "route_fraction" in exp2:
        route_frac = exp2["route_fraction"]
        speedup_if_free = exp2.get("speedup_if_routing_free", 1.0)
        p50_reduction = 1.0 - (1.0 / max(speedup_if_free, 1e-6))
        opportunities.append({
            "name": "Eliminate / reduce routing cost",
            "expected_speedup": round(speedup_if_free, 2),
            "expected_p50_reduction": round(p50_reduction, 2),
            "passes": speedup_if_free >= THRESHOLD_QPS_RATIO or p50_reduction >= THRESHOLD_P50_REDUCE,
            "evidence": f"route fraction={route_frac:.0%}, theoretical speedup if routing is free={speedup_if_free:.1f}x",
            "implementation": "GPU FAISS routing / ANN batching / lighter LEMUR feature extraction",
        })

    # --- Variable-length exact path (from Exp 3) ---
    if "padded_over_bucketed_ratio" in exp3:
        var_ratio = exp3["padded_over_bucketed_ratio"]
        est_p50_reduction = 1.0 - 1.0 / max(var_ratio, 1e-6)
        opportunities.append({
            "name": "Fix variable-length exact-scoring path (pack/bucket tokens)",
            "expected_speedup": round(var_ratio, 2),
            "expected_p50_reduction": round(est_p50_reduction, 2),
            "passes": var_ratio >= THRESHOLD_QPS_RATIO or est_p50_reduction >= THRESHOLD_P50_REDUCE,
            "evidence": (
                f"padded/bucketed score_all_docs_topk ratio={var_ratio:.2f}x; "
                f"bucketed/uniform={exp3.get('bucketed_over_uniform_ratio', 0.0):.2f}x"
            ),
            "implementation": "Use uniform token-length shards (already built with `uniform_shard_tokens=True`) or add bucket-pad in scorer.py to force is_uniform fast path",
        })

    # --- Fetch/data-movement (from Exp 4) ---
    if "warm_over_cache_ratio" in exp4:
        residency_gain = exp4["warm_over_cache_ratio"]
        warm_p50 = exp4["warm_fetch_p50_ms"]
        fetch_frac = float(exp1.get("baseline", {}).get("fetch_frac", 0.0))
        max_end2end_gain = fetch_frac * (1.0 - 1.0 / max(residency_gain, 1e-6))
        opportunities.append({
            "name": "Reduce fetch overhead (hotset cache / partial residency)",
            "expected_speedup": round(1.0 / max(1.0 - max_end2end_gain, 0.1), 2),
            "expected_p50_reduction": round(max_end2end_gain, 2),
            "passes": max_end2end_gain >= THRESHOLD_P50_REDUCE,
            "evidence": (
                f"warm_fetch_p50={warm_p50:.1f}ms  cold/warm ratio={exp4['cold_over_warm_ratio']:.1f}x  "
                f"warm/cache_floor={residency_gain:.1f}x  fetch_frac_of_total={fetch_frac:.0%}"
            ),
            "implementation": "Partial GPU-resident hotset (top-N most-fetched doc IDs) or candidate embedding cache keyed by doc_id",
        })

    # --- Concurrency ceiling (from Exp 1) ---
    if "concurrency" in exp1 and exp1["concurrency"]:
        single_qps = exp1["baseline"]["qps"]
        max_concurrent_qps = max(r["throughput_qps"] for r in exp1["concurrency"])
        concurrency_gain = max_concurrent_qps / max(single_qps, 1e-6)
        best_worker_row = max(exp1["concurrency"], key=lambda r: r["throughput_qps"])
        opportunities.append({
            "name": "Concurrency / multi-worker throughput",
            "expected_speedup": round(concurrency_gain, 2),
            "expected_p50_reduction": 0.0,
            "passes": concurrency_gain >= THRESHOLD_QPS_RATIO,
            "evidence": (
                f"best_QPS={max_concurrent_qps:.1f} at {best_worker_row['n_workers']} workers "
                f"(single={single_qps:.1f})  gain={concurrency_gain:.2f}x"
            ),
            "implementation": "Process-level multi-worker server (not just threading) to overcome GIL on the Python orchestration layer",
        })

    # Sort: passing ones first, then by speedup
    opportunities.sort(key=lambda o: (-int(o["passes"]), -o["expected_speedup"]))

    for rank, opp in enumerate(opportunities, 1):
        flag = "PASS ✓" if opp["passes"] else "BELOW threshold"
        print(f"  #{rank} [{flag}]  {opp['name']}")
        print(f"       expected speedup:     {opp['expected_speedup']:.2f}x")
        print(f"       expected p50 reduce:  {opp['expected_p50_reduction']:.0%}")
        print(f"       evidence:             {opp['evidence']}")
        print(f"       next implementation:  {opp['implementation']}")
        print()

    passing = [o for o in opportunities if o["passes"]]
    if passing:
        winner = passing[0]
        print("  RECOMMENDATION: Start with →", winner["name"])
        print("    " + winner["implementation"])
    else:
        print("  No single opportunity clears the threshold alone.")
        print("  Consider combining the top two or reducing k_candidates further.")
    print("=" * 72)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    queries = load_queries(N_EVAL + N_WARMUP + 5)
    log.info("Loaded %d queries", len(queries))
    gts = load_ground_truth(N_EVAL)
    log.info("Ground truth: %d entries", len(gts))

    log.info(
        "Config: device=%s  k_candidates=%d  max_docs_exact=%d  workers=%s  exps=%s",
        DEVICE, K_CAND, MAX_EXACT, WORKER_COUNTS, sorted(RUN_EXP),
    )

    mgr = build_manager()

    results = {}

    if "1" in RUN_EXP:
        results["exp1"] = exp1_phase_and_concurrency(mgr, queries, gts)
        gc.collect()

    if "2" in RUN_EXP:
        results["exp2"] = exp2_frozen_candidate_oracle(mgr, queries, gts)
        gc.collect()

    if "3" in RUN_EXP:
        results["exp3"] = exp3_exact_shape_microbench(
            DEVICE,
            n_docs=MAX_EXACT,
            dim=128,
            tokens_per_doc=128,
            n_reps=50,
            n_warmup=N_WARMUP,
        )
        gc.collect()

    if "4" in RUN_EXP:
        results["exp4"] = exp4_cache_residency(mgr, queries, gts)
        gc.collect()

    # Dump raw results
    print("\n" + "=" * 72)
    print("RAW RESULTS")
    print("=" * 72)
    print(json.dumps(results, indent=2))

    # Summary and ranking
    baseline_qps = results.get("exp1", {}).get("baseline", {}).get("qps", 0.0)
    baseline_p50 = results.get("exp1", {}).get("baseline", {}).get("p50_ms", 0.0)
    if baseline_qps == 0.0:
        # run a quick serial eval if exp1 was skipped
        baseline = serial_eval(mgr, queries, gts, "quick_baseline")
        baseline_qps = baseline["qps"]
        baseline_p50 = baseline["p50_ms"]

    rank_and_recommend(
        baseline_qps=baseline_qps,
        baseline_p50=baseline_p50,
        exp1=results.get("exp1", {}),
        exp2=results.get("exp2", {}),
        exp3=results.get("exp3", {}),
        exp4=results.get("exp4", {}),
    )

    mgr.close()


if __name__ == "__main__":
    main()
