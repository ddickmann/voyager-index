"""Stage-aware benchmark for LEMUR routing and exact scoring paths."""
from __future__ import annotations

from collections import Counter
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voyager_index._internal.inference.shard_engine.manager import (
    ShardEngineConfig,
    ShardSegmentManager,
)
from voyager_index._internal.inference.shard_engine.config import (
    AnnBackend,
    Compression,
    RouterType,
    StorageLayout,
    TransferMode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("--index-dir", type=str,
                     default="/workspace/.cache/shard-bench/index_100000_fp16_proxy_grouped_lemur_uniform")
_parser.add_argument("--npz", type=str, default="/workspace/.cache/voyager-qa/beir_100k.npz")
_parser.add_argument("--gt", type=str, default="/workspace/.cache/shard-bench/gt_100000.npz")
_parser.add_argument("--n-eval", type=int, default=100)
_parser.add_argument("--n-warmup", type=int, default=5)
_parser.add_argument("--k", type=int, default=10)
_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
_parser.add_argument("--n-shards", type=int, default=256)
_parser.add_argument("--k-candidates", type=int, default=2000)
_parser.add_argument("--max-docs-exact", type=int, default=0, help="0 means use k-candidates")
_parser.add_argument("--lemur-search-k-cap", type=int, default=2048, help="0 disables the explicit cap; faiss-gpu still clamps to 2048")
_parser.add_argument("--n-full-scores", type=int, default=4096)
_parser.add_argument("--approx-values", type=int, nargs="*", default=[0, 1024, 512, 256])
_parser.add_argument("--router-device", type=str, default="", help="Override router device independently from exact scoring device")
_parser.add_argument("--ann-backend", type=str, default=AnnBackend.FAISS_FLAT_IP.value, choices=[e.value for e in AnnBackend])
_args = _parser.parse_args()

INDEX_DIR = Path(_args.index_dir)
NPZ_PATH = Path(_args.npz)
GT_CACHE = Path(_args.gt)
N_EVAL = _args.n_eval
N_WARMUP = _args.n_warmup
K = _args.k
DEVICE = _args.device
K_CANDIDATES = _args.k_candidates
MAX_DOCS_EXACT = _args.max_docs_exact or K_CANDIDATES
LEMUR_SEARCH_K_CAP = None if _args.lemur_search_k_cap <= 0 else _args.lemur_search_k_cap
N_FULL_SCORES = _args.n_full_scores
APPROX_VALUES = _args.approx_values or [0]
ROUTER_DEVICE = _args.router_device or None
ANN_BACKEND = AnnBackend(_args.ann_backend)


def load_queries(n: int = 200) -> List[np.ndarray]:
    data = np.load(str(NPZ_PATH), allow_pickle=True)
    q_vecs = data["query_vectors"].astype(np.float32)
    q_offsets = data["query_offsets"]
    queries = []
    for i in range(min(n, len(q_offsets))):
        s, e = int(q_offsets[i, 0]), int(q_offsets[i, 1])
        queries.append(q_vecs[s:e])
    return queries


def load_ground_truth(n: int) -> List[List[int]]:
    if GT_CACHE.exists():
        log.info("Loading cached ground truth from %s", GT_CACHE)
        data = np.load(str(GT_CACHE), allow_pickle=True)
        return [row[row >= 0].tolist() for row in data["gt_ids"][:n]]
    raise FileNotFoundError(f"Ground truth not found at {GT_CACHE}.")


def recall_at_k(predicted: List[List[int]], ground_truth: List[List[int]], k: int) -> float:
    recalls = []
    for pred, gt in zip(predicted, ground_truth):
        pred_set = set(pred[:k])
        gt_set = set(gt[:k])
        if gt_set:
            recalls.append(len(pred_set & gt_set) / len(gt_set))
    return float(np.mean(recalls)) if recalls else 0.0


def candidate_recall_at_k(candidate_ids: List[int], ground_truth: List[int], k: int) -> float:
    gt_set = set(ground_truth[:k])
    if not gt_set:
        return 0.0
    return len(set(candidate_ids) & gt_set) / len(gt_set)


def run_pipeline(
    mgr: ShardSegmentManager,
    queries: List[np.ndarray],
    label: str,
    ground_truth: List[List[int]],
    k: int = K,
) -> dict:
    log.info("--- Pipeline: %s ---", label)

    for q in queries[:N_WARMUP]:
        mgr.inspect_query_pipeline(q, k=k)

    latencies = []
    all_results = []
    routed_recalls = []
    pruned_recalls = []
    exact_input_recalls = []
    route_ms = []
    prune_paths: Counter[str] = Counter()
    exact_paths: Counter[str] = Counter()
    actual_router_device = None
    actual_exact_device = None
    for q in queries[:N_EVAL]:
        t0 = time.perf_counter()
        trace = mgr.inspect_query_pipeline(q, k=k)
        latencies.append((time.perf_counter() - t0) * 1000)
        route_ms.append(float(trace["route_ms"]))
        prune_paths.update([trace["prune_path"]])
        exact_paths.update([trace["exact_path"]])
        actual_router_device = trace["router_device"]
        actual_exact_device = trace["exact_device"]
        all_results.append(trace["result_ids"])

        gt = ground_truth[len(all_results) - 1]
        routed_recalls.append(candidate_recall_at_k(trace["routed_ids"], gt, k))
        pruned_recalls.append(candidate_recall_at_k(trace["pruned_ids"], gt, k))
        exact_input_recalls.append(candidate_recall_at_k(trace["exact_candidate_ids"], gt, k))

    latencies_arr = np.array(latencies)
    recall_10 = recall_at_k(all_results, ground_truth[:N_EVAL], k)
    qps = N_EVAL / (sum(latencies) / 1000)

    stats = {
        "label": label,
        "n_queries": N_EVAL,
        "k": k,
        "router_device": actual_router_device or ROUTER_DEVICE or DEVICE,
        "exact_device": actual_exact_device or DEVICE,
        "ann_backend": ANN_BACKEND.value,
        "k_candidates": K_CANDIDATES,
        "max_docs_exact": MAX_DOCS_EXACT,
        "lemur_search_k_cap": LEMUR_SEARCH_K_CAP,
        "routed_recall@10": round(float(np.mean(routed_recalls)), 4),
        "pruned_recall@10": round(float(np.mean(pruned_recalls)), 4),
        "exact_input_recall@10": round(float(np.mean(exact_input_recalls)), 4),
        "final_recall@10": round(recall_10, 4),
        "qps": round(qps, 1),
        "p50_ms": round(float(np.percentile(latencies_arr, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies_arr, 99)), 2),
        "mean_ms": round(float(np.mean(latencies_arr)), 2),
        "route_p50_ms": round(float(np.percentile(np.array(route_ms), 50)), 2),
        "dominant_prune_path": prune_paths.most_common(1)[0][0] if prune_paths else "",
        "dominant_exact_path": exact_paths.most_common(1)[0][0] if exact_paths else "",
    }
    log.info(
        "  routed=%.4f  pruned=%.4f  exact_in=%.4f  final=%.4f  QPS=%.1f  p50=%.1fms  route_p50=%.1fms  path=%s/%s",
        stats["routed_recall@10"],
        stats["pruned_recall@10"],
        stats["exact_input_recall@10"],
        stats["final_recall@10"],
        stats["qps"],
        stats["p50_ms"],
        stats["route_p50_ms"],
        stats["dominant_prune_path"],
        stats["dominant_exact_path"],
    )
    return stats


def main():
    queries = load_queries(N_EVAL + N_WARMUP)
    log.info("Loaded %d queries", len(queries))
    ground_truth = load_ground_truth(N_EVAL)
    log.info("Ground truth loaded: %d queries", len(ground_truth))
    log.info(
        "Benchmark config: device=%s router_device=%s ann=%s k_candidates=%d max_docs_exact=%d search_k_cap=%s",
        DEVICE,
        ROUTER_DEVICE or DEVICE,
        ANN_BACKEND.value,
        K_CANDIDATES,
        MAX_DOCS_EXACT,
        "none" if LEMUR_SEARCH_K_CAP is None else LEMUR_SEARCH_K_CAP,
    )

    all_results = []
    configs = []
    for n_approx in APPROX_VALUES:
        label = "exact_no_approx" if n_approx <= 0 else f"centroid_approx_{n_approx}"
        configs.append((label, max(0, int(n_approx))))

    for label, n_approx in configs:
        cfg = ShardEngineConfig(
            dim=128,
            n_shards=_args.n_shards,
            compression=Compression.FP16,
            layout=StorageLayout.PROXY_GROUPED,
            router_type=RouterType.LEMUR,
            ann_backend=ANN_BACKEND,
            k_candidates=K_CANDIDATES,
            max_docs_exact=MAX_DOCS_EXACT,
            lemur_search_k_cap=LEMUR_SEARCH_K_CAP,
            n_full_scores=N_FULL_SCORES,
            n_centroid_approx=n_approx,
            transfer_mode=TransferMode.PINNED,
            router_device=ROUTER_DEVICE,
        )

        log.info("Loading index: %s (n_centroid_approx=%d)...", label, n_approx)
        mgr = ShardSegmentManager(INDEX_DIR, config=cfg, device=DEVICE)
        if not mgr._is_built:
            mgr.load()

        result = run_pipeline(mgr, queries, label, ground_truth, k=K)
        all_results.append(result)
        print(json.dumps(result, indent=2))

        mgr.close()
        gc.collect()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(
            f"  {r['label']:24s}  routed={r['routed_recall@10']:.4f}  "
            f"pruned={r['pruned_recall@10']:.4f}  exact_in={r['exact_input_recall@10']:.4f}  "
            f"final={r['final_recall@10']:.4f}  QPS={r['qps']:6.1f}  p50={r['p50_ms']:6.1f}ms  "
            f"path={r['dominant_prune_path']}/{r['dominant_exact_path']}"
        )


if __name__ == "__main__":
    main()
