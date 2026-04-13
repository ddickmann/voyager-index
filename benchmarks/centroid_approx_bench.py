"""
Centroid-code approximate scoring benchmark.

Compares the full shard engine pipeline with two configurations:
  A) Baseline: LEMUR -> doc-mean proxy -> exact MaxSim
  B) Centroid approx: LEMUR -> Rust centroid-code approx (Rayon) -> exact MaxSim

Measures: QPS, recall@10, latency percentiles.

OOM prevention: does NOT call _load_sealed_vectors / _try_gpu_preload.
The centroid-approx path bypasses GPU preload entirely.
"""
from __future__ import annotations

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
                     default="/root/.cache/shard-bench/index_100000_fp16_proxy_grouped_lemur_uniform")
_parser.add_argument("--npz", type=str, default="/root/.cache/voyager-qa/beir_100k.npz")
_parser.add_argument("--gt", type=str, default="/root/.cache/shard-bench/gt_100000.npz")
_parser.add_argument("--n-eval", type=int, default=100)
_parser.add_argument("--n-warmup", type=int, default=5)
_parser.add_argument("--k", type=int, default=10)
_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
_parser.add_argument("--n-shards", type=int, default=256)
_args = _parser.parse_args()

INDEX_DIR = Path(_args.index_dir)
NPZ_PATH = Path(_args.npz)
GT_CACHE = Path(_args.gt)
N_EVAL = _args.n_eval
N_WARMUP = _args.n_warmup
K = _args.k
DEVICE = _args.device


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


def run_pipeline(
    mgr: ShardSegmentManager,
    queries: List[np.ndarray],
    label: str,
    ground_truth: List[List[int]],
    k: int = K,
) -> dict:
    log.info("--- Pipeline: %s ---", label)

    for q in queries[:N_WARMUP]:
        mgr.search_multivector(q, k=k)

    latencies = []
    all_results = []
    for q in queries[:N_EVAL]:
        t0 = time.perf_counter()
        results = mgr.search_multivector(q, k=k)
        latencies.append((time.perf_counter() - t0) * 1000)
        all_results.append([r[0] for r in results])

    latencies_arr = np.array(latencies)
    recall_10 = recall_at_k(all_results, ground_truth[:N_EVAL], k)
    qps = N_EVAL / (sum(latencies) / 1000)

    stats = {
        "label": label,
        "n_queries": N_EVAL,
        "k": k,
        "recall@10": round(recall_10, 4),
        "qps": round(qps, 1),
        "p50_ms": round(float(np.percentile(latencies_arr, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies_arr, 99)), 2),
        "mean_ms": round(float(np.mean(latencies_arr)), 2),
    }
    log.info("  recall@10=%.4f  QPS=%.1f  p50=%.1fms  p95=%.1fms  mean=%.1fms",
             stats["recall@10"], stats["qps"], stats["p50_ms"], stats["p95_ms"], stats["mean_ms"])
    return stats


def main():
    queries = load_queries(N_EVAL + N_WARMUP)
    log.info("Loaded %d queries", len(queries))
    ground_truth = load_ground_truth(N_EVAL)
    log.info("Ground truth loaded: %d queries", len(ground_truth))

    all_results = []
    configs = [
        ("baseline_no_approx", 0),
        ("centroid_approx_1024", 1024),
        ("centroid_approx_512", 512),
        ("centroid_approx_256", 256),
    ]

    for label, n_approx in configs:
        cfg = ShardEngineConfig(
            dim=128,
            n_shards=_args.n_shards,
            compression=Compression.FP16,
            layout=StorageLayout.PROXY_GROUPED,
            router_type=RouterType.LEMUR,
            k_candidates=2000,
            n_full_scores=4096,
            n_centroid_approx=n_approx,
            transfer_mode=TransferMode.PINNED,
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
        print(f"  {r['label']:30s}  recall@10={r['recall@10']:.4f}  QPS={r['qps']:6.1f}  p50={r['p50_ms']:6.1f}ms  mean={r['mean_ms']:6.1f}ms")


if __name__ == "__main__":
    main()
